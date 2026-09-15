"""MCP server: the one app every plan tool registers on (DESIGN §4, §9).

Tool errors are MCP error results — the tool raises ``ToolError`` and the SDK
turns that into ``CallToolResult(isError=True)``; a failed call NEVER exits
the process, so the same session keeps serving subsequent requests. Captured
text is screened for credential-like content (DESIGN §11) and rejected with
an error naming the FIELD only — the matched content is never echoed back.
"""

import re
from typing import Any

import pgvector
import psycopg
from psycopg.rows import DictRow
from psycopg.types.json import Jsonb

try:  # mcp>=2 renamed FastMCP to MCPServer; keep both import spellings working
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError:  # fallback targets mcp 1.x, which is not installed here
    from mcp.server.fastmcp import FastMCP as MCPServer  # pyright: ignore[reportAttributeAccessIssue]
    from mcp.server.fastmcp.exceptions import ToolError  # pyright: ignore[reportMissingImports]

from agent_memory import db
from agent_memory.config import get_settings
from agent_memory.embed import load_embedder
from agent_memory.retrieval import run_retrieval
from agent_memory.usage import report_usage

_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{36}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"eyJhbGciOi[A-Za-z0-9._-]{20,}"),
)


def _screen_secrets(**fields: str | list[str] | None) -> None:
    """Reject any text field (or tag entry) carrying a credential pattern."""
    for name, value in fields.items():
        texts: list[str] = [value] if isinstance(value, str) else (value or [])
        if any(pattern.search(text) for text in texts for pattern in _SECRET_PATTERNS):
            raise ToolError(
                f"field {name!r} appears to contain a secret; refusing to store it; redact and retry"
            )


def create_server() -> MCPServer:
    """Server factory; every task's tool registers on this one app."""
    server: MCPServer = MCPServer(name="agent-memory")

    @server.tool()
    def memory_capture_episode(
        goal: str = "",
        expectation: str = "",
        action: str = "",
        outcome: str = "",
        surprise: float = 0.0,
        state_at_encoding: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        raw_text: str = "",
        namespace: str | None = None,
    ) -> dict[str, int]:
        """Record one episode at the moment of surprise; returns {"id": <int>}.

        state_at_encoding is mood/context JSON — stored for audit but NEVER
        embedded or matched (DESIGN P4); the embedding input is exactly
        "goal expectation action outcome".
        """
        settings = get_settings()
        _screen_secrets(
            goal=goal,
            expectation=expectation,
            action=action,
            outcome=outcome,
            raw_text=raw_text,
            tags=tags,
        )
        embedded_text = f"{goal} {expectation} {action} {outcome}"
        vector = load_embedder(settings).embed([embedded_text])[0]
        effective_namespace = (
            settings.MEMORY_NAMESPACE if namespace is None else namespace
        )
        clamped_surprise = max(0.0, min(surprise, 1.0))
        connection: psycopg.Connection[DictRow] = db.connect()
        try:
            row = connection.execute(
                """
                INSERT INTO episodes (
                    namespace, goal, expectation, action, outcome, surprise,
                    state_at_encoding, tags, raw_text, embedding
                ) VALUES (
                    %(namespace)s, %(goal)s, %(expectation)s, %(action)s, %(outcome)s,
                    %(surprise)s, %(state_at_encoding)s, %(tags)s, %(raw_text)s,
                    %(embedding)s
                )
                RETURNING id
                """,
                {
                    "namespace": effective_namespace,
                    "goal": goal,
                    "expectation": expectation,
                    "action": action,
                    "outcome": outcome,
                    "surprise": clamped_surprise,
                    "state_at_encoding": (
                        Jsonb(state_at_encoding) if state_at_encoding is not None else None
                    ),
                    "tags": tags if tags is not None else [],
                    "raw_text": raw_text,
                    "embedding": pgvector.Vector(vector),
                },
            ).fetchone()
        finally:
            connection.close()
        assert row is not None  # INSERT ... RETURNING always yields exactly one row
        return {"id": row["id"]}

    @server.tool()
    def memory_probe(
        current_goal: str,
        approach: str = "",
        k: int | None = None,
        namespace: str | None = None,
    ) -> dict[str, Any]:
        """Recall memories relevant to what the agent is about to do (P7).

        The query is the agent's current intent (goal + approach), not the
        user's words: hybrid keyword+vector RRF ranking, one hop of spreading
        activation over lesson_links, exposure logged to retrieval_events.
        """

        return run_retrieval(
            get_settings(),
            tool="probe",
            query_text=current_goal + " " + approach,
            k=k,
            namespace=namespace,
        )

    @server.tool()
    def memory_search(
        query: str, k: int | None = None, namespace: str | None = None
    ) -> dict[str, Any]:
        """Explicit recall ("what do we know about X"); same shape as probe."""
        return run_retrieval(
            get_settings(), tool="search", query_text=query, k=k, namespace=namespace
        )

    @server.tool()
    def memory_report_usage(
        retrieval_event_id: int,
        results: list[dict[str, Any]],
        namespace: str | None = None,
    ) -> dict[str, Any]:
        """Record explicit verdicts on records a retrieval showed (DESIGN §6).

        Each results entry is {"id": typed ref, "outcome":
        "used"|"helped"|"harmed"|"ignored"}; ids must be among the event's
        returned_ids. The batch is all-or-nothing and each shown record takes
        exactly one verdict. used/helped/harmed bump the record's access
        stats (lessons also move usefulness +-1); ignored logs the report
        only. namespace is accepted for signature uniformity — the
        retrieval_event_id alone scopes the report.
        """
        return report_usage(
            retrieval_event_id=retrieval_event_id, results=results
        )

    return server


def serve() -> None:
    create_server().run()


if __name__ == "__main__":
    serve()
