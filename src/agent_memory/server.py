"""MCP server: the one app every plan tool registers on (DESIGN §4, §9).

Tool errors are MCP error results — the tool raises ``ToolError`` and the SDK
turns that into ``CallToolResult(isError=True)``; a failed call NEVER exits
the process, so the same session keeps serving subsequent requests. Captured
text is screened for credential-like content (DESIGN §11) and rejected with
an error naming the FIELD only — the matched content is never echoed back.

SIZE_OK by plan contract: every plan tool registers on this one app, so the
file grows by one thin wrapper per task; logic lives in the tool modules.
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
from agent_memory.consolidation_tools import (
    consolidate_scan,
    corroborate,
    contradict,
    write_lesson,
)
from agent_memory.embed import load_embedder
from agent_memory.promotion import demote, promote
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
    def memory_consolidate_scan(
        pool: str = "fresh",
        min_cluster_size: int = 2,
        namespace: str | None = None,
    ) -> dict[str, Any]:
        """Scan for consolidation candidates (DESIGN §8 step 1; read-only).

        Age-eligible episodes (older than CONSOLIDATE_MIN_AGE_H) cluster
        greedily by embedding cosine > CLUSTER_COS with the earliest episode
        seeding; only clusters with min_cluster_size or more members return,
        each episode with full fields plus its citing lessons. pool='fresh'
        (default) limits to never-cited backlog episodes; pool='all'
        includes cited ones for abstraction passes. Every PENDING disputed
        lesson additionally returns a rederivation group holding ALL its
        source episodes regardless of pool, min_cluster_size, or age — the
        host re-derives it via memory_write_lesson with replaces_disputed.
        """
        return consolidate_scan(
            get_settings(),
            pool=pool,
            min_cluster_size=min_cluster_size,
            namespace=namespace,
        )

    @server.tool()
    def memory_write_lesson(
        claim: str,
        because: str,
        holds_when: str = "",
        fails_when: str = "",
        *,
        evidence: list[dict[str, Any]],
        contradicts: int | None = None,
        replaces_disputed: int | None = None,
        namespace: str | None = None,
    ) -> dict[str, Any]:
        """Store one drafted lesson with its evidence edges (DESIGN §8 step 3).

        Each evidence entry is {"episode_id": int, "relation":
        "support"|"refine"|"contradict", "reason": str}; at least one
        support/refine edge is required. Confidence is seeded SERVER-side
        from incident collapse + day diversity — never caller-supplied.
        Near-duplicate claims (cosine > DUP_CLAIM_COS to an existing lesson
        in the namespace) are rejected unless they supersede a disputed
        lesson via replaces_disputed, which also writes the refines link
        that completes the predecessor's pending re-derivation. Similar
        lessons are linked in both directions; contradicts names an
        opposing lesson (P9). Returns {"lesson_id": int,
        "seed_confidence": float}.
        """
        return write_lesson(
            get_settings(),
            claim=claim,
            because=because,
            holds_when=holds_when,
            fails_when=fails_when,
            evidence=evidence,
            contradicts=contradicts,
            replaces_disputed=replaces_disputed,
            namespace=namespace,
        )

    @server.tool()
    def memory_corroborate(
        lesson_id: int,
        episode_id: int,
        reason: str = "",
        namespace: str | None = None,
    ) -> dict[str, Any]:
        """Record one episode as further support for a lesson (DESIGN §6, §8 step 4).

        Inserts (or flips to) a support evidence edge and moves confidence
        +0.1 x novelty — novelty 1.0 on a new UTC date with a non-duplicate
        embedding vs the lesson's existing support evidence, 0.3 for a
        near-duplicate — capped at 0.95. An edge already in relation support
        is a no-op writing nothing. last_evidence_at refreshes to
        max(current, episode.created_at). namespace is accepted per the
        every-tool contract; the ids alone scope the move. Returns
        {"lesson_id", "episode_id", "relation", "confidence", "applied"}.
        """
        return corroborate(
            get_settings(), lesson_id=lesson_id, episode_id=episode_id, reason=reason
        )

    @server.tool()
    def memory_contradict(
        lesson_id: int,
        episode_id: int,
        reason: str = "",
        namespace: str | None = None,
    ) -> dict[str, Any]:
        """Record one episode as contradicting a lesson (DESIGN §6, §8 step 4).

        Inserts (or flips to) a contradict evidence edge and moves confidence
        -0.2 flat, floored at 0.05. An edge already in relation contradict is
        a no-op writing nothing — a retried contradiction never
        double-penalizes. No lesson-lesson contradicts links are written;
        write_lesson's contradicts param owns those. last_evidence_at
        refreshes to max(current, episode.created_at). namespace is accepted
        per the every-tool contract; the ids alone scope the move. Returns
        {"lesson_id", "episode_id", "relation", "confidence", "applied"}.
        """
        return contradict(
            get_settings(), lesson_id=lesson_id, episode_id=episode_id, reason=reason
        )

    @server.tool()
    def memory_promote(
        lesson_id: int,
        *,
        reason: str,
        target_namespace: str = "global",
        namespace: str | None = None,
    ) -> dict[str, int]:
        """Graduate one lesson to a broader namespace (DESIGN §11; manual only).

        Copies the lesson into target_namespace with promoted_from
        provenance, the human's reason, and promotion_seed_confidence frozen
        at graduation — the copy keeps the same claim/because/holds_when/
        fails_when, the same embedding, and the source's confidence, while
        its usage stats reset to zero. All evidence edges are copied to the
        same episode ids. The original row is untouched (copy, never move);
        promoting into the lesson's own namespace is an error. namespace is
        accepted per the every-tool contract; the lesson_id alone scopes the
        source. Returns {"promoted_lesson_id": int}.
        """
        return promote(
            lesson_id=lesson_id,
            reason=reason,
            target_namespace=target_namespace,
        )

    @server.tool()
    def memory_demote(
        lesson_id: int,
        *,
        reason: str,
        namespace: str | None = None,
    ) -> dict[str, Any]:
        """Retire one promoted copy with a tombstone (DESIGN §11; never a delete).

        Sets promotion_status='demoted' plus demoted_at and the demotion
        reason on the copy; the row and its evidence edges are retained so
        "why did the agent trust X in March?" stays answerable from the
        data. The copy becomes invisible to retrieval from every namespace,
        including 'global'. Only promoted copies (promoted_from_lesson_id
        set) can be demoted; the original lesson is untouched throughout.
        namespace is accepted per the every-tool contract; the lesson_id
        alone scopes the move. Returns {"lesson_id": int,
        "promotion_status": "demoted"}.
        """
        return demote(lesson_id=lesson_id, reason=reason)

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
