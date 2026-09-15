"""DB-facing usage-report pipeline for memory_report_usage (DESIGN §6, §7).

Owns verdict validation, the all-or-nothing usage_reports write, and the
access/usefulness stat moves on used records. Plain INSERTs only: a
duplicate verdict surfaces as a uniqueness violation converted to an MCP
error result — upsert semantics are forbidden (one verdict per shown
record, forever). The tool's namespace parameter exists for signature
uniformity (DESIGN §9); the retrieval_event_id alone scopes the report, so
no namespace value is consulted here.
"""

from typing import Any

import psycopg
from psycopg.rows import DictRow

try:  # mcp>=2 renamed FastMCP to MCPServer; keep both import spellings working
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError:  # fallback targets mcp 1.x, which is not installed here
    from mcp.server.fastmcp.exceptions import ToolError  # pyright: ignore[reportMissingImports]

from agent_memory import db

OUTCOMES = ("used", "helped", "harmed", "ignored")

_INSERT_REPORT_SQL = """
    INSERT INTO usage_reports (retrieval_event_id, record_id, record_type, outcome)
    VALUES (%(event_id)s, %(record_id)s, %(record_type)s, %(outcome)s)
"""
_UPDATE_EPISODE_SQL = """
    UPDATE episodes
    SET access_count = access_count + 1, last_accessed = now()
    WHERE id = %(record_pk)s
"""
_UPDATE_LESSON_SQL = """
    UPDATE lessons
    SET access_count = access_count + 1, last_accessed = now(),
        usefulness = usefulness + %(delta)s
    WHERE id = %(record_pk)s
"""
_USEFULNESS_DELTA = {"used": 0, "helped": 1, "harmed": -1}


def _validate_results(results: list[dict[str, Any]]) -> None:
    """Boundary parse of the one untrusted input: id + outcome vocabulary."""
    for entry in results:
        record_id = entry.get("id")
        if not isinstance(record_id, str):
            raise ToolError(f"verdict entry is missing a string 'id': {entry!r}")
        if entry.get("outcome") not in OUTCOMES:
            raise ToolError(
                f"invalid outcome {entry.get('outcome')!r} for record "
                f"{record_id!r}; expected one of {', '.join(OUTCOMES)}"
            )


def report_usage(
    *,
    retrieval_event_id: int,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Write one all-or-nothing batch of verdicts; returns {"reported": n}."""
    _validate_results(results)
    connection: psycopg.Connection[DictRow] = db.connect()
    try:
        try:
            with connection.transaction():
                event = connection.execute(
                    """
                    SELECT returned_ids FROM retrieval_events
                    WHERE id = %(event_id)s
                    """,
                    {"event_id": retrieval_event_id},
                ).fetchone()
                if event is None:
                    raise ToolError(
                        f"retrieval_event {retrieval_event_id} does not exist"
                    )
                shown = list(event["returned_ids"])
                for entry in results:
                    record_id = entry["id"]
                    if record_id not in shown:
                        raise ToolError(
                            f"record {record_id!r} was not among the results "
                            f"of retrieval_event {retrieval_event_id} "
                            f"(shown: {shown})"
                        )
                    record_type, _, raw_pk = record_id.partition(":")
                    connection.execute(
                        _INSERT_REPORT_SQL,
                        {
                            "event_id": retrieval_event_id,
                            "record_id": record_id,
                            "record_type": record_type,
                            "outcome": entry["outcome"],
                        },
                    )
                    if entry["outcome"] == "ignored":
                        continue  # report row only; stats untouched by design
                    record_pk = int(raw_pk)
                    if record_type == "episode":
                        connection.execute(
                            _UPDATE_EPISODE_SQL, {"record_pk": record_pk}
                        )
                    else:
                        connection.execute(
                            _UPDATE_LESSON_SQL,
                            {
                                "record_pk": record_pk,
                                "delta": _USEFULNESS_DELTA[entry["outcome"]],
                            },
                        )
        except psycopg.errors.UniqueViolation:
            raise ToolError(
                "duplicate verdict: each shown record takes exactly one "
                "report per retrieval_event"
            ) from None
    finally:
        connection.close()
    return {"reported": len(results)}
