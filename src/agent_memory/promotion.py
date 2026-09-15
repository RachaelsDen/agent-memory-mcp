"""DB-facing promotion pipeline for memory_promote / memory_demote (DESIGN §11).

Promotion is COPY-NEVER-MOVE: one transaction opened by a row lock on the
source, then a new row inserted into the target namespace carrying exactly
the plan's override field map (provenance, promotion reason and timestamp,
seed confidence frozen at graduation, active status, reset usage stats,
inherited confidence and embedding, same claim/because/holds_when/
fails_when). Every unlisted column takes its schema default — the copy runs
on fresh created_at/updated_at/last_evidence_at clocks. Evidence edges are
copied to the same episode ids verbatim (relations and reasons included);
lesson_links are NOT copied — promotion copies provenance, not the
similarity graph. The original row is never written.

Demotion is a TOMBSTONE, never a delete: promotion_status='demoted' +
demoted_at + demotion_reason on the copy, with the row and its evidence
edges retained for audit; retrieval's LessonVisibility hides demoted copies
from every probe, including namespace='global'.
"""

from typing import Any

import psycopg
from psycopg.rows import DictRow

try:  # mcp>=2 renamed FastMCP to MCPServer; keep both import spellings working
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError:  # fallback targets mcp 1.x, which is not installed here
    from mcp.server.fastmcp.exceptions import ToolError  # pyright: ignore[reportMissingImports]

from agent_memory import db

# The lock serializes the read-copy pair against concurrent movers on the
# source row (corroborate/contradict lock the same row); under READ COMMITTED
# the lock wait re-reads, so the copied confidence is the committed value.
PROMOTE_SOURCE_SQL = """
    SELECT id, namespace, claim, because, holds_when, fails_when, confidence,
           embedding
    FROM lessons
    WHERE id = %(id)s
    FOR UPDATE
"""

INSERT_PROMOTED_LESSON_SQL = """
    INSERT INTO lessons (
        namespace, claim, because, holds_when, fails_when, confidence,
        embedding, promoted_from_lesson_id, promotion_reason, promoted_at,
        promotion_seed_confidence, promotion_status
    ) VALUES (
        %(namespace)s, %(claim)s, %(because)s, %(holds_when)s, %(fails_when)s,
        %(confidence)s, %(embedding)s, %(promoted_from_lesson_id)s,
        %(promotion_reason)s, now(), %(promotion_seed_confidence)s, 'active'
    )
    RETURNING id
"""

COPY_EVIDENCE_SQL = """
    INSERT INTO lesson_evidence (lesson_id, episode_id, relation, reason)
    SELECT %(copy_id)s, episode_id, relation, reason
    FROM lesson_evidence
    WHERE lesson_id = %(source_id)s
"""

DEMOTE_SOURCE_SQL = """
    SELECT id, promoted_from_lesson_id
    FROM lessons
    WHERE id = %(id)s
    FOR UPDATE
"""

DEMOTE_SQL = """
    UPDATE lessons
    SET promotion_status = 'demoted', demoted_at = now(),
        demotion_reason = %(reason)s
    WHERE id = %(id)s
"""


def promote(
    *,
    lesson_id: int,
    reason: str,
    target_namespace: str = "global",
) -> dict[str, int]:
    """Copy one lesson into target_namespace; returns {"promoted_lesson_id": id}."""
    connection: psycopg.Connection[DictRow] = db.connect()
    try:
        with connection.transaction():
            source = connection.execute(
                PROMOTE_SOURCE_SQL, {"id": lesson_id}
            ).fetchone()
            if source is None:
                raise ToolError(f"lesson {lesson_id} does not exist")
            if source["namespace"] == target_namespace:
                raise ToolError(
                    f"lesson {lesson_id} is already in namespace "
                    f"{target_namespace!r}; promotion copies into a different "
                    "namespace (copy, never move)"
                )
            copy = connection.execute(
                INSERT_PROMOTED_LESSON_SQL,
                {
                    "namespace": target_namespace,
                    "claim": source["claim"],
                    "because": source["because"],
                    "holds_when": source["holds_when"],
                    "fails_when": source["fails_when"],
                    # both confidence columns inherit the source's value at
                    # promotion time — the ONLY sanctioned confidence write
                    # outside seeding and the evidence moves (F2)
                    "confidence": source["confidence"],
                    "embedding": source["embedding"],
                    "promoted_from_lesson_id": source["id"],
                    "promotion_reason": reason,
                    "promotion_seed_confidence": source["confidence"],
                },
            ).fetchone()
            assert copy is not None  # INSERT ... RETURNING yields one row
            connection.execute(
                COPY_EVIDENCE_SQL,
                {"copy_id": copy["id"], "source_id": lesson_id},
            )
    finally:
        connection.close()
    return {"promoted_lesson_id": int(copy["id"])}


def demote(*, lesson_id: int, reason: str) -> dict[str, Any]:
    """Tombstone one promoted copy; returns {"lesson_id", "promotion_status"}."""
    connection: psycopg.Connection[DictRow] = db.connect()
    try:
        with connection.transaction():
            lesson = connection.execute(
                DEMOTE_SOURCE_SQL, {"id": lesson_id}
            ).fetchone()
            if lesson is None:
                raise ToolError(f"lesson {lesson_id} does not exist")
            if lesson["promoted_from_lesson_id"] is None:
                raise ToolError(
                    f"lesson {lesson_id} is not a promotion; only promoted "
                    "copies can be demoted"
                )
            connection.execute(DEMOTE_SQL, {"id": lesson_id, "reason": reason})
    finally:
        connection.close()
    return {"lesson_id": lesson_id, "promotion_status": "demoted"}
