"""DB-facing oversight pipeline for memory_dispute / memory_stats (DESIGN S9, S10).

Dispute marks a lesson for human-flagged rederivation: ``disputed=true``
plus the human's reason in migration 002's ``lessons.dispute_reason``.
The reason is screened for credential-like content before any DB work
(Issue #9); a hit is an MCP error naming the field only, never the
matched content. The lesson STAYS retrievable — retrieval.py exposes
``"disputed": true`` and the reason on every probe result carrying it —
and the next consolidate_scan queues it as a pending rederivation group.
Re-disputing overwrites the reason (plain UPDATE semantics); a nonexistent
lesson is an MCP error result.

Stats is the namespace health check feeding the digest's audit sections:
two counts, the unconsolidated backlog, and four flag buckets. The health
buckets (popular_but_shaky, rare_critical_stale) cover ACTIVE lessons
only — a demoted copy is already retired, and its audit home is
demoted_promotions, never a second appearance in a health bucket. The
tombstone view lists demoted copies from BOTH the copy's namespace and
the source lesson's namespace (the promotion pair is that namespace's
audit history), and nothing from unrelated namespaces.

The staleness cutoff is the SAME parameterized formula as the probe
staleness note: with env_fresh = exp(-age_h / tau), env_fresh <
STALE_ENV_FRESH exactly when
``last_evidence_at < now() - make_interval(secs => 3600.0 * tau * ln(1 / stale))``
— make_interval takes INTEGER hours, so the fractional threshold MUST
cross via ``secs``. ``0 < STALE_ENV_FRESH < 1`` is validated before ln()
ever runs; a misconfigured setting is an MCP error result, not a NaN.
"""

from typing import Any, LiteralString

import psycopg
from psycopg.rows import DictRow

try:  # mcp>=2 renamed FastMCP to MCPServer; keep both import spellings working
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError:  # fallback targets mcp 1.x, which is not installed here
    from mcp.server.fastmcp.exceptions import ToolError  # pyright: ignore[reportMissingImports]

from agent_memory import db
from agent_memory.config import Settings
from agent_memory.secrets import screen_secrets
from agent_memory.session_state import resolve_namespace

POPULAR_SHAKY_CONF = 0.3
POPULAR_SHAKY_ACCESS = 5

ALL_NAMESPACES_SQL = """
    SELECT DISTINCT namespace FROM episodes
    UNION
    SELECT DISTINCT namespace FROM lessons
    UNION
    SELECT 'global'
        WHERE EXISTS (SELECT 1 FROM episodes) OR EXISTS (SELECT 1 FROM lessons)
    ORDER BY 1
"""

DISPUTE_SQL = """
    UPDATE lessons
    SET disputed = TRUE, dispute_reason = %(reason)s, updated_at = now()
    WHERE id = %(id)s
    RETURNING id
"""

EPISODE_COUNT_SQL = "SELECT count(*) AS n FROM episodes WHERE namespace = %(ns)s"

LESSON_COUNT_SQL = "SELECT count(*) AS n FROM lessons WHERE namespace = %(ns)s"

BACKLOG_SQL = """
    SELECT count(*) AS n
    FROM episodes e
    WHERE e.namespace = %(ns)s
      AND NOT EXISTS (SELECT 1 FROM lesson_evidence le WHERE le.episode_id = e.id)
"""

POPULAR_SHAKY_SQL = """
    SELECT id, claim, confidence, access_count
    FROM lessons
    WHERE namespace = %(ns)s
      AND promotion_status = 'active'
      AND confidence < %(shaky_conf)s
      AND access_count > %(shaky_access)s
    ORDER BY id
"""

RARE_CRITICAL_STALE_SQL = """
    SELECT id, claim, confidence, last_evidence_at
    FROM lessons
    WHERE namespace = %(ns)s
      AND promotion_status = 'active'
      AND confidence > %(salience_stale)s
      AND last_evidence_at < now()
          - make_interval(secs => 3600.0 * %(tau_env)s * ln(1.0 / %(stale_env)s))
    ORDER BY id
"""

CROSS_CUTTING_SQL = """
    SELECT e.id, array_agg(DISTINCT le.lesson_id) AS citers
    FROM episodes e
    JOIN lesson_evidence le ON le.episode_id = e.id
    WHERE e.namespace = %(ns)s
    GROUP BY e.id
    HAVING count(DISTINCT le.lesson_id) >= 2
    ORDER BY e.id
"""

DEMOTED_PROMOTIONS_SQL = """
    SELECT c.id, c.claim, c.namespace, c.promoted_from_lesson_id,
           c.promotion_reason, c.promoted_at, c.demotion_reason, c.demoted_at
    FROM lessons c
    LEFT JOIN lessons s ON s.id = c.promoted_from_lesson_id
    WHERE c.promotion_status = 'demoted'
      AND (c.namespace = %(ns)s OR s.namespace = %(ns)s)
    ORDER BY c.id
"""


def all_namespaces() -> list[str]:
    """Every namespace holding episodes or lessons, plus the literal 'global'
    whenever any memory exists at all, sorted.

    'global' never depends on holding rows of its own — it enumerates even
    while momentarily empty so its audit history keeps digesting — while a
    database with no episodes and no lessons anywhere enumerates zero
    namespaces: a fresh install's cron renders no files, not an empty global
    one.
    """
    connection: psycopg.Connection[DictRow] = db.connect()
    try:
        rows = connection.execute(ALL_NAMESPACES_SQL).fetchall()
    finally:
        connection.close()
    return [str(row["namespace"]) for row in rows]


def dispute(*, lesson_id: int, reason: str) -> dict[str, Any]:
    """Mark one lesson disputed with the human's reason; id-scoped like the
    other id-addressed mutations (the every-tool namespace param is inert)."""
    screen_secrets(reason=reason)
    connection: psycopg.Connection[DictRow] = db.connect()
    try:
        row = connection.execute(
            DISPUTE_SQL, {"id": lesson_id, "reason": reason}
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise ToolError(f"lesson {lesson_id} does not exist")
    return {"lesson_id": lesson_id, "disputed": True}


def _count(
    connection: psycopg.Connection[DictRow], query: LiteralString, bind: dict[str, Any]
) -> int:
    row = connection.execute(query, bind).fetchone()
    assert row is not None  # count(*) always returns exactly one row
    return int(row["n"])


def stats(settings: Settings, *, namespace: str | None = None) -> dict[str, Any]:
    """Namespace health check; None namespace -> resolved default."""
    if not 0.0 < settings.STALE_ENV_FRESH < 1.0:
        raise ToolError(
            f"STALE_ENV_FRESH must satisfy 0 < value < 1, got "
            f"{settings.STALE_ENV_FRESH!r}"
        )
    effective_ns = resolve_namespace(settings, namespace)
    bind = {"ns": effective_ns}
    staleness_bind = {"tau_env": settings.TAU_ENV_H, "stale_env": settings.STALE_ENV_FRESH}

    connection: psycopg.Connection[DictRow] = db.connect()
    try:
        shaky_rows = connection.execute(
            POPULAR_SHAKY_SQL,
            {**bind, "shaky_conf": POPULAR_SHAKY_CONF, "shaky_access": POPULAR_SHAKY_ACCESS},
        ).fetchall()
        stale_rows = connection.execute(
            RARE_CRITICAL_STALE_SQL,
            {**bind, "salience_stale": settings.SALIENCE_STALE, **staleness_bind},
        ).fetchall()
        cross_rows = connection.execute(CROSS_CUTTING_SQL, bind).fetchall()
        demoted_rows = connection.execute(DEMOTED_PROMOTIONS_SQL, bind).fetchall()
        result = {
            "namespace": effective_ns,
            "episode_count": _count(connection, EPISODE_COUNT_SQL, bind),
            "lesson_count": _count(connection, LESSON_COUNT_SQL, bind),
            "unconsolidated_backlog": _count(connection, BACKLOG_SQL, bind),
            "popular_but_shaky": [
                {
                    "id": f"lesson:{row['id']}",
                    "claim": row["claim"],
                    "confidence": row["confidence"],
                    "access_count": row["access_count"],
                }
                for row in shaky_rows
            ],
            "rare_critical_stale": [
                {
                    "id": f"lesson:{row['id']}",
                    "claim": row["claim"],
                    "confidence": row["confidence"],
                    "last_evidence_at": row["last_evidence_at"].isoformat(),
                }
                for row in stale_rows
            ],
            "cross_cutting_episodes": [
                {
                    "id": f"episode:{row['id']}",
                    "citing_lessons": [f"lesson:{id}" for id in sorted(row["citers"])],
                }
                for row in cross_rows
            ],
            "demoted_promotions": [
                {
                    "id": f"lesson:{row['id']}",
                    "claim": row["claim"],
                    "namespace": row["namespace"],
                    "promoted_from_lesson_id": row["promoted_from_lesson_id"],
                    "promotion_reason": row["promotion_reason"],
                    "promoted_at": row["promoted_at"].isoformat(),
                    "demotion_reason": row["demotion_reason"],
                    "demoted_at": row["demoted_at"].isoformat(),
                }
                for row in demoted_rows
            ],
        }
    finally:
        connection.close()
    return result
