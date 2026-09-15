"""DB-facing consolidation scan pipeline for memory_consolidate_scan (DESIGN §8 step 1, §9).

Owns the age-eligible episode pools (``fresh`` = no lesson_evidence citation,
``all`` = every age-eligible episode), the greedy scan clustering, and the
mandatory rederivation groups for PENDING disputed lessons. Consolidation is
derived: this scan is read-only — writing lessons/evidence is task 10/11.

Clustering here is NOT consolidate.collapse_incidents: that is the windowed
near-duplicate collapse for seeding (DEDUP_COS + 24h). The scan clusters on
cosine-to-seed > CLUSTER_COS with NO time window; the cosine helper pattern
is mirrored, not imported (consolidate._cosine is private).

A disputed lesson stays PENDING until some replacement cites it via
``replaces_disputed`` — concretely, until a lesson_links row of kind
``refines`` points AT it (the completion marker task 10 writes). Pending
rederivation groups ignore pool, min_cluster_size, and the age gate:
ordinary thresholds must never swallow re-derivation (§8 Reconsolidation).
"""

from collections.abc import Sequence
from math import sqrt
from typing import Any

import psycopg
from psycopg.rows import DictRow

try:  # mcp>=2 renamed FastMCP to MCPServer; keep both import spellings working
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError:  # fallback targets mcp 1.x, which is not installed here
    from mcp.server.fastmcp.exceptions import ToolError  # pyright: ignore[reportMissingImports]

from agent_memory import db
from agent_memory.config import Settings

POOLS = ("fresh", "all")

_AGE_ELIGIBLE_SQL = """
    SELECT id, created_at, goal, expectation, action, outcome, surprise,
           state_at_encoding, tags, embedding
    FROM episodes
    WHERE namespace = %(ns)s
      AND created_at < now() - make_interval(hours => %(min_age_h)s)
"""

ALL_POOL_SQL = _AGE_ELIGIBLE_SQL + " ORDER BY created_at ASC, id ASC"

FRESH_POOL_SQL = _AGE_ELIGIBLE_SQL + """
    AND NOT EXISTS (
        SELECT 1 FROM lesson_evidence le WHERE le.episode_id = episodes.id
    )
    ORDER BY created_at ASC, id ASC
"""

PENDING_DISPUTED_SQL = """
    SELECT id, claim, because, holds_when, fails_when
    FROM lessons
    WHERE namespace = %(ns)s
      AND disputed
      AND NOT EXISTS (
          SELECT 1 FROM lesson_links ll
          WHERE ll.related_lesson_id = lessons.id AND ll.kind = 'refines'
      )
    ORDER BY id ASC
"""

REDERIVATION_SOURCES_SQL = """
    SELECT le.lesson_id, e.id, e.created_at, e.goal, e.expectation, e.action,
           e.outcome, e.surprise, e.state_at_encoding, e.tags
    FROM lesson_evidence le
    JOIN episodes e ON e.id = le.episode_id
    WHERE le.lesson_id = ANY(%(lesson_ids)s)
    ORDER BY le.lesson_id ASC, e.created_at ASC, e.id ASC
"""

CITERS_SQL = """
    SELECT episode_id, lesson_id FROM lesson_evidence
    WHERE episode_id = ANY(%(ids)s)
    ORDER BY lesson_id ASC
"""


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sqrt(sum(x * x for x in a))
    norm_b = sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


def _greedy_clusters(rows: list[DictRow], cluster_cos: float) -> list[list[DictRow]]:
    """created_at ASC; earliest unclustered seeds; join on cosine-to-seed > cluster_cos."""
    remaining = sorted(rows, key=lambda row: (row["created_at"], row["id"]))
    groups: list[list[DictRow]] = []
    while remaining:
        seed = remaining[0]
        seed_vector = seed["embedding"].to_list()
        members = [seed]
        deferred: list[DictRow] = []
        for row in remaining[1:]:
            joins = _cosine(row["embedding"].to_list(), seed_vector) > cluster_cos
            if joins:
                members.append(row)
            else:
                deferred.append(row)
        groups.append(members)
        remaining = deferred
    return groups


def _episode_record(row: DictRow, citers: dict[int, list[str]]) -> dict[str, Any]:
    return {
        "id": f"episode:{row['id']}",
        "goal": row["goal"],
        "expectation": row["expectation"],
        "action": row["action"],
        "outcome": row["outcome"],
        "surprise": row["surprise"],
        "state_at_encoding": row["state_at_encoding"],
        "tags": row["tags"],
        "created_at": row["created_at"].isoformat(),
        "cited_by_lessons": citers.get(int(row["id"]), []),
    }


def consolidate_scan(
    settings: Settings,
    *,
    pool: str,
    min_cluster_size: int,
    namespace: str | None,
) -> dict[str, Any]:
    """Read-only consolidation scan; kwargs mirror the tool signature verbatim."""
    if pool not in POOLS:
        raise ToolError(f"invalid pool {pool!r}; expected 'fresh' or 'all'")
    effective_ns = settings.MEMORY_NAMESPACE if namespace is None else namespace
    pool_sql = FRESH_POOL_SQL if pool == "fresh" else ALL_POOL_SQL
    episode_bind = {"ns": effective_ns, "min_age_h": settings.CONSOLIDATE_MIN_AGE_H}

    connection: psycopg.Connection[DictRow] = db.connect()
    try:
        episode_rows = connection.execute(pool_sql, episode_bind).fetchall()
        disputed_rows = connection.execute(
            PENDING_DISPUTED_SQL, {"ns": effective_ns}
        ).fetchall()
        lesson_ids = [int(row["id"]) for row in disputed_rows]
        source_rows = (
            connection.execute(
                REDERIVATION_SOURCES_SQL, {"lesson_ids": lesson_ids}
            ).fetchall()
            if lesson_ids
            else []
        )
        result_ids = [int(row["id"]) for row in episode_rows] + [
            int(row["id"]) for row in source_rows
        ]
        citers: dict[int, list[str]] = {}
        if result_ids:
            for row in connection.execute(CITERS_SQL, {"ids": result_ids}):
                citers.setdefault(int(row["episode_id"]), []).append(
                    f"lesson:{int(row['lesson_id'])}"
                )
    finally:
        connection.close()

    clusters = [
        {"rederivation": False, "episodes": [_episode_record(m, citers) for m in members]}
        for members in _greedy_clusters(episode_rows, settings.CLUSTER_COS)
        if len(members) >= min_cluster_size
    ]

    sources_by_lesson: dict[int, list[DictRow]] = {}
    for row in source_rows:
        sources_by_lesson.setdefault(int(row["lesson_id"]), []).append(row)
    rederivation_groups = [
        {
            "rederivation": True,
            "lesson_id": int(lesson["id"]),
            "claim": lesson["claim"],
            "because": lesson["because"],
            "holds_when": lesson["holds_when"],
            "fails_when": lesson["fails_when"],
            "episodes": [
                _episode_record(row, citers)
                for row in sources_by_lesson.get(int(lesson["id"]), [])
            ],
        }
        for lesson in disputed_rows
    ]
    return {
        "pool": pool,
        "namespace": effective_ns,
        "clusters": clusters,
        "rederivation_groups": rederivation_groups,
    }
