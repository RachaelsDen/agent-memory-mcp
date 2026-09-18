"""DB-facing consolidation pipeline for the consolidate tools (DESIGN §8, §9).

Owns the age-eligible episode pools and greedy scan clustering
(``memory_consolidate_scan``, §8 step 1), the lesson write pipeline
(``memory_write_lesson``, §8 step 3): server-owned seed confidence, the
per-namespace duplicate-claim guard — a claim-identity bar with its
replacement-lineage exemption, composite cosine driving similar links only
(Issue #6) — and the similar/contradicts/refines link writes — and the
evidence moves (``memory_corroborate`` / ``memory_contradict``, §8 step 4):
novelty-scaled confidence transitions executed atomically under the lesson
row lock.

SIZE_OK by plan contract: tasks 9-11 deliberately share this one module
(scan + write + evidence moves); the scan portion is frozen and later
sections append.

Clustering here is NOT consolidate.collapse_incidents: that is the windowed
near-duplicate collapse for seeding (DEDUP_COS + 24h). The scan clusters on
cosine-to-seed > CLUSTER_COS with NO time window; the cosine helper pattern
is mirrored, not imported (consolidate._cosine is private).

A disputed lesson stays PENDING until some replacement cites it via
``replaces_disputed`` — concretely, until a lesson_links row of kind
``refines`` points AT it (the completion marker write_lesson writes). Pending
rederivation groups ignore pool, min_cluster_size, and the age gate:
ordinary thresholds must never swallow re-derivation (§8 Reconsolidation).

Every free-text reason (write_lesson evidence reasons and the corroborate/
contradict move reasons) is screened for credential-like content before
any DB work (Issue #9); a hit is an MCP error naming the field only,
never the matched content.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt
from typing import Any, Literal

import pgvector
import psycopg
from psycopg.rows import DictRow

try:  # mcp>=2 renamed FastMCP to MCPServer; keep both import spellings working
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError:  # fallback targets mcp 1.x, which is not installed here
    from mcp.server.fastmcp.exceptions import ToolError  # pyright: ignore[reportMissingImports]

from agent_memory import db
from agent_memory.config import Settings
from agent_memory.consolidate import (
    Episode,
    collapse_incidents,
    corroborate_delta,
    contradict_delta,
    diversity,
    novelty,
    seed_confidence,
)
from agent_memory.embed import load_embedder
from agent_memory.secrets import screen_secrets
from agent_memory.session_state import resolve_namespace

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
    effective_ns = resolve_namespace(settings, namespace)
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


# --- write pipeline (memory_write_lesson, §8 step 3) ------------------------

# Serialized heal+guard+insert per namespace, ONE lock window: the xact
# lock is taken INSIDE the transaction so two hosts racing near-identical
# claims on the shared DB cannot both pass the guard, and heal -> fetch ->
# guard-eval -> insert all happen under that single lock — an old-server
# writer can never slip a NULL-claim lesson between the heal and the guard
# (the guard's .to_list() would crash on it). The namespace crosses as a
# psycopg bind parameter — written as a bare identifier it would be an
# unbound column reference with no relation in scope and the statement
# would fail.
LESSON_WRITE_LOCK_SQL = """
    SELECT pg_advisory_xact_lock(hashtext('agent-memory-lesson-write:' || %(ns)s))
"""

EVIDENCE_EPISODES_SQL = """
    SELECT id, namespace, created_at, embedding
    FROM episodes
    WHERE id = ANY(%(ids)s)
"""

# Active-only candidates (promotion_status = 'active'): demoted copies are
# invisible to retrieval, so they must not block write_lesson via duplicate-claim
# guard nor receive similar links.
NAMESPACE_LESSONS_SQL = """
    SELECT id, claim, embedding, claim_embedding FROM lessons
    WHERE namespace = %(ns)s AND promotion_status = 'active'
"""

HEAL_CLAIM_EMBEDDING_SQL = """
    UPDATE lessons SET claim_embedding = %(claim_embedding)s WHERE id = %(id)s
"""

REPLACEMENT_TARGET_SQL = """
    SELECT id, disputed, namespace FROM lessons WHERE id = %(id)s
"""

# The replacement lineage: the named disputed predecessor plus every
# same-namespace ancestor reached by following refines links transitively.
# Every member is exempt from the dup guard, so repeated re-derivation works
# (replace disputed A with near-identical B, dispute B, replace B with
# near-identical C: C must not be rejected against retained A).
LINEAGE_SQL = """
    WITH RECURSIVE lineage AS (
        SELECT id FROM lessons WHERE id = %(root)s
        UNION
        SELECT l.id
        FROM lineage prev
        JOIN lesson_links ll ON ll.lesson_id = prev.id AND ll.kind = 'refines'
        JOIN lessons l ON l.id = ll.related_lesson_id AND l.namespace = %(ns)s
    )
    SELECT id FROM lineage
"""

INSERT_LESSON_SQL = """
    INSERT INTO lessons (
        namespace, claim, because, holds_when, fails_when, confidence,
        last_evidence_at, updated_at, embedding, claim_embedding
    ) VALUES (
        %(namespace)s, %(claim)s, %(because)s, %(holds_when)s, %(fails_when)s,
        %(confidence)s, %(last_evidence_at)s, now(), %(embedding)s,
        %(claim_embedding)s
    )
    RETURNING id
"""

INSERT_EVIDENCE_SQL = """
    INSERT INTO lesson_evidence (lesson_id, episode_id, relation, reason)
    VALUES (%(lesson_id)s, %(episode_id)s, %(relation)s, %(reason)s)
"""

INSERT_LINK_SQL = """
    INSERT INTO lesson_links (lesson_id, related_lesson_id, kind, weight)
    VALUES (%(from_id)s, %(to_id)s, %(kind)s, %(weight)s)
"""

_SEEDING_RELATIONS = ("support", "refine")
_EVIDENCE_RELATIONS = ("support", "refine", "contradict")


@dataclass(frozen=True, slots=True)
class _EvidenceEdge:
    """Boundary-parsed evidence item; relation vocabulary is closed."""

    episode_id: int
    relation: Literal["support", "refine", "contradict"]
    reason: str


def _parse_evidence(evidence: list[dict[str, Any]]) -> list[_EvidenceEdge]:
    """Parse the one untrusted input: ids, relation vocabulary, no dup pairs."""
    edges: list[_EvidenceEdge] = []
    seen: set[int] = set()
    for item in evidence:
        episode_id = item.get("episode_id")
        if isinstance(episode_id, bool) or not isinstance(episode_id, int):
            raise ToolError(
                f"evidence entry needs an integer 'episode_id', got {episode_id!r}"
            )
        if episode_id in seen:
            raise ToolError(
                f"episode {episode_id} cited more than once; lesson_evidence "
                "takes one aggregate verdict per (lesson, episode) pair"
            )
        seen.add(episode_id)
        relation = item.get("relation")
        if relation not in _EVIDENCE_RELATIONS:
            raise ToolError(
                f"invalid evidence relation {relation!r}; expected one of "
                f"{', '.join(_EVIDENCE_RELATIONS)}"
            )
        reason = item.get("reason") or ""
        if not isinstance(reason, str):
            raise ToolError(f"evidence reason must be a string, got {reason!r}")
        edges.append(_EvidenceEdge(episode_id, relation, reason))
    if not any(edge.relation in _SEEDING_RELATIONS for edge in edges):
        raise ToolError(
            "evidence must cite at least one support/refine episode; "
            "seed confidence is otherwise undefined"
        )
    return edges


def write_lesson(
    settings: Settings,
    *,
    claim: str,
    because: str,
    holds_when: str = "",
    fails_when: str = "",
    evidence: list[dict[str, Any]],
    contradicts: int | None = None,
    replaces_disputed: int | None = None,
    namespace: str | None = None,
) -> dict[str, Any]:
    """Write one drafted lesson + evidence edges + links; kwargs mirror the tool."""
    edges = _parse_evidence(evidence)
    screen_secrets(reason=[edge.reason for edge in edges])
    effective_ns = resolve_namespace(settings, namespace)
    if effective_ns == "global":
        # DESIGN §11: global lessons are created ONLY by memory_promote's
        # copy-with-provenance insert; a direct write would land a global row
        # with no promoted_from lineage (F4).
        raise ToolError(
            "namespace 'global' is promotion-only: global lessons are created "
            "by memory_promote (copy with provenance, DESIGN §11); write the "
            "lesson in its own namespace and promote it instead"
        )
    # Two embeddings, two jobs (Issue #6): the composite drives retrieval and
    # similar-links; the claim alone drives the duplicate-identity guard.
    composite_vector, claim_vector = load_embedder(settings).embed(
        [f"{claim} {because} {holds_when}", claim]
    )

    connection: psycopg.Connection[DictRow] = db.connect()
    try:
        # ONE lock window (round-4 review): heal -> fetch -> guard-eval ->
        # insert inside a single transaction, so an old-server writer can
        # never insert a NULL-claim lesson between the heal and the guard
        # (the guard's .to_list() would crash on NULL). Rejections are
        # verdicts raised AFTER the block — the heal commits even when the
        # write is rejected. Only a writer inside this same xact lock could
        # re-NULL a row after the post-heal fetch, and the lock serializes
        # namespace writers out.
        reject: str | None = None
        result: dict[str, Any] | None = None
        with connection.transaction():
            connection.execute(LESSON_WRITE_LOCK_SQL, {"ns": effective_ns})

            heal_candidates = connection.execute(
                NAMESPACE_LESSONS_SQL, {"ns": effective_ns}
            ).fetchall()
            unhealed = [
                row for row in heal_candidates if row["claim_embedding"] is None
            ]
            if unhealed:
                healed_vectors = load_embedder(settings).embed(
                    [row["claim"] for row in unhealed]
                )
                for row, vec in zip(unhealed, healed_vectors):
                    connection.execute(
                        HEAL_CLAIM_EMBEDDING_SQL,
                        {
                            "id": row["id"],
                            "claim_embedding": pgvector.Vector(vec),
                        },
                    )

            episode_rows = connection.execute(
                EVIDENCE_EPISODES_SQL,
                {"ids": [edge.episode_id for edge in edges]},
            ).fetchall()
            episodes = {int(row["id"]): row for row in episode_rows}
            missing = sorted(
                edge.episode_id for edge in edges if edge.episode_id not in episodes
            )
            if missing:
                reject = f"evidence cites nonexistent episode ids: {missing}"

            exempt: set[int] = set()
            if reject is None and replaces_disputed is not None:
                target = connection.execute(
                    REPLACEMENT_TARGET_SQL, {"id": replaces_disputed}
                ).fetchone()
                if target is None:
                    reject = (
                        f"replaces_disputed lesson {replaces_disputed} does not exist"
                    )
                elif not target["disputed"]:
                    reject = (
                        f"replaces_disputed lesson {replaces_disputed} is not disputed"
                    )
                elif target["namespace"] != effective_ns:
                    reject = (
                        f"replaces_disputed lesson {replaces_disputed} is in "
                        f"namespace {target['namespace']!r}, not {effective_ns!r}"
                    )
                else:
                    exempt = {
                        int(row["id"])
                        for row in connection.execute(
                            LINEAGE_SQL,
                            {"root": replaces_disputed, "ns": effective_ns},
                        )
                    }

            # Post-heal fetch: every claim_embedding is comparable here, so
            # no .to_list() below can see NULL.
            existing = connection.execute(
                NAMESPACE_LESSONS_SQL, {"ns": effective_ns}
            ).fetchall()
            if reject is None:
                # The rejection bar is claim identity over the now-complete
                # set; the composite cosine feeds similar-links ONLY
                # (Issue #6): identical rationale under a different claim
                # links, never rejects.
                for row in existing:
                    lesson_pk = int(row["id"])
                    if lesson_pk in exempt:
                        continue
                    claim_cosine = _cosine(
                        claim_vector, row["claim_embedding"].to_list()
                    )
                    if claim_cosine > settings.DUP_CLAIM_COS:
                        reject = (
                            f"duplicate claim: claim cosine {claim_cosine:.3f} to "
                            f"lesson {lesson_pk} in namespace {effective_ns!r} "
                            f"exceeds DUP_CLAIM_COS={settings.DUP_CLAIM_COS}; "
                            "supersede it via replaces_disputed on a disputed "
                            "lesson instead"
                        )
                        break

            if reject is None and contradicts is not None:
                opponent = connection.execute(
                    "SELECT id FROM lessons WHERE id = %(id)s", {"id": contradicts}
                ).fetchone()
                if opponent is None:
                    reject = f"contradicts lesson {contradicts} does not exist"

            if reject is None:
                similarities = [
                    (
                        int(row["id"]),
                        _cosine(composite_vector, row["embedding"].to_list()),
                    )
                    for row in existing
                ]
                seeding_rows = [
                    episodes[edge.episode_id]
                    for edge in edges
                    if edge.relation in _SEEDING_RELATIONS
                ]
                incidents = collapse_incidents(
                    [
                        Episode(
                            id=int(row["id"]),
                            namespace=row["namespace"],
                            created_at=row["created_at"],
                            embedding=tuple(row["embedding"].to_list()),
                        )
                        for row in seeding_rows
                    ],
                    settings.DEDUP_COS,
                    settings.DEDUP_WINDOW_H,
                )
                confidence = seed_confidence(
                    len(incidents),
                    diversity([row["created_at"] for row in seeding_rows]),
                )

                lesson_row = connection.execute(
                    INSERT_LESSON_SQL,
                    {
                        "namespace": effective_ns,
                        "claim": claim,
                        "because": because,
                        "holds_when": holds_when,
                        "fails_when": fails_when,
                        "confidence": confidence,
                        "last_evidence_at": max(
                            row["created_at"] for row in episode_rows
                        ),
                        "embedding": pgvector.Vector(composite_vector),
                        "claim_embedding": pgvector.Vector(claim_vector),
                    },
                ).fetchone()
                assert lesson_row is not None  # INSERT ... RETURNING yields one row
                lesson_id = int(lesson_row["id"])
                for edge in edges:
                    connection.execute(
                        INSERT_EVIDENCE_SQL,
                        {
                            "lesson_id": lesson_id,
                            "episode_id": edge.episode_id,
                            "relation": edge.relation,
                            "reason": edge.reason,
                        },
                    )
                for lesson_pk, cosine in similarities:
                    if cosine > settings.SIMILAR_LINK_COS:
                        for from_id, to_id in (
                            (lesson_id, lesson_pk),
                            (lesson_pk, lesson_id),
                        ):
                            connection.execute(
                                INSERT_LINK_SQL,
                                {
                                    "from_id": from_id,
                                    "to_id": to_id,
                                    "kind": "similar",
                                    "weight": cosine,
                                },
                            )
                if contradicts is not None:
                    connection.execute(
                        INSERT_LINK_SQL,
                        {
                            "from_id": lesson_id,
                            "to_id": contradicts,
                            "kind": "contradicts",
                            "weight": 0.5,
                        },
                    )
                if replaces_disputed is not None:
                    connection.execute(
                        INSERT_LINK_SQL,
                        {
                            "from_id": lesson_id,
                            "to_id": replaces_disputed,
                            "kind": "refines",
                            "weight": 0.5,
                        },
                    )
                result = {"lesson_id": lesson_id, "seed_confidence": confidence}

        if reject is not None:
            raise ToolError(reject)
        assert result is not None  # clean verdict => the insert path ran
    finally:
        connection.close()
    return result


# --- evidence-move pipeline (memory_corroborate / memory_contradict, §8 step 4)

# Atomicity: one transaction opened by a row lock on the lesson. SELECT ...
# FOR UPDATE serializes concurrent confidence movers — under READ COMMITTED
# the lock wait re-reads the row, so the second mover computes from the
# first mover's committed confidence (no lost update). The lesson row is
# locked BEFORE any confidence read or write.
LESSON_FOR_UPDATE_SQL = """
    SELECT id, confidence, last_evidence_at
    FROM lessons
    WHERE id = %(id)s
    FOR UPDATE
"""

MOVE_EPISODE_SQL = """
    SELECT id, namespace, created_at, embedding
    FROM episodes
    WHERE id = %(id)s
"""

CURRENT_EDGE_SQL = """
    SELECT relation FROM lesson_evidence
    WHERE lesson_id = %(lesson_id)s AND episode_id = %(episode_id)s
"""

SUPPORT_EPISODES_SQL = """
    SELECT e.id, e.namespace, e.created_at, e.embedding
    FROM lesson_evidence le
    JOIN episodes e ON e.id = le.episode_id
    WHERE le.lesson_id = %(lesson_id)s AND le.relation = 'support'
    ORDER BY e.id ASC
"""

# refine edges are INITIALIZATION-ONLY: the moves below never target refine,
# so refine-targeting transitions (support->refine, contradict->refine) are
# unrepresentable — only write_lesson's creation evidence produces refine.
CONVERT_EDGE_SQL = """
    UPDATE lesson_evidence
    SET relation = %(relation)s, reason = %(reason)s
    WHERE lesson_id = %(lesson_id)s AND episode_id = %(episode_id)s
"""

APPLY_MOVE_SQL = """
    UPDATE lessons
    SET confidence = %(confidence)s,
        last_evidence_at = greatest(last_evidence_at, %(evidence_at)s),
        updated_at = now()
    WHERE id = %(id)s
"""


def _as_episode(row: DictRow) -> Episode:
    return Episode(
        id=int(row["id"]),
        namespace=row["namespace"],
        created_at=row["created_at"],
        embedding=tuple(row["embedding"].to_list()),
    )


def _apply_evidence_move(
    settings: Settings,
    *,
    lesson_id: int,
    episode_id: int,
    relation: Literal["support", "contradict"],
    reason: str,
) -> dict[str, Any]:
    """One evidence move under the lesson row lock. Transition table:

    absent->support, refine->support, contradict->support : corroborate_delta
    absent->contradict, refine->contradict, support->contradict : contradict_delta
    support->support, contradict->contradict : NO-OP (nothing written; a
    retried contradiction must not double-penalize)

    Novelty is computed against the EXISTING support edges before the new
    edge exists (never against itself). Idempotence is CURRENT-STATE-ONLY by
    design: a replayed transition after an intervening change is an
    intentional new move (no operation-identity contract in v1).
    """
    screen_secrets(reason=reason)
    connection: psycopg.Connection[DictRow] = db.connect()
    try:
        with connection.transaction():
            lesson = connection.execute(
                LESSON_FOR_UPDATE_SQL, {"id": lesson_id}
            ).fetchone()
            if lesson is None:
                raise ToolError(f"lesson {lesson_id} does not exist")
            episode = connection.execute(
                MOVE_EPISODE_SQL, {"id": episode_id}
            ).fetchone()
            if episode is None:
                raise ToolError(f"episode {episode_id} does not exist")
            current = connection.execute(
                CURRENT_EDGE_SQL,
                {"lesson_id": lesson_id, "episode_id": episode_id},
            ).fetchone()

            if current is not None and current["relation"] == relation:
                result = {
                    "lesson_id": lesson_id,
                    "episode_id": episode_id,
                    "relation": relation,
                    "confidence": float(lesson["confidence"]),
                    "applied": False,
                }
            else:
                confidence = float(lesson["confidence"])
                if relation == "support":
                    support_rows = connection.execute(
                        SUPPORT_EPISODES_SQL, {"lesson_id": lesson_id}
                    ).fetchall()
                    confidence = corroborate_delta(
                        confidence,
                        novelty(
                            _as_episode(episode),
                            [_as_episode(row) for row in support_rows],
                            settings.DEDUP_COS,
                        ),
                    )
                else:
                    confidence = contradict_delta(confidence)
                if current is None:
                    connection.execute(
                        INSERT_EVIDENCE_SQL,
                        {
                            "lesson_id": lesson_id,
                            "episode_id": episode_id,
                            "relation": relation,
                            "reason": reason,
                        },
                    )
                else:
                    connection.execute(
                        CONVERT_EDGE_SQL,
                        {
                            "lesson_id": lesson_id,
                            "episode_id": episode_id,
                            "relation": relation,
                            "reason": reason,
                        },
                    )
                connection.execute(
                    APPLY_MOVE_SQL,
                    {
                        "id": lesson_id,
                        "confidence": confidence,
                        "evidence_at": episode["created_at"],
                    },
                )
                result = {
                    "lesson_id": lesson_id,
                    "episode_id": episode_id,
                    "relation": relation,
                    "confidence": confidence,
                    "applied": True,
                }
    finally:
        connection.close()
    return result


def corroborate(
    settings: Settings, *, lesson_id: int, episode_id: int, reason: str = ""
) -> dict[str, Any]:
    """Insert/flip one support edge; confidence +0.1 x novelty, cap 0.95."""
    return _apply_evidence_move(
        settings,
        lesson_id=lesson_id,
        episode_id=episode_id,
        relation="support",
        reason=reason,
    )


def contradict(
    settings: Settings, *, lesson_id: int, episode_id: int, reason: str = ""
) -> dict[str, Any]:
    """Insert/flip one contradict edge; confidence -0.2 flat, floor 0.05."""
    return _apply_evidence_move(
        settings,
        lesson_id=lesson_id,
        episode_id=episode_id,
        relation="contradict",
        reason=reason,
    )
