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
similarity graph. The original row is never written. Both reasons are
screened for credential-like content before any DB work (Issue #9); a
hit is an MCP error naming the field only, never the matched content.

Promotion also holds the TARGET namespace's claim-identity bar: under the
same per-namespace advisory xact lock write_lesson takes, the source's
claim vector is compared against every target claim embedding — NULL
target embeddings (old-server promotions) are healed in place under that
same lock before the comparison, so a NULL seat cannot let a duplicate
graduation slip past — and a cosine above DUP_CLAIM_COS rejects the
graduation: a claim already graduated cannot be duplicated; corroborate
the existing lesson instead. No lineage exemption applies: a promotion
copy is a new-namespace row, not a re-derivation. A source whose
claim_embedding is NULL (rolling upgrade / interrupted 003 backfill) is
healed at copy time — the vector computed through
load_embedder(get_settings()) (the serve dim invariant) is stored on BOTH
the source row and the copy inside the one transaction.

Demotion is a TOMBSTONE, never a delete: promotion_status='demoted' +
demoted_at + demotion_reason on the copy, with the row and its evidence
edges retained for audit; retrieval's LessonVisibility hides demoted copies
from every probe, including namespace='global'.
"""

from collections.abc import Sequence
from math import sqrt
from typing import Any

import pgvector
import psycopg
from psycopg.rows import DictRow

try:  # mcp>=2 renamed FastMCP to MCPServer; keep both import spellings working
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError:  # fallback targets mcp 1.x, which is not installed here
    from mcp.server.fastmcp.exceptions import ToolError  # pyright: ignore[reportMissingImports]

from agent_memory import db
from agent_memory.config import get_settings
from agent_memory.consolidation_tools import LESSON_WRITE_LOCK_SQL
from agent_memory.embed import load_embedder
from agent_memory.secrets import screen_secrets

# Mirrored, not imported (consolidation_tools._cosine is private): the
# claim-identity bar here and write_lesson's there must compare identically.
def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sqrt(sum(x * x for x in a))
    norm_b = sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


# The lock serializes the read-copy pair against concurrent movers on the
# source row (corroborate/contradict lock the same row); under READ COMMITTED
# the lock wait re-reads, so the copied confidence is the committed value.
PROMOTE_SOURCE_SQL = """
    SELECT id, namespace, claim, because, holds_when, fails_when, confidence,
           embedding, claim_embedding
    FROM lessons
    WHERE id = %(id)s
    FOR UPDATE
"""

INSERT_PROMOTED_LESSON_SQL = """
    INSERT INTO lessons (
        namespace, claim, because, holds_when, fails_when, confidence,
        embedding, claim_embedding, promoted_from_lesson_id, promotion_reason, promoted_at,
        promotion_seed_confidence, promotion_status
    ) VALUES (
        %(namespace)s, %(claim)s, %(because)s, %(holds_when)s, %(fails_when)s,
        %(confidence)s, %(embedding)s, %(claim_embedding)s, %(promoted_from_lesson_id)s,
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

# Backfill-repair semantics: only the vector column moves (updated_at stays,
# matching db._backfill_claim_embeddings — a storage repair, not an edit).
# Used for both the SOURCE heal and the target-namespace heals below.
HEAL_CLAIM_EMBEDDING_SQL = """
    UPDATE lessons SET claim_embedding = %(claim_embedding)s WHERE id = %(id)s
"""

# Target ACTIVE rows INCLUDING NULL claim embeddings: an old-server promotion
# creates NULL global copies, and a NOT NULL filter here would let a new
# identical-claim graduation slip past the bar. Demoted tombstones are excluded
# (promotion_status = 'active') because a demoted copy does not block re-promotion.
# The NULLs are healed under the already-held target advisory lock before the comparison.
TARGET_LESSONS_SQL = """
    SELECT id, claim, claim_embedding FROM lessons
    WHERE namespace = %(ns)s AND promotion_status = 'active'
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
    """Copy one lesson into target_namespace; returns {"promoted_lesson_id": id}.

    Holds the target namespace's claim-identity bar (duplicate graduations
    are rejected; NULL target claim embeddings are healed under the target
    lock before the comparison) and heals a NULL source claim_embedding in
    the same transaction — copy and source both leave with a claim vector.
    """
    screen_secrets(reason=reason)
    settings = get_settings()
    connection: psycopg.Connection[DictRow] = db.connect()
    try:
        with connection.transaction():
            source_info = connection.execute(
                "SELECT namespace FROM lessons WHERE id = %(id)s", {"id": lesson_id}
            ).fetchone()
            if source_info is None:
                raise ToolError(f"lesson {lesson_id} does not exist")
            source_ns = source_info["namespace"]
            if source_ns == target_namespace:
                raise ToolError(
                    f"lesson {lesson_id} is already in namespace "
                    f"{target_namespace!r}; promotion copies into a different "
                    "namespace (copy, never move)"
                )
            # LESSON_WRITE_LOCK_SQL is imported, not mirrored: the lock key
            # must stay byte-identical to write_lesson's or promotion stops
            # serializing against writes. Deterministic lock ordering on BOTH
            # namespaces in lexicographical order prevents deadlock cycles
            # between concurrent opposite-direction promotions (e.g. A->B vs B->A).
            for ns in sorted(set([source_ns, target_namespace])):
                connection.execute(LESSON_WRITE_LOCK_SQL, {"ns": ns})

            source = connection.execute(
                PROMOTE_SOURCE_SQL, {"id": lesson_id}
            ).fetchone()
            if source is None:
                raise ToolError(f"lesson {lesson_id} does not exist")

            # A NULL source claim_embedding (interrupted 003 backfill) would
            # copy guard-blind; compute it through the serve embedder and
            # heal the SOURCE row so both rows leave with claim identity.
            claim_embedding: pgvector.Vector
            if source["claim_embedding"] is None:
                claim_embedding = pgvector.Vector(
                    load_embedder(settings).embed([source["claim"]])[0]
                )
                connection.execute(
                    HEAL_CLAIM_EMBEDDING_SQL,
                    {"id": lesson_id, "claim_embedding": claim_embedding},
                )
            else:
                claim_embedding = source["claim_embedding"]

            # No lineage exemption here, unlike write_lesson's bar: a
            # promotion copy is a new-namespace row, not a re-derivation.
            # Target rows include NULL-claim ones; heal them in place under
            # this already-held target advisory lock, then compare ALL. A
            # rejected promotion rolls this target-heal back with the
            # transaction — harmless: nothing was inserted, so there is no
            # copy that needed the heal, and the next attempt (or the next
            # admission) re-heals. write_lesson keeps its stricter
            # raise-after-commit discipline because its heal must survive
            # rejection; promote's heal exists only to arm this check.
            target_rows = connection.execute(
                TARGET_LESSONS_SQL, {"ns": target_namespace}
            ).fetchall()
            unhealed_targets = [
                row for row in target_rows if row["claim_embedding"] is None
            ]
            if unhealed_targets:
                healed_vectors = load_embedder(settings).embed(
                    [row["claim"] for row in unhealed_targets]
                )
                for row, vec in zip(unhealed_targets, healed_vectors):
                    healed = pgvector.Vector(vec)
                    connection.execute(
                        HEAL_CLAIM_EMBEDDING_SQL,
                        {"id": row["id"], "claim_embedding": healed},
                    )
                    row["claim_embedding"] = healed
            for row in target_rows:
                cosine = _cosine(
                    claim_embedding.to_list(), row["claim_embedding"].to_list()
                )
                if cosine > settings.DUP_CLAIM_COS:
                    raise ToolError(
                        f"duplicate claim already graduated into namespace "
                        f"{target_namespace!r} (lesson {int(row['id'])}, claim "
                        f"cosine {cosine:.3f} exceeds "
                        f"DUP_CLAIM_COS={settings.DUP_CLAIM_COS}); corroborate "
                        "the existing lesson instead"
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
                    "claim_embedding": claim_embedding,
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
    screen_secrets(reason=reason)
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
