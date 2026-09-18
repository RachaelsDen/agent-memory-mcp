"""Contract tests for memory_promote / memory_demote (plan task 12, own file).

Promotion is COPY-NEVER-MOVE: the original row is byte-identical after the
call (row_to_json snapshot compared before/after), the copy carries exactly
the plan's override field map with every unlisted column on its schema
default, and BOTH confidence columns inherit the source's confidence at
promotion time — asserted SQL-side (float32 = float32 inside Postgres), so a
misleading success output cannot pass. Demotion is a TOMBSTONE, never a
delete: the copy persists with promotion_status='demoted' and its evidence
edges are retained, while retrieval's LessonVisibility hides it from probe —
including a probe with namespace='global' — driven here through the REAL
probe tool from a THIRD namespace. Lessons come from the REAL write_lesson
path (one exception: the already-in-global fixture seeds via SQL because
write_lesson rejects namespace='global' — F4); row/edge state is asserted
via SQL on the ``db`` fixture. Time is
pinned with backdate(at=<fixed tz-aware moments>) (no wall-clock-relative
assertions); per-test truncate in conftest covers stale state; tee'd runs
carry --durations=10 for the hung-command class.
"""

import json
from contextlib import AbstractAsyncContextManager
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

import pgvector
import psycopg
import pytest
from mcp import ClientSession
from mcp.types import CallToolResult, TextContent
from psycopg.rows import DictRow

from tests.conftest import backdate

DIM = 8
V_L = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # lesson + probe vector (cos 1.0)
V_E2 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
V_E3 = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
V_N = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]  # orthogonal novel corroboration

DEFAULT_NS = "default@local"  # server settings.MEMORY_NAMESPACE default
THIRD_NS = "third@watch"
GLOBAL = "global"

T_BASE = datetime(2026, 1, 10, 12, 0, tzinfo=timezone.utc)
DAY_1 = T_BASE
DAY_3 = T_BASE + timedelta(days=2)
DAY_5 = T_BASE + timedelta(days=4)
DAY_10 = T_BASE + timedelta(days=9)

CLAIM = "backoff pacing holds under load"
BECAUSE = "sync retries amplify packet storms"
# write_lesson embeds claim + " " + because + " " + holds_when; the empty
# holds_when keeps the trailing space (same as capture's default approach).
LESSON_TEXT = f"{CLAIM} {BECAUSE} "

PROBE_GOAL = "pace the retry loop"
PROBE_APPROACH = "check backoff pacing"
PROBE_TEXT = f"{PROBE_GOAL} {PROBE_APPROACH}"


def _insert_episode(
    db: psycopg.Connection[DictRow],
    *,
    goal: str,
    embedding: list[float],
    at: datetime,
) -> int:
    row = db.execute(
        """
        INSERT INTO episodes (namespace, goal, outcome, raw_text, embedding)
        VALUES (%(namespace)s, %(goal)s, '', '', %(embedding)s)
        RETURNING id
        """,
        {
            "namespace": DEFAULT_NS,
            "goal": goal,
            "embedding": pgvector.Vector(embedding),
        },
    ).fetchone()
    assert row is not None
    episode_id = int(row["id"])
    backdate("episodes", episode_id, at=at)
    return episode_id


def _ok(result: CallToolResult) -> dict[str, Any]:
    assert result.is_error is False, result.content
    block = result.content[0]
    assert isinstance(block, TextContent)
    payload: dict[str, Any] = json.loads(block.text)
    assert isinstance(payload, dict)
    return payload


def _err(result: CallToolResult) -> str:
    assert result.is_error is True, result.content
    block = result.content[0]
    assert isinstance(block, TextContent)
    return block.text


async def _write(
    session: ClientSession,
    *,
    evidence: list[dict[str, Any]],
    namespace: str | None = None,
) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "claim": CLAIM,
        "because": BECAUSE,
        "evidence": evidence,
    }
    if namespace is not None:
        arguments["namespace"] = namespace
    return _ok(await session.call_tool("memory_write_lesson", arguments))


async def _promote(
    session: ClientSession,
    lesson_id: int,
    reason: str,
    target_namespace: str | None = None,
    namespace: str | None = None,
) -> CallToolResult:
    arguments: dict[str, Any] = {"lesson_id": lesson_id, "reason": reason}
    if target_namespace is not None:
        arguments["target_namespace"] = target_namespace
    if namespace is not None:
        arguments["namespace"] = namespace
    return await session.call_tool("memory_promote", arguments)


async def _demote(session: ClientSession, lesson_id: int, reason: str) -> CallToolResult:
    return await session.call_tool(
        "memory_demote", {"lesson_id": lesson_id, "reason": reason}
    )


def _lesson_row(db: psycopg.Connection[DictRow], lesson_id: int) -> DictRow:
    row = db.execute(
        "SELECT * FROM lessons WHERE id = %(id)s", {"id": lesson_id}
    ).fetchone()
    assert row is not None
    return row


def _lesson_json(db: psycopg.Connection[DictRow], lesson_id: int) -> dict[str, Any]:
    """Full-row snapshot for the original-unchanged byte-comparison."""
    row = db.execute(
        "SELECT row_to_json(l.*) AS snapshot FROM lessons l WHERE l.id = %(id)s",
        {"id": lesson_id},
    ).fetchone()
    assert row is not None
    snapshot: dict[str, Any] = row["snapshot"]
    return snapshot


def _edges(db: psycopg.Connection[DictRow], lesson_id: int) -> list[tuple[int, str, str]]:
    rows = db.execute(
        """
        SELECT episode_id, relation, reason FROM lesson_evidence
        WHERE lesson_id = %(id)s
        ORDER BY episode_id
        """,
        {"id": lesson_id},
    ).fetchall()
    return [
        (int(row["episode_id"]), str(row["relation"]), row["reason"] or "")
        for row in rows
    ]


def _count(db: psycopg.Connection[DictRow], table: Literal["lessons", "lesson_evidence", "lesson_links"]) -> int:
    row = db.execute(f"SELECT count(*) AS n FROM {table}").fetchone()
    assert row is not None
    return int(row["n"])


async def _probe_lesson_refs(
    session: ClientSession, namespace: str | None = None
) -> set[str]:
    arguments: dict[str, Any] = {
        "current_goal": PROBE_GOAL,
        "approach": PROBE_APPROACH,
    }
    if namespace is not None:
        arguments["namespace"] = namespace
    payload = _ok(await session.call_tool("memory_probe", arguments))
    results: list[dict[str, Any]] = payload["results"]
    return {entry["id"] for entry in results if str(entry["id"]).startswith("lesson:")}


class TestPromoteCopy:
    """The happy path: field map, original byte-comparison, evidence-edge copy."""

    @pytest.fixture()
    def fake_embed_overrides(self) -> dict[str, list[float]]:
        return {LESSON_TEXT: V_L}

    async def test_copy_field_map_original_unchanged_edges_copied(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        # support day1 + refine day3 seed 0.50 (task-10 golden); the day5
        # contradict edge rides along without seeding.
        ep_support = _insert_episode(
            db, goal="support the pacing claim", embedding=V_L, at=DAY_1
        )
        ep_refine = _insert_episode(
            db, goal="refine the pacing boundary", embedding=V_E2, at=DAY_3
        )
        ep_contradict = _insert_episode(
            db, goal="contradict at tiny scale", embedding=V_E3, at=DAY_5
        )
        # Novel-day corroboration (orthogonal vector, fresh UTC date) moves
        # confidence off the 0.50 seed to 0.60 BEFORE promoting — the BOTH-
        # columns equality assert must discriminate an inherited 0.60 from
        # any hardcoded default (a mutation pinning 0.5 must fail here).
        ep_novel = _insert_episode(
            db, goal="corroborate on another project", embedding=V_N, at=DAY_10
        )
        evidence = [
            {"episode_id": ep_support, "relation": "support"},
            {"episode_id": ep_refine, "relation": "refine", "reason": "boundary case"},
            {"episode_id": ep_contradict, "relation": "contradict", "reason": "failed once"},
        ]
        before = datetime.now(tz=timezone.utc)
        async with client as session:
            written = await _write(session, evidence=evidence)
            lesson_id = int(written["lesson_id"])
            assert written["seed_confidence"] == pytest.approx(0.50)
            corroborated = _ok(
                await session.call_tool(
                    "memory_corroborate",
                    {
                        "lesson_id": lesson_id,
                        "episode_id": ep_novel,
                        "reason": "novel corroboration",
                    },
                )
            )
            assert corroborated["confidence"] == pytest.approx(0.60)
            source_snapshot = _lesson_json(db, lesson_id)
            payload = _ok(await _promote(session, lesson_id, reason="broadly useful"))
        copy_id = int(payload["promoted_lesson_id"])
        assert copy_id != lesson_id

        copy = _lesson_row(db, copy_id)
        assert copy["namespace"] == GLOBAL
        assert copy["claim"] == CLAIM
        assert copy["because"] == BECAUSE
        assert copy["holds_when"] == ""
        assert copy["fails_when"] == ""
        assert copy["promoted_from_lesson_id"] == lesson_id
        assert copy["promotion_reason"] == "broadly useful"
        assert copy["promoted_at"] is not None
        assert copy["promoted_at"] >= before
        assert copy["promotion_status"] == "active"
        assert copy["access_count"] == 0
        assert copy["usefulness"] == pytest.approx(0.0)
        assert copy["last_accessed"] is None
        assert copy["disputed"] is False
        assert copy["created_at"] >= before  # fresh copy clock, not the source's
        assert copy["demoted_at"] is None
        assert copy["demotion_reason"] is None

        # BOTH confidence columns equal the source's confidence at promotion
        # time — SQL-side equality (float32 = float32 inside Postgres).
        confidence_row = db.execute(
            """
            SELECT (c.confidence = c.promotion_seed_confidence) AS seed_matches,
                   (c.confidence = s.confidence) AS inherits_source
            FROM lessons c, lessons s
            WHERE c.id = %(copy)s AND s.id = %(src)s
            """,
            {"copy": copy_id, "src": lesson_id},
        ).fetchone()
        assert confidence_row is not None
        assert confidence_row["seed_matches"] is True
        assert confidence_row["inherits_source"] is True

        # SAME embedding, bit-for-bit, through the vector adapter round-trip.
        embedding_row = db.execute(
            """
            SELECT (c.embedding = s.embedding) AS same_embedding
            FROM lessons c, lessons s
            WHERE c.id = %(copy)s AND s.id = %(src)s
            """,
            {"copy": copy_id, "src": lesson_id},
        ).fetchone()
        assert embedding_row is not None
        assert embedding_row["same_embedding"] is True

        # The ORIGINAL is untouched: every column byte-identical.
        assert _lesson_json(db, lesson_id) == source_snapshot

        # Every source evidence edge exists on the copy — same episode ids,
        # relations, and reasons — and the source keeps its own (copy, not move).
        assert _edges(db, copy_id) == [
            (ep_support, "support", ""),
            (ep_refine, "refine", "boundary case"),
            (ep_contradict, "contradict", "failed once"),
            (ep_novel, "support", "novel corroboration"),
        ]
        assert len(_edges(db, lesson_id)) == 4

        # Promotion copies evidence provenance only — no lesson_links writes.
        assert _count(db, "lesson_links") == 0

    async def test_source_scoped_by_id_namespace_param_inert(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        """lesson_id alone scopes the source; the caller namespace is not a filter."""
        ep_support = _insert_episode(
            db, goal="support the pacing claim", embedding=V_L, at=DAY_1
        )
        async with client as session:
            written = await _write(session, evidence=[{"episode_id": ep_support, "relation": "support"}])
            lesson_id = int(written["lesson_id"])
            payload = _ok(
                await _promote(
                    session, lesson_id, reason="graduate", namespace=THIRD_NS
                )
            )
        assert int(payload["promoted_lesson_id"]) != lesson_id
        assert _lesson_row(db, int(payload["promoted_lesson_id"]))["namespace"] == GLOBAL


class TestPromoteVisibility:
    """Probe from a THIRD namespace before/after promote and demote."""

    @pytest.fixture()
    def fake_embed_overrides(self) -> dict[str, list[float]]:
        return {LESSON_TEXT: V_L, PROBE_TEXT: V_L}

    async def test_copy_visible_everywhere_then_demote_tombstones_everywhere(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        ep_support = _insert_episode(
            db, goal="support the pacing claim", embedding=V_L, at=DAY_1
        )
        async with client as session:
            written = await _write(
                session, evidence=[{"episode_id": ep_support, "relation": "support"}]
            )
            assert written["seed_confidence"] == pytest.approx(0.40)
            lesson_id = int(written["lesson_id"])

            # Before promotion the third namespace sees nothing.
            assert await _probe_lesson_refs(session, namespace=THIRD_NS) == set()

            payload = _ok(await _promote(session, lesson_id, reason="graduate"))
            copy_id = int(payload["promoted_lesson_id"])

            # Active copy: visible from the third namespace, from global
            # directly, and BOTH original+copy from the default namespace
            # (the (ns OR 'global') AND active union).
            assert await _probe_lesson_refs(session, namespace=THIRD_NS) == {
                f"lesson:{copy_id}"
            }
            assert await _probe_lesson_refs(session, namespace=GLOBAL) == {
                f"lesson:{copy_id}"
            }
            assert await _probe_lesson_refs(session) == {
                f"lesson:{lesson_id}",
                f"lesson:{copy_id}",
            }

            demoted = _ok(await _demote(session, copy_id, reason="stale globally"))
            assert demoted == {"lesson_id": copy_id, "promotion_status": "demoted"}

            # Demoted copy: invisible everywhere the union could surface it,
            # while the original stays visible in its own namespace.
            assert await _probe_lesson_refs(session, namespace=THIRD_NS) == set()
            assert await _probe_lesson_refs(session, namespace=GLOBAL) == set()
            assert await _probe_lesson_refs(session) == {f"lesson:{lesson_id}"}

        # The tombstone persists (never a delete): status, dates, reason,
        # provenance, and the evidence edges are all retained.
        copy = _lesson_row(db, copy_id)
        assert copy["promotion_status"] == "demoted"
        assert copy["demoted_at"] is not None
        assert copy["demotion_reason"] == "stale globally"
        assert copy["promoted_from_lesson_id"] == lesson_id
        assert _edges(db, copy_id) == _edges(db, lesson_id) == [
            (ep_support, "support", "")
        ]


class TestPromoteRejectsSameNamespace:
    """Plan QA (failure): promoting into the source's own namespace writes nothing."""

    @pytest.fixture()
    def fake_embed_overrides(self) -> dict[str, list[float]]:
        return {LESSON_TEXT: V_L}

    async def test_lesson_already_in_global_is_error_nothing_written(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        # F4: memory_write_lesson no longer accepts namespace='global' (the
        # promotion-only invariant this test exercises), so the already-global
        # fixture seeds via direct SQL — the sanctioned-fixture precedent of
        # _dispute in test_tools_consolidate.py.
        seeded = db.execute(
            """
            INSERT INTO lessons (namespace, claim, because, embedding)
            VALUES (%(namespace)s, %(claim)s, %(because)s, %(embedding)s)
            RETURNING id
            """,
            {
                "namespace": GLOBAL,
                "claim": CLAIM,
                "because": BECAUSE,
                "embedding": pgvector.Vector(V_L),
            },
        ).fetchone()
        assert seeded is not None
        lesson_id = int(seeded["id"])
        async with client as session:
            lessons_before = _count(db, "lessons")
            edges_before = _count(db, "lesson_evidence")
            message = _err(await _promote(session, lesson_id, reason="re-graduate"))
            assert "global" in message
            # NO rows written, and the session is still usable.
            assert _count(db, "lessons") == lessons_before
            assert _count(db, "lesson_evidence") == edges_before
            payload = _ok(await _promote(session, lesson_id, reason="cross-scope", target_namespace="other@proj"))
        assert _lesson_row(db, int(payload["promoted_lesson_id"]))["namespace"] == "other@proj"

    async def test_explicit_same_namespace_target_is_error(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        ep_support = _insert_episode(
            db, goal="support the pacing claim", embedding=V_L, at=DAY_1
        )
        async with client as session:
            written = await _write(
                session, evidence=[{"episode_id": ep_support, "relation": "support"}]
            )
            lesson_id = int(written["lesson_id"])
            lessons_before = _count(db, "lessons")
            message = _err(
                await _promote(
                    session, lesson_id, reason="no-op", target_namespace=DEFAULT_NS
                )
            )
            assert DEFAULT_NS in message
            assert _count(db, "lessons") == lessons_before


class TestDemoteValidation:
    """Demote requires a promotion; anything else is an MCP error result."""

    @pytest.fixture()
    def fake_embed_overrides(self) -> dict[str, list[float]]:
        return {LESSON_TEXT: V_L}

    async def test_demote_non_promotion_lesson_is_error_row_unchanged(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        ep_support = _insert_episode(
            db, goal="support the pacing claim", embedding=V_L, at=DAY_1
        )
        async with client as session:
            written = await _write(
                session, evidence=[{"episode_id": ep_support, "relation": "support"}]
            )
            lesson_id = int(written["lesson_id"])
            snapshot = _lesson_json(db, lesson_id)
            message = _err(await _demote(session, lesson_id, reason="not a promotion"))
            assert "promot" in message
        assert _lesson_json(db, lesson_id) == snapshot

    async def test_demote_nonexistent_lesson_is_error(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        async with client as session:
            message = _err(await _demote(session, 999999, reason="ghost"))
        assert "999999" in message

    async def test_promote_nonexistent_lesson_is_error_nothing_written(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        async with client as session:
            message = _err(await _promote(session, 999999, reason="ghost"))
            assert "999999" in message
            # The session survives the error (errors are results, not exits).
            probe = _ok(
                await session.call_tool(
                    "memory_probe", {"current_goal": "anything", "approach": ""}
                )
            )
        assert isinstance(probe["retrieval_event_id"], int)
        assert _count(db, "lessons") == 0


class TestReasonSecretScreen:
    """Issue #9: promote/demote reasons are screened (capture contract) —
    field named, secret never echoed, nothing written, same session recovers."""

    async def test_promote_reason_secret_rejected_then_clean_recovers(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        secret = "sk-AbCdEf0123456789AbCdEf0123456789"
        seed = _insert_episode(db, goal="seed", embedding=V_L, at=DAY_1)
        async with client as session:
            written = await _write(
                session, evidence=[{"episode_id": seed, "relation": "support"}]
            )
            lesson_id = int(written["lesson_id"])

            message = _err(
                await _promote(session, lesson_id, reason=f"pasted the wrong log: {secret}")
            )
            assert "reason" in message
            assert secret not in message
            assert _count(db, "lessons") == 1  # the source only — no copy row

            promoted = _ok(await _promote(session, lesson_id, reason="broadly useful"))
        row = db.execute(
            "SELECT promotion_reason FROM lessons WHERE id = %(id)s",
            {"id": promoted["promoted_lesson_id"]},
        ).fetchone()
        assert row is not None
        assert row["promotion_reason"] == "broadly useful"
        assert _count(db, "lessons") == 2

    async def test_demote_reason_secret_rejected_then_clean_recovers(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        secret = "sk-AbCdEf0123456789AbCdEf0123456789"
        seed = _insert_episode(db, goal="seed", embedding=V_L, at=DAY_1)
        async with client as session:
            written = await _write(
                session, evidence=[{"episode_id": seed, "relation": "support"}]
            )
            copy_id = int(
                _ok(
                    await _promote(
                        session, int(written["lesson_id"]), reason="graduate"
                    )
                )["promoted_lesson_id"]
            )

            message = _err(
                await _demote(session, copy_id, reason=f"pasted the wrong log: {secret}")
            )
            assert "reason" in message
            assert secret not in message
            status = db.execute(
                "SELECT promotion_status FROM lessons WHERE id = %(id)s", {"id": copy_id}
            ).fetchone()
            assert status is not None
            assert status["promotion_status"] == "active"

            demoted = _ok(await _demote(session, copy_id, reason="stale globally"))
        assert demoted == {"lesson_id": copy_id, "promotion_status": "demoted"}
