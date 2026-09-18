"""Contract tests for the consolidation tools (plan tasks 9-11, ONE shared file).

Task 9 section: memory_consolidate_scan — derived-consolidation pools
(``fresh`` = age-eligible episodes with no lesson_evidence citation,
``all`` = every age-eligible episode), greedy cosine clustering in Python,
and MANDATORY rederivation groups for pending disputed lessons (regardless
of pool and min_cluster_size). Group membership is asserted via the tool
result AND cross-checked against SQL rows on the ``db`` fixture — a
misleading success output must not pass. Timestamps are pinned via
``backdate`` with absolute ``at=`` moments (never wall-clock-relative
assertions); per-test truncate in conftest covers stale state.

Tasks 10 (memory_write_lesson) and 11 (corroborate/contradict) APPEND
their sections at the markers below — do not reorder this header block.
"""

import json
import os
import shutil
from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pgvector
import psycopg
import pytest
from mcp import ClientSession
from mcp.types import CallToolResult, TextContent
from psycopg import sql
from psycopg.rows import DictRow
from testcontainers.community.postgres import PostgresContainer

import agent_memory.db as agent_db
from agent_memory.config import get_settings
from agent_memory.embed import FakeEmbedder
from tests.conftest import backdate

DIM = 8
V_X = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
V_Y = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
DEFAULT_NS = "default@local"
OTHER_NS = "other@remote"

# Fixed pre-now moments: deterministic against DB now() for both the age
# gate (all are hours older than CONSOLIDATE_MIN_AGE_H) and cluster seeding
# order (T_BASE < T_BASE+1m < ...).
T_BASE = datetime(2026, 1, 10, 12, 0, tzinfo=timezone.utc)

CAPTURE_A = {
    "goal": "stabilize the deploy",
    "expectation": "rollout completes cleanly",
    "action": "retry on transient network errors",
    "outcome": "deploy finished green",
    "surprise": 0.8,
    "state_at_encoding": {"mood": "calm"},
    "tags": ["ops", "deploy"],
}
CAPTURE_B = {
    "goal": "stabilize the nightly deploy",
    "expectation": "rollout completes cleanly",
    "action": "retry on transient network errors",
    "outcome": "nightly deploy finished green",
}
CAPTURE_C = {
    "goal": "rename the widget struct",
    "expectation": "compiler catches all call sites",
    "action": "run grep before refactoring",
    "outcome": "one missed call site panicked at runtime",
}


def _capture_text(fields: dict[str, Any]) -> str:
    """The exact server-side embedded text of a capture (4-field join)."""
    return (
        f"{fields['goal']} {fields['expectation']} "
        f"{fields['action']} {fields['outcome']}"
    )


def _claim_vector(claim: str) -> list[float]:
    """The claim embedding the server's FakeEmbedder yields for this claim.

    No test pins an override on a BARE claim (override keys are the 3-field
    composite join), so test process and server derive the same hash vector.
    """
    return FakeEmbedder(dim=DIM).embed([claim])[0]


def _insert_episode(
    db: psycopg.Connection[DictRow],
    *,
    goal: str,
    embedding: Sequence[float],
    namespace: str = DEFAULT_NS,
) -> int:
    row = db.execute(
        """
        INSERT INTO episodes (namespace, goal, raw_text, embedding)
        VALUES (%(namespace)s, %(goal)s, '', %(embedding)s)
        RETURNING id
        """,
        {
            "namespace": namespace,
            "goal": goal,
            "embedding": pgvector.Vector(list(embedding)),
        },
    ).fetchone()
    assert row is not None
    return int(row["id"])


def _insert_lesson(
    db: psycopg.Connection[DictRow],
    *,
    claim: str,
    namespace: str = DEFAULT_NS,
) -> int:
    row = db.execute(
        """
        INSERT INTO lessons (namespace, claim, because, embedding, claim_embedding)
        VALUES (%(namespace)s, %(claim)s, 'causal gist', %(embedding)s,
                %(claim_embedding)s)
        RETURNING id
        """,
        {
            "namespace": namespace,
            "claim": claim,
            "embedding": pgvector.Vector(V_X),
            "claim_embedding": pgvector.Vector(_claim_vector(claim)),
        },
    ).fetchone()
    assert row is not None
    return int(row["id"])


def _dispute(db: psycopg.Connection[DictRow], lesson_id: int) -> None:
    """The plan's sanctioned dispute fixture: direct SQL, no tool (task 13)."""
    db.execute(
        "UPDATE lessons SET disputed = true WHERE id = %(id)s", {"id": lesson_id}
    )


def _insert_evidence(
    db: psycopg.Connection[DictRow],
    lesson_id: int,
    episode_id: int,
    relation: str = "support",
) -> None:
    db.execute(
        """
        INSERT INTO lesson_evidence (lesson_id, episode_id, relation, reason)
        VALUES (%(lesson_id)s, %(episode_id)s, %(relation)s, '')
        """,
        {"lesson_id": lesson_id, "episode_id": episode_id, "relation": relation},
    )


async def _scan(session: ClientSession, **kwargs: Any) -> CallToolResult:
    return await session.call_tool("memory_consolidate_scan", kwargs)


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


def _cluster_refs(payload: dict[str, Any]) -> list[list[str]]:
    """Typed refs per ordinary cluster in returned order (seed first)."""
    return [
        [str(record["id"]) for record in group["episodes"]]
        for group in payload["clusters"]
    ]


def _rederivations(payload: dict[str, Any]) -> dict[int, set[str]]:
    """lesson_id -> set of source-episode refs across rederivation groups."""
    return {
        int(group["lesson_id"]): {str(record["id"]) for record in group["episodes"]}
        for group in payload["rederivation_groups"]
    }


# === Task 9 section: memory_consolidate_scan ===============================
# (write_lesson tests append at the task-10 marker below)


class TestScanPools:
    async def test_fresh_excludes_cited_all_includes_and_reports_citers(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        uncited = _insert_episode(db, goal="ship the widget", embedding=V_X)
        cited = _insert_episode(db, goal="ship the widget again", embedding=V_X)
        backdate("episodes", uncited, at=T_BASE)
        backdate("episodes", cited, at=T_BASE + timedelta(minutes=1))
        lesson = _insert_lesson(db, claim="backoff before retry storms")
        other_lesson = _insert_lesson(db, claim="page only on sustained errors")
        _insert_evidence(db, lesson, cited)
        _insert_evidence(db, other_lesson, cited, relation="contradict")

        async with client as session:
            fresh = _ok(await _scan(session, pool="fresh", min_cluster_size=1))
            every = _ok(await _scan(session, pool="all", min_cluster_size=1))

        assert fresh["pool"] == "fresh"
        assert fresh["namespace"] == DEFAULT_NS
        assert _cluster_refs(fresh) == [[f"episode:{uncited}"]]
        assert fresh["rederivation_groups"] == []

        assert every["pool"] == "all"
        # uncited (T_BASE) seeds; cited (T_BASE+1m) joins on cosine 1.0
        assert _cluster_refs(every) == [
            [f"episode:{uncited}", f"episode:{cited}"]
        ]
        # cited_by_lessons: every citing lesson, any relation, deterministic order
        records = {
            str(record["id"]): record for record in every["clusters"][0]["episodes"]
        }
        assert records[f"episode:{cited}"]["cited_by_lessons"] == [
            f"lesson:{lesson}",
            f"lesson:{other_lesson}",
        ]
        assert records[f"episode:{uncited}"]["cited_by_lessons"] == []

        # DB cross-check: the citation edges the pools split on are real
        citers = db.execute(
            """
            SELECT lesson_id FROM lesson_evidence
            WHERE episode_id = %(id)s ORDER BY lesson_id
            """,
            {"id": cited},
        ).fetchall()
        assert [int(row["lesson_id"]) for row in citers] == [lesson, other_lesson]

    async def test_age_gate_boundary_excludes_young_includes_old(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        min_age_h = get_settings().CONSOLIDATE_MIN_AGE_H
        young = _insert_episode(db, goal="fresh capture", embedding=V_X)
        old = _insert_episode(db, goal="settled capture", embedding=V_Y)
        backdate("episodes", young, hours=-(min_age_h - 0.25))
        backdate("episodes", old, hours=-(min_age_h + 0.25))

        async with client as session:
            payload = _ok(await _scan(session, min_cluster_size=1))

        assert _cluster_refs(payload) == [[f"episode:{old}"]]

    async def test_namespace_scopes_episodes_and_disputed_lessons(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        foreign_episode = _insert_episode(
            db, goal="elsewhere", embedding=V_X, namespace=OTHER_NS
        )
        backdate("episodes", foreign_episode, at=T_BASE)
        foreign_lesson = _insert_lesson(
            db, claim="foreign rule", namespace=OTHER_NS
        )
        _insert_evidence(db, foreign_lesson, foreign_episode)
        _dispute(db, foreign_lesson)

        async with client as session:
            default_scan = _ok(await _scan(session, pool="all", min_cluster_size=1))
            foreign_scan = _ok(
                await _scan(
                    session, pool="all", min_cluster_size=1, namespace=OTHER_NS
                )
            )

        assert default_scan["clusters"] == []
        assert default_scan["rederivation_groups"] == []

        assert _cluster_refs(foreign_scan) == [[f"episode:{foreign_episode}"]]
        assert _rederivations(foreign_scan) == {
            foreign_lesson: {f"episode:{foreign_episode}"}
        }


class TestGreedyClustering:
    """Genuine path: capture via the tool with crafted FakeEmbedder vectors."""

    @pytest.fixture()
    def fake_embed_overrides(self) -> dict[str, list[float]]:
        return {
            _capture_text(CAPTURE_A): V_X,
            _capture_text(CAPTURE_B): V_X,  # cos 1.0 to A -> same cluster
            _capture_text(CAPTURE_C): V_Y,  # orthogonal -> stays single
        }

    async def test_two_cluster_one_single_and_full_episode_fields(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        async with client as session:
            ids: list[int] = []
            for fields in (CAPTURE_A, CAPTURE_B, CAPTURE_C):
                capture = _ok(
                    await session.call_tool("memory_capture_episode", fields)
                )
                ids.append(int(capture["id"]))
            episode_a, episode_b, episode_c = ids
            # deterministic age eligibility + seeding order (A earliest);
            # backdate rides its own connection, the session just idles
            backdate("episodes", episode_a, at=T_BASE)
            backdate("episodes", episode_b, at=T_BASE + timedelta(minutes=1))
            backdate("episodes", episode_c, at=T_BASE + timedelta(minutes=2))
            at_min_1 = _ok(await _scan(session, min_cluster_size=1))
            at_min_2 = _ok(await _scan(session, min_cluster_size=2))

        # min_cluster_size=1: the pair AND the singleton both return
        assert _cluster_refs(at_min_1) == [
            [f"episode:{episode_a}", f"episode:{episode_b}"],
            [f"episode:{episode_c}"],
        ]
        # below-min clusters are dropped: only the pair survives min=2
        assert _cluster_refs(at_min_2) == [
            [f"episode:{episode_a}", f"episode:{episode_b}"]
        ]
        assert at_min_2["rederivation_groups"] == []

        record = at_min_1["clusters"][0]["episodes"][0]
        assert record == {
            "id": f"episode:{episode_a}",
            "goal": CAPTURE_A["goal"],
            "expectation": CAPTURE_A["expectation"],
            "action": CAPTURE_A["action"],
            "outcome": CAPTURE_A["outcome"],
            "surprise": pytest.approx(CAPTURE_A["surprise"]),
            "state_at_encoding": CAPTURE_A["state_at_encoding"],
            "tags": CAPTURE_A["tags"],
            "created_at": T_BASE.isoformat(),
            "cited_by_lessons": [],
        }

        # DB cross-check: three age-eligible episodes exist for the scan
        count = db.execute(
            """
            SELECT count(*) AS n FROM episodes
            WHERE namespace = %(ns)s AND created_at < now() - make_interval(hours => %(h)s)
            """,
            {"ns": DEFAULT_NS, "h": get_settings().CONSOLIDATE_MIN_AGE_H},
        ).fetchone()
        assert count is not None and int(count["n"]) == 3


class TestRederivationGroups:
    async def test_disputed_sources_returned_in_fresh_scan_flagged(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        source_a = _insert_episode(db, goal="retry storm hit", embedding=V_X)
        source_b = _insert_episode(db, goal="retry storm again", embedding=V_Y)
        backdate("episodes", source_a, at=T_BASE)
        backdate("episodes", source_b, at=T_BASE + timedelta(minutes=1))
        lesson = _insert_lesson(db, claim="backoff solves retry storms")
        _insert_evidence(db, lesson, source_a)
        _insert_evidence(db, lesson, source_b, relation="contradict")
        _dispute(db, lesson)

        async with client as session:
            payload = _ok(await _scan(session))  # defaults: fresh, min=2

        # cited sources are excluded from ordinary fresh clusters ...
        assert payload["clusters"] == []
        # ... but the disputed lesson re-derives from ALL of them, any relation
        groups = payload["rederivation_groups"]
        assert len(groups) == 1
        group = groups[0]
        assert group["rederivation"] is True
        assert int(group["lesson_id"]) == lesson
        assert group["claim"] == "backoff solves retry storms"
        assert group["because"] == "causal gist"
        assert {str(record["id"]) for record in group["episodes"]} == {
            f"episode:{source_a}",
            f"episode:{source_b}",
        }
        by_id = {str(r["id"]): r for r in group["episodes"]}
        assert by_id[f"episode:{source_a}"]["cited_by_lessons"] == [
            f"lesson:{lesson}"
        ]
        # DB cross-check: the dispute flag and both evidence edges are real
        row = db.execute(
            "SELECT disputed FROM lessons WHERE id = %(id)s", {"id": lesson}
        ).fetchone()
        assert row is not None and row["disputed"] is True

    async def test_singleton_young_source_survives_min_cluster_size(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        # NOT backdated: younger than the age gate, still a rederivation
        # source — ordinary thresholds never swallow re-derivation.
        source = _insert_episode(db, goal="just captured contradiction", embedding=V_Y)
        lesson = _insert_lesson(db, claim="single-source rule")
        _insert_evidence(db, lesson, source)
        _dispute(db, lesson)

        async with client as session:
            payload = _ok(await _scan(session, min_cluster_size=2))

        assert payload["clusters"] == []
        assert _rederivations(payload) == {lesson: {f"episode:{source}"}}

    async def test_completed_rederivation_drops_out_dispute_record_stays(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        source = _insert_episode(db, goal="old incident", embedding=V_X)
        backdate("episodes", source, at=T_BASE)
        lesson = _insert_lesson(db, claim="superseded rule")
        _insert_evidence(db, lesson, source)
        _dispute(db, lesson)
        replacement = _insert_lesson(db, claim="replacement rule")
        # task 10's memory_write_lesson writes this link; fixture via SQL here
        db.execute(
            """
            INSERT INTO lesson_links (lesson_id, related_lesson_id, kind)
            VALUES (%(from)s, %(to)s, 'refines')
            """,
            {"from": replacement, "to": lesson},
        )

        async with client as session:
            payload = _ok(await _scan(session, pool="all", min_cluster_size=1))

        assert payload["rederivation_groups"] == []
        # historical dispute record remains on the row
        row = db.execute(
            "SELECT disputed FROM lessons WHERE id = %(id)s", {"id": lesson}
        ).fetchone()
        assert row is not None and row["disputed"] is True


class TestScanValidationAndEmpty:
    """Failure QA: zero age-eligible episodes -> empty result, NO error."""

    async def test_empty_db_scan_is_empty_not_an_error(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        async with client as session:
            payload = _ok(await _scan(session))

        assert payload == {
            "pool": "fresh",
            "namespace": DEFAULT_NS,
            "clusters": [],
            "rederivation_groups": [],
        }

    async def test_invalid_pool_is_error(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        async with client as session:
            error = _err(await _scan(session, pool="stale"))

        assert "pool" in error


# === Task 10 marker: memory_write_lesson tests append here =================

# === Task 10 section: memory_write_lesson ==================================
# Server-owned seeding, per-namespace dup guard with lineage exemption,
# similar/contradicts/refines links, one-transaction writes. Every DB effect
# is asserted via SQL on the ``db`` fixture (misleading_success_output);
# timestamps pinned via ``backdate(at=...)``; per-test truncate covers
# stale state; the concurrency test uses real threads + independent
# connections with a start barrier (no sleeps).

V_SIM = [0.8, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # cos 0.8 to V_X
V_MID = [0.5, -0.8660254, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # cos 0.5 to V_X, cos -0.12 to V_SIM
V_Z = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # orthogonal to V_X and V_Y


def _lesson_text(claim: str, because: str, holds_when: str) -> str:
    """The exact server-side embedded text of a write (3-field join)."""
    return f"{claim} {because} {holds_when}"


def _insert_lesson_vec(
    db: psycopg.Connection[DictRow],
    *,
    claim: str,
    vector: Sequence[float],
    namespace: str = DEFAULT_NS,
) -> int:
    row = db.execute(
        """
        INSERT INTO lessons (namespace, claim, because, embedding, claim_embedding)
        VALUES (%(namespace)s, %(claim)s, 'causal gist', %(embedding)s,
                %(claim_embedding)s)
        RETURNING id
        """,
        {
            "namespace": namespace,
            "claim": claim,
            "embedding": pgvector.Vector(list(vector)),
            "claim_embedding": pgvector.Vector(_claim_vector(claim)),
        },
    ).fetchone()
    assert row is not None
    return int(row["id"])


def _count(
    db: psycopg.Connection[DictRow], table: str, *, ns: str | None = None
) -> int:
    statement = sql.SQL("SELECT count(*) AS n FROM {}").format(sql.Identifier(table))
    if ns is not None:
        statement += sql.SQL(" WHERE namespace = %(ns)s")
    row = db.execute(statement, {"ns": ns}).fetchone()
    assert row is not None
    return int(row["n"])


async def _write(session: ClientSession, **payload: Any) -> CallToolResult:
    arguments = {
        "claim": payload["claim"],
        "because": payload["because"],
        "evidence": payload["evidence"],
    }
    for key in ("holds_when", "fails_when", "contradicts", "replaces_disputed", "namespace"):
        if payload.get(key) is not None:
            arguments[key] = payload[key]
    return await session.call_tool("memory_write_lesson", arguments)


class TestWriteLessonValidation:
    async def test_empty_evidence_is_error(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        async with client as session:
            error = _err(await _write(session, claim="c", because="b", evidence=[]))

        assert "support" in error or "refine" in error
        assert _count(db, "lessons") == 0

    async def test_contradiction_only_evidence_is_error(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        episode = _insert_episode(db, goal="contradicting incident", embedding=V_Y)

        async with client as session:
            error = _err(
                await _write(
                    session,
                    claim="c",
                    because="b",
                    evidence=[
                        {"episode_id": episode, "relation": "contradict", "reason": ""}
                    ],
                )
            )

        assert "support" in error or "refine" in error
        assert _count(db, "lessons") == 0

    async def test_invalid_relation_is_error(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        episode = _insert_episode(db, goal="some incident", embedding=V_Y)

        async with client as session:
            error = _err(
                await _write(
                    session,
                    claim="c",
                    because="b",
                    evidence=[{"episode_id": episode, "relation": "boosts"}],
                )
            )

        assert "relation" in error
        assert _count(db, "lessons") == 0

    async def test_duplicate_episode_in_evidence_is_error(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        episode = _insert_episode(db, goal="cited twice", embedding=V_Y)

        async with client as session:
            error = _err(
                await _write(
                    session,
                    claim="c",
                    because="b",
                    evidence=[
                        {"episode_id": episode, "relation": "support"},
                        {"episode_id": episode, "relation": "refine"},
                    ],
                )
            )

        assert "episode" in error
        assert _count(db, "lesson_evidence") == 0

    async def test_nonexistent_episode_rolls_back_everything(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        """Failure QA: bogus episode_id -> MCP error, ZERO rows anywhere."""
        episode = _insert_episode(db, goal="real incident", embedding=V_Y)

        async with client as session:
            error = _err(
                await _write(
                    session,
                    claim="real lesson",
                    because="real cause",
                    evidence=[
                        {"episode_id": episode, "relation": "support"},
                        {"episode_id": 987654, "relation": "support"},
                    ],
                )
            )
            # the SAME live session keeps serving after the error
            recovered = _ok(
                await _write(
                    session,
                    claim="recovered lesson",
                    because="cause",
                    evidence=[{"episode_id": episode, "relation": "support"}],
                )
            )

        assert "987654" in error
        assert set(recovered) == {"lesson_id", "seed_confidence"}
        assert _count(db, "lessons") == 1  # only the recovery write landed
        assert _count(db, "lesson_evidence") == 1

    async def test_replaces_disputed_requires_disputed_target(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        episode = _insert_episode(db, goal="incident", embedding=V_Y)
        undisputed = _insert_lesson(db, claim="never disputed")

        async with client as session:
            not_disputed = _err(
                await _write(
                    session,
                    claim="replacement",
                    because="cause",
                    evidence=[{"episode_id": episode, "relation": "support"}],
                    replaces_disputed=undisputed,
                )
            )
            nonexistent = _err(
                await _write(
                    session,
                    claim="replacement",
                    because="cause",
                    evidence=[{"episode_id": episode, "relation": "support"}],
                    replaces_disputed=555555,
                )
            )

        assert "disputed" in not_disputed
        assert "555555" in nonexistent
        assert _count(db, "lessons") == 1  # the fixture row only
        assert _count(db, "lesson_links") == 0


class TestWriteLessonSeeding:
    """Server-owned seed_confidence: goldens are the task-5 E2E math."""

    async def test_seed_golden_two_same_day_near_dup_plus_distinct_day(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        near_dup_a = _insert_episode(db, goal="retry storm", embedding=V_X)
        near_dup_b = _insert_episode(db, goal="retry storm again", embedding=V_X)
        distinct_day = _insert_episode(db, goal="calm-day confirmation", embedding=V_X)
        day_one = T_BASE
        day_one_later = T_BASE + timedelta(hours=1)  # same UTC day, <= 24h of seed
        day_three = T_BASE + timedelta(hours=48)  # outside the 24h incident window
        backdate("episodes", near_dup_a, at=day_one)
        backdate("episodes", near_dup_b, at=day_one_later)
        backdate("episodes", distinct_day, at=day_three)

        async with client as session:
            payload = _ok(
                await _write(
                    session,
                    claim="backoff before retrying",
                    because="sync retries amplify load",
                    holds_when="under packet loss",
                    evidence=[
                        {"episode_id": near_dup_a, "relation": "support"},
                        {"episode_id": near_dup_b, "relation": "support"},
                        {
                            "episode_id": distinct_day,
                            "relation": "refine",
                            "reason": "confirmed on a calm day",
                        },
                    ],
                )
            )

        # 2 incidents (near-dups collapse) + diversity 2 distinct days / 4
        # = 0.35 + 0.05 + 0.20 * 0.5 = 0.50 EXACT per the E2E math
        assert payload["seed_confidence"] == pytest.approx(0.50)
        lesson_id = int(payload["lesson_id"])

        row = db.execute(
            """
            SELECT confidence, last_evidence_at, updated_at, namespace
            FROM lessons WHERE id = %(id)s
            """,
            {"id": lesson_id},
        ).fetchone()
        assert row is not None
        assert float(row["confidence"]) == pytest.approx(0.50)
        assert row["last_evidence_at"] == day_three  # max over evidence episodes
        assert row["namespace"] == DEFAULT_NS
        assert row["updated_at"] is not None

        edges = db.execute(
            """
            SELECT episode_id, relation, reason FROM lesson_evidence
            WHERE lesson_id = %(id)s ORDER BY episode_id
            """,
            {"id": lesson_id},
        ).fetchall()
        assert [(int(e["episode_id"]), e["relation"], e["reason"]) for e in edges] == [
            (near_dup_a, "support", ""),
            (near_dup_b, "support", ""),
            (distinct_day, "refine", "confirmed on a calm day"),
        ]

    async def test_cited_episodes_leave_the_fresh_pool(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        cited_a = _insert_episode(db, goal="cited one", embedding=V_X)
        cited_b = _insert_episode(db, goal="cited two", embedding=V_X)
        uncited = _insert_episode(db, goal="never cited", embedding=V_Y)
        backdate("episodes", cited_a, at=T_BASE)
        backdate("episodes", cited_b, at=T_BASE + timedelta(minutes=30))
        backdate("episodes", uncited, at=T_BASE + timedelta(minutes=60))

        async with client as session:
            payload = _ok(
                await _write(
                    session,
                    claim="cited lessons leave the pool",
                    because="the edges are the consolidation",
                    evidence=[
                        {"episode_id": cited_a, "relation": "support"},
                        {"episode_id": cited_b, "relation": "support"},
                    ],
                )
            )
            scan = _ok(await _scan(session, min_cluster_size=1))

        assert int(payload["lesson_id"]) > 0
        assert _cluster_refs(scan) == [[f"episode:{uncited}"]]
        assert scan["rederivation_groups"] == []


class TestWriteLessonDupGuard:
    @pytest.fixture()
    def fake_embed_overrides(self) -> dict[str, list[float]]:
        return {
            _lesson_text("always backoff before retrying", "near-identical gist", "everywhere"): V_X,
            _lesson_text("local claim", "local cause", "local scope"): V_X,
            _lesson_text("a fresh orthogonal lesson", "different mechanism entirely", ""): V_Y,
        }

    async def test_duplicate_claim_rejected_and_rolls_back(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        episode = _insert_episode(db, goal="cited incident", embedding=V_Y)
        _insert_lesson(db, claim="always backoff before retrying")  # stores V_X

        async with client as session:
            error = _err(
                await _write(
                    session,
                    claim="always backoff before retrying",
                    because="near-identical gist",
                    holds_when="everywhere",
                    evidence=[{"episode_id": episode, "relation": "support"}],
                )
            )

        assert "duplicate" in error
        assert _count(db, "lessons") == 1  # the fixture only
        assert _count(db, "lesson_links") == 0

    async def test_guard_is_namespace_scoped(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        episode = _insert_episode(db, goal="incident", embedding=V_Y)
        # near-duplicate of the new claim (cos 1.0), but in ANOTHER namespace
        _insert_lesson_vec(db, claim="foreign near duplicate", vector=V_X, namespace=OTHER_NS)

        async with client as session:
            payload = _ok(
                await _write(
                    session,
                    claim="local claim",
                    because="local cause",
                    holds_when="local scope",
                    evidence=[{"episode_id": episode, "relation": "support"}],
                )
            )

        assert int(payload["lesson_id"]) > 0
        assert _count(db, "lessons", ns=DEFAULT_NS) == 1

    async def test_distinct_claim_passes_guard(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        episode = _insert_episode(db, goal="incident", embedding=V_Y)
        _insert_lesson_vec(db, claim="existing orthogonal rule", vector=V_X)

        async with client as session:
            payload = _ok(
                await _write(
                    session,
                    claim="a fresh orthogonal lesson",
                    because="different mechanism entirely",
                    evidence=[{"episode_id": episode, "relation": "support"}],
                )
            )

        assert int(payload["lesson_id"]) > 0

    async def test_write_lesson_admits_claim_when_copy_demoted(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        episode_src = _insert_episode(db, goal="src episode", namespace=DEFAULT_NS, embedding=V_Y)
        episode_tgt = _insert_episode(db, goal="tgt episode", namespace=OTHER_NS, embedding=V_Y)

        async with client as session:
            payload1 = _ok(
                await _write(
                    session,
                    claim="promoted then demoted claim",
                    because="source cause",
                    evidence=[{"episode_id": episode_src, "relation": "support"}],
                    namespace=DEFAULT_NS,
                )
            )
            src_id = int(payload1["lesson_id"])
            promoted = _ok(
                await session.call_tool(
                    "memory_promote",
                    {
                        "lesson_id": src_id,
                        "reason": "Promoting to OTHER_NS",
                        "target_namespace": OTHER_NS,
                    },
                )
            )
            copy_id = int(promoted["promoted_lesson_id"])
            _ok(
                await session.call_tool(
                    "memory_demote",
                    {
                        "lesson_id": copy_id,
                        "reason": "Demoting copy in OTHER_NS",
                    },
                )
            )
            payload2 = _ok(
                await _write(
                    session,
                    claim="promoted then demoted claim",
                    because="new independent cause in OTHER_NS",
                    evidence=[{"episode_id": episode_tgt, "relation": "support"}],
                    namespace=OTHER_NS,
                )
            )

        assert int(payload2["lesson_id"]) > 0


class TestWriteLessonLinks:
    @pytest.fixture()
    def fake_embed_overrides(self) -> dict[str, list[float]]:
        return {
            _lesson_text("moderately similar claim", "gist", "scope"): V_SIM,
            _lesson_text("a contradicting lesson", "split confidence per P9", ""): V_Y,
            _lesson_text("another lesson", "also valid", ""): V_Z,
        }

    async def test_similar_links_both_directions_with_cosine_weight(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        episode = _insert_episode(db, goal="incident", embedding=V_Y)
        linked = _insert_lesson_vec(db, claim="strongly related rule", vector=V_X)
        unlinked = _insert_lesson_vec(db, claim="unrelated rule", vector=V_MID)

        async with client as session:
            payload = _ok(
                await _write(
                    session,
                    claim="moderately similar claim",
                    because="gist",
                    holds_when="scope",
                    evidence=[{"episode_id": episode, "relation": "support"}],
                )
            )

        new_id = int(payload["lesson_id"])
        links = db.execute(
            """
            SELECT lesson_id, related_lesson_id, kind, weight FROM lesson_links
            WHERE kind = 'similar'
            """
        ).fetchall()
        pairs = {
            (int(row["lesson_id"]), int(row["related_lesson_id"])): float(row["weight"])
            for row in links
        }
        assert pairs == {
            (new_id, linked): pytest.approx(0.8),
            (linked, new_id): pytest.approx(0.8),
        }
        assert unlinked not in {target for _, target in pairs}

    async def test_contradicts_link_written_one_direction(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        episode = _insert_episode(db, goal="incident", embedding=V_Y)
        opponent = _insert_lesson_vec(db, claim="the opposite rule", vector=V_X)

        async with client as session:
            payload = _ok(
                await _write(
                    session,
                    claim="a contradicting lesson",
                    because="split confidence per P9",
                    evidence=[{"episode_id": episode, "relation": "support"}],
                    contradicts=opponent,
                )
            )
            bogus = _err(
                await _write(
                    session,
                    claim="another lesson",
                    because="also valid",
                    evidence=[{"episode_id": episode, "relation": "support"}],
                    contradicts=444444,
                )
            )

        rows = db.execute(
            """
            SELECT lesson_id, related_lesson_id, kind FROM lesson_links
            WHERE kind = 'contradicts'
            """
        ).fetchall()
        assert [
            (int(row["lesson_id"]), int(row["related_lesson_id"]))
            for row in rows
        ] == [(int(payload["lesson_id"]), opponent)]
        assert "444444" in bogus


class TestScanReplaceScanCycle:
    @pytest.fixture()
    def fake_embed_overrides(self) -> dict[str, list[float]]:
        return {
            _lesson_text("revised rule", "re-derived after dispute", ""): V_Y
        }

    async def test_predecessor_leaves_pending_then_replacement_queues(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        source = _insert_episode(db, goal="disputed source", embedding=V_X)
        backdate("episodes", source, at=T_BASE)
        lesson = _insert_lesson(db, claim="contested rule")
        _insert_evidence(db, lesson, source)
        _dispute(db, lesson)

        async with client as session:
            before = _ok(await _scan(session))
            assert _rederivations(before) == {lesson: {f"episode:{source}"}}

            replacement = _ok(
                await _write(
                    session,
                    claim="revised rule",
                    because="re-derived after dispute",
                    evidence=[{"episode_id": source, "relation": "support"}],
                    replaces_disputed=lesson,
                )
            )
            replacement_id = int(replacement["lesson_id"])
            after = _ok(await _scan(session))
            assert after["rederivation_groups"] == []

            _dispute(db, replacement_id)
            requeued = _ok(await _scan(session))
            assert _rederivations(requeued) == {
                replacement_id: {f"episode:{source}"}
            }

        # refines completion marker + the predecessor's dispute record stays
        refines = db.execute(
            """
            SELECT lesson_id, related_lesson_id FROM lesson_links
            WHERE kind = 'refines'
            """
        ).fetchall()
        assert [
            (int(row["lesson_id"]), int(row["related_lesson_id"])) for row in refines
        ] == [(replacement_id, lesson)]
        predecessor = db.execute(
            "SELECT disputed FROM lessons WHERE id = %(id)s", {"id": lesson}
        ).fetchone()
        assert predecessor is not None and predecessor["disputed"] is True


class TestFullReplacementCycleABC:
    """Round-05 review fix: the FULL A->B->C lineage exemption cycle.

    Issue #6 restatement: every re-derivation shares the disputed
    predecessor's claim VERBATIM, so each replacement is claim-identical
    (cosine exactly 1.0 under FakeEmbedder) to the whole lineage — only the
    exemption admits it, and an unexempted verbatim twin is still rejected.
    """

    CYCLE_CLAIM = "backoff tames retry storms"

    @pytest.fixture()
    def fake_embed_overrides(self) -> dict[str, list[float]]:
        # both composites in the cycle embed to V_X (cos 1.0 pairwise)
        return {
            _lesson_text(self.CYCLE_CLAIM, "storms amplify load", "packet loss"): V_X,
            _lesson_text(
                self.CYCLE_CLAIM, "re-derived after the dispute", "packet loss"
            ): V_X,
        }

    async def test_full_a_b_c_replacement_cycle(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        claim = self.CYCLE_CLAIM
        source = _insert_episode(db, goal="cycle source", embedding=V_X)
        evidence = [{"episode_id": source, "relation": "support"}]

        async with client as session:
            a = _ok(
                await _write(
                    session,
                    claim=claim,
                    because="storms amplify load",
                    holds_when="packet loss",
                    evidence=evidence,
                )
            )
            a_id = int(a["lesson_id"])
            _dispute(db, a_id)

            # B is claim-identical to A (verbatim re-derivation): exempt
            b = _ok(
                await _write(
                    session,
                    claim=claim,
                    because="re-derived after the dispute",
                    holds_when="packet loss",
                    evidence=evidence,
                    replaces_disputed=a_id,
                )
            )
            b_id = int(b["lesson_id"])
            _dispute(db, b_id)

            # C is claim-identical to B AND ancestor A: lineage exempts both
            c = _ok(
                await _write(
                    session,
                    claim=claim,
                    because="storms amplify load",
                    holds_when="packet loss",
                    evidence=evidence,
                    replaces_disputed=b_id,
                )
            )
            c_id = int(c["lesson_id"])

            # a verbatim twin WITHOUT replaces_disputed is rejected
            rejected = _err(
                await _write(
                    session,
                    claim=claim,
                    because="re-derived after the dispute",
                    holds_when="packet loss",
                    evidence=evidence,
                )
            )

        assert "duplicate" in rejected
        refines = db.execute(
            """
            SELECT lesson_id, related_lesson_id FROM lesson_links
            WHERE kind = 'refines' ORDER BY lesson_id
            """
        ).fetchall()
        assert [
            (int(row["lesson_id"]), int(row["related_lesson_id"])) for row in refines
        ] == [(b_id, a_id), (c_id, b_id)]
        assert _count(db, "lessons") == 3
        disputed_rows = db.execute(
            "SELECT count(*) AS n FROM lessons WHERE disputed"
        ).fetchone()
        assert disputed_rows is not None and int(disputed_rows["n"]) == 2  # A and B


class TestConcurrentWriteLesson:
    def test_two_connections_near_identical_claims_exactly_one_wins(
        self,
        pg: str,
        db: psycopg.Connection[DictRow],
    ) -> None:
        """Advisory xact lock: real threads, independent connections, no sleeps."""
        import threading

        from mcp.server.mcpserver.exceptions import ToolError

        from agent_memory.config import Settings
        from agent_memory.consolidation_tools import write_lesson

        episode = _insert_episode(db, goal="raced incident", embedding=V_Y)
        evidence = [{"episode_id": episode, "relation": "support"}]
        settings = Settings(
            DATABASE_URL=pg,
            EMBED_IMPL="fake",
            PGVECTOR_DIM=DIM,
            FAKE_EMBED_OVERRIDES=json.dumps(
                {
                    # one VERBATIM claim, two rationale wordings: the loser
                    # is claim-identical to the winner (Issue #6 bar)
                    _lesson_text("raced claim", "raced cause one", ""): V_X,
                    _lesson_text("raced claim", "raced cause two", ""): V_X,
                }
            ),
        )

        barrier = threading.Barrier(2)
        outcomes: list[tuple[str, Any]] = []
        lock = threading.Lock()

        def racer(because: str) -> None:
            barrier.wait(timeout=30.0)
            try:
                result = write_lesson(
                    settings,
                    claim="raced claim",
                    because=because,
                    holds_when="",
                    evidence=evidence,
                )
                with lock:
                    outcomes.append(("ok", result))
            except ToolError as exc:
                with lock:
                    outcomes.append(("error", str(exc)))

        threads = [
            threading.Thread(target=racer, args=(because,))
            for because in ("raced cause one", "raced cause two")
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60.0)

        kinds = sorted(kind for kind, _ in outcomes)
        assert kinds == ["error", "ok"], outcomes
        winner = next(value for kind, value in outcomes if kind == "ok")
        loser = next(value for kind, value in outcomes if kind == "error")
        assert set(winner) == {"lesson_id", "seed_confidence"}
        assert "duplicate" in loser
        assert _count(db, "lessons") == 1


# === Task 11 marker: memory_corroborate / memory_contradict append here ===

# === Task 11 section: memory_corroborate / memory_contradict ================
# Evidence moves per DESIGN §6 (+0.1 x novelty / -0.2 flat, cap 0.95, floor
# 0.05) and the review-hardened transition table. Adversarial classes:
# flaky_tests — barrier-sync threads + absolute backdate(at=) dates, never
# wall-clock reads; misleading_success_output — every move cross-checked via
# SQL on confidence/edges/last_evidence_at/updated_at; stale_state — per-test
# truncate in conftest; hung_commands — --durations=10 in the tee'd evidence
# runs; scope_drift / sloppy_patching / fake_data_bypass /
# verification_by_vibes — n/a (real stdio server, real Postgres, no mocks).
#
# Seed discipline: lessons come from write_lesson (the REAL path). The 0.50
# seed composition is the task-10 golden — episodes A,B = same-day near-dup
# pair (V_X, day1 12:00/13:00 -> one incident), C = distinct day (V_Y, day3);
# incidents=2, occasions={day1, day3} -> 0.35 + 0.05 + 0.20*0.5 = 0.50 EXACT.
# Novel corroboration episodes sit on UTC dates distinct from EVERY existing
# support episode's date AND carry vectors orthogonal (cos exactly 0.0) to all
# of them — novelty 1.0 by construction, deterministic against DB now().

SEED_CLAIM = "backoff pacing holds under load"
SEED_BECAUSE = "sync retries amplify packet storms"

# mutually-orthogonal novel-episode vectors (dims 4-8; V_X/V_Y own dims 1-2)
V_N1 = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
V_N2 = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
V_N3 = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
V_N4 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
V_N5 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

DAY = timedelta(days=1)
DAY_1 = T_BASE                 # seed near-dup date
DAY_2 = T_BASE + DAY           # between the seed dates (max() no-rewind)
DAY_3 = T_BASE + 2 * DAY       # seed distinct-day date
NOVEL_DAY = T_BASE + 10 * DAY  # first corroboration date (fresh UTC date)

SEED_OVERRIDES = {_lesson_text(SEED_CLAIM, SEED_BECAUSE, ""): V_Z}


def _seed_overrides() -> dict[str, list[float]]:
    """Override channel for the shared 0.50 seed lesson (write_lesson embed)."""
    return dict(SEED_OVERRIDES)


async def _seed_half_confidence_lesson(
    db: psycopg.Connection[DictRow], session: ClientSession
) -> tuple[int, int, int, int]:
    """write_lesson (REAL path) -> confidence exactly 0.50 over 3 support edges.

    Returns (lesson_id, ep_a, ep_b, ep_c); last_evidence_at lands on DAY_3.
    """
    ep_a = _insert_episode(db, goal="seed near-dup one", embedding=V_X)
    ep_b = _insert_episode(db, goal="seed near-dup two", embedding=V_X)
    ep_c = _insert_episode(db, goal="seed distinct day", embedding=V_Y)
    backdate("episodes", ep_a, at=DAY_1)
    backdate("episodes", ep_b, at=DAY_1 + timedelta(hours=1))
    backdate("episodes", ep_c, at=DAY_3)
    payload = _ok(
        await _write(
            session,
            claim=SEED_CLAIM,
            because=SEED_BECAUSE,
            evidence=[
                {"episode_id": ep_a, "relation": "support"},
                {"episode_id": ep_b, "relation": "support"},
                {"episode_id": ep_c, "relation": "support"},
            ],
        )
    )
    assert payload["seed_confidence"] == pytest.approx(0.50)
    return int(payload["lesson_id"]), ep_a, ep_b, ep_c


async def _seed_refine_edge_lesson(
    db: psycopg.Connection[DictRow], session: ClientSession
) -> tuple[int, int, int]:
    """write_lesson (REAL path) with a refine edge: A support day1 + R refine day3.

    Same seed math (incidents 2, occasions {day1, day3}) -> 0.50. Returns
    (lesson_id, ep_a, ep_r); the (lesson, ep_r) edge is relation 'refine'.
    """
    ep_a = _insert_episode(db, goal="refine seed support", embedding=V_X)
    ep_r = _insert_episode(db, goal="refine seed refinement", embedding=V_Y)
    backdate("episodes", ep_a, at=DAY_1)
    backdate("episodes", ep_r, at=DAY_3)
    payload = _ok(
        await _write(
            session,
            claim=SEED_CLAIM,
            because=SEED_BECAUSE,
            evidence=[
                {"episode_id": ep_a, "relation": "support"},
                {"episode_id": ep_r, "relation": "refine", "reason": "boundary case"},
            ],
        )
    )
    assert payload["seed_confidence"] == pytest.approx(0.50)
    return int(payload["lesson_id"]), ep_a, ep_r


async def _corroborate(
    session: ClientSession, lesson_id: int, episode_id: int, reason: str = ""
) -> dict[str, Any]:
    result = _ok(
        await session.call_tool(
            "memory_corroborate",
            {"lesson_id": lesson_id, "episode_id": episode_id, "reason": reason},
        )
    )
    assert result["lesson_id"] == lesson_id
    assert result["episode_id"] == episode_id
    return result


async def _contradict(
    session: ClientSession, lesson_id: int, episode_id: int, reason: str = ""
) -> dict[str, Any]:
    result = _ok(
        await session.call_tool(
            "memory_contradict",
            {"lesson_id": lesson_id, "episode_id": episode_id, "reason": reason},
        )
    )
    assert result["lesson_id"] == lesson_id
    assert result["episode_id"] == episode_id
    return result


def _lesson_state(
    db: psycopg.Connection[DictRow], lesson_id: int
) -> dict[str, Any]:
    row = db.execute(
        """
        SELECT confidence, last_evidence_at, updated_at FROM lessons
        WHERE id = %(id)s
        """,
        {"id": lesson_id},
    ).fetchone()
    assert row is not None
    return {
        "confidence": float(row["confidence"]),
        "last_evidence_at": row["last_evidence_at"],
        "updated_at": row["updated_at"],
    }


def _edge_relations(
    db: psycopg.Connection[DictRow], lesson_id: int
) -> dict[int, str]:
    rows = db.execute(
        "SELECT episode_id, relation FROM lesson_evidence WHERE lesson_id = %(id)s",
        {"id": lesson_id},
    ).fetchall()
    return {int(row["episode_id"]): str(row["relation"]) for row in rows}


def _edge_reason(
    db: psycopg.Connection[DictRow], lesson_id: int, episode_id: int
) -> str:
    row = db.execute(
        """
        SELECT reason FROM lesson_evidence
        WHERE lesson_id = %(lesson_id)s AND episode_id = %(episode_id)s
        """,
        {"lesson_id": lesson_id, "episode_id": episode_id},
    ).fetchone()
    assert row is not None
    return str(row["reason"])


class TestEvidenceMoveValidation:
    @pytest.fixture()
    def fake_embed_overrides(self) -> dict[str, list[float]]:
        return _seed_overrides()

    async def test_nonexistent_lesson_is_error_nothing_written(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        """Failure QA: MCP error result, zero writes, same session recovers."""
        async with client as session:
            lesson, _ep_a, _ep_b, _ep_c = await _seed_half_confidence_lesson(
                db, session
            )
            episode = _insert_episode(db, goal="late incident", embedding=V_N1)
            backdate("episodes", episode, at=NOVEL_DAY)

            bogus_lesson = _err(
                await session.call_tool(
                    "memory_corroborate",
                    {"lesson_id": 555555, "episode_id": episode, "reason": ""},
                )
            )
            bogus_contradict = _err(
                await session.call_tool(
                    "memory_contradict",
                    {"lesson_id": 555555, "episode_id": episode, "reason": ""},
                )
            )
            # the SAME live session keeps serving after the errors
            recovered = await _corroborate(session, lesson, episode)

        assert "555555" in bogus_lesson
        assert "555555" in bogus_contradict
        assert recovered["applied"] is True
        # nothing from the failed calls: 3 seed edges + the one recovery edge
        assert len(_edge_relations(db, lesson)) == 4
        assert _lesson_state(db, lesson)["confidence"] == pytest.approx(0.60)

    async def test_nonexistent_episode_is_error_nothing_written(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        async with client as session:
            lesson, _ep_a, _ep_b, _ep_c = await _seed_half_confidence_lesson(
                db, session
            )
            error = _err(
                await session.call_tool(
                    "memory_corroborate",
                    {"lesson_id": lesson, "episode_id": 987654, "reason": ""},
                )
            )

        assert "987654" in error
        assert len(_edge_relations(db, lesson)) == 3
        assert _lesson_state(db, lesson)["confidence"] == pytest.approx(0.50)


class TestCorroborateTransitions:
    @pytest.fixture()
    def fake_embed_overrides(self) -> dict[str, list[float]]:
        return _seed_overrides()

    async def test_absent_to_support_novel_day_moves_plus_0_1(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        async with client as session:
            lesson, _ep_a, _ep_b, _ep_c = await _seed_half_confidence_lesson(
                db, session
            )
            episode = _insert_episode(db, goal="novel-day confirmation", embedding=V_N1)
            backdate("episodes", episode, at=NOVEL_DAY)
            # namespace override accepted per the every-tool contract (the ids
            # alone scope a move — same inert-namespace precedent as usage)
            moved = _ok(
                await session.call_tool(
                    "memory_corroborate",
                    {
                        "lesson_id": lesson,
                        "episode_id": episode,
                        "reason": "independent confirmation",
                        "namespace": OTHER_NS,
                    },
                )
            )

        assert moved["applied"] is True
        assert moved["relation"] == "support"
        assert moved["confidence"] == pytest.approx(0.60)
        state = _lesson_state(db, lesson)
        assert state["confidence"] == pytest.approx(0.60)
        assert state["last_evidence_at"] == NOVEL_DAY  # refreshed to episode date
        assert _edge_relations(db, lesson)[episode] == "support"

    async def test_absent_to_support_same_day_near_dup_moves_plus_0_03(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        async with client as session:
            lesson, _ep_a, _ep_b, _ep_c = await _seed_half_confidence_lesson(
                db, session
            )
            # same UTC date as the day-1 support episodes -> novelty 0.3
            episode = _insert_episode(db, goal="same-day echo", embedding=V_N1)
            backdate("episodes", episode, at=DAY_1 + timedelta(hours=2))
            moved = await _corroborate(session, lesson, episode)

        assert moved["confidence"] == pytest.approx(0.50 + 0.1 * 0.3)  # 0.53
        assert _lesson_state(db, lesson)["confidence"] == pytest.approx(0.53)
        assert _edge_relations(db, lesson)[episode] == "support"

    async def test_cap_at_0_95_across_novel_corroborations(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        async with client as session:
            lesson, _ep_a, _ep_b, _ep_c = await _seed_half_confidence_lesson(
                db, session
            )
            ladder: list[float] = []
            for index, vector in enumerate((V_N1, V_N2, V_N3, V_N4, V_N5)):
                episode = _insert_episode(
                    db, goal=f"corroboration {index}", embedding=vector
                )
                backdate("episodes", episode, at=NOVEL_DAY + index * DAY)
                moved = await _corroborate(session, lesson, episode)
                assert moved["applied"] is True
                ladder.append(moved["confidence"])

        assert ladder == [pytest.approx(x) for x in (0.60, 0.70, 0.80, 0.90, 0.95)]
        assert _lesson_state(db, lesson)["confidence"] == pytest.approx(0.95)
        assert len(_edge_relations(db, lesson)) == 8  # 3 seed + 5 corroborations

    async def test_refine_to_support_applies_corroborate_delta(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        async with client as session:
            lesson, _ep_a, ep_r = await _seed_refine_edge_lesson(db, session)
            moved = await _corroborate(session, lesson, ep_r)

        # novelty vs the support set {ep_a}: distinct day, orthogonal -> 1.0
        assert moved["applied"] is True
        assert moved["confidence"] == pytest.approx(0.60)
        assert _edge_relations(db, lesson)[ep_r] == "support"  # converted
        assert _lesson_state(db, lesson)["confidence"] == pytest.approx(0.60)

    async def test_contradict_to_support_applies_corroborate_delta(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        async with client as session:
            lesson, _ep_a, _ep_b, _ep_c = await _seed_half_confidence_lesson(
                db, session
            )
            episode = _insert_episode(db, goal="flip target", embedding=V_N1)
            backdate("episodes", episode, at=NOVEL_DAY)
            first = await _contradict(session, lesson, episode)
            second = await _corroborate(session, lesson, episode)

        assert first["confidence"] == pytest.approx(0.30)
        assert second["confidence"] == pytest.approx(0.40)
        assert _edge_relations(db, lesson)[episode] == "support"
        assert _lesson_state(db, lesson)["confidence"] == pytest.approx(0.40)

    async def test_identical_support_retry_is_noop_writing_nothing(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        async with client as session:
            lesson, ep_a, _ep_b, _ep_c = await _seed_half_confidence_lesson(
                db, session
            )
            # pin updated_at so "unchanged" is assertable without now()
            backdate("lessons.updated_at", lesson, at=T_BASE)
            retried = await _corroborate(session, lesson, ep_a, reason="second look")

        assert retried["applied"] is False
        assert retried["confidence"] == pytest.approx(0.50)
        state = _lesson_state(db, lesson)
        assert state["confidence"] == pytest.approx(0.50)
        assert state["last_evidence_at"] == DAY_3  # untouched
        assert state["updated_at"] == T_BASE  # untouched: no write at all
        assert _edge_relations(db, lesson)[ep_a] == "support"
        assert _edge_reason(db, lesson, ep_a) == ""  # reason not overwritten

    async def test_last_evidence_at_never_rewinds_past_current(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        async with client as session:
            lesson, _ep_a, _ep_b, _ep_c = await _seed_half_confidence_lesson(
                db, session
            )
            # day2 sits BETWEEN the seed dates: novel, but older than DAY_3
            episode = _insert_episode(db, goal="older corroboration", embedding=V_N1)
            backdate("episodes", episode, at=DAY_2)
            moved = await _corroborate(session, lesson, episode)

        assert moved["confidence"] == pytest.approx(0.60)
        state = _lesson_state(db, lesson)
        assert state["confidence"] == pytest.approx(0.60)
        assert state["last_evidence_at"] == DAY_3  # max(current, day2) = day3
        assert state["updated_at"] > T_BASE  # the move DID touch the row


class TestContradictTransitions:
    @pytest.fixture()
    def fake_embed_overrides(self) -> dict[str, list[float]]:
        return _seed_overrides()

    async def test_absent_to_contradict_is_flat_minus_0_2_even_for_near_dup(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        async with client as session:
            lesson, _ep_a, _ep_b, _ep_c = await _seed_half_confidence_lesson(
                db, session
            )
            # same-day near-dup of the support evidence: novelty would scale a
            # corroboration to 0.3, but contradiction is FLAT -0.2
            episode = _insert_episode(db, goal="same-day contradiction", embedding=V_N1)
            backdate("episodes", episode, at=DAY_1 + timedelta(hours=3))
            moved = await _contradict(session, lesson, episode)

        assert moved["applied"] is True
        assert moved["relation"] == "contradict"
        assert moved["confidence"] == pytest.approx(0.30)
        assert _lesson_state(db, lesson)["confidence"] == pytest.approx(0.30)
        assert _edge_relations(db, lesson)[episode] == "contradict"

    async def test_floor_at_0_05(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        async with client as session:
            lesson, _ep_a, _ep_b, _ep_c = await _seed_half_confidence_lesson(
                db, session
            )
            ladder: list[float] = []
            for index, vector in enumerate((V_N1, V_N2, V_N3, V_N4)):
                episode = _insert_episode(
                    db, goal=f"contradiction {index}", embedding=vector
                )
                backdate("episodes", episode, at=NOVEL_DAY + index * DAY)
                moved = await _contradict(session, lesson, episode)
                assert moved["applied"] is True
                ladder.append(moved["confidence"])

        assert ladder == [pytest.approx(x) for x in (0.30, 0.10, 0.05, 0.05)]
        state = _lesson_state(db, lesson)
        assert state["confidence"] == pytest.approx(0.05)
        assert len(_edge_relations(db, lesson)) == 7  # every edge still written
        assert state["last_evidence_at"] == NOVEL_DAY + 3 * DAY

    async def test_refine_to_contradict_applies_minus_0_2(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        async with client as session:
            lesson, _ep_a, ep_r = await _seed_refine_edge_lesson(db, session)
            moved = await _contradict(session, lesson, ep_r)

        assert moved["applied"] is True
        assert moved["confidence"] == pytest.approx(0.30)
        assert _edge_relations(db, lesson)[ep_r] == "contradict"
        assert _lesson_state(db, lesson)["confidence"] == pytest.approx(0.30)

    async def test_support_to_contradict_applies_minus_0_2_once(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        """A retried contradiction must not double-penalize (hardening)."""
        async with client as session:
            lesson, _ep_a, _ep_b, _ep_c = await _seed_half_confidence_lesson(
                db, session
            )
            episode = _insert_episode(db, goal="once only", embedding=V_N1)
            backdate("episodes", episode, at=NOVEL_DAY)
            corroborated = await _corroborate(session, lesson, episode)
            contradicted = await _contradict(session, lesson, episode)
            retried = await _contradict(session, lesson, episode)

        assert corroborated["confidence"] == pytest.approx(0.60)
        assert contradicted["confidence"] == pytest.approx(0.40)
        assert retried["applied"] is False
        assert retried["confidence"] == pytest.approx(0.40)
        assert _edge_relations(db, lesson)[episode] == "contradict"
        assert _lesson_state(db, lesson)["confidence"] == pytest.approx(0.40)

    async def test_identical_contradict_retry_is_noop(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        async with client as session:
            lesson, _ep_a, _ep_b, _ep_c = await _seed_half_confidence_lesson(
                db, session
            )
            episode = _insert_episode(db, goal="retry target", embedding=V_N1)
            backdate("episodes", episode, at=NOVEL_DAY)
            first = await _contradict(session, lesson, episode)
            second = await _contradict(session, lesson, episode)

        assert first["confidence"] == pytest.approx(0.30)
        assert second["applied"] is False
        assert second["confidence"] == pytest.approx(0.30)
        assert _lesson_state(db, lesson)["confidence"] == pytest.approx(0.30)
        assert len(_edge_relations(db, lesson)) == 4  # one edge, not two


class TestInterleavedCurrentStateIdempotence:
    """0.50 -> contradict 0.30 -> support 0.40 -> contradict 0.20 (sanctioned).

    Idempotence is CURRENT-STATE-ONLY by design: a replayed transition after
    an intervening change is indistinguishable from an intentional new move
    (no operation-identity contract in v1) — this test DOCUMENTS it.
    """

    @pytest.fixture()
    def fake_embed_overrides(self) -> dict[str, list[float]]:
        return _seed_overrides()

    async def test_support_contradict_support_contradict_lands_at_0_20(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        async with client as session:
            lesson, _ep_a, _ep_b, _ep_c = await _seed_half_confidence_lesson(
                db, session
            )
            episode = _insert_episode(db, goal="interleaved target", embedding=V_N1)
            backdate("episodes", episode, at=NOVEL_DAY)
            first = await _contradict(session, lesson, episode)
            second = await _corroborate(session, lesson, episode)
            third = await _contradict(session, lesson, episode)

        assert first["confidence"] == pytest.approx(0.30)
        assert second["confidence"] == pytest.approx(0.40)
        assert third["confidence"] == pytest.approx(0.20)
        assert third["applied"] is True  # an intentional new move, not a retry
        assert _lesson_state(db, lesson)["confidence"] == pytest.approx(0.20)
        assert _edge_relations(db, lesson)[episode] == "contradict"
        assert len(_edge_relations(db, lesson)) == 4  # one edge per (lesson, ep)


class TestConcurrentCorroborations:
    def test_two_threads_two_connections_both_moves_land(
        self,
        pg: str,
        db: psycopg.Connection[DictRow],
    ) -> None:
        """FOR UPDATE serialization: no lost update (0.50 -> 0.70, not 0.60)."""
        import threading

        from agent_memory.config import Settings
        from agent_memory.consolidation_tools import corroborate, write_lesson

        settings = Settings(
            DATABASE_URL=pg,
            EMBED_IMPL="fake",
            PGVECTOR_DIM=DIM,
            FAKE_EMBED_OVERRIDES=json.dumps(_seed_overrides()),
        )
        ep_a = _insert_episode(db, goal="race near-dup one", embedding=V_X)
        ep_b = _insert_episode(db, goal="race near-dup two", embedding=V_X)
        ep_c = _insert_episode(db, goal="race distinct day", embedding=V_Y)
        backdate("episodes", ep_a, at=DAY_1)
        backdate("episodes", ep_b, at=DAY_1 + timedelta(hours=1))
        backdate("episodes", ep_c, at=DAY_3)
        seeded = write_lesson(
            settings,
            claim=SEED_CLAIM,
            because=SEED_BECAUSE,
            evidence=[
                {"episode_id": ep_a, "relation": "support"},
                {"episode_id": ep_b, "relation": "support"},
                {"episode_id": ep_c, "relation": "support"},
            ],
        )
        lesson = int(seeded["lesson_id"])

        racers: list[tuple[datetime, Sequence[float]]] = [
            (NOVEL_DAY, V_N1),
            (NOVEL_DAY + DAY, V_N2),
        ]
        episodes: list[int] = []
        for day, vector in racers:
            episode = _insert_episode(db, goal=f"raced {day.date()}", embedding=vector)
            backdate("episodes", episode, at=day)
            episodes.append(episode)

        barrier = threading.Barrier(2)
        outcomes: list[tuple[str, Any]] = []
        lock = threading.Lock()

        def racer(episode_id: int) -> None:
            barrier.wait(timeout=30.0)
            try:
                result = corroborate(
                    settings,
                    lesson_id=lesson,
                    episode_id=episode_id,
                    reason="raced corroboration",
                )
                with lock:
                    outcomes.append(("ok", result))
            except Exception as exc:  # noqa: BLE001 - recorded, asserted below
                with lock:
                    outcomes.append(("error", repr(exc)))

        threads = [
            threading.Thread(target=racer, args=(episode_id,))
            for episode_id in episodes
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60.0)

        kinds = sorted(kind for kind, _ in outcomes)
        assert kinds == ["ok", "ok"], outcomes
        confidences = sorted(
            float(payload["confidence"]) for _, payload in outcomes
        )
        # serialized: one move lands on 0.60, the second reads 0.60 -> 0.70
        assert confidences == [pytest.approx(0.60), pytest.approx(0.70)]
        state = _lesson_state(db, lesson)
        assert state["confidence"] == pytest.approx(0.70)  # no lost update
        assert len(_edge_relations(db, lesson)) == 5  # both edges landed


# === F4 fix: namespace='global' is promotion-only ===========================
# DESIGN §11 copy-with-provenance invariant: global lessons exist ONLY as
# promoted copies (INSERT_PROMOTED_LESSON_SQL carries promoted_from_lesson_id
# provenance); memory_write_lesson must refuse 'global' as its effective
# target namespace — mirrors promote's own source==target rejection style.


class TestWriteLessonGlobalNamespace:
    @pytest.fixture()
    def fake_embed_overrides(self) -> dict[str, list[float]]:
        return {
            _lesson_text("global via write", "bypass attempt", ""): V_Y,
            _lesson_text("local lesson", "local cause", ""): V_X,
        }

    async def test_global_namespace_is_error_session_recovers(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        """Failure QA: global target -> MCP error, zero rows, same session
        keeps serving a normal-namespace write afterwards."""
        episode = _insert_episode(db, goal="incident", embedding=V_Y)

        async with client as session:
            error = _err(
                await _write(
                    session,
                    claim="global via write",
                    because="bypass attempt",
                    evidence=[{"episode_id": episode, "relation": "support"}],
                    namespace="global",
                )
            )
            recovered = _ok(
                await _write(
                    session,
                    claim="local lesson",
                    because="local cause",
                    evidence=[{"episode_id": episode, "relation": "support"}],
                )
            )

        assert "global" in error
        assert "promote" in error
        assert _count(db, "lessons", ns="global") == 0
        assert _count(db, "lessons") == 1  # only the recovery write landed
        assert int(recovered["lesson_id"]) > 0

    async def test_promote_still_lands_global_lesson(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        """Pairing assertion: the SANCTIONED global path still works."""
        episode = _insert_episode(db, goal="incident", embedding=V_Y)

        async with client as session:
            written = _ok(
                await _write(
                    session,
                    claim="local lesson",
                    because="local cause",
                    evidence=[{"episode_id": episode, "relation": "support"}],
                )
            )
            promoted = _ok(
                await session.call_tool(
                    "memory_promote",
                    {"lesson_id": written["lesson_id"], "reason": "broadly useful"},
                )
            )

        row = db.execute(
            """
            SELECT namespace, promoted_from_lesson_id FROM lessons
            WHERE id = %(id)s
            """,
            {"id": promoted["promoted_lesson_id"]},
        ).fetchone()
        assert row is not None
        assert row["namespace"] == "global"
        assert int(row["promoted_from_lesson_id"]) == int(written["lesson_id"])
        assert _count(db, "lessons", ns="global") == 1


# === Issue #9 section: reason-field secret screening on the write paths =====


class TestReasonSecretScreen:
    """Issue #9: evidence-reason strings on write_lesson and the evidence
    moves are screened (capture contract) — field named, secret never echoed,
    nothing written, same session recovers."""

    async def test_write_lesson_evidence_reason_secret_rejected_then_clean_recovers(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        secret = "sk-AbCdEf0123456789AbCdEf0123456789"
        episode = _insert_episode(db, goal="incident", embedding=V_Y)
        async with client as session:
            error = _err(
                await _write(
                    session,
                    claim="c",
                    because="b",
                    evidence=[
                        {"episode_id": episode, "relation": "support", "reason": f"see {secret}"}
                    ],
                )
            )
            assert "reason" in error
            assert secret not in error
            assert _count(db, "lessons") == 0
            assert _count(db, "lesson_evidence") == 0

            recovered = _ok(
                await _write(
                    session,
                    claim="c",
                    because="b",
                    evidence=[
                        {"episode_id": episode, "relation": "support", "reason": "clean"}
                    ],
                )
            )
        assert set(recovered) == {"lesson_id", "seed_confidence"}
        assert _count(db, "lessons") == 1  # only the recovery write landed
        assert _count(db, "lesson_evidence") == 1

    async def test_corroborate_reason_secret_rejected_then_clean_recovers(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        secret = "sk-AbCdEf0123456789AbCdEf0123456789"
        seed = _insert_episode(db, goal="seed incident", embedding=V_X)
        novel = _insert_episode(db, goal="novel incident", embedding=V_Y)
        async with client as session:
            written = _ok(
                await _write(
                    session,
                    claim="c",
                    because="b",
                    evidence=[{"episode_id": seed, "relation": "support"}],
                )
            )
            lesson_id = int(written["lesson_id"])
            before = _lesson_state(db, lesson_id)["confidence"]

            message = _err(
                await session.call_tool(
                    "memory_corroborate",
                    {
                        "lesson_id": lesson_id,
                        "episode_id": novel,
                        "reason": f"see {secret}",
                    },
                )
            )
            assert "reason" in message
            assert secret not in message
            assert _count(db, "lesson_evidence") == 1  # the seeding edge only
            assert _lesson_state(db, lesson_id)["confidence"] == before

            moved = await _corroborate(session, lesson_id, novel, "clean")
        assert moved["applied"] is True
        assert _count(db, "lesson_evidence") == 2

    async def test_contradict_reason_secret_rejected_then_clean_recovers(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        secret = "sk-AbCdEf0123456789AbCdEf0123456789"
        seed = _insert_episode(db, goal="seed incident", embedding=V_X)
        counter = _insert_episode(db, goal="counter incident", embedding=V_Y)
        async with client as session:
            written = _ok(
                await _write(
                    session,
                    claim="c",
                    because="b",
                    evidence=[{"episode_id": seed, "relation": "support"}],
                )
            )
            lesson_id = int(written["lesson_id"])
            before = _lesson_state(db, lesson_id)["confidence"]

            message = _err(
                await session.call_tool(
                    "memory_contradict",
                    {
                        "lesson_id": lesson_id,
                        "episode_id": counter,
                        "reason": f"see {secret}",
                    },
                )
            )
            assert "reason" in message
            assert secret not in message
            assert _count(db, "lesson_evidence") == 1  # the seeding edge only
            assert _lesson_state(db, lesson_id)["confidence"] == before

            moved = await _contradict(session, lesson_id, counter, "clean")
        assert moved["applied"] is True
        after = _lesson_state(db, lesson_id)["confidence"]
        assert after == pytest.approx(max(before - 0.2, 0.05))
        assert _count(db, "lesson_evidence") == 2


# === Issue #6 section: two-threshold duplicate-lesson guard ================
# Claim identity is the rejection bar: claim-only cosine > DUP_CLAIM_COS
# rejects (verbatim claims with different rationale no longer slip past);
# the replaces_disputed lineage is the sanctioned verbatim path; composite
# cosine governs ONLY similar links. Under FakeEmbedder, verbatim claims
# embed to cosine exactly 1.0 (same text -> same hash vector) while distinct
# claims land near-random (<< 0.95): claim pairs come from plain text reuse,
# paraphrase pairs from distinct claim texts.

_LEGACY_CLAIM = "back up before schema sweeps"


class TestTwoThresholdClaimGuard:
    @pytest.fixture()
    def fake_embed_overrides(self) -> dict[str, list[float]]:
        return {
            # paraphrase test: composite cos 0.8 to the fixture's stored V_X
            _lesson_text(
                "load shedders trip well before overload",
                "eager tripping keeps latency bounded",
                "traffic bursts",
            ): V_SIM,
        }

    async def test_verbatim_claim_different_because_rejected(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        """The Issue #6 repro shape: identical claim, different rationale."""
        episode = _insert_episode(db, goal="incident", embedding=V_Y)
        _insert_lesson(db, claim="always pin the base image digest")

        async with client as session:
            error = _err(
                await _write(
                    session,
                    claim="always pin the base image digest",
                    because="a wholly different one-line rationale",
                    evidence=[{"episode_id": episode, "relation": "support"}],
                )
            )

        assert "duplicate" in error
        assert _count(db, "lessons") == 1  # the fixture only
        assert _count(db, "lesson_evidence") == 0

    async def test_claim_cosine_exactly_one_rejected(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        """Boundary: the claim bar fires at the maximum cosine, 1.000."""
        episode = _insert_episode(db, goal="incident", embedding=V_Y)
        _insert_lesson(db, claim="retry budgets cap total attempts")

        async with client as session:
            error = _err(
                await _write(
                    session,
                    claim="retry budgets cap total attempts",
                    because="identical claim and identical rationale text",
                    evidence=[{"episode_id": episode, "relation": "support"}],
                )
            )

        assert "duplicate" in error
        assert "1.000" in error

    async def test_verbatim_claim_via_replaces_disputed_lineage_admitted(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        """Re-derivations share their claim by design: lineage exempts."""
        episode = _insert_episode(db, goal="incident", embedding=V_Y)
        predecessor = _insert_lesson(db, claim="page only on sustained errors")
        _dispute(db, predecessor)

        async with client as session:
            replacement = _ok(
                await _write(
                    session,
                    claim="page only on sustained errors",
                    because="re-derived after the dispute",
                    evidence=[{"episode_id": episode, "relation": "support"}],
                    replaces_disputed=predecessor,
                )
            )

        replacement_id = int(replacement["lesson_id"])
        assert _count(db, "lessons") == 2
        refines = db.execute(
            """
            SELECT lesson_id, related_lesson_id FROM lesson_links
            WHERE kind = 'refines'
            """
        ).fetchall()
        assert [
            (int(row["lesson_id"]), int(row["related_lesson_id"])) for row in refines
        ] == [(replacement_id, predecessor)]

    async def test_near_claim_paraphrase_admitted_with_similar_links(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        """Different claim, similar document: linked, never rejected."""
        episode = _insert_episode(db, goal="incident", embedding=V_Y)
        existing = _insert_lesson_vec(
            db, claim="circuit breakers shed load gracefully", vector=V_X
        )

        async with client as session:
            payload = _ok(
                await _write(
                    session,
                    claim="load shedders trip well before overload",
                    because="eager tripping keeps latency bounded",
                    holds_when="traffic bursts",
                    evidence=[{"episode_id": episode, "relation": "support"}],
                )
            )

        new_id = int(payload["lesson_id"])
        links = db.execute(
            """
            SELECT lesson_id, related_lesson_id, weight FROM lesson_links
            WHERE kind = 'similar'
            """
        ).fetchall()
        pairs = {
            (int(row["lesson_id"]), int(row["related_lesson_id"])): float(row["weight"])
            for row in links
        }
        assert (new_id, existing) in pairs
        assert (existing, new_id) in pairs


class TestLegacyNullClaimEmbedding:
    @pytest.fixture()
    def fake_embed_overrides(self) -> dict[str, list[float]]:
        return {
            # composite pinned to the legacy row's V_X -> similar links fire
            _lesson_text(
                "distinct claim about schema migration",
                "mid-flight rows corrupt otherwise",
                "",
            ): V_X,
        }

    async def test_null_claim_row_healed_and_different_claim_still_links(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        """Pre-backfill rows (claim_embedding NULL) carry no claim identity initially:
        the lazy heal populates their claim_embedding, non-duplicate claims are admitted,
        and composite similarity links are created.
        """
        episode = _insert_episode(db, goal="incident", embedding=V_Y)

        async with client as session:
            row = db.execute(
                """
                INSERT INTO lessons (namespace, claim, because, embedding)
                VALUES (%(ns)s, %(claim)s, 'legacy gist', %(embedding)s)
                RETURNING id
                """,
                {
                    "ns": DEFAULT_NS,
                    "claim": _LEGACY_CLAIM,
                    "embedding": pgvector.Vector(V_X),
                },
            ).fetchone()
            assert row is not None
            legacy = int(row["id"])
            db.commit()

            payload = _ok(
                await _write(
                    session,
                    claim="distinct claim about schema migration",
                    because="mid-flight rows corrupt otherwise",
                    evidence=[{"episode_id": episode, "relation": "support"}],
                )
            )

        new_id = int(payload["lesson_id"])
        db.commit()
        links = db.execute(
            """
            SELECT lesson_id, related_lesson_id FROM lesson_links
            WHERE kind = 'similar'
            """
        ).fetchall()
        assert {
            (int(link["lesson_id"]), int(link["related_lesson_id"])) for link in links
        } == {(new_id, legacy), (legacy, new_id)}

        legacy_after = db.execute(
            "SELECT claim_embedding FROM lessons WHERE id = %(id)s",
            {"id": legacy},
        ).fetchone()
        assert legacy_after is not None
        assert legacy_after["claim_embedding"] is not None

    async def test_null_claim_embedding_healed_before_duplicate_bar(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        """Rolling-upgrade simulation: an older server inserts a lesson with
        claim_embedding = NULL after migration 003's startup backfill.

        write_lesson with the verbatim claim must lazily heal the legacy row's
        NULL claim_embedding in DB, then reject the new claim as a duplicate.
        """
        episode = _insert_episode(db, goal="incident", embedding=V_Y)

        async with client as session:
            row = db.execute(
                """
                INSERT INTO lessons (namespace, claim, because, embedding)
                VALUES (%(ns)s, %(claim)s, 'legacy gist', %(embedding)s)
                RETURNING id
                """,
                {
                    "ns": DEFAULT_NS,
                    "claim": _LEGACY_CLAIM,
                    "embedding": pgvector.Vector(V_X),
                },
            ).fetchone()
            assert row is not None
            legacy = int(row["id"])
            db.commit()

            legacy_before = db.execute(
                "SELECT claim_embedding FROM lessons WHERE id = %(id)s",
                {"id": legacy},
            ).fetchone()
            assert legacy_before is not None
            assert legacy_before["claim_embedding"] is None

            error = _err(
                await _write(
                    session,
                    claim=_LEGACY_CLAIM,
                    because="mid-flight rows corrupt otherwise",
                    evidence=[{"episode_id": episode, "relation": "support"}],
                )
            )
            assert "duplicate claim" in error

            db.commit()
            legacy_after = db.execute(
                "SELECT claim_embedding FROM lessons WHERE id = %(id)s",
                {"id": legacy},
            ).fetchone()
            assert legacy_after is not None
            assert legacy_after["claim_embedding"] is not None

    async def test_between_windows_null_row_rejected_not_crashed_heal_commits(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        """Round-4 review: heal and guard share ONE lock window.

        The two-window shape let an old-server writer slip a NULL-claim
        lesson between the heal transaction and the guard's re-lock; the
        guard's cosine then called .to_list() on NULL and crashed the write
        instead of rejecting it. A serial test cannot interleave a writer
        INSIDE the lock window (the xact lock serializes namespace writers —
        injecting one there would also break the fixed code's contract), so
        this pins the visible behavior of the single window: the state the
        between-windows writer leaves behind (a NULL-claim row) meets the
        verbatim write as a structured duplicate-claim REJECTION — text a
        crash can never produce — and the heal COMMITS despite the
        rejection because the reject is raised outside the transaction.
        """
        episode = _insert_episode(db, goal="racing incident", embedding=V_Y)

        async with client as session:
            row = db.execute(
                """
                INSERT INTO lessons (namespace, claim, because, embedding)
                VALUES (%(ns)s, %(claim)s, 'racing writer gist', %(embedding)s)
                RETURNING id
                """,
                {
                    "ns": DEFAULT_NS,
                    "claim": _LEGACY_CLAIM,
                    "embedding": pgvector.Vector(V_X),
                },
            ).fetchone()
            assert row is not None
            racer = int(row["id"])
            db.commit()

            rejected = _err(
                await _write(
                    session,
                    claim=_LEGACY_CLAIM,
                    because="a verbatim twin of the racing claim",
                    evidence=[{"episode_id": episode, "relation": "support"}],
                )
            )
            assert "duplicate claim" in rejected

            # The heal committed even though the write was rejected.
            db.commit()
            healed = db.execute(
                """
                SELECT claim_embedding IS NOT NULL AS healed,
                       claim_embedding <=> %(vec)s AS dist
                FROM lessons WHERE id = %(id)s
                """,
                {"id": racer, "vec": pgvector.Vector(_claim_vector(_LEGACY_CLAIM))},
            ).fetchone()
            assert healed is not None
            assert healed["healed"] is True
            assert healed["dist"] == pytest.approx(0.0)

            # A different claim in the same window shape is still admitted.
            admitted = _ok(
                await _write(
                    session,
                    claim="unrelated to the racing claim",
                    because="a genuinely distinct mechanism",
                    evidence=[{"episode_id": episode, "relation": "support"}],
                )
            )
            assert int(admitted["lesson_id"]) > 0
        assert _count(db, "lessons") == 2  # the racing row + the admitted write


class TestClaimEmbeddingMigrationBackfill:
    def test_003_applies_after_002_and_backfills_preexisting_lessons(
        self,
        pg: str,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Upgrade path: a lesson written pre-003 gets its claim embedded by
        the migrate() Python backfill (embed(claim), dim-matched), and a
        second migrate pass is a full no-op."""
        real_dir = agent_db.MIGRATIONS_DIR
        pre_003 = tmp_path / "pre003"
        pre_003.mkdir()
        for name in ("001_init.sql", "002_dispute_reason.sql"):
            shutil.copy(real_dir / name, pre_003 / name)
        legacy_claim = "migrations must run before serving traffic"
        with PostgresContainer("pgvector/pgvector:pg16") as container:
            url = container.get_connection_url(driver=None)
            previous = os.environ.get("DATABASE_URL")
            os.environ["DATABASE_URL"] = url
            get_settings.cache_clear()
            try:
                monkeypatch.setattr(agent_db, "MIGRATIONS_DIR", pre_003)
                assert agent_db.migrate(dim=DIM) == [
                    "001_init.sql",
                    "002_dispute_reason.sql",
                ]
                with psycopg.connect(url, autocommit=True) as conn:
                    legacy = conn.execute(
                        """
                        INSERT INTO lessons (namespace, claim, because, embedding)
                        VALUES (
                            'default@local', %(claim)s, 'legacy gist',
                            '[1,0,0,0,0,0,0,0]'
                        )
                        RETURNING id
                        """,
                        {"claim": legacy_claim},
                    ).fetchone()
                assert legacy is not None

                monkeypatch.setattr(agent_db, "MIGRATIONS_DIR", real_dir)
                assert agent_db.migrate(dim=DIM) == [
                    "003_claim_embedding.sql",
                    "004_lessons_namespace_index.sql",
                ]
                assert agent_db.migrate(dim=DIM) == []  # idempotent no-op
            finally:
                if previous is None:
                    os.environ.pop("DATABASE_URL", None)
                else:
                    os.environ["DATABASE_URL"] = previous
                get_settings.cache_clear()
            with psycopg.connect(url, autocommit=True) as conn:
                stored = conn.execute(
                    "SELECT claim_embedding::text FROM lessons WHERE id = %(id)s",
                    {"id": legacy[0]},
                ).fetchone()
        assert stored is not None and stored[0] is not None
        floats = [float(part) for part in str(stored[0]).strip("[]").split(",")]
        expected = FakeEmbedder(dim=DIM).embed([legacy_claim])[0]
        assert floats == pytest.approx(expected)  # float32 round-trip tolerance
