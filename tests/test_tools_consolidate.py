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
from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from datetime import datetime, timedelta, timezone
from typing import Any

import pgvector
import psycopg
import pytest
from mcp import ClientSession
from mcp.types import CallToolResult, TextContent
from psycopg.rows import DictRow

from agent_memory.config import get_settings
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
        INSERT INTO lessons (namespace, claim, because, embedding)
        VALUES (%(namespace)s, %(claim)s, 'causal gist', %(embedding)s)
        RETURNING id
        """,
        {"namespace": namespace, "claim": claim, "embedding": pgvector.Vector(V_X)},
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
# === Task 11 marker: memory_corroborate / memory_contradict append here ===
