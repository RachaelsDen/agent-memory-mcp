"""Contract tests for memory_report_usage: explicit verdicts, stat moves, atomicity.

Verdicts attach to a retrieval_event's returned_ids (typed refs
"episode:<id>"/"lesson:<id>"); every batch is all-or-nothing. usage_reports
rows and episodes/lessons stats are asserted via SQL on the ``db`` fixture —
never via tool replies alone (a misleading success output must not pass).
Most retrieval_events are hand-inserted via direct SQL for determinism; one
test drives the genuine path end-to-end (capture -> probe -> report on the
probe's own retrieval_event_id).
"""

import json
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from typing import Any

import pgvector
import psycopg
import pytest
from mcp import ClientSession
from mcp.types import CallToolResult, TextContent
from psycopg.rows import DictRow

DIM = 8
V_Q = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
DEFAULT_NS = "default@local"

CAPTURE_TEXT = (
    "stabilize blue pipeline flaky specs stay green "
    "quarantine flaky spec main pipeline went green"
)
PROBE_TEXT = "stabilize blue pipeline quarantine flaky spec"


def _insert_episode(
    db: psycopg.Connection[DictRow],
    *,
    goal: str,
    outcome: str = "",
    embedding: list[float],
) -> int:
    row = db.execute(
        """
        INSERT INTO episodes (namespace, goal, outcome, raw_text, embedding)
        VALUES (%(namespace)s, %(goal)s, %(outcome)s, '', %(embedding)s)
        RETURNING id
        """,
        {
            "namespace": DEFAULT_NS,
            "goal": goal,
            "outcome": outcome,
            "embedding": pgvector.Vector(embedding),
        },
    ).fetchone()
    assert row is not None
    return int(row["id"])


def _insert_lesson(
    db: psycopg.Connection[DictRow],
    *,
    claim: str,
    embedding: list[float],
) -> int:
    row = db.execute(
        """
        INSERT INTO lessons (namespace, claim, because, confidence, embedding,
                             promotion_status, promoted_from_lesson_id, disputed)
        VALUES (%(namespace)s, %(claim)s, 'causal gist', 0.5, %(embedding)s,
                'active', NULL, FALSE)
        RETURNING id
        """,
        {
            "namespace": DEFAULT_NS,
            "claim": claim,
            "embedding": pgvector.Vector(embedding),
        },
    ).fetchone()
    assert row is not None
    return int(row["id"])


def _insert_event(
    db: psycopg.Connection[DictRow], returned_ids: list[str]
) -> int:
    row = db.execute(
        """
        INSERT INTO retrieval_events (namespace, tool, query_context, returned_ids)
        VALUES (%(namespace)s, 'probe', 'qc', %(returned_ids)s)
        RETURNING id
        """,
        {"namespace": DEFAULT_NS, "returned_ids": returned_ids},
    ).fetchone()
    assert row is not None
    return int(row["id"])


async def _report(
    session: ClientSession,
    retrieval_event_id: int,
    results: list[dict[str, str]],
    namespace: str | None = None,
) -> CallToolResult:
    args: dict[str, Any] = {
        "retrieval_event_id": retrieval_event_id,
        "results": results,
    }
    if namespace is not None:
        args["namespace"] = namespace
    return await session.call_tool("memory_report_usage", args)


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


def _episode_stats(db: psycopg.Connection[DictRow], episode_id: int) -> DictRow:
    row = db.execute(
        "SELECT access_count, last_accessed FROM episodes WHERE id = %(id)s",
        {"id": episode_id},
    ).fetchone()
    assert row is not None
    return row


def _lesson_stats(db: psycopg.Connection[DictRow], lesson_id: int) -> DictRow:
    row = db.execute(
        """
        SELECT access_count, last_accessed, usefulness
        FROM lessons WHERE id = %(id)s
        """,
        {"id": lesson_id},
    ).fetchone()
    assert row is not None
    return row


def _usage_rows(db: psycopg.Connection[DictRow]) -> list[DictRow]:
    return db.execute(
        """
        SELECT retrieval_event_id, record_id, record_type, outcome
        FROM usage_reports ORDER BY id
        """
    ).fetchall()


def _usage_count(db: psycopg.Connection[DictRow]) -> int:
    row = db.execute("SELECT count(*) AS n FROM usage_reports").fetchone()
    assert row is not None
    return int(row["n"])


class TestStatMoves:
    async def test_used_updates_episode_stats(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        episode_id = _insert_episode(db, goal="ship the widget", embedding=V_Q)
        event_id = _insert_event(db, [f"episode:{episode_id}"])
        async with client as session:
            payload = _ok(await _report(session, event_id, [
                {"id": f"episode:{episode_id}", "outcome": "used"}
            ]))
        assert payload["reported"] == 1
        stats = _episode_stats(db, episode_id)
        assert stats["access_count"] == 1
        assert stats["last_accessed"] is not None
        rows = _usage_rows(db)
        assert len(rows) == 1
        assert rows[0]["retrieval_event_id"] == event_id
        assert rows[0]["record_id"] == f"episode:{episode_id}"
        assert rows[0]["record_type"] == "episode"
        assert rows[0]["outcome"] == "used"

    async def test_helped_and_harmed_move_lesson_usefulness(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        helped_id = _insert_lesson(db, claim="cache the digest", embedding=V_Q)
        harmed_id = _insert_lesson(db, claim="skip the tests", embedding=V_Q)
        event_id = _insert_event(
            db, [f"lesson:{helped_id}", f"lesson:{harmed_id}"]
        )
        async with client as session:
            _ok(await _report(session, event_id, [
                {"id": f"lesson:{helped_id}", "outcome": "helped"},
                {"id": f"lesson:{harmed_id}", "outcome": "harmed"},
            ]))
        helped = _lesson_stats(db, helped_id)
        assert helped["access_count"] == 1
        assert helped["last_accessed"] is not None
        assert helped["usefulness"] == pytest.approx(1.0)
        harmed = _lesson_stats(db, harmed_id)
        assert harmed["access_count"] == 1
        assert harmed["last_accessed"] is not None
        assert harmed["usefulness"] == pytest.approx(-1.0)

    async def test_ignored_writes_report_only(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        episode_id = _insert_episode(db, goal="ship the widget", embedding=V_Q)
        lesson_id = _insert_lesson(db, claim="cache the digest", embedding=V_Q)
        event_id = _insert_event(
            db, [f"episode:{episode_id}", f"lesson:{lesson_id}"]
        )
        async with client as session:
            _ok(await _report(session, event_id, [
                {"id": f"episode:{episode_id}", "outcome": "ignored"},
                {"id": f"lesson:{lesson_id}", "outcome": "ignored"},
            ]))
        rows = _usage_rows(db)
        assert [row["outcome"] for row in rows] == ["ignored", "ignored"]
        episode = _episode_stats(db, episode_id)
        assert episode["access_count"] == 0
        assert episode["last_accessed"] is None
        lesson = _lesson_stats(db, lesson_id)
        assert lesson["access_count"] == 0
        assert lesson["last_accessed"] is None
        assert lesson["usefulness"] == pytest.approx(0.0)


class TestBatchAtomicity:
    async def test_multi_verdict_batch_applies_all(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        episode_id = _insert_episode(db, goal="ship the widget", embedding=V_Q)
        helped_id = _insert_lesson(db, claim="cache the digest", embedding=V_Q)
        ignored_id = _insert_lesson(db, claim="skip the tests", embedding=V_Q)
        event_id = _insert_event(
            db,
            [
                f"episode:{episode_id}",
                f"lesson:{helped_id}",
                f"lesson:{ignored_id}",
            ],
        )
        async with client as session:
            payload = _ok(await _report(
                session,
                event_id,
                [
                    {"id": f"episode:{episode_id}", "outcome": "used"},
                    {"id": f"lesson:{helped_id}", "outcome": "helped"},
                    {"id": f"lesson:{ignored_id}", "outcome": "ignored"},
                ],
                namespace=DEFAULT_NS,
            ))
        assert payload["reported"] == 3
        assert _usage_count(db) == 3
        episode = _episode_stats(db, episode_id)
        assert episode["access_count"] == 1
        assert episode["last_accessed"] is not None
        helped = _lesson_stats(db, helped_id)
        assert helped["access_count"] == 1
        assert helped["usefulness"] == pytest.approx(1.0)
        ignored = _lesson_stats(db, ignored_id)
        assert ignored["access_count"] == 0
        assert ignored["last_accessed"] is None
        assert ignored["usefulness"] == pytest.approx(0.0)

    async def test_batch_with_one_invalid_entry_writes_nothing(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        episode_id = _insert_episode(db, goal="ship the widget", embedding=V_Q)
        lesson_id = _insert_lesson(db, claim="cache the digest", embedding=V_Q)
        event_id = _insert_event(
            db, [f"episode:{episode_id}", f"lesson:{lesson_id}"]
        )
        async with client as session:
            error = _err(await _report(session, event_id, [
                {"id": f"episode:{episode_id}", "outcome": "used"},
                {"id": f"lesson:{lesson_id}", "outcome": "harmed"},
                {"id": "episode:999", "outcome": "used"},
            ]))
        assert "not among" in error
        assert _usage_count(db) == 0
        episode = _episode_stats(db, episode_id)
        assert episode["access_count"] == 0
        assert episode["last_accessed"] is None
        lesson = _lesson_stats(db, lesson_id)
        assert lesson["access_count"] == 0
        assert lesson["usefulness"] == pytest.approx(0.0)


class TestValidation:
    async def test_id_not_in_returned_ids_is_error(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        episode_id = _insert_episode(db, goal="ship the widget", embedding=V_Q)
        _insert_lesson(db, claim="cache the digest", embedding=V_Q)
        event_id = _insert_event(db, [f"episode:{episode_id}"])
        async with client as session:
            error = _err(await _report(session, event_id, [
                {"id": "lesson:1", "outcome": "used"}
            ]))
        assert "not among" in error
        assert _usage_count(db) == 0
        assert _episode_stats(db, episode_id)["access_count"] == 0

    async def test_invalid_outcome_is_error(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        episode_id = _insert_episode(db, goal="ship the widget", embedding=V_Q)
        event_id = _insert_event(db, [f"episode:{episode_id}"])
        async with client as session:
            error = _err(await _report(session, event_id, [
                {"id": f"episode:{episode_id}", "outcome": "maybe"}
            ]))
        assert "outcome" in error
        assert _usage_count(db) == 0
        assert _episode_stats(db, episode_id)["access_count"] == 0

    async def test_duplicate_verdict_second_call_error_original_untouched(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        episode_id = _insert_episode(db, goal="ship the widget", embedding=V_Q)
        event_id = _insert_event(db, [f"episode:{episode_id}"])
        async with client as session:
            _ok(await _report(session, event_id, [
                {"id": f"episode:{episode_id}", "outcome": "used"}
            ]))
            error = _err(await _report(session, event_id, [
                {"id": f"episode:{episode_id}", "outcome": "harmed"}
            ]))
        assert "duplicate" in error
        rows = _usage_rows(db)
        assert len(rows) == 1
        assert rows[0]["outcome"] == "used"  # first verdict survives untouched
        assert _episode_stats(db, episode_id)["access_count"] == 1

    async def test_duplicate_verdict_within_batch_is_error(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        episode_id = _insert_episode(db, goal="ship the widget", embedding=V_Q)
        event_id = _insert_event(db, [f"episode:{episode_id}"])
        async with client as session:
            error = _err(await _report(session, event_id, [
                {"id": f"episode:{episode_id}", "outcome": "used"},
                {"id": f"episode:{episode_id}", "outcome": "helped"},
            ]))
        assert "duplicate" in error
        assert _usage_count(db) == 0
        assert _episode_stats(db, episode_id)["access_count"] == 0
        assert _episode_stats(db, episode_id)["last_accessed"] is None


class TestFabricatedEvent:
    """Failure QA: fabricated retrieval_event_id -> error, nothing written, session usable."""

    async def test_fabricated_event_is_error_nothing_written_session_usable(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        episode_id = _insert_episode(db, goal="ship the widget", embedding=V_Q)
        async with client as session:
            error = _err(await _report(session, 424242, [
                {"id": f"episode:{episode_id}", "outcome": "used"}
            ]))
            assert "does not exist" in error
            assert _usage_count(db) == 0

            # same live session must keep serving: a valid report succeeds
            event_id = _insert_event(db, [f"episode:{episode_id}"])
            _ok(await _report(session, event_id, [
                {"id": f"episode:{episode_id}", "outcome": "used"}
            ]))
        assert _usage_count(db) == 1
        assert _episode_stats(db, episode_id)["access_count"] == 1


class TestGenuineProbeEvent:
    """End-to-end: capture -> probe -> report against the probe's own event."""

    @pytest.fixture()
    def fake_embed_overrides(self) -> dict[str, list[float]]:
        return {CAPTURE_TEXT: V_Q, PROBE_TEXT: V_Q}

    async def test_report_against_real_probe_event(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        async with client as session:
            result = await session.call_tool(
                "memory_capture_episode",
                {
                    "goal": "stabilize blue pipeline",
                    "expectation": "flaky specs stay green",
                    "action": "quarantine flaky spec",
                    "outcome": "main pipeline went green",
                },
            )
            capture = _ok(result)
            episode_id = capture["id"]
            assert isinstance(episode_id, int)

            probe = _ok(await session.call_tool("memory_probe", {
                "current_goal": "stabilize blue pipeline",
                "approach": "quarantine flaky spec",
            }))
            event_id = probe["retrieval_event_id"]
            returned = [str(record["id"]) for record in probe["results"]]
            assert f"episode:{episode_id}" in returned

            # retrieval itself must not move usage stats (exposure != usage)
            stats = _episode_stats(db, episode_id)
            assert stats["access_count"] == 0
            assert stats["last_accessed"] is None

            _ok(await _report(session, event_id, [
                {"id": f"episode:{episode_id}", "outcome": "used"}
            ]))
        stats = _episode_stats(db, episode_id)
        assert stats["access_count"] == 1
        assert stats["last_accessed"] is not None
        rows = _usage_rows(db)
        assert len(rows) == 1
        assert rows[0]["retrieval_event_id"] == event_id
        assert rows[0]["record_id"] == f"episode:{episode_id}"
        assert rows[0]["outcome"] == "used"
