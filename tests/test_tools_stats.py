"""Contract tests for memory_dispute / memory_stats (plan task 13, own file).

Dispute marks a lesson for human-flagged rederivation: disputed=true plus
the reason (migration 002's lessons.dispute_reason), the lesson STAYS
retrievable, every probe result carrying it includes "disputed": true and
"dispute_reason", and the next consolidate_scan queues it as a pending
rederivation group — the task-9 cycle this tool feeds. Stats is the
namespace health check: counts, the unconsolidated backlog, and four flag
buckets, each pinned here from crafted fixtures built on the REAL
write_lesson / corroborate / contradict / promote / demote / probe /
report_usage paths (no hand-inserted lessons); row state is asserted via
SQL on the ``db`` fixture so a misleading success payload cannot pass.

The rare_critical_stale boundary is anchored to ONE DB reference
timestamp: the cutoff instant is computed in Python with the SAME formula
the SQL parameterizes (make_interval(secs => 3600*tau*ln(1/stale))) and
one record is placed just inside, one just outside via backdate(at=).
Access-count fixtures ride the real probe -> report_usage cycle (six
verdicts cross access_count > 5; five do not). The migration upgrade test
proves the task-2 filename-compare runner catches a PENDING 002 against a
database that already has 001 applied — not a table-existence check.
"""

import json
import math
import os
import shutil
from contextlib import AbstractAsyncContextManager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pgvector
import psycopg
import pytest
from mcp import ClientSession
from mcp.types import CallToolResult, TextContent
from psycopg.rows import DictRow
from testcontainers.community.postgres import PostgresContainer

import agent_memory.db as agent_db
from agent_memory.config import get_settings
from tests.conftest import backdate

DIM = 8
V1 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
V2 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
V3 = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
V4 = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
V5 = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
V6 = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]

DEFAULT_NS = "default@local"  # server settings.MEMORY_NAMESPACE default
THIRD_NS = "third@watch"
GLOBAL = "global"

# Fixed past anchor for episode dates (novelty wants distinct UTC dates).
T_BASE = datetime(2026, 3, 10, 12, 0, tzinfo=timezone.utc)
DAY_1 = T_BASE
DAY_3 = T_BASE + timedelta(days=2)
DAY_5 = T_BASE + timedelta(days=4)
DAY_7 = T_BASE + timedelta(days=6)
DAY_9 = T_BASE + timedelta(days=8)

CLAIM_1 = "backoff pacing holds under load"
BECAUSE_1 = "sync retries amplify packet storms"
CLAIM_2 = "index hints poison the planner"
BECAUSE_2 = "stale statistics mislead the optimizer"

# write_lesson embeds claim + " " + because + " " + holds_when (trailing
# space with the empty default holds_when, same as capture's approach).
LESSON_TEXT_1 = f"{CLAIM_1} {BECAUSE_1} "
LESSON_TEXT_2 = f"{CLAIM_2} {BECAUSE_2} "

PROBE_SUFFIX = "probe approach"


def _probe_text(claim: str) -> str:
    """Probe query text; keyed to the lesson's vector via fake overrides."""
    return f"{claim} {PROBE_SUFFIX}"


def _insert_episode(
    db: psycopg.Connection[DictRow],
    *,
    goal: str,
    embedding: list[float],
    at: datetime,
    namespace: str = DEFAULT_NS,
) -> int:
    row = db.execute(
        """
        INSERT INTO episodes (namespace, goal, outcome, raw_text, embedding)
        VALUES (%(namespace)s, %(goal)s, '', '', %(embedding)s)
        RETURNING id
        """,
        {
            "namespace": namespace,
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
    claim: str,
    because: str,
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    return _ok(
        await session.call_tool(
            "memory_write_lesson",
            {"claim": claim, "because": because, "evidence": evidence},
        )
    )


async def _dispute(
    session: ClientSession, lesson_id: int, reason: str, namespace: str | None = None
) -> CallToolResult:
    arguments: dict[str, Any] = {"lesson_id": lesson_id, "reason": reason}
    if namespace is not None:
        arguments["namespace"] = namespace
    return await session.call_tool("memory_dispute", arguments)


async def _stats(session: ClientSession, namespace: str | None = None) -> dict[str, Any]:
    arguments: dict[str, Any] = {}
    if namespace is not None:
        arguments["namespace"] = namespace
    return _ok(await session.call_tool("memory_stats", arguments))


async def _bump_access(
    session: ClientSession, *, goal: str, record_ref: str, times: int
) -> None:
    """Real usage path: probe (embeds goal + " " + PROBE_SUFFIX), report used."""
    for _ in range(times):
        payload = _ok(
            await session.call_tool(
                "memory_probe",
                {"current_goal": goal, "approach": PROBE_SUFFIX},
            )
        )
        assert any(entry["id"] == record_ref for entry in payload["results"])
        reported = _ok(
            await session.call_tool(
                "memory_report_usage",
                {
                    "retrieval_event_id": payload["retrieval_event_id"],
                    "results": [{"id": record_ref, "outcome": "used"}],
                },
            )
        )
        assert reported == {"reported": 1}


def _lesson_confidence(db: psycopg.Connection[DictRow], lesson_id: int) -> float:
    row = db.execute(
        "SELECT confidence FROM lessons WHERE id = %(id)s", {"id": lesson_id}
    ).fetchone()
    assert row is not None
    return float(row["confidence"])


async def _corroborate_to(
    session: ClientSession,
    *,
    lesson_id: int,
    novel_episodes: list[int],
) -> float:
    """Novelty-1.0 corroboration ladder (+0.1 each); returns final confidence."""
    confidence = 0.0
    for episode_id in novel_episodes:
        payload = _ok(
            await session.call_tool(
                "memory_corroborate",
                {"lesson_id": lesson_id, "episode_id": episode_id},
            )
        )
        confidence = float(payload["confidence"])
    return confidence


class TestStatsEmpty:
    @pytest.fixture()
    def fake_embed_overrides(self) -> dict[str, list[float]]:
        return {}

    async def test_empty_namespace_all_zero(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        async with client as session:
            payload = await _stats(session)
        assert payload == {
            "namespace": DEFAULT_NS,
            "episode_count": 0,
            "lesson_count": 0,
            "unconsolidated_backlog": 0,
            "popular_but_shaky": [],
            "rare_critical_stale": [],
            "cross_cutting_episodes": [],
            "demoted_promotions": [],
        }


class TestStatsCountsAndBacklog:
    """episode_count / lesson_count / unconsolidated_backlog + namespace scope."""

    @pytest.fixture()
    def fake_embed_overrides(self) -> dict[str, list[float]]:
        return {LESSON_TEXT_1: V1}

    async def test_counts_and_backlog_from_crafted_fixtures(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        cited = _insert_episode(db, goal="cited once", embedding=V3, at=DAY_1)
        _insert_episode(db, goal="never cited", embedding=V4, at=DAY_3)
        async with client as session:
            await _write(
                session,
                claim=CLAIM_1,
                because=BECAUSE_1,
                evidence=[{"episode_id": cited, "relation": "support"}],
            )
            payload = await _stats(session)
            other_ns = await _stats(session, namespace=THIRD_NS)
        assert payload["episode_count"] == 2
        assert payload["lesson_count"] == 1
        assert payload["unconsolidated_backlog"] == 1
        # Strict namespace scoping: the third namespace sees none of it.
        assert other_ns == {
            "namespace": THIRD_NS,
            "episode_count": 0,
            "lesson_count": 0,
            "unconsolidated_backlog": 0,
            "popular_but_shaky": [],
            "rare_critical_stale": [],
            "cross_cutting_episodes": [],
            "demoted_promotions": [],
        }


class TestPopularButShaky:
    """confidence < 0.3 AND access_count > 5 — both conjuncts, access boundary."""

    @pytest.fixture()
    def fake_embed_overrides(self) -> dict[str, list[float]]:
        return {
            LESSON_TEXT_1: V1,
            _probe_text(CLAIM_1): V1,
            LESSON_TEXT_2: V2,
            _probe_text(CLAIM_2): V2,
        }

    async def test_boundary_and_conjuncts(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        shaky_seed = _insert_episode(db, goal="shaky seed", embedding=V1, at=DAY_1)
        contradiction = _insert_episode(db, goal="shaky contradict", embedding=V3, at=DAY_3)
        high_seed = _insert_episode(db, goal="high conf seed", embedding=V2, at=DAY_1)
        async with client as session:
            shaky = await _write(
                session,
                claim=CLAIM_1,
                because=BECAUSE_1,
                evidence=[{"episode_id": shaky_seed, "relation": "support"}],
            )
            high = await _write(
                session,
                claim=CLAIM_2,
                because=BECAUSE_2,
                evidence=[{"episode_id": high_seed, "relation": "support"}],
            )
            shaky_id = int(shaky["lesson_id"])
            high_id = int(high["lesson_id"])
            # one flat contradict: 0.40 seed -> 0.20 < 0.3
            demoted_conf = _ok(
                await session.call_tool(
                    "memory_contradict",
                    {"lesson_id": shaky_id, "episode_id": contradiction},
                )
            )
            assert demoted_conf["confidence"] == pytest.approx(0.20)

            # five usage verdicts leave access_count at 5 — NOT > 5.
            await _bump_access(
                session,
                goal=CLAIM_1,
                record_ref=f"lesson:{shaky_id}",
                times=5,
            )
            payload = await _stats(session)
            assert payload["popular_but_shaky"] == []

            # the sixth crosses the boundary; the high-confidence control
            # (0.40, access 6) proves the confidence conjunct.
            await _bump_access(
                session,
                goal=CLAIM_1,
                record_ref=f"lesson:{shaky_id}",
                times=1,
            )
            await _bump_access(
                session,
                goal=CLAIM_2,
                record_ref=f"lesson:{high_id}",
                times=6,
            )
            payload = await _stats(session)
        assert payload["popular_but_shaky"] == [
            {
                "id": f"lesson:{shaky_id}",
                "claim": CLAIM_1,
                "confidence": pytest.approx(0.20),
                "access_count": 6,
            }
        ]
        assert _lesson_confidence(db, high_id) == pytest.approx(0.40)


class TestRareCriticalStale:
    """confidence > SALIENCE_STALE AND env_fresh < STALE_ENV_FRESH.

    Boundary anchored to ONE DB reference timestamp; the expected cutoff is
    computed in Python with the same formula the SQL parameterizes.
    """

    @pytest.fixture()
    def fake_embed_overrides(self) -> dict[str, list[float]]:
        return {LESSON_TEXT_1: V1, LESSON_TEXT_2: V2}

    async def test_boundary_both_sides_of_one_db_reference(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        settings = get_settings()
        assert settings.TAU_ENV_H == 4320
        assert settings.STALE_ENV_FRESH == pytest.approx(0.2)
        assert settings.SALIENCE_STALE == pytest.approx(0.7)

        stale_seed = _insert_episode(db, goal="stale seed", embedding=V1, at=DAY_1)
        fresh_seed = _insert_episode(db, goal="fresh seed", embedding=V2, at=DAY_1)
        low_seed = _insert_episode(db, goal="low conf seed", embedding=V5, at=DAY_1)
        # same vectors are per-lesson-novel: novelty compares only within one
        # lesson's own support evidence, so both lessons reuse V3..V6.
        novels = [
            _insert_episode(db, goal=f"novel {i}", embedding=vector, at=day)
            for i, (vector, day) in enumerate(
                [(V3, DAY_3), (V4, DAY_5), (V5, DAY_7), (V6, DAY_9)], start=1
            )
        ]
        ref_row = db.execute("SELECT now() AS now").fetchone()
        assert ref_row is not None
        db_now: datetime = ref_row["now"]
        threshold = timedelta(
            seconds=3600.0 * settings.TAU_ENV_H * math.log(1.0 / settings.STALE_ENV_FRESH)
        )
        margin = timedelta(hours=1)  # dwarfs query-execution drift; deterministic
        older_than_cutoff = db_now - threshold - margin  # stale side
        newer_than_cutoff = db_now - threshold + margin  # fresh side

        async with client as session:
            stale = await _write(
                session,
                claim=CLAIM_1,
                because=BECAUSE_1,
                evidence=[{"episode_id": stale_seed, "relation": "support"}],
            )
            fresh = await _write(
                session,
                claim=CLAIM_2,
                because=BECAUSE_2,
                evidence=[{"episode_id": fresh_seed, "relation": "support"}],
            )
            low = await _write(
                session,
                claim="low confidence stale lesson",
                because="seeded once, never corroborated",
                evidence=[{"episode_id": low_seed, "relation": "support"}],
            )
            stale_id = int(stale["lesson_id"])
            fresh_id = int(fresh["lesson_id"])
            low_id = int(low["lesson_id"])
            # 0.40 seed + 4 x (+0.1 novelty-1.0) = 0.80 — clear of the 0.7
            # conjunct and of float32 rounding.
            assert await _corroborate_to(
                session, lesson_id=stale_id, novel_episodes=novels
            ) == pytest.approx(0.80)
            assert await _corroborate_to(
                session, lesson_id=fresh_id, novel_episodes=novels
            ) == pytest.approx(0.80)

            # A demoted high-confidence copy with stale evidence must NOT
            # reappear in a health bucket (its audit home is the tombstone).
            promoted = _ok(
                await session.call_tool(
                    "memory_promote",
                    {"lesson_id": stale_id, "reason": "graduate"},
                )
            )
            copy_id = int(promoted["promoted_lesson_id"])
            _ok(
                await session.call_tool(
                    "memory_demote",
                    {"lesson_id": copy_id, "reason": "retired"},
                )
            )

            # Backdates ride their own connection inside the one client
            # block (the conftest client is one-shot per test).
            backdate("lessons.last_evidence_at", stale_id, at=older_than_cutoff)
            backdate("lessons.last_evidence_at", fresh_id, at=newer_than_cutoff)
            backdate("lessons.last_evidence_at", copy_id, at=older_than_cutoff)
            backdate("lessons.last_evidence_at", low_id, at=db_now - timedelta(days=400))

            payload = await _stats(session)
        assert payload["rare_critical_stale"] == [
            {
                "id": f"lesson:{stale_id}",
                "claim": CLAIM_1,
                "confidence": pytest.approx(0.80),
                "last_evidence_at": older_than_cutoff.isoformat(),
            }
        ]
        listed_ids = {entry["id"] for entry in payload["rare_critical_stale"]}
        assert f"lesson:{fresh_id}" not in listed_ids  # just inside the cutoff
        assert f"lesson:{low_id}" not in listed_ids  # stale but confidence 0.40
        assert f"lesson:{copy_id}" not in listed_ids  # demoted tombstone


class TestCrossCuttingEpisodes:
    """Episodes cited by >= 2 DISTINCT lessons vs one vs none."""

    @pytest.fixture()
    def fake_embed_overrides(self) -> dict[str, list[float]]:
        return {LESSON_TEXT_1: V1, LESSON_TEXT_2: V2}

    async def test_two_citing_lessons(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        shared = _insert_episode(db, goal="shared episode", embedding=V3, at=DAY_1)
        single = _insert_episode(db, goal="single-cite episode", embedding=V4, at=DAY_3)
        _insert_episode(db, goal="uncited episode", embedding=V5, at=DAY_5)
        async with client as session:
            first = await _write(
                session,
                claim=CLAIM_1,
                because=BECAUSE_1,
                evidence=[
                    {"episode_id": shared, "relation": "support"},
                    {"episode_id": single, "relation": "support"},
                ],
            )
            second = await _write(
                session,
                claim=CLAIM_2,
                because=BECAUSE_2,
                evidence=[{"episode_id": shared, "relation": "support"}],
            )
            payload = await _stats(session)
        assert payload["cross_cutting_episodes"] == [
            {
                "id": f"episode:{shared}",
                "citing_lessons": [
                    f"lesson:{int(first['lesson_id'])}",
                    f"lesson:{int(second['lesson_id'])}",
                ],
            }
        ]


class TestDemotedPromotions:
    """Tombstone view: reason + dates, visible from the copy's and the
    source's namespaces, invisible from unrelated ones."""

    @pytest.fixture()
    def fake_embed_overrides(self) -> dict[str, list[float]]:
        return {LESSON_TEXT_1: V1, LESSON_TEXT_2: V2}

    async def test_tombstone_fields_and_scoping(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        retired_seed = _insert_episode(db, goal="retired seed", embedding=V1, at=DAY_1)
        active_seed = _insert_episode(db, goal="active seed", embedding=V2, at=DAY_1)
        async with client as session:
            retired = await _write(
                session,
                claim=CLAIM_1,
                because=BECAUSE_1,
                evidence=[{"episode_id": retired_seed, "relation": "support"}],
            )
            active = await _write(
                session,
                claim=CLAIM_2,
                because=BECAUSE_2,
                evidence=[{"episode_id": active_seed, "relation": "support"}],
            )
            retired_id = int(retired["lesson_id"])
            promoted_retired = _ok(
                await session.call_tool(
                    "memory_promote",
                    {"lesson_id": retired_id, "reason": "was broadly useful"},
                )
            )
            promoted_active = _ok(
                await session.call_tool(
                    "memory_promote",
                    {"lesson_id": int(active["lesson_id"]), "reason": "still useful"},
                )
            )
            copy_id = int(promoted_retired["promoted_lesson_id"])
            _ok(
                await session.call_tool(
                    "memory_demote",
                    {"lesson_id": copy_id, "reason": "broke under new load"},
                )
            )
            from_default = await _stats(session)
            from_global = await _stats(session, namespace=GLOBAL)
            from_third = await _stats(session, namespace=THIRD_NS)

        entry = {
            "id": f"lesson:{copy_id}",
            "claim": CLAIM_1,
            "namespace": GLOBAL,
            "promoted_from_lesson_id": retired_id,
            "promotion_reason": "was broadly useful",
            "demotion_reason": "broke under new load",
        }
        for payload in (from_default, from_global):
            assert len(payload["demoted_promotions"]) == 1
            listed = payload["demoted_promotions"][0]
            assert {key: listed[key] for key in entry} == entry
            datetime.fromisoformat(listed["promoted_at"])
            datetime.fromisoformat(listed["demoted_at"])
        # the active promotion is not a tombstone; an unrelated namespace
        # sees neither the copy's nor the source's tombstone.
        assert int(promoted_active["promoted_lesson_id"]) != copy_id
        assert from_default["demoted_promotions"][0]["id"] == f"lesson:{copy_id}"
        assert from_third["demoted_promotions"] == []


class TestDispute:
    """Marks + reason storage, probe exposure, scan feed, id scoping."""

    @pytest.fixture()
    def fake_embed_overrides(self) -> dict[str, list[float]]:
        return {
            LESSON_TEXT_1: V1,
            _probe_text(CLAIM_1): V1,
            LESSON_TEXT_2: V2,
            _probe_text(CLAIM_2): V2,
        }

    async def test_mark_expose_and_rederive(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        reason_1 = "wrong premise: the fixture drifted"
        reason_2 = "updated reason after review"
        seed_1 = _insert_episode(db, goal="seed one", embedding=V1, at=DAY_1)
        seed_2 = _insert_episode(db, goal="seed two", embedding=V2, at=DAY_1)
        async with client as session:
            first = await _write(
                session,
                claim=CLAIM_1,
                because=BECAUSE_1,
                evidence=[{"episode_id": seed_1, "relation": "support"}],
            )
            second = await _write(
                session,
                claim=CLAIM_2,
                because=BECAUSE_2,
                evidence=[{"episode_id": seed_2, "relation": "support"}],
            )
            lesson_id = int(first["lesson_id"])
            other_id = int(second["lesson_id"])

            # namespace is accepted per the every-tool contract and inert:
            # the lesson_id alone scopes the dispute.
            payload = _ok(await _dispute(session, lesson_id, reason_1, namespace=THIRD_NS))
            assert payload == {"lesson_id": lesson_id, "disputed": True}

            row = db.execute(
                "SELECT disputed, dispute_reason FROM lessons WHERE id = %(id)s",
                {"id": lesson_id},
            ).fetchone()
            assert row is not None
            assert row["disputed"] is True
            assert row["dispute_reason"] == reason_1

            # Disputed lessons REMAIN retrievable, now carrying both keys;
            # every other probe entry (the cited episode shares V1, the
            # undisputed lesson probes separately) carries neither.
            probed = _ok(
                await session.call_tool(
                    "memory_probe",
                    {"current_goal": CLAIM_1, "approach": PROBE_SUFFIX},
                )
            )
            seen_disputed = 0
            for entry in probed["results"]:
                if entry["id"] == f"lesson:{lesson_id}":
                    seen_disputed += 1
                    assert entry["disputed"] is True
                    assert entry["dispute_reason"] == reason_1
                else:
                    assert "disputed" not in entry
                    assert "dispute_reason" not in entry
            assert seen_disputed == 1
            other_probed = _ok(
                await session.call_tool(
                    "memory_probe",
                    {"current_goal": CLAIM_2, "approach": PROBE_SUFFIX},
                )
            )
            other = [e for e in other_probed["results"] if e["id"] == f"lesson:{other_id}"]
            assert other and "disputed" not in other[0] and "dispute_reason" not in other[0]

            # The dispute feeds the task-9 cycle: the next scan queues the
            # lesson for rederivation.
            scan = _ok(await session.call_tool("memory_consolidate_scan", {}))
            assert [group["lesson_id"] for group in scan["rederivation_groups"]] == [
                lesson_id
            ]

            # Re-dispute overwrites the reason (plain UPDATE semantics).
            _ok(await _dispute(session, lesson_id, reason_2))
            updated = db.execute(
                "SELECT disputed, dispute_reason FROM lessons WHERE id = %(id)s",
                {"id": lesson_id},
            ).fetchone()
            assert updated is not None
            assert updated["dispute_reason"] == reason_2

    async def test_dispute_nonexistent_lesson_is_error(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        """QA (failure): disputing a lesson that does not exist."""
        async with client as session:
            message = _err(await _dispute(session, 424242, "no such lesson"))
        assert "424242" in message


class TestStaleEnvFreshValidation:
    """0 < STALE_ENV_FRESH < 1 is validated before ln() ever runs."""

    def test_out_of_range_settings_raise_tool_error(self, pg: str) -> None:
        from mcp.server.mcpserver.exceptions import ToolError

        from agent_memory.config import Settings
        from agent_memory.oversight import stats as oversight_stats

        for bad in (0.0, 1.0, 1.5, -0.2):
            settings = Settings(
                DATABASE_URL=pg,
                EMBED_IMPL="fake",
                PGVECTOR_DIM=DIM,
                STALE_ENV_FRESH=bad,
            )
            with pytest.raises(ToolError, match="STALE_ENV_FRESH"):
                oversight_stats(settings)


class TestMigrationUpgrade:
    """With 001 applied and 002 added, migrate applies 002 — proving the
    task-2 filename-compare runner (not a table-existence check) catches
    pending migrations on upgrade, which serve() startup relies on."""

    def test_pending_002_caught_by_filename_compare_runner(
        self, pg: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        real_dir = agent_db.MIGRATIONS_DIR
        only_001 = tmp_path / "only001"
        only_001.mkdir()
        shutil.copy(real_dir / "001_init.sql", only_001 / "001_init.sql")
        with PostgresContainer("pgvector/pgvector:pg16") as container:
            url = container.get_connection_url(driver=None)
            previous = os.environ.get("DATABASE_URL")
            os.environ["DATABASE_URL"] = url
            get_settings.cache_clear()
            try:
                monkeypatch.setattr(agent_db, "MIGRATIONS_DIR", only_001)
                assert agent_db.migrate(dim=8) == ["001_init.sql"]
                monkeypatch.setattr(agent_db, "MIGRATIONS_DIR", real_dir)
                # every table from 001 already exists — only the bookkeeping
                # comparison can see that 002 is pending.
                assert agent_db.migrate(dim=8) == ["002_dispute_reason.sql"]
                assert agent_db.migrate(dim=8) == []
            finally:
                if previous is None:
                    os.environ.pop("DATABASE_URL", None)
                else:
                    os.environ["DATABASE_URL"] = previous
                get_settings.cache_clear()
            with psycopg.connect(url, autocommit=True) as conn:
                names = {
                    row[0] for row in conn.execute("SELECT name FROM agent_memory_migrations")
                }
                assert names == {"001_init.sql", "002_dispute_reason.sql"}
                column = conn.execute(
                    """
                    SELECT count(*) FROM information_schema.columns
                    WHERE table_name = 'lessons' AND column_name = 'dispute_reason'
                    """
                ).fetchone()
                assert column is not None
                assert column[0] == 1
