"""Contract tests for memory_digest (plan task 14, own file).

The digest renders <DIGEST_DIR>/<slug>-<sha256(namespace)[:8]>-<date>.md with
DESIGN S10's six audit sections in verbatim order. Adversarial classes:

- stale_state: the snapshot source is the LATEST VALID PRIOR DIGEST across ALL
  dates (cross-date rename tests; the weekly cron must not regress to the
  no-snapshot fallback just because the date rolled), and the snapshot is READ
  BEFORE the replacement digest is written (second same-day run must compare
  against the pre-overwrite values, not a truncated file).
- misleading_success_output: every assertion parses the WRITTEN FILE
  (frontmatter via yaml.safe_load + section bodies) instead of trusting the
  returned path/flagged payload; row state comes via SQL on the ``db`` fixture.
- flaky_tests: dates are anchored (T_BASE fixed UTC instants; the DB-side
  staleness boundary uses one SELECT now() reference), and filename dates are
  parsed from the tool's returned path rather than assumed.
- hung_commands: tee'd runs use --durations=10.

Collision fixtures are the REAL generated pair from planning: '//_//___///__/
/_/' and '____///_///_//___/' share an 18-underscore slug AND sha256 prefix
a777eebc (re-verified by assertion in-test), plus the a/b vs a_b same-slug
pair. The floor-adjacent 0.055 -> 0.05 drop pins REAL float32 storage
precision: the snapshot serializes the float64 of the stored float32 with full
repr precision, compared with only a 1e-6 tolerance.
"""

import hashlib
import json
import re
import sys
import tempfile
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
from typing import Any

import pgvector
import psycopg
import pytest
import yaml
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import CallToolResult, TextContent
from psycopg.rows import DictRow

from tests.conftest import backdate

DIM = 8
V1 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
V2 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
V3 = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
V4 = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
V5 = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
V6 = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
V7 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]

DEFAULT_NS = "default@local"  # server settings.MEMORY_NAMESPACE default

# The REAL collision pair from planning (slug AND sha256[:8] identical).
NS_A = "//_//___///__//_/_"
NS_B = "____///_///_//___/"
COLLISION_SLUG = "_" * 18
COLLISION_BASE = "a777eebc"
RETRY_B1 = hashlib.sha256((NS_B + "#1").encode()).hexdigest()[:8]

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
CLAIM_3 = "canary deploys catch regressions early"
BECAUSE_3 = "small blast radius limits customer harm"

LESSON_TEXT_1 = f"{CLAIM_1} {BECAUSE_1} "
LESSON_TEXT_2 = f"{CLAIM_2} {BECAUSE_2} "
LESSON_TEXT_3 = f"{CLAIM_3} {BECAUSE_3} "

PROBE_SUFFIX = "probe approach"

# 160 chars: pins the 140-char excerpt truncation convention.
EXCERPT_OUTCOME = "excerpt-marker " + "d" * 160

SECTION_HEADERS = (
    "## 1. Disputed",
    "## 2. Recently contradicted",
    "## 3. Popular-but-shaky",
    "## 4. Rare-critical-stale",
    "## 5. Recently demoted promotions",
    "## 6. Unconsolidated backlog",
)


def _slug(namespace: str) -> str:
    """Independent slug implementation (never imported from the module)."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", namespace)


def _base_hash(namespace: str) -> str:
    return hashlib.sha256(namespace.encode()).hexdigest()[:8]


def _insert_episode(
    db: psycopg.Connection[DictRow],
    *,
    goal: str,
    embedding: list[float],
    at: datetime,
    outcome: str = "",
    namespace: str = DEFAULT_NS,
) -> int:
    row = db.execute(
        """
        INSERT INTO episodes (namespace, goal, outcome, raw_text, embedding)
        VALUES (%(namespace)s, %(goal)s, %(outcome)s, '', %(embedding)s)
        RETURNING id
        """,
        {
            "namespace": namespace,
            "goal": goal,
            "outcome": outcome,
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
    namespace: str | None = None,
) -> dict[str, Any]:
    arguments: dict[str, Any] = {"claim": claim, "because": because, "evidence": evidence}
    if namespace is not None:
        arguments["namespace"] = namespace
    return _ok(await session.call_tool("memory_write_lesson", arguments))


async def _digest(
    session: ClientSession, namespace: str | None = None
) -> dict[str, Any]:
    arguments: dict[str, Any] = {}
    if namespace is not None:
        arguments["namespace"] = namespace
    return _ok(await session.call_tool("memory_digest", arguments))


def _read_digest(path: Path) -> tuple[dict[str, Any], str]:
    """Parse the written file itself — never trust the returned payload."""
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "---"
    end = lines.index("---", 1)
    frontmatter: dict[str, Any] = yaml.safe_load("\n".join(lines[1:end]))
    return frontmatter, "\n".join(lines[end + 1 :])


def _digest_date(path: Path) -> str:
    matched = re.search(r"(\d{4}-\d{2}-\d{2})\.md$", path.name)
    assert matched is not None, path.name
    return matched.group(1)


def _yesterday_name(path: Path) -> str:
    """Same file, date component rewound one day (frontmatter has no date)."""
    date = _digest_date(path)
    yesterday = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=1)).strftime(
        "%Y-%m-%d"
    )
    return path.name.replace(date, yesterday)


def _prefix_files(path: Path, prefix: str) -> list[Path]:
    return sorted(entry for entry in path.parent.iterdir() if entry.name.startswith(prefix))


def _set_confidence(db: psycopg.Connection[DictRow], lesson_id: int, value: float) -> None:
    db.execute(
        "UPDATE lessons SET confidence = %(value)s WHERE id = %(id)s",
        {"value": value, "id": lesson_id},
    )


def _confidence(db: psycopg.Connection[DictRow], lesson_id: int) -> float:
    row = db.execute(
        "SELECT confidence FROM lessons WHERE id = %(id)s", {"id": lesson_id}
    ).fetchone()
    assert row is not None
    return float(row["confidence"])


@asynccontextmanager
async def _spawn(pg: str, *, digest_dir: str) -> AsyncIterator[ClientSession]:
    """Server with an explicit DIGEST_DIR (conftest's client pins its own)."""
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "agent_memory"],
        env={
            "DATABASE_URL": pg,
            "EMBED_IMPL": "fake",
            "PGVECTOR_DIM": str(DIM),
            "FAKE_EMBED_OVERRIDES": "{}",
            "DIGEST_DIR": digest_dir,
        },
    )
    async with stdio_client(parameters) as (read_stream, write_stream):
        async with ClientSession(
            read_stream, write_stream, read_timeout_seconds=60.0
        ) as session:
            await session.initialize()
            yield session


class TestDigestBasics:
    @pytest.fixture()
    def fake_embed_overrides(self) -> dict[str, list[float]]:
        return {LESSON_TEXT_1: V1}

    async def test_expected_filename_frontmatter_and_flagged_count(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        seed = _insert_episode(db, goal="seed", embedding=V1, at=DAY_1)
        async with client as session:
            written = await _write(
                session,
                claim=CLAIM_1,
                because=BECAUSE_1,
                evidence=[{"episode_id": seed, "relation": "support"}],
            )
            lesson_id = int(written["lesson_id"])
            payload = await _digest(session)
            # File reads stay INSIDE the block: the client fixture removes
            # the digest dir at teardown.
            parsed = _read_digest(Path(payload["path"]))

        path = Path(payload["path"])
        date = _digest_date(path)
        assert path.name == f"{_slug(DEFAULT_NS)}-{_base_hash(DEFAULT_NS)}-{date}.md"
        assert payload["flagged"] == []
        assert payload["flagged_count"] == 0

        frontmatter, body = parsed
        assert frontmatter["namespace"] == DEFAULT_NS
        # Full float precision: the snapshot value is EXACTLY the float64 of
        # the stored float32 (repr round-trip), not a formatted shortening.
        snapshot: dict[str, float] = frontmatter["confidence_snapshot"]
        assert snapshot[str(lesson_id)] == _confidence(db, lesson_id)
        for header in SECTION_HEADERS:
            assert header in body
        assert f"[[lesson:{lesson_id}]]" in body

    async def test_empty_namespace_renders_cleanly(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        async with client as session:
            payload = await _digest(session)
            parsed = _read_digest(Path(payload["path"]))
        frontmatter, body = parsed
        assert frontmatter == {"namespace": DEFAULT_NS, "confidence_snapshot": {}}
        assert payload["flagged_count"] == 0
        assert "0 episode" in body

    async def test_traversal_namespace_stays_inside_digest_dir(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        async with client as session:
            payload = await _digest(session, namespace="../../evil")
            path = Path(payload["path"])
            parsed = _read_digest(path)
        assert ".." not in path.parts
        assert "/" not in path.name
        assert path.name.startswith(f"{_slug('../../evil')}-")
        frontmatter, _ = parsed
        assert frontmatter["namespace"] == "../../evil"


class TestAuditSectionsRender:
    """Sections 1, 3, 4, 5, 6 from real write/dispute/contradict/corroborate/
    promote/demote/probe/report_usage paths; section 2 hits the first-run
    no-snapshot fallback (contradict-edge lessons)."""

    @pytest.fixture()
    def fake_embed_overrides(self) -> dict[str, list[float]]:
        return {
            LESSON_TEXT_1: V1,
            f"{CLAIM_1} {PROBE_SUFFIX}": V1,
            LESSON_TEXT_2: V2,
            LESSON_TEXT_3: V3,
        }

    async def test_all_six_sections_from_crafted_fixtures(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        from agent_memory.config import get_settings

        shaky_seed = _insert_episode(
            db, goal="shaky seed", embedding=V1, at=DAY_1, outcome=EXCERPT_OUTCOME
        )
        shaky_contradiction = _insert_episode(
            db, goal="shaky contradict", embedding=V5, at=DAY_3
        )
        stale_seed = _insert_episode(db, goal="stale seed", embedding=V2, at=DAY_1)
        novels = [
            _insert_episode(db, goal=f"novel {i}", embedding=vector, at=day)
            for i, (vector, day) in enumerate(
                [(V4, DAY_3), (V5, DAY_5), (V6, DAY_7), (V7, DAY_9)], start=1
            )
        ]
        demoted_seed = _insert_episode(db, goal="demoted seed", embedding=V3, at=DAY_1)
        _insert_episode(db, goal="uncited backlog episode", embedding=V6, at=DAY_5)

        dispute_reason = "wrong premise: load test was misconfigured"
        demotion_reason = "broke under multi-region replication"
        async with client as session:
            shaky = await _write(
                session,
                claim=CLAIM_1,
                because=BECAUSE_1,
                evidence=[{"episode_id": shaky_seed, "relation": "support"}],
            )
            stale = await _write(
                session,
                claim=CLAIM_2,
                because=BECAUSE_2,
                evidence=[{"episode_id": stale_seed, "relation": "support"}],
            )
            retired = await _write(
                session,
                claim=CLAIM_3,
                because=BECAUSE_3,
                evidence=[{"episode_id": demoted_seed, "relation": "support"}],
            )
            shaky_id = int(shaky["lesson_id"])
            stale_id = int(stale["lesson_id"])
            retired_id = int(retired["lesson_id"])

            demoted_conf = _ok(
                await session.call_tool(
                    "memory_contradict",
                    {"lesson_id": shaky_id, "episode_id": shaky_contradiction},
                )
            )
            assert demoted_conf["confidence"] == pytest.approx(0.20)
            for episode_id in novels:
                _ok(
                    await session.call_tool(
                        "memory_corroborate",
                        {"lesson_id": stale_id, "episode_id": episode_id},
                    )
                )
            assert _confidence(db, stale_id) == pytest.approx(0.80)
            _ok(
                await session.call_tool(
                    "memory_dispute", {"lesson_id": shaky_id, "reason": dispute_reason}
                )
            )
            promoted = _ok(
                await session.call_tool(
                    "memory_promote", {"lesson_id": retired_id, "reason": "graduate"}
                )
            )
            copy_id = int(promoted["promoted_lesson_id"])
            _ok(
                await session.call_tool(
                    "memory_demote", {"lesson_id": copy_id, "reason": demotion_reason}
                )
            )

            for _ in range(6):
                probed = _ok(
                    await session.call_tool(
                        "memory_probe",
                        {"current_goal": CLAIM_1, "approach": PROBE_SUFFIX},
                    )
                )
                assert any(e["id"] == f"lesson:{shaky_id}" for e in probed["results"])
                _ok(
                    await session.call_tool(
                        "memory_report_usage",
                        {
                            "retrieval_event_id": probed["retrieval_event_id"],
                            "results": [{"id": f"lesson:{shaky_id}", "outcome": "used"}],
                        },
                    )
                )

            settings = get_settings()
            ref_row = db.execute("SELECT now() AS now").fetchone()
            assert ref_row is not None
            threshold = timedelta(
                seconds=3600.0
                * settings.TAU_ENV_H
                * math.log(1.0 / settings.STALE_ENV_FRESH)
            )
            backdate(
                "lessons.last_evidence_at",
                stale_id,
                at=ref_row["now"] - threshold - timedelta(hours=1),
            )

            payload = await _digest(session)
            body = _read_digest(Path(payload["path"]))[1]

        path = Path(payload["path"])
        for header in SECTION_HEADERS:
            assert header in body

        # Section 1: disputed with reason + 140-char source excerpt.
        assert dispute_reason in body
        assert EXCERPT_OUTCOME[:140] in body
        assert EXCERPT_OUTCOME[:141] not in body

        # Section 2: no prior snapshot anywhere -> contradict-edge fallback.
        # Section 3: access_count 6 crosses > 5. Section 4: stale side of the
        # one-reference boundary. Section 5: tombstone with demotion reason.
        flagged = payload["flagged"]
        assert payload["flagged_count"] == len(flagged) == 5
        by_marker = {
            marker: [entry for entry in flagged if marker in entry]
            for marker in (
                "disputed",
                "contradicted",
                "popular-but-shaky",
                "rare-critical-stale",
                "demoted",
            )
        }
        assert all(len(entries) == 1 for entries in by_marker.values())
        assert f"[[lesson:{shaky_id}]]" in by_marker["disputed"][0]
        assert f"[[lesson:{shaky_id}]]" in by_marker["contradicted"][0]
        assert "contradict evidence" in by_marker["contradicted"][0]
        assert f"[[lesson:{shaky_id}]]" in by_marker["popular-but-shaky"][0]
        assert f"[[lesson:{stale_id}]]" in by_marker["rare-critical-stale"][0]
        assert f"[[lesson:{copy_id}]]" in by_marker["demoted"][0]
        assert demotion_reason in body
        assert f"[[lesson:{stale_id}]]" not in by_marker["popular-but-shaky"][0]

        # Section 6: one uncited episode; plain listing with confidences.
        assert "1 episode" in body
        for lesson_ref in (shaky_id, stale_id, retired_id):
            assert f"[[lesson:{lesson_ref}]]" in body


class TestContradictedSnapshot:
    """Snapshot comparison: a REAL decrease flags, unchanged does not, and the
    second same-day run overwrites cleanly against the pre-write snapshot."""

    @pytest.fixture()
    def fake_embed_overrides(self) -> dict[str, list[float]]:
        return {LESSON_TEXT_1: V1, LESSON_TEXT_2: V2}

    async def test_decrease_flags_unchanged_does_not_and_rerun_overwrites(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        seed_1 = _insert_episode(db, goal="seed one", embedding=V1, at=DAY_1)
        seed_2 = _insert_episode(db, goal="seed two", embedding=V2, at=DAY_1)
        contradiction = _insert_episode(
            db, goal="flat contradiction", embedding=V5, at=DAY_3
        )
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
            lesson_1 = int(first["lesson_id"])
            lesson_2 = int(second["lesson_id"])

            run_one = await _digest(session)
            path_one = Path(run_one["path"])
            assert run_one["flagged"] == []

            moved = _ok(
                await session.call_tool(
                    "memory_contradict",
                    {"lesson_id": lesson_1, "episode_id": contradiction},
                )
            )
            assert moved["confidence"] == pytest.approx(0.20)

            run_two = await _digest(session)
            path_two = Path(run_two["path"])
            body_two = _read_digest(path_two)[1]
            prefix_files = _prefix_files(
                path_two, f"{_slug(DEFAULT_NS)}-{_base_hash(DEFAULT_NS)}-"
            )

        # Same-day rerun overwrites cleanly: same path, still one file.
        assert path_two == path_one
        assert len(prefix_files) == 1

        flagged = run_two["flagged"]
        assert run_two["flagged_count"] == 1
        assert "contradicted" in flagged[0]
        assert f"[[lesson:{lesson_1}]]" in flagged[0]
        assert f"[[lesson:{lesson_2}]]" not in flagged[0]
        assert "0.4000 -> 0.2000" in body_two
        assert f"[[lesson:{lesson_2}]]" in body_two  # section 6 listing only


class TestFloorAdjacentDecrease:
    """0.055 -> 0.05 (drop of 0.005, floor-adjacent float32 values) must flag
    against a 1e-6 tolerance — the round-03 review fix."""

    @pytest.fixture()
    def fake_embed_overrides(self) -> dict[str, list[float]]:
        return {LESSON_TEXT_1: V1}

    async def test_floor_adjacent_drop_flags(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        seed = _insert_episode(db, goal="floor seed", embedding=V1, at=DAY_1)
        async with client as session:
            written = await _write(
                session,
                claim=CLAIM_1,
                because=BECAUSE_1,
                evidence=[{"episode_id": seed, "relation": "support"}],
            )
            lesson_id = int(written["lesson_id"])
            _set_confidence(db, lesson_id, 0.055)
            # psycopg yields float4 as its shortest-round-trip decimal in
            # float64 (0.055, not the raw float32 expansion); the snapshot
            # must carry exactly that value at full repr precision.
            snap_value = _confidence(db, lesson_id)
            assert snap_value == pytest.approx(0.055)
            run_one = await _digest(session)
            assert run_one["flagged"] == []
            frontmatter, _ = _read_digest(Path(run_one["path"]))
            assert frontmatter["confidence_snapshot"][str(lesson_id)] == snap_value

            _set_confidence(db, lesson_id, 0.05)
            run_two = await _digest(session)
            body_two = _read_digest(Path(run_two["path"]))[1]

        flagged = run_two["flagged"]
        assert run_two["flagged_count"] == 1
        assert f"[[lesson:{lesson_id}]]" in flagged[0]
        assert "0.0550 -> 0.0500" in body_two


class TestCrossDateSnapshot:
    """Yesterday's snapshot, today's run: decrease flags, unchanged does not
    (the weekly cron must not regress to fallback when the date rolls)."""

    @pytest.fixture()
    def fake_embed_overrides(self) -> dict[str, list[float]]:
        return {LESSON_TEXT_1: V1, LESSON_TEXT_2: V2}

    async def test_cross_date_decrease_and_unchanged(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        seed_1 = _insert_episode(db, goal="seed one", embedding=V1, at=DAY_1)
        seed_2 = _insert_episode(db, goal="seed two", embedding=V2, at=DAY_1)
        contradiction = _insert_episode(
            db, goal="cross-date contradiction", embedding=V5, at=DAY_3
        )
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
            lesson_1 = int(first["lesson_id"])
            lesson_2 = int(second["lesson_id"])

            run_one = await _digest(session)
            path_one = Path(run_one["path"])
            # Backdate the snapshot by renaming: the date lives in the NAME
            # and the frontmatter schema carries no date field.
            path_one.rename(path_one.with_name(_yesterday_name(path_one)))

            _ok(
                await session.call_tool(
                    "memory_contradict",
                    {"lesson_id": lesson_1, "episode_id": contradiction},
                )
            )
            run_two = await _digest(session)
            prefix_files = _prefix_files(
                Path(run_two["path"]), f"{_slug(DEFAULT_NS)}-{_base_hash(DEFAULT_NS)}-"
            )

        assert len(prefix_files) == 2
        flagged = run_two["flagged"]
        assert run_two["flagged_count"] == 1
        assert "contradicted" in flagged[0]
        assert f"[[lesson:{lesson_1}]]" in flagged[0]
        # Unchanged lesson_2 (0.40 -> 0.40) is not flagged, cross-date.
        assert f"[[lesson:{lesson_2}]]" not in flagged[0]


class TestCollisionPairs:
    """The REAL generated collision pair plus a/b vs a_b: two distinct files
    each, the base prefix never rewritten, neither overwriting the other."""

    @pytest.fixture()
    def fake_embed_overrides(self) -> dict[str, list[float]]:
        return {}

    async def test_real_collision_pair_and_same_slug_pair(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        # Fixture reality check (learnings note from planning).
        assert _slug(NS_A) == _slug(NS_B) == COLLISION_SLUG
        assert _base_hash(NS_A) == _base_hash(NS_B) == COLLISION_BASE

        async with client as session:
            payload_a = await _digest(session, namespace=NS_A)
            payload_b = await _digest(session, namespace=NS_B)
            payload_slash = await _digest(session, namespace="a/b")
            payload_flat = await _digest(session, namespace="a_b")
            path_a = Path(payload_a["path"])
            path_b = Path(payload_b["path"])
            frontmatter_a = _read_digest(path_a)[0]
            frontmatter_b = _read_digest(path_b)[0]

        date = _digest_date(path_a)
        assert path_a.name == f"{COLLISION_SLUG}-{COLLISION_BASE}-{date}.md"
        assert path_b.name == (
            f"{COLLISION_SLUG}-{COLLISION_BASE}-{RETRY_B1}-{_digest_date(path_b)}.md"
        )
        # The base file still belongs to NS_A — not overwritten by NS_B.
        assert frontmatter_a["namespace"] == NS_A
        assert frontmatter_b["namespace"] == NS_B
        assert path_a != path_b

        # Same slug, different hash: no retry, distinct base files.
        assert Path(payload_slash["path"]).name.startswith(f"a_b-{_base_hash('a/b')}-")
        assert Path(payload_flat["path"]).name.startswith(f"a_b-{_base_hash('a_b')}-")
        names = {
            payload_a["path"],
            payload_b["path"],
            payload_slash["path"],
            payload_flat["path"],
        }
        assert len(names) == 4


class TestCollisionResolvedCrossDate:
    """The namespace whose filename took a retry hash still finds its OWN
    prior snapshot the next DAY: unchanged confidence does NOT hit the
    no-snapshot fallback (despite a contradict edge), a decrease DOES flag."""

    @pytest.fixture()
    def fake_embed_overrides(self) -> dict[str, list[float]]:
        return {f"{CLAIM_1} {BECAUSE_1} ": V1}

    async def test_retry_path_snapshot_discovery_next_day(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        seed = _insert_episode(db, goal="retry seed", embedding=V1, at=DAY_1)
        contradiction = _insert_episode(
            db, goal="retry contradiction", embedding=V5, at=DAY_3
        )
        async with client as session:
            written = await _write(
                session,
                claim=CLAIM_1,
                because=BECAUSE_1,
                evidence=[{"episode_id": seed, "relation": "support"}],
                namespace=NS_B,
            )
            lesson_id = int(written["lesson_id"])
            moved = _ok(
                await session.call_tool(
                    "memory_contradict",
                    {"lesson_id": lesson_id, "episode_id": contradiction},
                )
            )
            assert moved["confidence"] == pytest.approx(0.20)

            # NS_A occupies the base name; NS_B must retry deterministically.
            payload_a = await _digest(session, namespace=NS_A)
            path_a = Path(payload_a["path"])
            run_one = await _digest(session, namespace=NS_B)
            path_retry = Path(run_one["path"])
            date = _digest_date(path_retry)
            assert path_retry.name == (
                f"{COLLISION_SLUG}-{COLLISION_BASE}-{RETRY_B1}-{date}.md"
            )

            # Next day: yesterday's retry file is this namespace's snapshot.
            path_retry.rename(path_retry.with_name(_yesterday_name(path_retry)))
            run_two = await _digest(session, namespace=NS_B)

            # Unchanged 0.20 -> no flag, and NOT the contradict-edge fallback
            # (this lesson carries one — the fallback WOULD list it).
            assert run_two["flagged"] == []
            assert run_two["flagged_count"] == 0

            _set_confidence(db, lesson_id, 0.10)
            run_three = await _digest(session, namespace=NS_B)
            flagged = run_three["flagged"]
            files = _prefix_files(
                Path(run_three["path"]), f"{COLLISION_SLUG}-{COLLISION_BASE}-"
            )
            frontmatter_a = _read_digest(path_a)[0]

        assert run_three["flagged_count"] == 1
        assert f"[[lesson:{lesson_id}]]" in flagged[0]
        assert "0.2000 -> 0.1000" in flagged[0]
        # NS_A's base file survived every NS_B write, and the prefix scan sees
        # exactly: NS_A today + NS_B yesterday + NS_B today.
        assert len(files) == 3
        assert frontmatter_a["namespace"] == NS_A


class TestContainmentDirect:
    def test_digest_path_containment_in_process(
        self, pg: str, tmp_path: Path
    ) -> None:
        """Resolved path stays under DIGEST_DIR even for a traversal-shaped
        namespace (slug sanitization makes it inert; containment verified)."""
        from agent_memory.config import Settings
        from agent_memory.digest import digest as digest_fn

        settings = Settings(
            DATABASE_URL=pg,
            EMBED_IMPL="fake",
            PGVECTOR_DIM=DIM,
            DIGEST_DIR=str(tmp_path),
        )
        payload = digest_fn(settings, namespace="../../evil")
        path = Path(payload["path"])
        assert path.parent == tmp_path
        assert ".." not in path.parts
        assert path.is_file()
        frontmatter, _ = _read_digest(path)
        assert frontmatter["namespace"] == "../../evil"


class TestUnwritableDigestDir:
    async def test_unwritable_digest_dir_is_clean_error(
        self,
        db: psycopg.Connection[DictRow],
        pg: str,
    ) -> None:
        """QA (failure): DIGEST_DIR under a file -> MCP error result, and the
        session stays alive for the next call."""
        blocker = Path(tempfile.mkdtemp(prefix="agent-memory-digest-block-")) / "blocker"
        blocker.write_text("i am a file, not a directory", encoding="utf-8")
        bad_dir = str(blocker / "sub")
        async with _spawn(pg, digest_dir=bad_dir) as session:
            result = await session.call_tool("memory_digest", {})
            message = _err(result)
            assert "DIGEST_DIR" in message
            # Same-session recovery: the failure never exits the process.
            stats = _ok(await session.call_tool("memory_stats", {}))
        assert stats["lesson_count"] == 0
