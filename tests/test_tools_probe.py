"""Contract tests for memory_probe / memory_search: hybrid RRF, activation, exposure log.

Three distinct SERVER session configs, never shared (plan task 7):
- PRODUCTION: the shared conftest ``client`` fixture (no planner overrides) —
  correctness-only assertions, never plan shapes.
- GOLDEN: module-local server whose DATABASE_URL carries libpq
  ``options='-c enable_indexscan=off'`` — deterministic exact-cosine ranking
  goldens (ties broken by the SQL total order).
- DIAGNOSTIC: server with ``options='-c enable_seqscan=off'`` — the
  representative vector query executes under a forced-HNSW planner, and an
  EXPLAIN on a test-side connection carrying the same libpq options (in its
  own URL, never SET LOCAL) proves the plan is an HNSW index scan.

FakeEmbedder override keys are EXACT raw texts: the probe query text is
``current_goal + " " + approach``; episode captures embed the 4-field join.
Vectors are crafted orthogonal/angled unit pairs; lessons are inserted via
direct SQL (write_lesson arrives in task 10).
"""

import json
import math
import shutil
import sys
import tempfile
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import pgvector
import pgvector.psycopg
import psycopg
import pytest
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import CallToolResult, TextContent
from psycopg.rows import DictRow

from tests.conftest import backdate

DIM = 8
V_Q = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # query axis
V_ALT = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # exactly orthogonal to V_Q
V_MID = [0.5, math.sqrt(0.75), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # cos(V_MID, V_Q) = 0.5

DEFAULT_NS = "default@local"
T_FIXED = datetime(2026, 9, 15, 10, 0, tzinfo=timezone.utc)

OPTS_INDEXSCAN_OFF = "options=-c%20enable_indexscan%3Doff"
OPTS_SEQSCAN_OFF = "options=-c%20enable_seqscan%3Doff"


def _angled(step: int) -> list[float]:
    """Unit vector at step * 5 degrees from V_Q: cos strictly decreasing in step.

    Distinct distances make vector-channel ordering deterministic under ANY
    planner (HNSW tie order is arbitrary; distinct keys are not).
    """
    theta = step * math.pi / 36.0
    return [round(math.cos(theta), 6), round(math.sin(theta), 6), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def _with_options(url: str, options: str) -> str:
    """Append libpq session options to a URL (space=%20, inner ==%3D — verified)."""
    return f"{url}{'&' if '?' in url else '?'}{options}"


def _server(
    pg_url: str,
    overrides: dict[str, list[float]],
    *,
    url_options: str | None = None,
    env_extra: dict[str, str] | None = None,
) -> AbstractAsyncContextManager[ClientSession]:
    """Module-local stdio server with custom DATABASE_URL options / env.

    Mirrors conftest's ``client`` (fake embedder, dim 8, temp digest dir)
    without touching it; per-test planner overrides reach the SERVER process
    through its own DATABASE_URL, never the test's connection.
    """
    digest_dir = tempfile.mkdtemp(prefix="agent-memory-digest-")
    env = {
        "DATABASE_URL": _with_options(pg_url, url_options) if url_options else pg_url,
        "EMBED_IMPL": "fake",
        "PGVECTOR_DIM": "8",
        "FAKE_EMBED_OVERRIDES": json.dumps(overrides),
        "DIGEST_DIR": digest_dir,
    }
    if env_extra:
        env.update(env_extra)
    parameters = StdioServerParameters(command=sys.executable, args=["-m", "agent_memory"], env=env)

    @asynccontextmanager
    async def enter() -> AsyncIterator[ClientSession]:
        try:
            async with stdio_client(parameters) as (read_stream, write_stream):
                async with ClientSession(
                    read_stream, write_stream, read_timeout_seconds=60.0
                ) as session:
                    await session.initialize()
                    yield session
        finally:
            shutil.rmtree(digest_dir, ignore_errors=True)

    return enter()


def _insert_episode(
    db: psycopg.Connection[DictRow],
    *,
    goal: str,
    expectation: str = "",
    action: str = "",
    outcome: str = "",
    surprise: float = 0.5,
    namespace: str = DEFAULT_NS,
    embedding: list[float],
) -> int:
    row = db.execute(
        """
        INSERT INTO episodes (namespace, goal, expectation, action, outcome, surprise,
                              raw_text, embedding)
        VALUES (%(namespace)s, %(goal)s, %(expectation)s, %(action)s, %(outcome)s,
                %(surprise)s, '', %(embedding)s)
        RETURNING id
        """,
        {
            "namespace": namespace,
            "goal": goal,
            "expectation": expectation,
            "action": action,
            "outcome": outcome,
            "surprise": surprise,
            "embedding": pgvector.Vector(embedding),
        },
    ).fetchone()
    assert row is not None
    return int(row["id"])


def _insert_lesson(
    db: psycopg.Connection[DictRow],
    *,
    claim: str,
    because: str = "causal gist",
    holds_when: str | None = None,
    fails_when: str | None = None,
    confidence: float = 0.5,
    namespace: str = DEFAULT_NS,
    promotion_status: str = "active",
    promoted_from_lesson_id: int | None = None,
    disputed: bool = False,
    embedding: list[float],
) -> int:
    row = db.execute(
        """
        INSERT INTO lessons (namespace, claim, because, holds_when, fails_when, confidence,
                             embedding, promotion_status, promoted_from_lesson_id, disputed)
        VALUES (%(namespace)s, %(claim)s, %(because)s, %(holds_when)s, %(fails_when)s,
                %(confidence)s, %(embedding)s, %(promotion_status)s, %(promoted_from)s,
                %(disputed)s)
        RETURNING id
        """,
        {
            "namespace": namespace,
            "claim": claim,
            "because": because,
            "holds_when": holds_when,
            "fails_when": fails_when,
            "confidence": confidence,
            "embedding": pgvector.Vector(embedding),
            "promotion_status": promotion_status,
            "promoted_from": promoted_from_lesson_id,
            "disputed": disputed,
        },
    ).fetchone()
    assert row is not None
    return int(row["id"])


def _insert_evidence(
    db: psycopg.Connection[DictRow],
    lesson_id: int,
    episode_id: int,
    relation: str,
    reason: str | None,
) -> None:
    db.execute(
        """
        INSERT INTO lesson_evidence (lesson_id, episode_id, relation, reason)
        VALUES (%(lesson_id)s, %(episode_id)s, %(relation)s, %(reason)s)
        """,
        {"lesson_id": lesson_id, "episode_id": episode_id, "relation": relation, "reason": reason},
    )


def _insert_link(
    db: psycopg.Connection[DictRow],
    lesson_id: int,
    related_lesson_id: int,
    *,
    kind: str = "similar",
    weight: float = 0.5,
) -> None:
    db.execute(
        """
        INSERT INTO lesson_links (lesson_id, related_lesson_id, kind, weight)
        VALUES (%(lesson_id)s, %(related)s, %(kind)s, %(weight)s)
        """,
        {"lesson_id": lesson_id, "related": related_lesson_id, "kind": kind, "weight": weight},
    )


def _payload(result: CallToolResult) -> dict[str, Any]:
    assert result.is_error is False, result.content
    block = result.content[0]
    assert isinstance(block, TextContent)
    payload: dict[str, Any] = json.loads(block.text)
    assert isinstance(payload, dict)
    return payload


async def _probe(
    session: ClientSession,
    goal: str,
    approach: str | None = None,
    k: int | None = None,
    namespace: str | None = None,
) -> dict[str, Any]:
    args: dict[str, Any] = {"current_goal": goal}
    if approach is not None:
        args["approach"] = approach
    if k is not None:
        args["k"] = k
    if namespace is not None:
        args["namespace"] = namespace
    return _payload(await session.call_tool("memory_probe", args))


async def _search(
    session: ClientSession,
    query: str,
    k: int | None = None,
    namespace: str | None = None,
) -> dict[str, Any]:
    args: dict[str, Any] = {"query": query}
    if k is not None:
        args["k"] = k
    if namespace is not None:
        args["namespace"] = namespace
    return _payload(await session.call_tool("memory_search", args))


def _result_ids(payload: dict[str, Any]) -> list[str]:
    return [str(record["id"]) for record in payload["results"]]


def _plan_nodes(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten an EXPLAIN (FORMAT JSON) plan tree (Title Case keys) into nodes."""
    nodes = [plan]
    for child in plan.get("Plans", []):
        nodes.extend(_plan_nodes(child))
    return nodes


def _event_row(db: psycopg.Connection[DictRow]) -> DictRow:
    row = db.execute("SELECT * FROM retrieval_events ORDER BY id").fetchone()
    assert row is not None
    return row


# ---------------------------------------------------------------------------
# PRODUCTION-config session (conftest client): correctness only, no plan shapes.
# ---------------------------------------------------------------------------


class TestGateAndExposure:
    """Failure QA: orthogonal vectors + disjoint keywords -> miss, exposure logged."""

    @pytest.fixture()
    def fake_embed_overrides(self) -> dict[str, list[float]]:
        return {"zephyr quilting horizon": V_Q}

    async def test_irrelevant_query_returns_empty_but_logs_exposure(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        _insert_episode(
            db,
            goal="hiking the ridge trail",
            outcome="reached the summit by noon",
            embedding=V_ALT,
        )
        _insert_lesson(db, claim="always carry spare batteries", embedding=V_ALT)
        async with client as session:
            payload = await _probe(session, "zephyr quilting", approach="horizon")
        assert payload["results"] == []
        event = _event_row(db)
        assert event["tool"] == "probe"
        assert event["namespace"] == DEFAULT_NS
        assert event["query_context"] == "zephyr quilting horizon"
        assert event["returned_ids"] == []
        assert isinstance(payload["retrieval_event_id"], int)
        assert payload["retrieval_event_id"] == event["id"]


class TestEpisodeProvenance:
    """Production representative vector query: correct results only."""

    @pytest.fixture()
    def fake_embed_overrides(self) -> dict[str, list[float]]:
        return {"deploy green service canary": V_Q}

    async def test_relevant_episode_retrieved_with_provenance(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        episode_id = _insert_episode(
            db,
            goal="deploy green service",
            expectation="canary passes quickly",
            action="rolled canary to 5 percent",
            outcome="latency spike absorbed",
            surprise=0.6,
            embedding=V_Q,
        )
        async with client as session:
            payload = await _probe(session, "deploy green service", approach="canary")
        assert _result_ids(payload) == [f"episode:{episode_id}"]
        record = payload["results"][0]
        assert record["record_type"] == "episode"
        assert record["goal"] == "deploy green service"
        assert record["outcome"] == "latency spike absorbed"
        assert record["surprise"] == pytest.approx(0.6)
        assert record["match_strength"] == pytest.approx(1.0)
        assert isinstance(record["created_at"], str)
        event = _event_row(db)
        assert event["returned_ids"] == [f"episode:{episode_id}"]
        assert payload["retrieval_event_id"] == event["id"]


class TestLessonProvenance:
    @pytest.fixture()
    def fake_embed_overrides(self) -> dict[str, list[float]]:
        return {"cache eviction pressure": V_Q}

    async def test_evidence_edges_with_excerpt_and_relation(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        long_outcome = "redis eviction storms cascaded " + "x" * 200
        supporting = _insert_episode(
            db,
            goal="tune cache eviction",
            outcome=long_outcome,
            embedding=V_ALT,
        )
        contradicting = _insert_episode(
            db,
            goal="tune cache eviction",
            outcome="no measurable change in hit rate",
            embedding=V_ALT,
        )
        lesson_id = _insert_lesson(
            db,
            claim="cache eviction pressure needs headroom",
            because="eviction storms follow memory pressure",
            holds_when="redis under 80 percent memory",
            confidence=0.7,
            embedding=V_Q,
        )
        _insert_evidence(db, lesson_id, supporting, "support", "direct storm trace")
        _insert_evidence(db, lesson_id, contradicting, "contradict", "no effect observed")
        async with client as session:
            payload = await _probe(session, "cache eviction", approach="pressure")
        assert _result_ids(payload) == [f"lesson:{lesson_id}"]
        record = payload["results"][0]
        assert record["record_type"] == "lesson"
        assert record["confidence"] == pytest.approx(0.7)
        assert record["match_strength"] == pytest.approx(1.0)
        assert "disputed" not in record  # not set -> key absent
        by_relation = {edge["relation"]: edge for edge in record["evidence"]}
        assert set(by_relation) == {"support", "contradict"}
        assert by_relation["support"]["episode"] == f"episode:{supporting}"
        assert by_relation["support"]["reason"] == "direct storm trace"
        assert by_relation["support"]["excerpt"] == long_outcome[:140]
        assert by_relation["contradict"]["excerpt"] == "no measurable change in hit rate"

    async def test_disputed_lesson_flagged_when_set(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        lesson_id = _insert_lesson(
            db,
            claim="indexes never help small tables",
            because="seq scan dominates below ten thousand rows",
            confidence=0.5,
            disputed=True,
            embedding=V_Q,
        )
        async with client as session:
            payload = await _search(session, "indexes never help small tables")
        assert _result_ids(payload) == [f"lesson:{lesson_id}"]
        assert payload["results"][0]["disputed"] is True
        # task-13 contract: disputed results always carry dispute_reason;
        # this fixture SQL-sets disputed without one, so the value is null.
        assert payload["results"][0]["dispute_reason"] is None


class TestAccessStatsUntouched:
    @pytest.fixture()
    def fake_embed_overrides(self) -> dict[str, list[float]]:
        return {"roll back migrations safely": V_Q}

    async def test_probe_does_not_touch_access_stats(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        episode_id = _insert_episode(db, goal="roll back migrations", embedding=V_Q, surprise=0.9)
        lesson_id = _insert_lesson(
            db, claim="roll back migrations safely with backups", embedding=V_Q
        )
        accessed = datetime(2026, 9, 1, tzinfo=timezone.utc)
        db.execute(
            "UPDATE episodes SET access_count = 7, last_accessed = %(at)s WHERE id = %(id)s",
            {"at": accessed, "id": episode_id},
        )
        db.execute(
            "UPDATE lessons SET access_count = 3, last_accessed = %(at)s WHERE id = %(id)s",
            {"at": accessed, "id": lesson_id},
        )
        async with client as session:
            payload = await _probe(session, "roll back migrations", approach="safely")
        assert set(_result_ids(payload)) == {f"episode:{episode_id}", f"lesson:{lesson_id}"}
        episode = db.execute(
            "SELECT access_count, last_accessed FROM episodes WHERE id = %(id)s",
            {"id": episode_id},
        ).fetchone()
        lesson = db.execute(
            "SELECT access_count, last_accessed FROM lessons WHERE id = %(id)s",
            {"id": lesson_id},
        ).fetchone()
        assert episode is not None and episode["access_count"] == 7
        assert episode["last_accessed"] == accessed
        assert lesson is not None and lesson["access_count"] == 3
        assert lesson["last_accessed"] == accessed


class TestNamespaceVisibility:
    @pytest.fixture()
    def fake_embed_overrides(self) -> dict[str, list[float]]:
        return {"rotate credentials quarterly": V_Q}

    async def test_active_global_lesson_retrievable_from_project_namespace(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        local_noise = _insert_lesson(
            db, claim="unrelated local claim", embedding=V_ALT, confidence=0.9
        )
        global_id = _insert_lesson(
            db,
            claim="rotate credentials quarterly",
            namespace="global",
            promoted_from_lesson_id=None,
            embedding=V_Q,
        )
        async with client as session:
            payload = await _probe(session, "rotate credentials", approach="quarterly")
        assert _result_ids(payload) == [f"lesson:{global_id}"]
        assert f"lesson:{local_noise}" not in _result_ids(payload)
        assert payload["results"][0]["namespace"] == "global"

    async def test_demoted_global_lesson_invisible_even_when_probing_global(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        original = _insert_lesson(
            db, claim="rotate credentials quarterly source", embedding=V_ALT
        )
        active_global = _insert_lesson(
            db,
            claim="rotate credentials quarterly",
            namespace="global",
            promoted_from_lesson_id=original,
            embedding=V_Q,
        )
        demoted_global = _insert_lesson(
            db,
            claim="rotate credentials quarterly old rule",
            namespace="global",
            promotion_status="demoted",
            promoted_from_lesson_id=original,
            embedding=V_Q,
        )
        async with client as session:
            from_global = await _probe(
                session, "rotate credentials", approach="quarterly", namespace="global"
            )
            from_default = await _probe(session, "rotate credentials", approach="quarterly")
        assert _result_ids(from_global) == [f"lesson:{active_global}"]
        assert f"lesson:{demoted_global}" not in _result_ids(from_global)
        # From the default namespace the LOCAL original is still visible next
        # to its active global copy; only the demoted copy disappears. The
        # order between the two near-identical lessons is not under test.
        assert set(_result_ids(from_default)) == {
            f"lesson:{original}",
            f"lesson:{active_global}",
        }
        events = db.execute(
            "SELECT namespace FROM retrieval_events ORDER BY id"
        ).fetchall()
        assert [row["namespace"] for row in events] == ["global", DEFAULT_NS]


class TestSearchToolMode:
    @pytest.fixture()
    def fake_embed_overrides(self) -> dict[str, list[float]]:
        return {"green service rotisserie": V_Q}

    async def test_search_logs_tool_and_query_context(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        episode_id = _insert_episode(
            db, goal="green service rotisserie", outcome="evenly browned", embedding=V_Q
        )
        async with client as session:
            payload = await _search(session, "green service rotisserie")
        assert _result_ids(payload) == [f"episode:{episode_id}"]
        event = _event_row(db)
        assert event["tool"] == "search"
        assert event["query_context"] == "green service rotisserie"


class TestStalenessNote:
    @pytest.fixture()
    def fake_embed_overrides(self) -> dict[str, list[float]]:
        return {"vendor quota limits": V_Q}

    async def test_stale_high_salience_record_gets_note(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        stale_lesson = _insert_lesson(
            db,
            claim="vendor quota limits bite in Q4",
            because="quota resets in January",
            confidence=0.9,  # salience > SALIENCE_STALE (0.7)
            embedding=V_Q,
        )
        backdate("lessons.last_evidence_at", stale_lesson, days=-400)  # env_fresh < 0.2
        fresh_episode = _insert_episode(
            db, goal="vendor quota limits", outcome="hit the ceiling", surprise=0.9, embedding=V_Q
        )
        async with client as session:
            payload = await _probe(session, "vendor quota", approach="limits")
        by_id = {record["id"]: record for record in payload["results"]}
        assert "staleness_note" in by_id[f"lesson:{stale_lesson}"]
        assert "staleness_note" not in by_id[f"episode:{fresh_episode}"]


class TestPerLessonEvidenceIsolation:
    """Issue #7 regression: multi-lesson probe results must not bleed evidence.

    ``dict.fromkeys(lesson_ids, [])`` bound every lesson id to ONE shared
    list, so each returned lesson displayed the union of ALL lessons'
    evidence edges. Both lessons are crafted through the REAL
    memory_write_lesson path with disjoint evidence episodes and share the
    distinctive "quokka jam" vocabulary so one keyword-channel probe
    (websearch_to_tsquery ANDs its terms) returns both.
    """

    async def test_each_lesson_lists_exactly_its_own_evidence_edges(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        async with client as session:
            ep_ingest = _payload(
                await session.call_tool(
                    "memory_capture_episode",
                    {
                        "goal": "index quokka jam telemetry",
                        "outcome": "index rebuild finished overnight",
                        "surprise": 0.6,
                    },
                )
            )["id"]
            ep_backpressure = _payload(
                await session.call_tool(
                    "memory_capture_episode",
                    {
                        "goal": "index quokka jam backpressure",
                        "outcome": "backpressure drained after batch tuning",
                        "surprise": 0.5,
                    },
                )
            )["id"]
            ep_render = _payload(
                await session.call_tool(
                    "memory_capture_episode",
                    {
                        "goal": "render quokka jam dashboards",
                        "outcome": "dashboard render cached successfully",
                        "surprise": 0.4,
                    },
                )
            )["id"]
            ingest_lesson = _payload(
                await session.call_tool(
                    "memory_write_lesson",
                    {
                        "claim": "quokka jam ingest needs batch indexing",
                        "because": "quokka jam telemetry overloads single-row inserts",
                        "holds_when": "quokka jam pipeline runs nightly",
                        "evidence": [
                            {
                                "episode_id": ep_ingest,
                                "relation": "support",
                                "reason": "ingest storm trace",
                            },
                            {
                                "episode_id": ep_backpressure,
                                "relation": "refine",
                                "reason": "backpressure variant",
                            },
                        ],
                    },
                )
            )["lesson_id"]
            render_lesson = _payload(
                await session.call_tool(
                    "memory_write_lesson",
                    {
                        "claim": "quokka jam dashboards need render caching",
                        "because": "dashboard panels recompute every request",
                        "holds_when": "quokka jam panels have repeat viewers",
                        "evidence": [
                            {
                                "episode_id": ep_render,
                                "relation": "support",
                                "reason": "render cache trace",
                            },
                        ],
                    },
                )
            )["lesson_id"]
            payload = await _probe(session, "quokka jam")
        lessons_by_id = {
            record["id"]: record
            for record in payload["results"]
            if record["record_type"] == "lesson"
        }
        assert set(lessons_by_id) == {
            f"lesson:{ingest_lesson}",
            f"lesson:{render_lesson}",
        }
        ingest_edges = {
            (edge["episode"], edge["relation"], edge["reason"])
            for edge in lessons_by_id[f"lesson:{ingest_lesson}"]["evidence"]
        }
        render_edges = {
            (edge["episode"], edge["relation"], edge["reason"])
            for edge in lessons_by_id[f"lesson:{render_lesson}"]["evidence"]
        }
        assert ingest_edges == {
            (f"episode:{ep_ingest}", "support", "ingest storm trace"),
            (f"episode:{ep_backpressure}", "refine", "backpressure variant"),
        }
        assert render_edges == {(f"episode:{ep_render}", "support", "render cache trace")}


# ---------------------------------------------------------------------------
# GOLDEN session (enable_indexscan=off in the SERVER's DATABASE_URL):
# deterministic ranking goldens with exact-cosine sequential scans.
# ---------------------------------------------------------------------------


class TestVectorOnlyHit:
    """Vector-channel hit with zero keyword rank must not soak a keyword slot."""

    async def test_no_keyword_rrf_contribution_for_zero_rank_row(
        self,
        db: psycopg.Connection[DictRow],
        pg: str,
    ) -> None:
        goal, approach = "sprocket analysis", "depth"
        qtext = f"{goal} {approach}"
        episode_id = _insert_episode(
            db,
            goal="hiking the ridge trail",
            outcome="reached the summit by noon",
            surprise=0.5,
            embedding=V_Q,  # vector-only hit: no shared keywords with qtext
        )
        lesson_id = _insert_lesson(
            db,
            claim="sprocket analysis depth charts decay",
            because="measured over many rides",
            confidence=0.5,
            embedding=V_ALT,  # orthogonal vector, keyword-only hit
        )
        backdate("episodes", episode_id, at=T_FIXED)
        backdate("lessons.last_evidence_at", lesson_id, at=T_FIXED)
        async with _server(pg, {qtext: V_Q}, url_options=OPTS_INDEXSCAN_OFF) as session:
            payload = await _probe(session, goal, approach)
        # Correct: lesson (keyword+vector RRF) strictly above the vector-only
        # episode. If the episode soaked a keyword slot despite rank 0, both
        # rel_norms would be 1.0, scores exactly equal, and the total order
        # would flip this to [episode, lesson].
        assert _result_ids(payload) == [f"lesson:{lesson_id}", f"episode:{episode_id}"]
        strengths = {record["id"]: record["match_strength"] for record in payload["results"]}
        assert strengths[f"episode:{episode_id}"] == pytest.approx(1.0)


class TestEqualNumericIds:
    async def test_episode_1_and_lesson_1_no_collision(
        self,
        db: psycopg.Connection[DictRow],
        pg: str,
    ) -> None:
        episode_id = _insert_episode(
            db, goal="hiking the ridge trail", outcome="summit", embedding=V_Q
        )
        lesson_id = _insert_lesson(
            db, claim="always carry spare batteries", embedding=V_ALT
        )
        assert episode_id == 1 and lesson_id == 1
        async with _server(
            pg,
            {"ridge trail north": V_Q, "battery lantern road": V_ALT},
            url_options=OPTS_INDEXSCAN_OFF,
        ) as session:
            toward_episode = await _probe(session, "ridge trail", approach="north")
            toward_lesson = await _probe(session, "battery lantern", approach="road")
        assert _result_ids(toward_episode) == ["episode:1"]
        assert _result_ids(toward_lesson) == ["lesson:1"]

    async def test_equal_score_equal_numeric_id_total_tiebreak(
        self,
        db: psycopg.Connection[DictRow],
        pg: str,
    ) -> None:
        episode_id = _insert_episode(
            db,
            goal="hiking the ridge trail",
            outcome="summit by noon",
            surprise=0.5,
            embedding=V_Q,
        )
        lesson_id = _insert_lesson(
            db,
            claim="always carry spare batteries",
            because="headlamps fail cold",
            confidence=0.5,
            embedding=V_Q,
        )
        assert episode_id == 1 and lesson_id == 1
        backdate("episodes", episode_id, at=T_FIXED)
        backdate("lessons.last_evidence_at", lesson_id, at=T_FIXED)
        async with _server(pg, {"quantum telemetry band": V_Q}, url_options=OPTS_INDEXSCAN_OFF) as session:
            payload = await _probe(session, "quantum telemetry", approach="band", k=2)
        # Identical signals -> identical scores -> total order (record_type, id):
        # episode:1 sorts before lesson:1 and both survive as DISTINCT records.
        assert _result_ids(payload) == ["episode:1", "lesson:1"]


class TestActivation:
    async def test_one_hop_spread_lifts_linked_neighbor(
        self,
        db: psycopg.Connection[DictRow],
        pg: str,
    ) -> None:
        goal, approach = "kafka lag runbook", "partition"
        qtext = f"{goal} {approach}"
        top = _insert_lesson(
            db,
            claim="kafka lag runbook partition rebalance",
            because="measured in production",
            confidence=0.5,
            embedding=V_Q,
        )
        unlinked = _insert_lesson(
            db,
            claim="unrelated ledger note alpha",
            because="bookkeeping",
            confidence=0.5,
            embedding=V_MID,
        )
        linked = _insert_lesson(
            db,
            claim="unrelated ledger note beta",
            because="bookkeeping",
            confidence=0.5,
            embedding=V_MID,
        )
        # unlinked (id 2) beats linked (id 3) on rel_norm via the vector tie
        # break; only the activation edge can flip them.
        _insert_link(db, top, linked, weight=0.9)
        backdate("lessons.last_evidence_at", top, at=T_FIXED)
        backdate("lessons.last_evidence_at", unlinked, at=T_FIXED)
        backdate("lessons.last_evidence_at", linked, at=T_FIXED)
        async with _server(pg, {qtext: V_Q}, url_options=OPTS_INDEXSCAN_OFF) as session:
            payload = await _probe(session, goal, approach)
        assert _result_ids(payload) == [
            f"lesson:{top}",
            f"lesson:{linked}",
            f"lesson:{unlinked}",
        ]

    async def test_neighbor_expansion_reapplies_visibility_restrictions(
        self,
        db: psycopg.Connection[DictRow],
        client: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        top = _insert_lesson(
            db,
            claim="rotate credentials quarterly",
            because="security posture",
            embedding=V_Q,
        )
        demoted_global = _insert_lesson(
            db,
            claim="rotate credentials quarterly loudly",
            namespace="global",
            promotion_status="demoted",
            promoted_from_lesson_id=1,
            embedding=V_Q,
        )
        foreign = _insert_lesson(
            db,
            claim="rotate credentials quarterly quietly",
            namespace="other@proj",
            embedding=V_Q,
        )
        _insert_link(db, top, demoted_global, weight=1.0)
        _insert_link(db, top, foreign, weight=1.0)
        async with client as session:
            payload = await _probe(session, "rotate credentials", approach="quarterly", k=8)
        ids = _result_ids(payload)
        assert f"lesson:{top}" in ids
        assert f"lesson:{demoted_global}" not in ids
        assert f"lesson:{foreign}" not in ids


class TestKResolution:
    """k omitted -> settings.FINAL_K (env-effective); explicit k wins."""

    async def test_env_final_k_when_omitted_and_explicit_k_wins(
        self,
        db: psycopg.Connection[DictRow],
        pg: str,
    ) -> None:
        for step in range(5):
            _insert_episode(
                db,
                goal=f"orbital station log entry {step}",
                outcome=f"routine orbit {step}",
                embedding=_angled(step),
            )
        goal, approach = "orbital station", "log"
        async with _server(
            pg, {f"{goal} {approach}": V_Q}, env_extra={"FINAL_K": "3"}
        ) as session:
            omitted = await _probe(session, goal, approach)
            explicit_two = await _probe(session, goal, approach, k=2)
            explicit_five = await _probe(session, goal, approach, k=5)
            search_one = await _search(session, "orbital station log", k=1)
        assert len(omitted["results"]) == 3
        assert len(explicit_two["results"]) == 2
        assert len(explicit_five["results"]) == 5
        assert len(search_one["results"]) == 1
        # distinct distances -> deterministic id order regardless of planner
        assert _result_ids(omitted) == ["episode:1", "episode:2", "episode:3"]


# ---------------------------------------------------------------------------
# DIAGNOSTIC session (enable_seqscan=off): forced-HNSW execution + EXPLAIN.
# ---------------------------------------------------------------------------

_REPRESENTATIVE_VECTOR_SQL = """
SELECT id FROM lessons
WHERE ((namespace = %(ns)s OR namespace = 'global') AND promotion_status = 'active')
  AND embedding IS NOT NULL
ORDER BY embedding <=> %(qvec)s
LIMIT %(topk)s
"""


class TestHnswDiagnostic:
    @pytest.fixture()
    def fake_embed_overrides(self) -> dict[str, list[float]]:
        return {"aurora telemetry surge": V_Q}

    async def test_representative_vector_query_plans_hnsw_scan(
        self,
        db: psycopg.Connection[DictRow],
        pg: str,
    ) -> None:
        episode_id = _insert_episode(
            db, goal="aurora telemetry surge", outcome="photon count doubled", embedding=V_Q
        )
        _insert_lesson(
            db, claim="quiet observatory baseline", because="dark run", embedding=V_ALT
        )
        async with _server(pg, {"aurora telemetry surge": V_Q}, url_options=OPTS_SEQSCAN_OFF) as session:
            payload = await _probe(session, "aurora telemetry", approach="surge")
        # Correct results under the forced-HNSW planner (execution proof).
        assert _result_ids(payload) == [f"episode:{episode_id}"]
        # EXPLAIN on a diagnostic connection carrying the SAME libpq options in
        # its own URL (session-scoped by contract; never SET LOCAL).
        with psycopg.connect(_with_options(pg, OPTS_SEQSCAN_OFF), autocommit=True) as conn:
            pgvector.psycopg.register_vector(conn)
            explained = conn.execute(
                "EXPLAIN (FORMAT JSON) " + _REPRESENTATIVE_VECTOR_SQL,
                {"ns": DEFAULT_NS, "qvec": pgvector.Vector(V_Q), "topk": 12},
            ).fetchone()
            assert explained is not None
            plan = explained[0][0]["Plan"]
            scanned = {
                node["Index Name"]
                for node in _plan_nodes(plan)
                if node.get("Node Type") == "Index Scan" and "Index Name" in node
            }
            assert scanned, plan
            # The default index name carries no "hnsw" substring; prove the
            # access method through the catalog instead.
            methods = conn.execute(
                """
                SELECT am.amname
                FROM pg_index x
                JOIN pg_class ic ON ic.oid = x.indexrelid
                JOIN pg_am am ON am.oid = ic.relam
                WHERE ic.relname = ANY(%(names)s)
                """,
                {"names": sorted(scanned)},
            ).fetchall()
        assert [row[0] for row in methods] == ["hnsw"] * len(scanned)
