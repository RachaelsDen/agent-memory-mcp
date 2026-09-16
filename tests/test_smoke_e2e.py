"""E2E smoke (plan task 16): the full memory loop in ONE test / ONE session.

capture -> probe -> report_usage -> consolidate_scan -> write_lesson ->
corroborate -> promote -> probe(other namespace) -> demote -> digest, all
through a real stdio ClientSession against the testcontainer Postgres.
Every step is cross-checked with SQL on the ``db`` fixture between tool
calls — a misleading success payload cannot pass.

Vector geometry (deterministic orthonormal basis, dim 8):
- V_PAIR (dim 1): the near-duplicate pair, the lesson text, and every probe
  query -> pair cos 1.0 (one incident, one cluster), probe cos 1.0
  (guaranteed retrieval, match_strength exactly 1.0).
- W_SIDE (dim 2): the distinct-day episode -> cos 0.0 to V_PAIR, strictly
  below CLUSTER_COS (0.82), so the scan returns exactly the 2-cluster; its
  vocabulary shares no lexeme with the probe query, so the keyword channel
  cannot surface it either.
- V_CORR (dim 3): the corroborating episode -> orthogonal to both, on a
  fresh UTC date, so corroboration novelty is 1.0 by construction (+0.1).

Dates are fixed 2026-01 UTC instants backdated with ``at=`` (boundary-safe:
the pair sits 1h apart inside the 24h incident window on one UTC day, the
distinct-day episode lands 48h later — outside the window by a full day —
and every date is months past the 1h consolidation age gate), deterministic
against DB now().

Adversarial classes: flaky_tests — deterministic override vectors + absolute
backdate(at=) moments, no wall-clock reads; misleading_success_output — SQL
asserts between every step (usage rows, stats, edges, lesson rows, exposure
rows, digest FILES parsed from disk); stale_state — per-test truncate in
conftest; hung_commands — --durations=10 in the tee'd evidence runs;
scope_drift / sloppy_patching / fake_data_bypass / verification_by_vibes —
n/a (one real server subprocess, one real Postgres, no mocks; every row in
the DB got there through an MCP tool call).

The plan mandates ONE test function sharing ONE client session (the conftest
client is one-shot), so all ten steps assert sequentially inside a single
``async with`` block. Setting TASK16_EVIDENCE_DIR copies both step-10 digest
files there before the client fixture removes its temp digest dir (the
plan's evidence mandate; inert when unset).

# allow: SIZE_OK — plan task 16 mandates ONE test function holding the full
# ten-step loop; the steps share one client session by design.
"""

import json
import os
import shutil
from contextlib import AbstractAsyncContextManager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import psycopg
import pytest
import yaml
from mcp import ClientSession
from mcp.types import CallToolResult, TextContent
from psycopg.rows import DictRow

from tests.conftest import backdate

V_PAIR = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
W_SIDE = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
V_CORR = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

DEFAULT_NS = "default@local"  # server settings.MEMORY_NAMESPACE default
OTHER_NS = "other@proj"
GLOBAL = "global"

T_BASE = datetime(2026, 1, 10, 12, 0, tzinfo=timezone.utc)
DAY_1 = T_BASE
DAY_1_LATER = T_BASE + timedelta(hours=1)  # same UTC day as DAY_1
DAY_3 = T_BASE + timedelta(hours=48)
DAY_5 = T_BASE + timedelta(hours=96)

CAPTURE_A = {
    "goal": "stabilize the nightly deploy",
    "expectation": "rollout completes cleanly",
    "action": "retry on transient network errors",
    "outcome": "nightly deploy finished green",
    "surprise": 0.8,
    "state_at_encoding": {"mood": "calm"},
    "tags": ["ops", "deploy"],
}
CAPTURE_B = {
    "goal": "stabilize the canary deploy",
    "expectation": "rollout completes cleanly",
    "action": "retry on transient network errors",
    "outcome": "canary deploy finished green",
    "surprise": 0.8,
    "tags": ["ops"],
}
# No lexeme overlaps with PROBE_QUERY: the keyword channel must not surface it.
CAPTURE_C = {
    "goal": "rename the widget struct",
    "expectation": "compiler catches all call sites",
    "action": "run grep before refactoring",
    "outcome": "one missed call site panicked at runtime",
    "surprise": 0.3,
    "tags": ["refactor"],
}
CAPTURE_D = {
    "goal": "confirm pacing on the batch job",
    "expectation": "throughput holds under load",
    "action": "apply backoff pacing to workers",
    "outcome": "batch job held steady throughput",
    "surprise": 0.1,
    "tags": ["batch"],
}

CLAIM = "backoff pacing holds under load"
BECAUSE = "sync retries amplify packet storms"
HOLDS_WHEN = "under packet loss"
LESSON_TEXT = f"{CLAIM} {BECAUSE} {HOLDS_WHEN}"

PROBE_GOAL = "stabilize the deploy"
PROBE_APPROACH = "retry on transient network errors"
PROBE_QUERY = f"{PROBE_GOAL} {PROBE_APPROACH}"

PROMOTION_REASON = "broadly useful across projects"
DEMOTION_REASON = "superseded by a finer-grained pacing rule"


def _capture_text(fields: dict[str, Any]) -> str:
    """The exact server-side embedded text of a capture (4-field join)."""
    return (
        f"{fields['goal']} {fields['expectation']} "
        f"{fields['action']} {fields['outcome']}"
    )


def _ok(result: CallToolResult) -> dict[str, Any]:
    assert result.is_error is False, result.content
    block = result.content[0]
    assert isinstance(block, TextContent)
    payload: dict[str, Any] = json.loads(block.text)
    assert isinstance(payload, dict)
    return payload


def _read_digest(path: Path) -> tuple[dict[str, Any], str]:
    """Parse the written file itself — never trust the returned payload."""
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "---"
    end = lines.index("---", 1)
    frontmatter: dict[str, Any] = yaml.safe_load("\n".join(lines[1:end]))
    return frontmatter, "\n".join(lines[end + 1 :])


@pytest.fixture()
def fake_embed_overrides() -> dict[str, list[float]]:
    return {
        _capture_text(CAPTURE_A): V_PAIR,
        _capture_text(CAPTURE_B): V_PAIR,
        _capture_text(CAPTURE_C): W_SIDE,
        _capture_text(CAPTURE_D): V_CORR,
        LESSON_TEXT: V_PAIR,
        PROBE_QUERY: V_PAIR,
    }


async def test_full_memory_loop(
    db: psycopg.Connection[DictRow],
    client: AbstractAsyncContextManager[ClientSession],
) -> None:
    async with client as session:
        # Step 1: capture 3 episodes in one namespace — a same-day
        # near-duplicate pair (identical vector) and a distinct-day episode
        # whose vector sits below CLUSTER_COS similarity to the pair.
        capture_a = _ok(await session.call_tool("memory_capture_episode", CAPTURE_A))
        capture_b = _ok(await session.call_tool("memory_capture_episode", CAPTURE_B))
        capture_c = _ok(await session.call_tool("memory_capture_episode", CAPTURE_C))
        ep_a = int(capture_a["id"])
        ep_b = int(capture_b["id"])
        ep_c = int(capture_c["id"])
        backdate("episodes", ep_a, at=DAY_1)
        backdate("episodes", ep_b, at=DAY_1_LATER)
        backdate("episodes", ep_c, at=DAY_3)

        episodes = db.execute(
            "SELECT namespace, embedding FROM episodes ORDER BY id"
        ).fetchall()
        assert len(episodes) == 3
        for row in episodes:
            assert row["namespace"] == DEFAULT_NS
        assert [row["embedding"].to_list() for row in episodes] == [
            pytest.approx(V_PAIR),
            pytest.approx(V_PAIR),
            pytest.approx(W_SIDE),
        ]

        # Step 2: probe retrieves the relevant episode with provenance.
        probe = _ok(
            await session.call_tool(
                "memory_probe",
                {"current_goal": PROBE_GOAL, "approach": PROBE_APPROACH},
            )
        )
        assert isinstance(probe["retrieval_event_id"], int)
        result_ids = [str(record["id"]) for record in probe["results"]]
        assert len(result_ids) == 2
        assert set(result_ids) == {f"episode:{ep_a}", f"episode:{ep_b}"}
        assert f"episode:{ep_c}" not in result_ids
        record_a = next(
            record
            for record in probe["results"]
            if str(record["id"]) == f"episode:{ep_a}"
        )
        assert record_a["record_type"] == "episode"
        assert record_a["namespace"] == DEFAULT_NS
        assert record_a["goal"] == CAPTURE_A["goal"]
        assert record_a["expectation"] == CAPTURE_A["expectation"]
        assert record_a["action"] == CAPTURE_A["action"]
        assert record_a["outcome"] == CAPTURE_A["outcome"]
        assert record_a["surprise"] == pytest.approx(CAPTURE_A["surprise"])
        assert record_a["tags"] == CAPTURE_A["tags"]
        assert isinstance(record_a["created_at"], str)
        assert record_a["match_strength"] == pytest.approx(1.0)  # cos 1.0 hit
        event_id = probe["retrieval_event_id"]
        event = db.execute(
            """
            SELECT tool, namespace, query_context, returned_ids
            FROM retrieval_events WHERE id = %(id)s
            """,
            {"id": event_id},
        ).fetchone()
        assert event is not None
        assert event["tool"] == "probe"
        assert event["namespace"] == DEFAULT_NS
        assert event["query_context"] == PROBE_QUERY
        assert set(event["returned_ids"]) == {f"episode:{ep_a}", f"episode:{ep_b}"}
        stats = db.execute(
            "SELECT access_count, last_accessed FROM episodes WHERE id = %(id)s",
            {"id": ep_a},
        ).fetchone()
        assert stats is not None
        assert stats["access_count"] == 0  # exposure is not usage
        assert stats["last_accessed"] is None

        # Step 3: report_usage — helped on the surfaced episode.
        _ok(
            await session.call_tool(
                "memory_report_usage",
                {
                    "retrieval_event_id": event_id,
                    "results": [{"id": f"episode:{ep_a}", "outcome": "helped"}],
                },
            )
        )
        usage_rows = db.execute(
            "SELECT record_id, record_type, outcome FROM usage_reports"
        ).fetchall()
        assert len(usage_rows) == 1
        assert usage_rows[0]["record_id"] == f"episode:{ep_a}"
        assert usage_rows[0]["record_type"] == "episode"
        assert usage_rows[0]["outcome"] == "helped"
        stats = db.execute(
            "SELECT access_count, last_accessed FROM episodes WHERE id = %(id)s",
            {"id": ep_a},
        ).fetchone()
        assert stats is not None
        assert stats["access_count"] == 1
        assert stats["last_accessed"] is not None

        # Step 4: consolidate_scan (fresh) returns exactly the 2-cluster.
        scan = _ok(
            await session.call_tool(
                "memory_consolidate_scan",
                {"pool": "fresh", "min_cluster_size": 2},
            )
        )
        assert scan["pool"] == "fresh"
        assert scan["namespace"] == DEFAULT_NS
        assert scan["rederivation_groups"] == []
        assert [
            [str(record["id"]) for record in cluster["episodes"]]
            for cluster in scan["clusters"]
        ] == [[f"episode:{ep_a}", f"episode:{ep_b}"]]
        scan_record = scan["clusters"][0]["episodes"][0]
        assert scan_record["goal"] == CAPTURE_A["goal"]
        assert scan_record["surprise"] == pytest.approx(CAPTURE_A["surprise"])
        assert scan_record["tags"] == CAPTURE_A["tags"]
        assert scan_record["state_at_encoding"] == CAPTURE_A["state_at_encoding"]
        assert scan_record["cited_by_lessons"] == []
        edge_count = db.execute(
            "SELECT count(*) AS n FROM lesson_evidence"
        ).fetchone()
        assert edge_count is not None
        assert int(edge_count["n"]) == 0  # nothing consolidated yet

        # Step 5: write_lesson over all 3 episodes — seed EXACTLY
        # 0.35 + min(0.15, 0.05 x (2-1)) + 0.20 x (2/4) = 0.50.
        written = _ok(
            await session.call_tool(
                "memory_write_lesson",
                {
                    "claim": CLAIM,
                    "because": BECAUSE,
                    "holds_when": HOLDS_WHEN,
                    "evidence": [
                        {"episode_id": ep_a, "relation": "support"},
                        {"episode_id": ep_b, "relation": "support"},
                        {
                            "episode_id": ep_c,
                            "relation": "refine",
                            "reason": "confirmed on a distinct day",
                        },
                    ],
                },
            )
        )
        assert written["seed_confidence"] == pytest.approx(0.50)
        lesson_id = int(written["lesson_id"])
        lesson = db.execute(
            """
            SELECT namespace, confidence, last_evidence_at
            FROM lessons WHERE id = %(id)s
            """,
            {"id": lesson_id},
        ).fetchone()
        assert lesson is not None
        assert lesson["namespace"] == DEFAULT_NS
        assert float(lesson["confidence"]) == pytest.approx(0.50)
        assert lesson["last_evidence_at"] == DAY_3  # max over evidence episodes
        edges = db.execute(
            """
            SELECT episode_id, relation, reason FROM lesson_evidence
            WHERE lesson_id = %(id)s ORDER BY episode_id
            """,
            {"id": lesson_id},
        ).fetchall()
        assert [
            (int(edge["episode_id"]), edge["relation"], edge["reason"])
            for edge in edges
        ] == [
            (ep_a, "support", ""),
            (ep_b, "support", ""),
            (ep_c, "refine", "confirmed on a distinct day"),
        ]

        # Step 6: corroborate with a new distinct-day episode — confidence
        # rises by +0.1 x novelty (novelty 1.0), under the 0.95 cap.
        capture_d = _ok(await session.call_tool("memory_capture_episode", CAPTURE_D))
        ep_d = int(capture_d["id"])
        backdate("episodes", ep_d, at=DAY_5)
        moved = _ok(
            await session.call_tool(
                "memory_corroborate",
                {
                    "lesson_id": lesson_id,
                    "episode_id": ep_d,
                    "reason": "independent confirmation",
                },
            )
        )
        assert moved["applied"] is True
        assert moved["relation"] == "support"
        assert moved["confidence"] - 0.50 == pytest.approx(0.1)
        assert moved["confidence"] == pytest.approx(0.60)
        assert moved["confidence"] <= 0.95
        lesson = db.execute(
            """
            SELECT confidence, last_evidence_at FROM lessons WHERE id = %(id)s
            """,
            {"id": lesson_id},
        ).fetchone()
        assert lesson is not None
        assert float(lesson["confidence"]) == pytest.approx(0.60)
        assert lesson["last_evidence_at"] == DAY_5
        edge = db.execute(
            """
            SELECT relation, reason FROM lesson_evidence
            WHERE lesson_id = %(lesson)s AND episode_id = %(episode)s
            """,
            {"lesson": lesson_id, "episode": ep_d},
        ).fetchone()
        assert edge is not None
        assert edge["relation"] == "support"
        assert edge["reason"] == "independent confirmation"

        # Step 7: promote to global — a copy, never a move.
        promoted = _ok(
            await session.call_tool(
                "memory_promote",
                {"lesson_id": lesson_id, "reason": PROMOTION_REASON},
            )
        )
        copy_id = int(promoted["promoted_lesson_id"])
        assert copy_id != lesson_id
        copy = db.execute(
            """
            SELECT namespace, promotion_status, promoted_from_lesson_id,
                   promotion_reason
            FROM lessons WHERE id = %(id)s
            """,
            {"id": copy_id},
        ).fetchone()
        assert copy is not None
        assert copy["namespace"] == GLOBAL
        assert copy["promotion_status"] == "active"
        assert copy["promoted_from_lesson_id"] == lesson_id
        assert copy["promotion_reason"] == PROMOTION_REASON
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
        copy_edges = db.execute(
            "SELECT count(*) AS n FROM lesson_evidence WHERE lesson_id = %(id)s",
            {"id": copy_id},
        ).fetchone()
        assert copy_edges is not None
        assert int(copy_edges["n"]) == 4  # all 4 edges copied with the lesson

        # Step 8: a probe from ANOTHER namespace sees the global lesson.
        probe_other = _ok(
            await session.call_tool(
                "memory_probe",
                {
                    "current_goal": PROBE_GOAL,
                    "approach": PROBE_APPROACH,
                    "namespace": OTHER_NS,
                },
            )
        )
        assert [str(record["id"]) for record in probe_other["results"]] == [
            f"lesson:{copy_id}"
        ]
        global_record = probe_other["results"][0]
        assert global_record["record_type"] == "lesson"
        assert global_record["namespace"] == GLOBAL
        assert global_record["claim"] == CLAIM
        assert global_record["because"] == BECAUSE
        assert global_record["confidence"] == pytest.approx(0.60)
        assert global_record["match_strength"] == pytest.approx(1.0)
        by_episode = {
            str(edge["episode"]): edge for edge in global_record["evidence"]
        }
        assert set(by_episode) == {
            f"episode:{ep_a}",
            f"episode:{ep_b}",
            f"episode:{ep_c}",
            f"episode:{ep_d}",
        }
        assert by_episode[f"episode:{ep_a}"]["relation"] == "support"
        assert by_episode[f"episode:{ep_c}"]["relation"] == "refine"
        assert by_episode[f"episode:{ep_c}"]["reason"] == "confirmed on a distinct day"
        assert by_episode[f"episode:{ep_d}"]["reason"] == "independent confirmation"
        assert by_episode[f"episode:{ep_a}"]["excerpt"] == CAPTURE_A["outcome"]
        assert by_episode[f"episode:{ep_d}"]["excerpt"] == CAPTURE_D["outcome"]

        # Step 9: demote the copy — the tombstone hides it everywhere.
        demoted = _ok(
            await session.call_tool(
                "memory_demote",
                {"lesson_id": copy_id, "reason": DEMOTION_REASON},
            )
        )
        assert demoted == {"lesson_id": copy_id, "promotion_status": "demoted"}
        probe_gone = _ok(
            await session.call_tool(
                "memory_probe",
                {
                    "current_goal": PROBE_GOAL,
                    "approach": PROBE_APPROACH,
                    "namespace": OTHER_NS,
                },
            )
        )
        gone_ids = [str(record["id"]) for record in probe_gone["results"]]
        assert f"lesson:{copy_id}" not in gone_ids
        assert gone_ids == []
        gone_event = db.execute(
            "SELECT returned_ids FROM retrieval_events WHERE id = %(id)s",
            {"id": probe_gone["retrieval_event_id"]},
        ).fetchone()
        assert gone_event is not None
        assert gone_event["returned_ids"] == []
        tombstone = db.execute(
            """
            SELECT promotion_status, demoted_at, demotion_reason,
                   promoted_from_lesson_id
            FROM lessons WHERE id = %(id)s
            """,
            {"id": copy_id},
        ).fetchone()
        assert tombstone is not None
        assert tombstone["promotion_status"] == "demoted"
        assert tombstone["demoted_at"] is not None
        assert tombstone["demotion_reason"] == DEMOTION_REASON
        assert tombstone["promoted_from_lesson_id"] == lesson_id
        original = db.execute(
            """
            SELECT confidence, promotion_status, promoted_from_lesson_id
            FROM lessons WHERE id = %(id)s
            """,
            {"id": lesson_id},
        ).fetchone()
        assert original is not None
        assert float(original["confidence"]) == pytest.approx(0.60)
        assert original["promotion_status"] == "active"
        assert original["promoted_from_lesson_id"] is None

        # Step 10: the global digest flags the demoted promotion; the
        # project-namespace digest renders the plain view.
        global_digest = _ok(
            await session.call_tool("memory_digest", {"namespace": GLOBAL})
        )
        global_path = Path(global_digest["path"])
        assert global_path.exists()
        global_frontmatter, global_body = _read_digest(global_path)
        assert global_frontmatter["namespace"] == GLOBAL
        assert "## 5. Recently demoted promotions" in global_body
        assert DEMOTION_REASON in global_body
        assert global_digest["flagged_count"] == 1
        assert len(global_digest["flagged"]) == 1
        assert "demoted" in global_digest["flagged"][0]
        assert f"[[lesson:{copy_id}]]" in global_digest["flagged"][0]

        project_digest = _ok(await session.call_tool("memory_digest", {}))
        project_path = Path(project_digest["path"])
        assert project_path.exists()
        project_frontmatter, project_body = _read_digest(project_path)
        assert project_frontmatter["namespace"] == DEFAULT_NS
        # The demotion stays auditable from the source lesson's namespace
        # too (tombstones surface from BOTH namespaces — task-13 contract).
        assert project_digest["flagged_count"] == 1
        assert "demoted" in project_digest["flagged"][0]
        assert f"[[lesson:{copy_id}]]" in project_digest["flagged"][0]
        assert f"[[lesson:{lesson_id}]]" in project_body
        snapshot: dict[str, float] = project_frontmatter["confidence_snapshot"]
        assert float(snapshot[str(lesson_id)]) == pytest.approx(0.60)

        # Evidence hook: copy the digest files out before the client fixture
        # removes its temp digest dir (inert when the env var is unset).
        evidence_dir = os.environ.get("TASK16_EVIDENCE_DIR")
        if evidence_dir:
            shutil.copy2(global_path, Path(evidence_dir) / "digest-global.md")
            shutil.copy2(project_path, Path(evidence_dir) / "digest-project.md")
