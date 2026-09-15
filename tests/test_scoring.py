"""Golden battery for the retrieval scoring contract (DESIGN.md S7, plan task 4).

Deterministic pure math throughout: `now` is a fixed module constant; no clocks,
no randomness, no DB. Float goldens use pytest.approx; contract-exact values
(top rel_norm 1.0, boost-only use_fresh 0.0, guard zeros) use `==`.
"""

# allow: SIZE_OK — plan task 4 mandates ONE tests/test_scoring.py holding the full golden battery

from __future__ import annotations

import math
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from agent_memory.config import Settings
from agent_memory.retrieve import (
    Candidate,
    Edge,
    RecordType,
    ScoringConfigError,
    ScoringParams,
    activation_hop,
    env_fresh,
    gate,
    keyword_strength,
    match_strength,
    rrf,
    rel_norm,
    score,
    use_fresh,
    vector_strength,
)

NOW = datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc)


def params_from_settings(settings: Settings) -> ScoringParams:
    return ScoringParams(
        w_rel=settings.W_REL,
        w_sal=settings.W_SAL,
        w_env=settings.W_ENV,
        w_use=settings.W_USE,
        w_spread=settings.W_SPREAD,
        sim_floor=settings.SIM_FLOOR,
        ts_rank_sat=settings.TS_RANK_SAT,
        tau_env_h=float(settings.TAU_ENV_H),
        tau_use_h=float(settings.TAU_USE_H),
        probe_topk=settings.PROBE_TOPK,
    )


PARAMS = params_from_settings(Settings())


def candidate(
    record_type: RecordType,
    record_id: int,
    *,
    salience: float = 0.0,
    cosine: float = 0.9,
    ts_rank: float = 0.0,
    evidence_ts: datetime = NOW,
    last_accessed: datetime | None = None,
) -> Candidate:
    return Candidate(
        record_type=record_type,
        record_id=record_id,
        salience=salience,
        cosine=cosine,
        ts_rank=ts_rank,
        evidence_ts=evidence_ts,
        last_accessed=last_accessed,
    )


# --- RRF: typed refs, 1-based ranks, k = 60 --------------------------------


def test_rrf_when_ref_is_rank1_in_both_channels():
    fused = rrf([["lesson:1"], ["lesson:1"]])
    assert fused["lesson:1"] == pytest.approx(0.03278688524590164)
    assert fused["lesson:1"] == pytest.approx(2 / 61)


def test_rrf_when_numeric_ids_match_across_record_types():
    fused = rrf([["episode:1"], ["lesson:1"]])
    assert set(fused) == {"episode:1", "lesson:1"}
    assert fused["episode:1"] == pytest.approx(1 / 61)
    assert fused["lesson:1"] == pytest.approx(1 / 61)


def test_rrf_when_channel_list_mixes_episodes_and_lessons():
    fused = rrf([["lesson:3", "episode:5"]])
    assert fused["lesson:3"] == pytest.approx(1 / 61)
    assert fused["episode:5"] == pytest.approx(1 / 62)


def test_rrf_when_no_lists_or_empty_list():
    assert rrf([]) == {}
    assert rrf([[]]) == {}


# --- Normalization contract -------------------------------------------------


def test_rel_norm_when_pool_nonempty_crowns_top_at_1():
    fused = rrf([["episode:1", "episode:2"], ["episode:1", "episode:2"]])
    normalized = rel_norm(fused)
    assert normalized["episode:1"] == 1.0
    assert normalized["episode:2"] == pytest.approx((2 / 62) / (2 / 61))


def test_rel_norm_when_pool_empty_returns_empty_map():
    assert rel_norm(rrf([])) == {}


def test_thin_pool_when_single_weak_candidate():
    normalized = rel_norm(rrf([["episode:1"]]))
    assert normalized == {"episode:1": 1.0}
    weak = match_strength(0.26, 0.0, 0.25, 0.1)
    assert weak == pytest.approx(0.013333333333333334)
    assert weak < 0.1


# --- Channel strengths ------------------------------------------------------


def test_vector_strength_when_cosine_just_above_floor():
    assert vector_strength(0.30, 0.25) == pytest.approx(0.06666666666666665)


def test_vector_strength_when_cosine_at_or_above_one_clips_to_1():
    assert vector_strength(1.0, 0.25) == 1.0
    assert vector_strength(2.0, 0.25) == 1.0


def test_vector_strength_when_cosine_at_or_below_floor_floors_to_0():
    assert vector_strength(0.25, 0.25) == 0.0
    assert vector_strength(0.1, 0.25) == 0.0


def test_keyword_strength_when_below_saturation_scales_linearly():
    assert keyword_strength(0.05, 0.1) == pytest.approx(0.5)


def test_keyword_strength_when_at_or_above_saturation_clips_to_1():
    assert keyword_strength(0.1, 0.1) == pytest.approx(1.0)
    assert keyword_strength(0.5, 0.1) == 1.0


def test_match_strength_when_vector_channel_dominates():
    # max((0.9-0.25)/0.75, 0.01/0.1) = 0.8667
    assert match_strength(0.9, 0.01, 0.25, 0.1) == pytest.approx(0.8666666666666667)


def test_match_strength_when_keyword_channel_dominates():
    # max((0.3-0.25)/0.75, 0.05/0.1) = 0.5
    assert match_strength(0.3, 0.05, 0.25, 0.1) == pytest.approx(0.5)


# --- Gate -------------------------------------------------------------------


def test_gate_when_either_channel_above_noise():
    assert gate(0.26, 0.0, 0.25) is True
    assert gate(0.0, 0.0001, 0.25) is True
    assert gate(0.9, 0.05, 0.25) is True


def test_gate_when_both_channels_at_or_below_noise():
    assert gate(0.25, 0.0, 0.25) is False
    assert gate(0.2, 0.0, 0.25) is False
    assert gate(0.0, 0.0, 0.25) is False


# --- Freshness --------------------------------------------------------------


def test_env_fresh_when_evidence_is_now():
    assert env_fresh(NOW, NOW, 4320.0) == pytest.approx(1.0)


def test_env_fresh_when_one_tau_elapsed():
    aged = NOW - timedelta(hours=4320)
    assert env_fresh(aged, NOW, 4320.0) == pytest.approx(math.exp(-1.0))
    assert env_fresh(aged, NOW, 4320.0) == pytest.approx(0.36787944117144233)


def test_env_fresh_when_evidence_in_future_clamps_to_fresh():
    future = NOW + timedelta(hours=48)
    assert env_fresh(future, NOW, 4320.0) == pytest.approx(1.0)


def test_use_fresh_when_never_used_is_exactly_zero():
    assert use_fresh(None, NOW, 720.0) == 0.0


def test_use_fresh_when_one_tau_elapsed():
    last = NOW - timedelta(hours=720)
    assert use_fresh(last, NOW, 720.0) == pytest.approx(0.36787944117144233)


# --- Score ------------------------------------------------------------------


def test_score_when_every_term_hand_computable():
    record = candidate(
        "episode", 1, salience=0.8, cosine=0.9, ts_rank=0.05, last_accessed=NOW
    )
    total = score(record, relevance=0.5, activation=0.0, params=PARAMS, now=NOW)
    # 0.45*0.5 + 0.20*0.8 + 0.15*1 + 0.10*1 + 0.10*0
    assert total == pytest.approx(0.635)


def test_score_when_never_used_vs_just_used_differs_by_use_weight_only():
    never = candidate("lesson", 1, salience=0.5, last_accessed=None)
    just_used = candidate("lesson", 1, salience=0.5, last_accessed=NOW)
    delta = score(just_used, relevance=0.5, activation=0.0, params=PARAMS, now=NOW) - (
        score(never, relevance=0.5, activation=0.0, params=PARAMS, now=NOW)
    )
    assert delta == pytest.approx(PARAMS.w_use)


def test_scoring_params_when_built_from_settings_defaults():
    weights = (PARAMS.w_rel, PARAMS.w_sal, PARAMS.w_env, PARAMS.w_use, PARAMS.w_spread)
    assert weights == (0.45, 0.20, 0.15, 0.10, 0.10)
    assert PARAMS.sim_floor == 0.25
    assert PARAMS.ts_rank_sat == pytest.approx(0.1)
    assert (PARAMS.tau_env_h, PARAMS.tau_use_h) == (4320.0, 720.0)
    assert PARAMS.probe_topk == 12


@pytest.mark.xfail(
    strict=True,
    reason=(
        "normalization contract trap: raw RRF (dual rank-1 = 0.0328) must never "
        "enter the weighted sum un-normalized"
    ),
)
def test_trap_when_raw_rrf_mixed_into_weighted_sum():
    raw = rrf([["episode:1"], ["episode:1"]])
    wrong = PARAMS.w_rel * raw["episode:1"]
    right = PARAMS.w_rel * rel_norm(raw)["episode:1"]
    assert wrong == pytest.approx(right)


# --- ScoringParams numeric domains ------------------------------------------


def test_scoring_params_when_sim_floor_outside_half_open_unit():
    with pytest.raises(ScoringConfigError):
        replace(PARAMS, sim_floor=1.0)
    with pytest.raises(ScoringConfigError):
        replace(PARAMS, sim_floor=-0.1)


def test_scoring_params_when_tau_not_finite_positive():
    with pytest.raises(ScoringConfigError):
        replace(PARAMS, tau_env_h=0.0)
    with pytest.raises(ScoringConfigError):
        replace(PARAMS, tau_use_h=math.nan)


def test_scoring_params_when_weight_negative_or_nonfinite():
    with pytest.raises(ScoringConfigError):
        replace(PARAMS, w_rel=-0.01)
    with pytest.raises(ScoringConfigError):
        replace(PARAMS, w_spread=math.inf)


def test_scoring_params_when_saturation_not_positive():
    with pytest.raises(ScoringConfigError):
        replace(PARAMS, ts_rank_sat=0.0)


def test_scoring_params_when_all_weights_zero_is_legal():
    inert = replace(PARAMS, w_rel=0.0, w_sal=0.0, w_env=0.0, w_use=0.0, w_spread=0.0)
    assert issubclass(ScoringConfigError, ValueError)
    assert inert.w_spread == 0.0


# --- Spreading activation: two phases, one hop ------------------------------


def test_activation_hop_when_two_phase_golden():
    records = {
        "lesson:1": candidate("lesson", 1, salience=1.0, cosine=0.9, ts_rank=0.05),
        "lesson:2": candidate("lesson", 2, salience=0.0, cosine=0.9, ts_rank=0.0),
    }
    relevance = rel_norm(rrf([["lesson:1"], ["lesson:1"]]))
    edges = [Edge(source="lesson:1", target="lesson:2", weight=0.5)]

    ranked = activation_hop(records, relevance, edges, PARAMS, NOW)

    assert [item.ref for item in ranked] == ["lesson:1", "lesson:2"]
    parent, neighbor = ranked
    assert parent.base_score == pytest.approx(0.80)  # 0.45*1 + 0.20*1 + 0.15*1
    assert parent.activation == 0.0
    assert parent.score == pytest.approx(0.80)
    assert parent.rel_norm == 1.0
    assert neighbor.rel_norm == 0.0  # expansion-only: frozen, never recomputed
    assert neighbor.base_score == pytest.approx(0.15)  # env_fresh term only
    assert neighbor.activation == pytest.approx(0.5)  # clip(0.5 * parent_norm 1.0)
    assert neighbor.score == pytest.approx(0.20)  # base + W_SPREAD * 0.5
    assert neighbor.match_strength == pytest.approx(0.8666666666666667)
    assert neighbor.env_fresh == pytest.approx(1.0)
    assert neighbor.use_fresh == 0.0


def test_activation_hop_when_parents_mutually_linked_no_circularity():
    records = {
        "lesson:1": candidate("lesson", 1, salience=1.0),
        "lesson:2": candidate("lesson", 2, salience=1.0),
    }
    relevance = {"lesson:1": 1.0, "lesson:2": 0.5}  # hand-frozen map
    edges = [
        Edge(source="lesson:1", target="lesson:2", weight=0.6),
        Edge(source="lesson:2", target="lesson:1", weight=0.6),
    ]

    ranked = activation_hop(records, relevance, edges, PARAMS, NOW)
    by_ref = {item.ref: item for item in ranked}

    # Bases 0.80 / 0.575 -> parent norms 1.0 / 0.71875. Activation reads BASE
    # norms only, so a mutual pair can never chase its own final scores.
    assert by_ref["lesson:1"].score == pytest.approx(0.843125)  # 0.80 + 0.10*0.6*0.71875
    assert by_ref["lesson:2"].score == pytest.approx(0.635)  # 0.575 + 0.10*0.6*1.0

    reversed_records = dict(reversed(list(records.items())))
    rerun = activation_hop(reversed_records, relevance, edges[::-1], PARAMS, NOW)
    assert rerun == ranked


def test_activation_hop_when_edge_weight_exceeds_one_clips():
    records = {
        "lesson:1": candidate("lesson", 1, salience=1.0, ts_rank=0.05),
        "lesson:2": candidate("lesson", 2, salience=0.0),
    }
    relevance = {"lesson:1": 1.0}
    edges = [Edge(source="lesson:1", target="lesson:2", weight=1.5)]

    ranked = activation_hop(records, relevance, edges, PARAMS, NOW)

    assert ranked[1].activation == 1.0
    assert ranked[1].score == pytest.approx(0.25)  # 0.15 base + 0.10 * 1.0


def test_activation_hop_when_activated_node_has_zero_base_spreads_nothing():
    spread_only = replace(PARAMS, w_sal=0.0, w_env=0.0, w_use=0.0)
    records = {
        "lesson:1": candidate("lesson", 1),
        "lesson:2": candidate("lesson", 2),
        "lesson:3": candidate("lesson", 3),
    }
    relevance = {"lesson:1": 1.0}  # lessons 2 and 3 are expansion-only: base 0
    edges = [
        Edge(source="lesson:1", target="lesson:2", weight=1.0),
        Edge(source="lesson:2", target="lesson:3", weight=1.0),
    ]

    ranked = activation_hop(records, relevance, edges, spread_only, NOW)
    by_ref = {item.ref: item for item in ranked}

    assert by_ref["lesson:2"].activation == 1.0  # from the real parent
    assert by_ref["lesson:2"].score == pytest.approx(0.10)
    assert by_ref["lesson:3"].activation == 0.0  # one hop: lesson:2 has base 0
    assert by_ref["lesson:3"].score == 0.0


def test_activation_hop_when_all_weights_zero_guard_holds():
    inert = replace(PARAMS, w_rel=0.0, w_sal=0.0, w_env=0.0, w_use=0.0, w_spread=0.0)
    records = {"lesson:1": candidate("lesson", 1, salience=1.0)}
    edges = [Edge(source="lesson:1", target="lesson:1", weight=1.0)]

    ranked = activation_hop(records, relevance={}, edges=edges, params=inert, now=NOW)

    assert len(ranked) == 1
    assert ranked[0].activation == 0.0
    assert ranked[0].score == 0.0


def test_activation_hop_when_pool_empty():
    assert activation_hop({}, {}, [], PARAMS, NOW) == []


def test_activation_hop_when_scores_tie_break_on_type_then_numeric_id():
    inert = replace(PARAMS, w_rel=0.0, w_sal=0.0, w_env=0.0, w_use=0.0, w_spread=0.0)
    records = {
        "lesson:10": candidate("lesson", 10),
        "lesson:2": candidate("lesson", 2),
        "lesson:1": candidate("lesson", 1),
        "episode:1": candidate("episode", 1),
    }

    ranked = activation_hop(records, relevance={}, edges=[], params=inert, now=NOW)

    assert [item.ref for item in ranked] == ["episode:1", "lesson:1", "lesson:2", "lesson:10"]


def test_activation_hop_when_record_fails_gate_excluded():
    records = {
        "episode:1": candidate("episode", 1, salience=1.0, ts_rank=0.05),
        "episode:9": candidate("episode", 9, salience=1.0, cosine=0.2, ts_rank=0.0),
    }
    relevance = {"episode:1": 1.0, "episode:9": 0.5}

    ranked = activation_hop(records, relevance, [], PARAMS, NOW)

    assert [item.ref for item in ranked] == ["episode:1"]

