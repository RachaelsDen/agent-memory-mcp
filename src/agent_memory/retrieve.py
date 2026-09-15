"""Pure retrieval scoring: hybrid RRF, normalization contract, spreading activation.

Formulas are DESIGN.md S7 verbatim: every feature is scaled to [0, 1] BEFORE the
weighted sum so the weights are true mix shares within a query's candidate set.
No DB imports -- channel signals (cosine, ts_rank) and graph edges arrive as
arguments; config plumbing is the caller's job.

Refs are TYPED ("episode:<id>" / "lesson:<id>") end-to-end so equal numeric ids
across the two tables can never collide. Every ordered list this module produces
sorts by the total order (score DESC, record_type ASC, id ASC).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

RecordType = Literal["episode", "lesson"]

RRF_K = 60.0


class ScoringConfigError(ValueError):
    """A ScoringParams field is outside its documented numeric domain."""


@dataclass(frozen=True, slots=True)
class ScoringParams:
    """Weights and shaping constants for the scoring pipeline (DESIGN S7)."""

    w_rel: float
    w_sal: float
    w_env: float
    w_use: float
    w_spread: float
    sim_floor: float
    ts_rank_sat: float
    tau_env_h: float
    tau_use_h: float
    probe_topk: int

    def __post_init__(self) -> None:
        for name, weight in (
            ("w_rel", self.w_rel),
            ("w_sal", self.w_sal),
            ("w_env", self.w_env),
            ("w_use", self.w_use),
            ("w_spread", self.w_spread),
        ):
            if not math.isfinite(weight) or weight < 0:
                raise ScoringConfigError(f"{name} must be finite and >= 0, got {weight!r}")
        if not 0.0 <= self.sim_floor < 1.0:
            raise ScoringConfigError(f"sim_floor must be in [0, 1), got {self.sim_floor!r}")
        for name, value in (
            ("ts_rank_sat", self.ts_rank_sat),
            ("tau_env_h", self.tau_env_h),
            ("tau_use_h", self.tau_use_h),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ScoringConfigError(f"{name} must be finite and > 0, got {value!r}")
        if self.probe_topk < 1:
            raise ScoringConfigError(f"probe_topk must be >= 1, got {self.probe_topk!r}")


@dataclass(frozen=True, slots=True)
class Candidate:
    """One retrievable record with its per-query channel signals."""

    record_type: RecordType
    record_id: int
    salience: float  # surprise (episode) | confidence (lesson), already 0..1
    cosine: float
    ts_rank: float
    evidence_ts: datetime
    last_accessed: datetime | None

    @property
    def ref(self) -> str:
        return f"{self.record_type}:{self.record_id}"


@dataclass(frozen=True, slots=True)
class Edge:
    """Directed weighted graph link; activation flows source -> target."""

    source: str
    target: str
    weight: float


@dataclass(frozen=True, slots=True)
class Scored:
    """Final ranked result with the features that produced it."""

    ref: str
    record_type: RecordType
    record_id: int
    score: float
    base_score: float
    activation: float
    rel_norm: float
    match_strength: float
    env_fresh: float
    use_fresh: float


def rrf(rank_lists: list[list[str]]) -> dict[str, float]:
    """Reciprocal Rank Fusion over typed refs: sum of 1/(RRF_K + rank), 1-based.

    Each element of ``rank_lists`` is one channel's GLOBAL ranking (episodes and
    lessons merged into a single list before rank assignment).
    """
    fused: dict[str, float] = {}
    for ranking in rank_lists:
        for rank, ref in enumerate(ranking, start=1):
            fused[ref] = fused.get(ref, 0.0) + 1.0 / (RRF_K + rank)
    return fused


def rel_norm(rrf_map: dict[str, float]) -> dict[str, float]:
    """Max-normalize RRF to [0, 1] for mixing; empty pool is a no-op, not an error."""
    if not rrf_map:
        return {}
    top = max(rrf_map.values())
    return {ref: value / top for ref, value in rrf_map.items()}


def vector_strength(cosine: float, sim_floor: float) -> float:
    """Vector channel strength above the similarity floor, clipped to [0, 1]."""
    return min(1.0, max(0.0, (cosine - sim_floor) / (1.0 - sim_floor)))


def keyword_strength(ts_rank: float, sat: float) -> float:
    """Keyword channel strength: ts_rank against a saturation ceiling."""
    return min(1.0, ts_rank / sat)


def match_strength(cosine: float, ts_rank: float, sim_floor: float, sat: float) -> float:
    """Absolute 0..1 match quality from the underlying channels, never from RRF."""
    return max(vector_strength(cosine, sim_floor), keyword_strength(ts_rank, sat))


def gate(cosine: float, ts_rank: float, sim_floor: float) -> bool:
    """Pool eligibility: either channel above noise (cosine comparison strict)."""
    return ts_rank > 0 or cosine > sim_floor


def env_fresh(evidence_ts: datetime, now: datetime, tau_h: float) -> float:
    """Evidence-clock freshness; future evidence clamps to fully fresh."""
    elapsed_h = max(0.0, (now - evidence_ts).total_seconds() / 3600.0)
    return math.exp(-elapsed_h / tau_h)


def use_fresh(last_accessed: datetime | None, now: datetime, tau_h: float) -> float:
    """Usage freshness, boost-only: never-used contributes exactly 0.0."""
    if last_accessed is None:
        return 0.0
    return env_fresh(last_accessed, now, tau_h)


def score(
    candidate: Candidate,
    *,
    relevance: float,
    activation: float,
    params: ScoringParams,
    now: datetime,
) -> float:
    """Weighted feature sum; every feature is [0, 1] going in (DESIGN S7)."""
    return (
        params.w_rel * relevance
        + params.w_sal * candidate.salience
        + params.w_env * env_fresh(candidate.evidence_ts, now, params.tau_env_h)
        + params.w_use * use_fresh(candidate.last_accessed, now, params.tau_use_h)
        + params.w_spread * activation
    )


def activation_hop(
    records: Mapping[str, Candidate],
    relevance: Mapping[str, float],
    edges: Iterable[Edge],
    params: ScoringParams,
    now: datetime,
) -> list[Scored]:
    """Two-phase ranking with exactly ONE hop of spreading activation.

    ``relevance`` is the FROZEN rel_norm map over channel candidates: records
    linked but outside both channel cutoffs simply miss from it and keep
    rel_norm = 0 while their gate, strengths, salience, and freshness compute
    independently. Phase 1 scores with activation = 0; phase 2 picks the top
    ``probe_topk`` parents by base score, normalizes each parent by the best
    base, and spreads ``clip(weight * parent_norm, 0, 1)`` along edges (max
    aggregation). Finals re-rank once; neighbors never trigger another hop.
    """
    pool = {
        ref: record
        for ref, record in records.items()
        if gate(record.cosine, record.ts_rank, params.sim_floor)
    }
    base = {
        ref: score(
            record,
            relevance=relevance.get(ref, 0.0),
            activation=0.0,
            params=params,
            now=now,
        )
        for ref, record in pool.items()
    }
    by_total_order = sorted(
        pool,
        key=lambda ref: (-base[ref], pool[ref].record_type, pool[ref].record_id),
    )
    parents = by_total_order[: params.probe_topk]

    activation = dict.fromkeys(pool, 0.0)
    parent_max = max((base[ref] for ref in parents), default=0.0)
    if parent_max > 0.0:  # division guard: no parents or all-zero bases -> no spread
        parent_norms = {ref: base[ref] / parent_max for ref in parents}
        for edge in edges:
            parent_norm = parent_norms.get(edge.source)
            if parent_norm is None or edge.target not in pool:
                continue
            spread = min(1.0, max(0.0, edge.weight * parent_norm))
            if spread > activation[edge.target]:
                activation[edge.target] = spread

    scored = [
        Scored(
            ref=ref,
            record_type=record.record_type,
            record_id=record.record_id,
            score=base[ref] + params.w_spread * activation[ref],
            base_score=base[ref],
            activation=activation[ref],
            rel_norm=relevance.get(ref, 0.0),
            match_strength=match_strength(
                record.cosine, record.ts_rank, params.sim_floor, params.ts_rank_sat
            ),
            env_fresh=env_fresh(record.evidence_ts, now, params.tau_env_h),
            use_fresh=use_fresh(record.last_accessed, now, params.tau_use_h),
        )
        for ref, record in pool.items()
    ]
    return sorted(scored, key=lambda item: (-item.score, item.record_type, item.record_id))
