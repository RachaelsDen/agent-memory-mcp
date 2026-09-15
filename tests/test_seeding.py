"""Golden tests for consolidation pure functions (DESIGN §6/§8, plan task 5).

Goldens 0.40 / 0.45 / 0.70 are review-hardened; float comparison via
pytest.approx is mandatory everywhere. Occasions are distinct UTC dates
computed over SOURCE episodes, never incident representatives.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta, timezone

import pytest

from agent_memory.consolidate import (
    Episode,
    collapse_incidents,
    corroborate_delta,
    contradict_delta,
    diversity,
    novelty,
    seed_confidence,
)

UTC = timezone.utc
NS = "default@local"
OTHER_NS = "other@local"
VEC_A = (1.0, 0.0)
VEC_B = (0.0, 1.0)  # orthogonal to VEC_A: cosine 0.0
DEDUP_COS = 0.95
WINDOW_H = 24


def utc_dt(
    year: int, month: int, day: int, hour: int = 0, minute: int = 0
) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


def make_episode(
    identifier: int,
    created_at: datetime,
    embedding: tuple[float, ...] = VEC_A,
    namespace: str = NS,
) -> Episode:
    return Episode(
        id=identifier, namespace=namespace, created_at=created_at, embedding=embedding
    )


def occasions_of(episodes: Sequence[Episode]) -> list[datetime]:
    return [episode.created_at for episode in episodes]


class TestCollapseIncidents:
    def test_collapse_when_five_same_day_near_duplicates(self):
        # Given: five near-duplicate episodes (hourly, identical vectors) on ONE UTC day
        episodes = [
            make_episode(identifier, utc_dt(2026, 1, 10, 9 + offset))
            for offset, identifier in enumerate(range(5))
        ]

        # When: collapse + seed
        groups = collapse_incidents(episodes, dedup_cos=DEDUP_COS, window_h=WINDOW_H)
        seed = seed_confidence(
            incidents=len(groups), diversity=diversity(occasions_of(episodes))
        )

        # Then: one incident of five members, earliest is the seed, golden 0.40
        assert len(groups) == 1
        assert [ep.id for ep in groups[0]] == [0, 1, 2, 3, 4]
        assert seed == pytest.approx(0.40)

    def test_seed_when_incident_window_spans_midnight(self):
        # Given: one incident at 23:50 -> 00:10 (20 min apart, same vector)
        episodes = [
            make_episode(1, utc_dt(2026, 1, 10, 23, 50)),
            make_episode(2, utc_dt(2026, 1, 11, 0, 10)),
        ]

        # When: collapse + seed
        groups = collapse_incidents(episodes, dedup_cos=DEDUP_COS, window_h=WINDOW_H)
        seed = seed_confidence(
            incidents=len(groups), diversity=diversity(occasions_of(episodes))
        )

        # Then: one incident, but occasions count BOTH source UTC dates -> golden 0.45
        assert len(groups) == 1
        assert len(groups[0]) == 2
        assert seed == pytest.approx(0.45)

    def test_seed_when_five_incidents_on_five_distinct_days(self):
        # Given: identical vectors 48h apart (outside the 24h window) on five distinct UTC dates
        base = utc_dt(2026, 1, 10, 12)
        episodes = [
            make_episode(identifier, base + timedelta(hours=48 * offset))
            for offset, identifier in enumerate(range(5))
        ]

        # When: collapse + seed
        groups = collapse_incidents(episodes, dedup_cos=DEDUP_COS, window_h=WINDOW_H)
        seed = seed_confidence(
            incidents=len(groups), diversity=diversity(occasions_of(episodes))
        )

        # Then: five incidents, diversity capped at 1.0 -> golden 0.70
        assert len(groups) == 5
        assert diversity(occasions_of(episodes)) == pytest.approx(1.0)
        assert seed == pytest.approx(0.70)

    def test_collapse_when_cross_namespace_never_merges(self):
        # Given: near-duplicates 5 minutes apart in DIFFERENT namespaces
        episodes = [
            make_episode(1, utc_dt(2026, 1, 10, 12, 0), namespace=NS),
            make_episode(2, utc_dt(2026, 1, 10, 12, 5), namespace=OTHER_NS),
        ]

        # When
        groups = collapse_incidents(episodes, dedup_cos=DEDUP_COS, window_h=WINDOW_H)

        # Then: two single-member incidents, one per namespace
        assert [len(group) for group in groups] == [1, 1]
        assert {group[0].namespace for group in groups} == {NS, OTHER_NS}

    def test_collapse_when_beyond_window_h(self):
        # Given: identical vectors 25h apart (outside the 24h seed window)
        episodes = [
            make_episode(1, utc_dt(2026, 1, 10, 12)),
            make_episode(2, utc_dt(2026, 1, 11, 13)),
        ]

        # When
        groups = collapse_incidents(episodes, dedup_cos=DEDUP_COS, window_h=WINDOW_H)

        # Then: two incidents
        assert [len(group) for group in groups] == [1, 1]

    def test_collapse_when_cosine_at_or_below_dedup_cos(self):
        # Given: same timestamp, orthogonal vectors (cosine 0.0 < 0.95)
        episodes = [
            make_episode(1, utc_dt(2026, 1, 10, 12), embedding=VEC_A),
            make_episode(2, utc_dt(2026, 1, 10, 12), embedding=VEC_B),
        ]

        # When
        groups = collapse_incidents(episodes, dedup_cos=DEDUP_COS, window_h=WINDOW_H)

        # Then: two incidents
        assert [len(group) for group in groups] == [1, 1]

    def test_collapse_when_member_window_measured_against_seed_not_chain(self):
        # Given: A at 00:00, B at +10h (joins A), C at +30h (within 24h of B,
        # but 30h > 24h from seed A) — all identical vectors
        base = utc_dt(2026, 1, 10, 0)
        episode_a = make_episode(1, base)
        episode_b = make_episode(2, base + timedelta(hours=10))
        episode_c = make_episode(3, base + timedelta(hours=30))

        # When
        groups = collapse_incidents(
            [episode_a, episode_b, episode_c], dedup_cos=DEDUP_COS, window_h=WINDOW_H
        )

        # Then: A+B form one incident, C seeds its own — no transitive chaining
        assert [[ep.id for ep in group] for group in groups] == [[1, 2], [3]]

    def test_collapse_when_input_arrives_unsorted(self):
        # Given: the same-day cluster handed over in reverse chronological order
        episodes = [
            make_episode(4, utc_dt(2026, 1, 10, 13)),
            make_episode(2, utc_dt(2026, 1, 10, 11)),
            make_episode(0, utc_dt(2026, 1, 10, 9)),
            make_episode(3, utc_dt(2026, 1, 10, 12)),
            make_episode(1, utc_dt(2026, 1, 10, 10)),
        ]

        # When
        groups = collapse_incidents(episodes, dedup_cos=DEDUP_COS, window_h=WINDOW_H)

        # Then: one incident, members ordered chronologically with earliest as seed
        assert [ep.id for ep in groups[0]] == [0, 1, 2, 3, 4]

    def test_collapse_when_no_episodes(self):
        # Given: no episodes
        # When
        groups = collapse_incidents([], dedup_cos=DEDUP_COS, window_h=WINDOW_H)

        # Then: no incidents
        assert groups == []


class TestDiversity:
    @pytest.mark.parametrize(
        ("date_count", "expected"),
        [(0, 0.0), (1, 0.25), (3, 0.75), (4, 1.0), (9, 1.0)],
    )
    def test_diversity_when_n_distinct_utc_dates(
        self, date_count: int, expected: float
    ):
        # Given: date_count timestamps, one per distinct UTC day
        occasions = [
            utc_dt(2026, 1, 1, 12) + timedelta(days=offset)
            for offset in range(date_count)
        ]

        # When
        result = diversity(occasions)

        # Then: distinct UTC dates over cap 4, clipped at 1.0
        assert result == pytest.approx(expected)

    def test_diversity_when_offsets_share_one_utc_date(self):
        # Given: 23:30 UTC and 01:30+02:00 — the SAME instant (23:30 UTC the
        # previous calendar day in local terms), so ONE distinct UTC date
        occasions = [
            utc_dt(2026, 1, 10, 23, 30),
            datetime(2026, 1, 11, 1, 30, tzinfo=timezone(timedelta(hours=2))),
        ]

        # When
        result = diversity(occasions)

        # Then: 1/4 — a naive .date() (no UTC normalization) would yield 2/4
        assert result == pytest.approx(0.25)


class TestSeedConfidence:
    def test_seed_when_repeat_bonus_caps_at_0_15(self):
        # Given: saturation-level incidents and full diversity
        # When
        result = seed_confidence(incidents=100, diversity=1.0)

        # Then: 0.35 + 0.15 (capped) + 0.20 = 0.70 — the cap math cannot exceed 0.70
        assert result == pytest.approx(0.70)

    def test_seed_when_first_incident_no_diversity(self):
        # Given: single incident, single occasion
        # When
        result = seed_confidence(incidents=1, diversity=0.0)

        # Then: base rate only
        assert result == pytest.approx(0.35)


class TestNovelty:
    def test_novelty_when_no_existing_support(self):
        # Given: a fresh episode and zero existing support
        episode = make_episode(1, utc_dt(2026, 1, 10, 12), embedding=VEC_A)

        # When
        result = novelty(episode, [], dedup_cos=DEDUP_COS)

        # Then: vacuously novel
        assert result == pytest.approx(1.0)

    def test_novelty_when_new_date_and_dissimilar_vector(self):
        # Given: different UTC date and orthogonal vector to all existing support
        episode = make_episode(1, utc_dt(2026, 1, 12, 12), embedding=VEC_A)
        existing = [make_episode(9, utc_dt(2026, 1, 10, 12), embedding=VEC_B)]

        # When
        result = novelty(episode, existing, dedup_cos=DEDUP_COS)

        # Then: fully novel
        assert result == pytest.approx(1.0)

    def test_novelty_when_near_duplicate_date_and_vector(self):
        # Given: same UTC date AND near-duplicate vector (golden 0.3)
        episode = make_episode(1, utc_dt(2026, 1, 10, 15), embedding=VEC_A)
        existing = [make_episode(9, utc_dt(2026, 1, 10, 9), embedding=VEC_A)]

        # When
        result = novelty(episode, existing, dedup_cos=DEDUP_COS)

        # Then: near-duplicate of existing support
        assert result == pytest.approx(0.3)

    def test_novelty_when_same_utc_date_only(self):
        # Given: same UTC date but orthogonal vector
        episode = make_episode(1, utc_dt(2026, 1, 10, 15), embedding=VEC_B)
        existing = [make_episode(9, utc_dt(2026, 1, 10, 9), embedding=VEC_A)]

        # When
        result = novelty(episode, existing, dedup_cos=DEDUP_COS)

        # Then: date collision alone drops novelty
        assert result == pytest.approx(0.3)

    def test_novelty_when_near_duplicate_vector_only(self):
        # Given: different UTC date but near-duplicate vector
        episode = make_episode(1, utc_dt(2026, 1, 12, 15), embedding=VEC_A)
        existing = [make_episode(9, utc_dt(2026, 1, 10, 9), embedding=VEC_A)]

        # When
        result = novelty(episode, existing, dedup_cos=DEDUP_COS)

        # Then: vector collision alone drops novelty
        assert result == pytest.approx(0.3)

    def test_novelty_when_offsets_share_one_utc_date(self):
        # Given: episode at 00:10+02:00 (22:10 UTC Jan 10) vs existing 23:30 UTC
        # Jan 10 — same UTC date despite different calendar days locally
        episode = make_episode(
            1, datetime(2026, 1, 11, 0, 10, tzinfo=timezone(timedelta(hours=2)))
        )
        existing = [make_episode(9, utc_dt(2026, 1, 10, 23, 30))]

        # When
        result = novelty(episode, existing, dedup_cos=DEDUP_COS)

        # Then: a naive .date() comparison would call these different days
        assert result == pytest.approx(0.3)


class TestEvidenceMoves:
    def test_corroborate_when_novelty_scales_the_boost(self):
        # Given: mid confidence and both novelty regimes
        # When / Then: +0.1 x novelty
        assert corroborate_delta(0.50, nov=0.3) == pytest.approx(0.53)
        assert corroborate_delta(0.50, nov=1.0) == pytest.approx(0.60)

    def test_corroborate_when_confidence_caps_at_0_95(self):
        # Given: confidence already near the cap
        # When / Then
        assert corroborate_delta(0.90, nov=1.0) == pytest.approx(0.95)
        assert corroborate_delta(0.99, nov=1.0) == pytest.approx(0.95)

    def test_contradict_when_confidence_floors_at_0_05(self):
        # Given: low confidence heading lower
        # When / Then
        assert contradict_delta(0.50) == pytest.approx(0.30)
        assert contradict_delta(0.10) == pytest.approx(0.05)
        assert contradict_delta(0.01) == pytest.approx(0.05)

    def test_evidence_moves_when_compared_directly(self):
        # Given: identical starting confidence (DESIGN §6: 2:1 asymmetry)
        confidence = 0.50

        # When: a fully novel corroborate vs a contradict
        corroborate_move = corroborate_delta(confidence, nov=1.0) - confidence
        contradict_move = confidence - contradict_delta(confidence)

        # Then: contradiction moves twice as far as the best corroboration
        assert corroborate_move == pytest.approx(0.10)
        assert contradict_move == pytest.approx(0.20)
