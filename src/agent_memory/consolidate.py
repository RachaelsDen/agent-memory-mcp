"""Pure consolidation functions: near-duplicate collapse, diversity seeding,
novelty-scaled evidence moves (DESIGN §6, §8).

No DB imports: callers parse rows into `Episode` values at the boundary.
Occasions are distinct UTC dates computed over SOURCE episodes, never over
incident representatives (a midnight-crossing incident contributes both dates).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from math import sqrt


@dataclass(frozen=True, slots=True)
class Episode:
    """Boundary-parsed episode shape consumed by consolidation."""

    id: int
    namespace: str
    created_at: datetime  # tz-aware
    embedding: tuple[float, ...]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sqrt(sum(x * x for x in a))
    norm_b = sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


def _utc_date(timestamp: datetime) -> date:
    return timestamp.astimezone(timezone.utc).date()


def collapse_incidents(
    episodes: Sequence[Episode], dedup_cos: float, window_h: float
) -> list[list[Episode]]:
    """Greedy near-duplicate collapse (DESIGN §8).

    Sort by `created_at`; the earliest ungrouped episode seeds an incident;
    a later episode joins when it shares the seed's namespace, falls within
    `window_h` of the SEED (never transitively of another member), and has
    cosine-to-seed > `dedup_cos`. Cross-namespace evidence never merges.
    """
    remaining = sorted(episodes, key=lambda episode: (episode.created_at, episode.id))
    window = timedelta(hours=window_h)
    groups: list[list[Episode]] = []
    while remaining:
        seed = remaining[0]
        members = [seed]
        deferred: list[Episode] = []
        for episode in remaining[1:]:
            joins = (
                episode.namespace == seed.namespace
                and episode.created_at - seed.created_at <= window
                and _cosine(episode.embedding, seed.embedding) > dedup_cos
            )
            if joins:
                members.append(episode)
            else:
                deferred.append(episode)
        groups.append(members)
        remaining = deferred
    return groups


def diversity(occasions: Sequence[datetime], cap: int = 4) -> float:
    """Fraction of distinct-source-episode UTC dates, clipped at 1.0."""
    distinct_dates = {_utc_date(timestamp) for timestamp in occasions}
    return min(1.0, len(distinct_dates) / cap)


def seed_confidence(incidents: int, diversity: float) -> float:
    """DESIGN §8 seed formula; repeats pay a capped rate, diversity full rate."""
    return 0.35 + min(0.15, 0.05 * (incidents - 1)) + 0.20 * diversity


def novelty(
    episode: Episode, existing_support: Sequence[Episode], dedup_cos: float
) -> float:
    """1.0 only on a new UTC date with cosine < `dedup_cos` to every existing."""
    for existing in existing_support:
        shares_date = _utc_date(episode.created_at) == _utc_date(existing.created_at)
        near_duplicate = (
            _cosine(episode.embedding, existing.embedding) >= dedup_cos
        )
        if shares_date or near_duplicate:
            return 0.3
    return 1.0


def corroborate_delta(confidence: float, nov: float) -> float:
    """Novelty-scaled corroboration boost, capped at 0.95 (DESIGN §6)."""
    return min(0.95, confidence + 0.1 * nov)


def contradict_delta(confidence: float) -> float:
    """Flat contradiction penalty, floored at 0.05 (DESIGN §6)."""
    return max(0.05, confidence - 0.2)
