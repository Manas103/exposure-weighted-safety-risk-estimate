"""Benchmark the simulated fleet's exposure-weighted collision rate against
a published human crash-rate figure, in matched units (per 100M vehicle
miles traveled).

The human figure is a stated public number, not a live data pull: NHTSA's
Traffic Safety Facts reporting puts total police-reported motor vehicle
crashes (all severities) at roughly 6.0 million per year against roughly
3.2 trillion vehicle miles traveled per year in recent pre-pandemic-trend
US data, which works out to about 188 crashes per 100 million vehicle
miles traveled. That arithmetic, and its source, are stated here rather
than re-derived at runtime so the benchmark number cannot silently drift.
"""
from __future__ import annotations

HUMAN_CRASHES_PER_100M_VMT = 188.0
HUMAN_BENCHMARK_SOURCE = (
    "NHTSA Traffic Safety Facts (police-reported crashes, all severities): "
    "~6.0M crashes/year over ~3.2T vehicle miles traveled/year implies "
    "~188 crashes per 100M VMT. Stated as a fixed, cited public figure, not "
    "fetched live by this repository."
)


def rate_per_mile_to_per_100m_vmt(rate_per_mile: float) -> float:
    return rate_per_mile * 100_000_000.0


def miles_per_event(rate_per_mile: float) -> float:
    if rate_per_mile <= 0:
        raise ValueError("rate_per_mile must be positive to invert to miles-per-event")
    return 1.0 / rate_per_mile


def benchmark_ratio(simulated_rate_per_mile: float) -> float:
    """Simulated per-100M-VMT rate divided by the published human rate.
    Greater than 1 means the simulated fleet's collision rate is higher
    than the human benchmark at that exposure mix; less than 1 means lower.
    """
    simulated_per_100m = rate_per_mile_to_per_100m_vmt(simulated_rate_per_mile)
    return simulated_per_100m / HUMAN_CRASHES_PER_100M_VMT
