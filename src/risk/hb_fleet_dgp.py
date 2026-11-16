"""A more granular, 12-bin version of the fleet road-speed structure used
elsewhere in this repository (`fleet_dgp.py`'s 5 bins), designed so that a
feasible simulated exposure per bin lands most bins at zero observed events
in a single run. This is the data-generating process behind the
hierarchical Bayesian partial-pooling model in `hb_glm.py`; it deliberately
disagrees with the 5-bin fleet used by the naive/importance-sampling
estimators, since the point here is to stress-test pooling across many
sparse bins, not to describe a better fleet.

Nothing here is real driving data; the bin split, the true per-mile rates
and the exposure mix are design choices for this extension, same as the
original `fleet_dgp.py`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 12 road-speed-by-context bins: the original 5 road-speed bands from
# fleet_dgp.py, each further split by a dense/light-traffic context, plus
# one highway-construction and one rural-light bin. Raw weights are
# normalized to sum to exactly 1.0 below rather than hand-rounded, so the
# invariant check is exact rather than approximate.
_NAMES = (
    "urban_0_25_dense",
    "urban_0_25_light",
    "arterial_25_35_dense",
    "arterial_25_35_light",
    "arterial_35_45_dense",
    "arterial_35_45_light",
    "suburban_45_55_dense",
    "suburban_45_55_light",
    "highway_55plus_cruise",
    "highway_55plus_merge",
    "highway_55plus_construction",
    "rural_55plus_light",
)
_TRUE_RATE_PER_MILE = (
    3.40e-6, 2.60e-6, 1.50e-6, 1.10e-6, 1.70e-6, 1.30e-6,
    2.20e-6, 1.80e-6, 2.40e-6, 3.00e-6, 3.80e-6, 1.00e-6,
)
_RAW_EXPOSURE_WEIGHTS = (12, 10, 9, 8, 7, 6, 6, 5, 9, 5, 3, 3)


@dataclass(frozen=True)
class HBBin:
    name: str
    true_rate_per_mile: float
    fleet_exposure_weight: float


def _build_bins() -> tuple[HBBin, ...]:
    raw = np.array(_RAW_EXPOSURE_WEIGHTS, dtype=float)
    weights = raw / raw.sum()
    return tuple(
        HBBin(name=n, true_rate_per_mile=r, fleet_exposure_weight=float(w))
        for n, r, w in zip(_NAMES, _TRUE_RATE_PER_MILE, weights)
    )


HB_BINS: tuple[HBBin, ...] = _build_bins()
N_BINS = len(HB_BINS)

# Chosen so the expected count per bin (true_rate * MILES_PER_BIN_PERIOD)
# ranges from about 0.06 to 0.23: low enough that most of the 12 bins land
# at exactly zero observed events in a typical simulated period, which is
# the property this extension needs to stress-test partial pooling.
MILES_PER_BIN_PERIOD = 60_000

# The held-out period a coverage check is scored against is simulated at
# 50x the training exposure (same idea as the extreme-value project's 40x
# longer held-out run): long enough that its own observed rate is a much
# less noisy proxy for the true rate than the sparse, often-zero training
# period, without reaching into the model and reading off the true rate
# the model was never shown.
HOLDOUT_EXPOSURE_MULTIPLIER = 50


def validate_hb_bins(bins: tuple[HBBin, ...] = HB_BINS) -> None:
    total_weight = sum(b.fleet_exposure_weight for b in bins)
    if not np.isclose(total_weight, 1.0, atol=1e-9):
        raise ValueError(f"fleet exposure weights must sum to 1.0, got {total_weight}")
    for b in bins:
        if not (0.0 < b.true_rate_per_mile < 1.0):
            raise ValueError(f"{b.name}: true rate per mile out of (0, 1)")
        if b.fleet_exposure_weight <= 0.0:
            raise ValueError(f"{b.name}: exposure weight must be positive")


def true_rates_array(bins: tuple[HBBin, ...] = HB_BINS) -> np.ndarray:
    return np.array([b.true_rate_per_mile for b in bins])


def exposure_weights_array(bins: tuple[HBBin, ...] = HB_BINS) -> np.ndarray:
    return np.array([b.fleet_exposure_weight for b in bins])


def simulate_period_counts(
    rng: np.random.Generator,
    exposure_miles: np.ndarray,
    bins: tuple[HBBin, ...] = HB_BINS,
) -> np.ndarray:
    """Poisson draws of observed event counts for one period, one draw per
    bin, given each bin's true per-mile rate and that period's own simulated
    exposure in miles (exposure_miles may differ by bin and by period; it is
    not required to equal MILES_PER_BIN_PERIOD).
    """
    rates = true_rates_array(bins)
    mu = rates * exposure_miles
    return rng.poisson(mu)
