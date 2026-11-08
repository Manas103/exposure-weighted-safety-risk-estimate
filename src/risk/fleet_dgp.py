"""The simulated driving fleet: road-speed bins, a true per-mile collision
probability for each bin, and the fleet's own measured mileage exposure mix
across those bins.

Nothing here is real driving data. The bin structure (five road-speed bands)
and the shape of the exposure mix (more low-speed urban miles than highway
miles) are modeled on how a real AV safety case splits exposure, but every
number is a design choice made for this repository, not a measurement of
any real fleet.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RoadSpeedBin:
    name: str
    speed_mph_low: float
    speed_mph_high: float
    true_collision_prob_per_mile: float  # the ground truth this project never gets to see directly
    fleet_exposure_weight: float  # fraction of total fleet miles driven in this bin


# Five road-speed bins. True collision probability per mile is deliberately
# non-monotonic in speed (lowest at mid-speed arterials, higher in dense
# low-speed intersection-heavy driving and again at highway speed where
# severity and closing speed both rise), which is also why a flat,
# unweighted average across bins would be the wrong number to report.
FLEET_BINS: tuple[RoadSpeedBin, ...] = (
    RoadSpeedBin("urban_0_25mph", 0, 25, 3.00e-6, 0.38),
    RoadSpeedBin("arterial_25_35mph", 25, 35, 1.30e-6, 0.27),
    RoadSpeedBin("arterial_35_45mph", 35, 45, 1.50e-6, 0.16),
    RoadSpeedBin("suburban_45_55mph", 45, 55, 2.00e-6, 0.11),
    RoadSpeedBin("highway_55plus_mph", 55, 999, 2.60e-6, 0.08),
)


def validate_fleet_bins(bins: tuple[RoadSpeedBin, ...] = FLEET_BINS) -> None:
    total_weight = sum(b.fleet_exposure_weight for b in bins)
    if not np.isclose(total_weight, 1.0, atol=1e-9):
        raise ValueError(f"fleet exposure weights must sum to 1.0, got {total_weight}")
    for b in bins:
        if not (0.0 < b.true_collision_prob_per_mile < 1.0):
            raise ValueError(f"{b.name}: collision probability per mile out of (0, 1)")
        if b.fleet_exposure_weight <= 0.0:
            raise ValueError(f"{b.name}: exposure weight must be positive")


def exposure_weighted_true_rate(bins: tuple[RoadSpeedBin, ...] = FLEET_BINS) -> float:
    """The ground-truth exposure-weighted collision probability per mile,
    used only to check this project's estimators against a known answer;
    the estimators themselves never see this value.
    """
    return sum(b.fleet_exposure_weight * b.true_collision_prob_per_mile for b in bins)
