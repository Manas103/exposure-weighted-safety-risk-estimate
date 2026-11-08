"""Jeffreys credible intervals for a Poisson rate, used instead of a point
estimate of zero whenever a bin's simulated exposure produced no observed
events.

A bin with 0 observed collisions over, say, 50,000 simulated miles does not
have a collision rate of 0 per mile; it has a rate that is consistent with
anything up to roughly 3/50,000 at 95% confidence, and reporting the point
estimate alone throws that uncertainty away. The Jeffreys interval for a
Poisson rate lambda given x events over exposure T uses the (improper)
Jeffreys prior lambda ~ Gamma(0.5, rate=0), giving a Gamma(x + 0.5, scale=1/T)
posterior; by convention the lower bound is reported as exactly 0 when x=0
(the textbook treatment, e.g. Fay 2010), since the technical shape=0.5
lower quantile is a real but not independently meaningful number for a
0-event bin.
"""
from __future__ import annotations

from dataclasses import dataclass

from scipy import stats


@dataclass
class JeffreysInterval:
    x_events: int
    exposure_miles: float
    point_estimate: float
    lower: float
    upper: float
    confidence: float


def jeffreys_poisson_interval(
    x_events: int, exposure_miles: float, confidence: float = 0.95
) -> JeffreysInterval:
    if exposure_miles <= 0:
        raise ValueError("exposure_miles must be positive")
    if x_events < 0:
        raise ValueError("x_events must be non-negative")
    alpha = 1.0 - confidence
    shape = x_events + 0.5
    lower = 0.0 if x_events == 0 else float(stats.gamma.ppf(alpha / 2.0, a=shape) / exposure_miles)
    upper = float(stats.gamma.ppf(1.0 - alpha / 2.0, a=shape) / exposure_miles)
    point = x_events / exposure_miles
    return JeffreysInterval(
        x_events=x_events,
        exposure_miles=exposure_miles,
        point_estimate=point,
        lower=lower,
        upper=upper,
        confidence=confidence,
    )
