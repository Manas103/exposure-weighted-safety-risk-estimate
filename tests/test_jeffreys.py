import numpy as np
import pytest

from risk.jeffreys import jeffreys_poisson_interval


def test_zero_events_gives_nonzero_upper_bound_not_a_zero_rate():
    interval = jeffreys_poisson_interval(0, 50_000.0)
    assert interval.point_estimate == 0.0
    assert interval.lower == 0.0
    assert interval.upper > 0.0


def test_interval_contains_point_estimate_for_positive_counts():
    interval = jeffreys_poisson_interval(12, 1_000_000.0)
    assert interval.lower <= interval.point_estimate <= interval.upper


def test_interval_width_shrinks_as_exposure_grows():
    small = jeffreys_poisson_interval(3, 10_000.0)
    big = jeffreys_poisson_interval(300, 1_000_000.0)  # same rate, 100x exposure
    assert (big.upper - big.lower) < (small.upper - small.lower)


def test_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        jeffreys_poisson_interval(-1, 100.0)
    with pytest.raises(ValueError):
        jeffreys_poisson_interval(1, 0.0)


def test_empirical_coverage_near_nominal_by_simulation():
    """Reference check: simulate many Poisson(true_rate * exposure) draws
    and measure how often the resulting Jeffreys interval actually covers
    the true rate. This is the non-obvious correctness check for an
    interval formula: it diffs the formula's behavior against repeated
    simulation, not against another copy of the same formula."""
    rng = np.random.default_rng(7)
    true_rate = 2.0e-5
    exposure = 30_000.0
    lam = true_rate * exposure
    n_trials = 4000
    draws = rng.poisson(lam, size=n_trials)
    covered = 0
    for x in draws:
        interval = jeffreys_poisson_interval(int(x), exposure, confidence=0.95)
        if interval.lower <= true_rate <= interval.upper:
            covered += 1
    coverage = covered / n_trials
    assert 0.90 <= coverage <= 0.99
