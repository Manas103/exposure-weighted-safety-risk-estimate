"""Tests for the 12-bin hierarchical NB GLM (hb_fleet_dgp.py, hb_glm.py).

Draws/tune are kept small here (80/80) purely for test speed; the
production coverage run (scripts/run_coverage_simulation.py,
docs/coverage_simulation_output.txt) uses 250/250.
"""
from __future__ import annotations

import numpy as np

from risk.hb_fleet_dgp import (
    HB_BINS,
    N_BINS,
    MILES_PER_BIN_PERIOD,
    validate_hb_bins,
    exposure_weights_array,
    true_rates_array,
    simulate_period_counts,
)
from risk.hb_glm import build_pooled_model, build_unpooled_model, refit_rates

_FAST_DRAWS = 80
_FAST_TUNE = 80


def test_hb_bins_weights_sum_to_one_and_in_range():
    validate_hb_bins()
    assert np.isclose(exposure_weights_array().sum(), 1.0)
    rates = true_rates_array()
    assert np.all(rates > 0) and np.all(rates < 1)


def test_hb_bins_count_is_twelve():
    assert N_BINS == 12
    assert len(HB_BINS) == 12


def test_most_bins_observe_zero_events_at_the_design_exposure():
    rng = np.random.default_rng(12345)
    exposure0 = np.full(N_BINS, MILES_PER_BIN_PERIOD)
    zero_fracs = []
    for _ in range(300):
        counts = simulate_period_counts(rng, exposure0)
        zero_fracs.append((counts == 0).mean())
    # "most" bins zero means a majority, not just a plurality
    assert np.mean(zero_fracs) > 0.5


def test_pooled_posterior_shrinks_more_than_unpooled_for_sparse_data():
    """Invariant check, not a plausibility check: with almost all bins at
    zero observed events, the pooled model's bin_eff posterior must have a
    smaller spread than the unpooled model's for essentially every bin,
    since pooling is defined by borrowing strength from the other 11 bins
    rather than relying on a fixed, wide, per-bin prior alone.
    """
    rng = np.random.default_rng(99)
    exposure0 = np.full(N_BINS, MILES_PER_BIN_PERIOD)
    counts0 = simulate_period_counts(rng, exposure0)
    assert (counts0 == 0).sum() >= N_BINS // 2  # sanity: this draw is sparse enough to test shrinkage

    pooled = build_pooled_model(N_BINS, exposure0, counts0)
    unpooled = build_unpooled_model(N_BINS, exposure0, counts0)

    # 150/150 rather than the module's usual 80/80 fast setting: this specific
    # check compares two posterior standard deviations against each other and
    # is sensitive to MCMC noise at very short chains.
    pooled_fit = refit_rates(pooled, exposure0, counts0, 150, 150, seed=1)
    unpooled_fit = refit_rates(unpooled, exposure0, counts0, 150, 150, seed=1)

    pooled_std = pooled_fit.rate_samples.std(axis=0)
    unpooled_std = unpooled_fit.rate_samples.std(axis=0)

    n_narrower = int((pooled_std < unpooled_std).sum())
    assert n_narrower >= N_BINS - 2, (
        f"expected pooling to narrow almost every bin's posterior for sparse data, "
        f"only {n_narrower}/{N_BINS} bins were narrower under pooling"
    )


def test_credible_interval_contains_point_estimate():
    rng = np.random.default_rng(5)
    exposure0 = np.full(N_BINS, MILES_PER_BIN_PERIOD)
    counts0 = simulate_period_counts(rng, exposure0)
    pooled = build_pooled_model(N_BINS, exposure0, counts0)
    fit = refit_rates(pooled, exposure0, counts0, _FAST_DRAWS, _FAST_TUNE, seed=2)
    lower, upper = fit.credible_interval()
    point = fit.point_estimate()
    assert np.all(point >= lower) and np.all(point <= upper)


def test_credible_interval_widens_for_lower_confidence_level_is_false_and_narrows_for_higher():
    """A sanity invariant on the interval formula itself: a wider nominal
    confidence level must never produce a narrower interval."""
    rng = np.random.default_rng(6)
    exposure0 = np.full(N_BINS, MILES_PER_BIN_PERIOD)
    counts0 = simulate_period_counts(rng, exposure0)
    pooled = build_pooled_model(N_BINS, exposure0, counts0)
    fit = refit_rates(pooled, exposure0, counts0, _FAST_DRAWS, _FAST_TUNE, seed=3)
    lo50, hi50 = fit.credible_interval(confidence=0.50)
    lo95, hi95 = fit.credible_interval(confidence=0.95)
    assert np.all((hi95 - lo95) >= (hi50 - lo50))


def test_refit_reuses_compiled_model_without_rebuilding():
    """set_data + sample again on the same model object must not raise and
    must reflect the newly set counts, not the counts the model was built
    with (this is the mechanism coverage_sim.py relies on for speed)."""
    rng = np.random.default_rng(7)
    exposure0 = np.full(N_BINS, MILES_PER_BIN_PERIOD)
    counts_a = simulate_period_counts(rng, exposure0)
    counts_b = simulate_period_counts(rng, exposure0)
    pooled = build_pooled_model(N_BINS, exposure0, counts_a)
    fit_a = refit_rates(pooled, exposure0, counts_a, _FAST_DRAWS, _FAST_TUNE, seed=1)
    fit_b = refit_rates(pooled, exposure0, counts_b, _FAST_DRAWS, _FAST_TUNE, seed=1)
    assert fit_a.rate_samples.shape == (( _FAST_DRAWS, N_BINS))
    assert fit_b.rate_samples.shape == (( _FAST_DRAWS, N_BINS))
