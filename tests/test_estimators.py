import numpy as np
import pytest

from risk.estimators import (
    importance_sampling,
    naive_monte_carlo,
    required_samples_for_relative_halfwidth,
)


def test_naive_monte_carlo_unbiased_on_average():
    """Average the naive estimator over many independent repeats; the mean
    of repeat estimates should land close to the true probability (a
    statistical, not exact, check, since each repeat is itself noisy)."""
    rng = np.random.default_rng(1)
    true_p = 0.02  # not microscopically rare, so n_miles stays small in a test
    estimates = [naive_monte_carlo(true_p, 20_000, rng).rate_hat for _ in range(200)]
    assert abs(np.mean(estimates) - true_p) < 0.001


def test_importance_sampling_unbiased_on_average():
    rng = np.random.default_rng(2)
    true_p = 2e-4
    proposal_p = 2e-2
    estimates = [
        importance_sampling(true_p, proposal_p, 5_000, rng).rate_hat for _ in range(500)
    ]
    # relative tolerance since true_p is small
    assert abs(np.mean(estimates) - true_p) / true_p < 0.15


def test_importance_sampling_rejects_proposal_not_greater_than_true():
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError):
        importance_sampling(0.01, 0.01, 100, rng)
    with pytest.raises(ValueError):
        importance_sampling(0.01, 0.005, 100, rng)


def test_importance_sampling_has_lower_variance_than_naive_for_rare_event():
    """The core claim this project measures: for a sufficiently rare event,
    importance sampling's per-sample variance is lower than naive Monte
    Carlo's, at matched sample size. This is the mechanism behind the
    required-sample-reduction measurement in scripts/run_estimate.py."""
    rng = np.random.default_rng(4)
    true_p = 1e-5
    n = 200_000
    naive = naive_monte_carlo(true_p, n, rng)
    importance = importance_sampling(true_p, 50 * true_p, n, rng)
    naive_var_per_sample = naive.variance_hat * naive.n_miles
    is_var_per_sample = importance.variance_hat * importance.n_miles
    assert is_var_per_sample < naive_var_per_sample


def test_required_samples_formula_scales_inversely_with_target_squared():
    base = required_samples_for_relative_halfwidth(1.0, 0.01, 0.20)
    tighter = required_samples_for_relative_halfwidth(1.0, 0.01, 0.10)
    # halving the target half-width should roughly quadruple required N
    assert tighter / base == pytest.approx(4.0, rel=1e-9)
