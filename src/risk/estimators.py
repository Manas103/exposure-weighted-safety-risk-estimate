"""Per-bin rare-event rate estimators: naive (uniform) Monte Carlo and
importance sampling, both over simulated Bernoulli-per-mile collision
trials.

A collision is rare enough per mile (on the order of 1e-5) that a naive
estimator needs an enormous number of simulated miles to pin down a
relative confidence interval, because its variance is p(1-p)/N, almost all
of which is wasted simulating miles that can never produce an event. The
importance sampling estimator instead simulates miles under a proposal
distribution with an inflated event probability, then reweights every
outcome by the likelihood ratio between the true and proposal
distributions so the estimator stays unbiased while spending its samples
where they are informative.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EstimateResult:
    rate_hat: float
    variance_hat: float  # variance of the per-mile rate estimator itself
    n_miles: int


def naive_monte_carlo(
    true_prob_per_mile: float, n_miles: int, rng: np.random.Generator
) -> EstimateResult:
    """Direct Bernoulli(true_prob_per_mile) draws, no reweighting. This is
    the baseline every real analyst would reach for first.
    """
    draws = rng.random(n_miles) < true_prob_per_mile
    phat = draws.mean()
    # unbiased sample variance of a Bernoulli mean
    var_phat = phat * (1.0 - phat) / n_miles
    return EstimateResult(rate_hat=float(phat), variance_hat=float(var_phat), n_miles=n_miles)


def importance_sampling(
    true_prob_per_mile: float,
    proposal_prob_per_mile: float,
    n_miles: int,
    rng: np.random.Generator,
) -> EstimateResult:
    """Draw each simulated mile's collision indicator from
    Bernoulli(proposal_prob_per_mile) instead of the true probability, then
    reweight every outcome by the likelihood ratio w = p_true(x)/p_proposal(x)
    so that mean(w * indicator) is still an unbiased estimator of
    true_prob_per_mile. proposal_prob_per_mile must exceed
    true_prob_per_mile for this to concentrate simulated draws where
    collisions actually happen.
    """
    if proposal_prob_per_mile <= true_prob_per_mile:
        raise ValueError("proposal probability must exceed the true probability")
    draws = rng.random(n_miles) < proposal_prob_per_mile
    w_event = true_prob_per_mile / proposal_prob_per_mile
    w_non_event = (1.0 - true_prob_per_mile) / (1.0 - proposal_prob_per_mile)
    weights = np.where(draws, w_event, w_non_event)
    # the estimator is mean(weight * indicator); indicator is 1 only where
    # draws is True, so this collapses to (count of draws) * w_event / n
    phat = float(draws.sum()) * w_event / n_miles
    # per-sample contribution is w_event*1{draw} (0 when draw is False,
    # since weight*indicator is 0 there); variance of that per-sample value
    contributions = np.where(draws, w_event, 0.0)
    var_phat = float(contributions.var(ddof=1) / n_miles)
    return EstimateResult(rate_hat=phat, variance_hat=var_phat, n_miles=n_miles)


def required_samples_for_relative_halfwidth(
    variance_per_sample_times_n: float, rate_hat: float, target_relative_halfwidth: float, z: float = 1.96
) -> float:
    """Samples needed to drive a normal-approximation CI's relative
    half-width (z * sqrt(Var(rate_hat)) / rate_hat) down to
    target_relative_halfwidth, given Var(rate_hat) scales as
    variance_per_sample_times_n / N (variance_per_sample_times_n is
    Var(rate_hat) * N at whatever N it was measured at, i.e. the per-sample
    variance).
    """
    # halfwidth(N) = z * sqrt(variance_per_sample_times_n / N) / rate_hat = target
    # => N = (z * sqrt(variance_per_sample_times_n) / (target * rate_hat)) ** 2
    return (z * np.sqrt(variance_per_sample_times_n) / (target_relative_halfwidth * rate_hat)) ** 2
