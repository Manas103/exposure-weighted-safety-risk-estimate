"""Negative-binomial GLM with a log-exposure offset for the 12-bin fleet in
`hb_fleet_dgp.py`, fit by MCMC (PyMC, NUTS via the `nutpie` backend; see the
README's "What broke" section for why `nutpie` and not PyMC's default
Python-mode NUTS). Two variants of the same model:

- `build_pooled_model`: partial pooling across bins. Each bin's log-rate
  offset `bin_eff` is drawn from a shared `Normal(0, sigma)`, with `sigma`
  itself estimated from the data, so a sparse bin's posterior is pulled
  toward the population mean by however much the other 11 bins support
  that pooling.
- `build_unpooled_model`: the same likelihood and the same `mu0` intercept,
  but `bin_eff` is given a fixed, wide `Normal(0, 5)` prior per bin instead
  of a shared, estimated `sigma`. Nothing links one bin's posterior to
  another's; a zero-count bin's posterior is shaped almost entirely by that
  wide prior.

Both models share a `pm.Data` container for the observed counts and the
period's exposure so the same compiled model can be reused across many
simulated refits (`pm.set_data` + `pm.sample` again) without paying the
~10-15s one-time PyTensor/nutpie compile cost on every refit; see
`coverage_sim.py`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pymc as pm


@dataclass
class FittedBinRates:
    """Posterior samples of the per-mile rate for every bin, flattened
    across chains and draws, shape (n_samples, n_bins)."""
    rate_samples: np.ndarray
    n_divergences: int

    def credible_interval(self, confidence: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        alpha = 1.0 - confidence
        lower = np.percentile(self.rate_samples, 100 * alpha / 2.0, axis=0)
        upper = np.percentile(self.rate_samples, 100 * (1.0 - alpha / 2.0), axis=0)
        return lower, upper

    def point_estimate(self) -> np.ndarray:
        return np.median(self.rate_samples, axis=0)


def build_pooled_model(n_bins: int, exposure0: np.ndarray, counts0: np.ndarray) -> pm.Model:
    with pm.Model() as model:
        exposure = pm.Data("exposure", exposure0)
        y_obs = pm.Data("y_obs", counts0)
        sigma = pm.HalfNormal("sigma", 1.0)
        mu0 = pm.Normal("mu0", mu=np.log(2e-6), sigma=2.0)
        bin_eff = pm.Normal("bin_eff", 0.0, sigma, shape=n_bins)
        alpha = pm.HalfNormal("alpha", 2.0)
        log_rate = mu0 + bin_eff
        lam = pm.math.exp(log_rate) * exposure
        pm.NegativeBinomial("y", mu=lam, alpha=alpha, observed=y_obs, shape=n_bins)
    return model


def build_unpooled_model(n_bins: int, exposure0: np.ndarray, counts0: np.ndarray) -> pm.Model:
    with pm.Model() as model:
        exposure = pm.Data("exposure", exposure0)
        y_obs = pm.Data("y_obs", counts0)
        mu0 = pm.Normal("mu0", mu=np.log(2e-6), sigma=2.0)
        # Fixed, wide, not-estimated-from-data prior: no sigma hyperparameter,
        # so no bin's posterior can be pulled toward another bin's estimate.
        bin_eff = pm.Normal("bin_eff", 0.0, 5.0, shape=n_bins)
        alpha = pm.HalfNormal("alpha", 2.0)
        log_rate = mu0 + bin_eff
        lam = pm.math.exp(log_rate) * exposure
        pm.NegativeBinomial("y", mu=lam, alpha=alpha, observed=y_obs, shape=n_bins)
    return model


def refit_rates(
    model: pm.Model,
    exposure: np.ndarray,
    counts: np.ndarray,
    draws: int,
    tune: int,
    seed: int,
) -> FittedBinRates:
    """Update the model's observed data in place and re-sample. Returns the
    posterior per-mile rate samples (exp(mu0 + bin_eff)), flattened across
    the single chain's draws.
    """
    with model:
        pm.set_data({"exposure": exposure, "y_obs": counts})
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=1,
            cores=1,
            progressbar=False,
            random_seed=seed,
            nuts_sampler="nutpie",
            compute_convergence_checks=False,
        )
    mu0 = idata.posterior["mu0"].values[0]  # (draws,)
    bin_eff = idata.posterior["bin_eff"].values[0]  # (draws, n_bins)
    rate_samples = np.exp(mu0[:, None] + bin_eff)
    n_div = int(idata.sample_stats["diverging"].values.sum())
    return FittedBinRates(rate_samples=rate_samples, n_divergences=n_div)
