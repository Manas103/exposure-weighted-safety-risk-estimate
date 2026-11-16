"""Coverage simulation: repeatedly simulate a fresh train period and a fresh
held-out period from the same 12-bin fleet (`hb_fleet_dgp`), fit the pooled
and unpooled negative-binomial GLMs (`hb_glm`) to the train period only, and
check whether each model's 95% posterior interval for a bin's per-mile rate
contains that bin's own held-out observed rate
(`held-out counts / held-out exposure`), not the (unobservable, in a real
deployment) true rate. This is a harder, more realistic check than coverage
against the known ground truth: the held-out rate is itself a noisy
estimate, especially for the many bins that land at zero both periods.

Coverage is aggregated across every (refit, bin) pair, not just across
refits, since each refit already produces 12 independent bin-level checks.
"""
from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import get_context

import numpy as np

from risk.hb_fleet_dgp import (
    HB_BINS,
    N_BINS,
    MILES_PER_BIN_PERIOD,
    HOLDOUT_EXPOSURE_MULTIPLIER,
    simulate_period_counts,
)
from risk.hb_glm import build_pooled_model, build_unpooled_model, refit_rates

# Module-level, per-worker-process state: each worker builds (and compiles)
# its own pooled/unpooled models once via _init_worker, then reuses them for
# every refit it is assigned, avoiding PyTensor/nutpie's one-time compile
# cost on every one of thousands of refits.
_WORKER_POOLED = None
_WORKER_UNPOOLED = None


@dataclass
class RefitCoverageResult:
    seed: int
    pooled_covered: np.ndarray  # shape (N_BINS,), bool
    unpooled_covered: np.ndarray
    pooled_divergences: int
    unpooled_divergences: int


def _init_worker() -> None:
    global _WORKER_POOLED, _WORKER_UNPOOLED
    exposure0 = np.full(N_BINS, MILES_PER_BIN_PERIOD)
    placeholder_counts = np.zeros(N_BINS, dtype=int)
    _WORKER_POOLED = build_pooled_model(N_BINS, exposure0, placeholder_counts)
    _WORKER_UNPOOLED = build_unpooled_model(N_BINS, exposure0, placeholder_counts)


def _run_one_refit(args: tuple[int, int, int]) -> RefitCoverageResult:
    seed, draws, tune = args
    rng = np.random.default_rng(seed)
    exposure0 = np.full(N_BINS, MILES_PER_BIN_PERIOD)
    holdout_exposure = exposure0 * HOLDOUT_EXPOSURE_MULTIPLIER
    train_counts = simulate_period_counts(rng, exposure0)
    holdout_counts = simulate_period_counts(rng, holdout_exposure)
    holdout_rate = holdout_counts / holdout_exposure

    pooled_fit = refit_rates(_WORKER_POOLED, exposure0, train_counts, draws, tune, seed)
    unpooled_fit = refit_rates(_WORKER_UNPOOLED, exposure0, train_counts, draws, tune, seed)

    p_lo, p_hi = pooled_fit.credible_interval()
    u_lo, u_hi = unpooled_fit.credible_interval()

    pooled_covered = (holdout_rate >= p_lo) & (holdout_rate <= p_hi)
    unpooled_covered = (holdout_rate >= u_lo) & (holdout_rate <= u_hi)

    return RefitCoverageResult(
        seed=seed,
        pooled_covered=pooled_covered,
        unpooled_covered=unpooled_covered,
        pooled_divergences=pooled_fit.n_divergences,
        unpooled_divergences=unpooled_fit.n_divergences,
    )


@dataclass
class CoverageSummary:
    n_refits: int
    n_bins: int
    pooled_coverage: float
    unpooled_coverage: float
    pooled_total_divergences: int
    unpooled_total_divergences: int
    zero_count_frac: float


def run_coverage_simulation(
    n_refits: int,
    draws: int = 250,
    tune: int = 250,
    n_workers: int = 4,
    base_seed: int = 20261101,
) -> CoverageSummary:
    seeds = [base_seed + i for i in range(n_refits)]
    args = [(s, draws, tune) for s in seeds]

    ctx = get_context("spawn")
    with ctx.Pool(processes=n_workers, initializer=_init_worker) as pool:
        results: list[RefitCoverageResult] = pool.map(_run_one_refit, args)

    pooled_hits = np.concatenate([r.pooled_covered for r in results])
    unpooled_hits = np.concatenate([r.unpooled_covered for r in results])

    rng = np.random.default_rng(base_seed)
    exposure0 = np.full(N_BINS, MILES_PER_BIN_PERIOD)
    zero_fracs = []
    for _ in range(200):
        c = simulate_period_counts(rng, exposure0)
        zero_fracs.append((c == 0).mean())

    return CoverageSummary(
        n_refits=n_refits,
        n_bins=N_BINS,
        pooled_coverage=float(pooled_hits.mean()),
        unpooled_coverage=float(unpooled_hits.mean()),
        pooled_total_divergences=sum(r.pooled_divergences for r in results),
        unpooled_total_divergences=sum(r.unpooled_divergences for r in results),
        zero_count_frac=float(np.mean(zero_fracs)),
    )
