"""Run the pooled-vs-unpooled posterior-coverage simulation
(risk.coverage_sim) and print the result. Takes the refit count, draws/tune
per fit and worker-process count as CLI args so the committed raw output
under docs/ can state exactly which run produced the headline numbers.

Usage:
    python scripts/run_coverage_simulation.py [n_refits] [draws] [tune] [n_workers]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from risk.coverage_sim import run_coverage_simulation


def main() -> None:
    n_refits = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    draws = int(sys.argv[2]) if len(sys.argv) > 2 else 250
    tune = int(sys.argv[3]) if len(sys.argv) > 3 else 250
    n_workers = int(sys.argv[4]) if len(sys.argv) > 4 else 4

    print(f"Running {n_refits} refits, draws={draws}, tune={tune}, n_workers={n_workers}")
    t0 = time.perf_counter()
    summary = run_coverage_simulation(n_refits, draws=draws, tune=tune, n_workers=n_workers)
    elapsed = time.perf_counter() - t0

    print(f"\nWall-clock: {elapsed:.1f}s ({elapsed / n_refits:.3f}s/refit average, {n_workers} workers)")
    print(f"Bins: {summary.n_bins}, average fraction of bins with zero observed events per period: {summary.zero_count_frac:.3f}")
    print(f"\nPooled (partial-pooling) 95% posterior-interval coverage of the held-out rate: {summary.pooled_coverage:.4f} over {summary.n_refits * summary.n_bins} (refit, bin) checks")
    print(f"Unpooled 95% posterior-interval coverage of the held-out rate: {summary.unpooled_coverage:.4f} over {summary.n_refits * summary.n_bins} (refit, bin) checks")
    print(f"\nTotal NUTS divergences: pooled={summary.pooled_total_divergences}, unpooled={summary.unpooled_total_divergences}")


if __name__ == "__main__":
    main()
