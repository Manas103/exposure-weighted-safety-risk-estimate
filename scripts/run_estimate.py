"""End-to-end run: simulate every fleet road-speed bin, reweight by
exposure, benchmark against the published human rate, and report Jeffreys
intervals for any bin with zero observed events. This is the script behind
every headline number in the README's Measured results table except the
importance-sampling sample-size reduction, which run_importance_sampling_benchmark.py
measures on its own.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import duckdb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from risk.benchmark import benchmark_ratio, miles_per_event, rate_per_mile_to_per_100m_vmt, HUMAN_CRASHES_PER_100M_VMT, HUMAN_BENCHMARK_SOURCE
from risk.estimators import naive_monte_carlo
from risk.fleet_dgp import FLEET_BINS, exposure_weighted_true_rate, validate_fleet_bins
from risk.jeffreys import jeffreys_poisson_interval
from risk.mart import BinTrialRecord, build_trials_table, exposure_weighted_rate_sql, total_events_and_miles_sql

MILES_PER_BIN = 500_000  # chosen so bins plausibly land at 0-3 events, exercising Jeffreys on the sparsest


def main() -> None:
    validate_fleet_bins()
    rng = np.random.default_rng(20261008)

    print(f"Fleet bins: {len(FLEET_BINS)}, miles simulated per bin: {MILES_PER_BIN}")
    print(f"Ground-truth exposure-weighted rate (never seen by the estimator): {exposure_weighted_true_rate():.6e} per mile\n")

    t0 = time.perf_counter()
    records = []
    zero_event_bins = []
    for b in FLEET_BINS:
        result = naive_monte_carlo(b.true_collision_prob_per_mile, MILES_PER_BIN, rng)
        n_events = int(round(result.rate_hat * MILES_PER_BIN))
        records.append(
            BinTrialRecord(
                bin_name=b.name,
                fleet_exposure_weight=b.fleet_exposure_weight,
                n_miles=MILES_PER_BIN,
                n_events=n_events,
                rate_hat=result.rate_hat,
            )
        )
        print(f"  bin={b.name:22s} true_p={b.true_collision_prob_per_mile:.3e} n_events={n_events:3d} rate_hat={result.rate_hat:.3e}")
        if n_events == 0:
            zero_event_bins.append(b.name)
    elapsed = time.perf_counter() - t0

    con = duckdb.connect(":memory:")
    build_trials_table(con, records)
    fleet_rate = exposure_weighted_rate_sql(con)
    total_events, total_miles = total_events_and_miles_sql(con)

    print(f"\nSimulation wall-clock: {elapsed:.3f}s")
    print(f"Total simulated events: {total_events} over {total_miles} total miles across all bins")
    print(f"Exposure-weighted fleet collision rate (DuckDB SQL aggregation): {fleet_rate:.6e} per mile")
    print(f"  -> {miles_per_event(fleet_rate):,.0f} miles per event")
    print(f"  -> {rate_per_mile_to_per_100m_vmt(fleet_rate):.2f} events per 100M vehicle miles")
    print(f"\nHuman benchmark: {HUMAN_CRASHES_PER_100M_VMT:.2f} crashes per 100M VMT")
    print(f"  Source: {HUMAN_BENCHMARK_SOURCE}")
    ratio = benchmark_ratio(fleet_rate)
    print(f"  Simulated-fleet-rate / human-rate ratio: {ratio:.3f}x")

    print(f"\nZero-event bins this run ({len(zero_event_bins)} of {len(FLEET_BINS)}): {zero_event_bins or 'none'}")
    for b in FLEET_BINS:
        rec = next(r for r in records if r.bin_name == b.name)
        interval = jeffreys_poisson_interval(rec.n_events, rec.n_miles)
        flag = "  <- zero-event bin, interval is the honest report, not a rate of zero" if rec.n_events == 0 else ""
        print(
            f"  {b.name:22s} x={rec.n_events:3d}  point={interval.point_estimate:.3e}  "
            f"95% CI=[{interval.lower:.3e}, {interval.upper:.3e}]{flag}"
        )


if __name__ == "__main__":
    main()
