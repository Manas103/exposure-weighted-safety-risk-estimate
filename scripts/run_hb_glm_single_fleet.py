"""One simulated period of the 12-bin fleet: most bins land at zero
observed events, the pooled hierarchical NB GLM is fit to that single
period, and the resulting exposure-weighted fleet rate (DuckDB SQL, reusing
risk.mart, cross-checked against the independent NumPy reference the same
way scripts/run_estimate.py does) is benchmarked against the published
human crash rate (risk.benchmark). This script produces the single-fleet
numbers in the README; scripts/run_coverage_simulation.py produces the
pooled-vs-unpooled coverage numbers from many repeated fleets.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import duckdb
import numpy as np

from risk.benchmark import (
    HUMAN_BENCHMARK_SOURCE,
    HUMAN_CRASHES_PER_100M_VMT,
    benchmark_ratio,
    miles_per_event,
    rate_per_mile_to_per_100m_vmt,
)
from risk.hb_fleet_dgp import HB_BINS, N_BINS, MILES_PER_BIN_PERIOD, simulate_period_counts, validate_hb_bins
from risk.hb_glm import build_pooled_model, refit_rates
from risk.mart import BinTrialRecord, build_trials_table, exposure_weighted_rate_sql, total_events_and_miles_sql


def main() -> None:
    validate_hb_bins()
    rng = np.random.default_rng(20261101)
    exposure0 = np.full(N_BINS, MILES_PER_BIN_PERIOD)
    counts0 = simulate_period_counts(rng, exposure0)

    zero_bins = [b.name for b, c in zip(HB_BINS, counts0) if c == 0]
    print(f"Fleet bins: {N_BINS}, miles simulated per bin: {MILES_PER_BIN_PERIOD}")
    print(f"Zero-event bins this run: {len(zero_bins)} of {N_BINS} ({len(zero_bins) / N_BINS:.1%})")
    for b, c in zip(HB_BINS, counts0):
        print(f"  bin={b.name:30s} true_rate={b.true_rate_per_mile:.3e} n_events={c}")

    model = build_pooled_model(N_BINS, exposure0, counts0)
    fit = refit_rates(model, exposure0, counts0, draws=1000, tune=1000, seed=1)
    lower, upper = fit.credible_interval()
    point = fit.point_estimate()

    print("\nPooled (partial-pooling) posterior per-bin rate, 95% credible interval:")
    for b, p, lo, hi in zip(HB_BINS, point, lower, upper):
        flag = "  <- zero-event bin" if next(c for bb, c in zip(HB_BINS, counts0) if bb.name == b.name) == 0 else ""
        print(f"  {b.name:30s} point={p:.3e}  95% CI=[{lo:.3e}, {hi:.3e}]{flag}")

    records = [
        BinTrialRecord(
            bin_name=b.name,
            fleet_exposure_weight=b.fleet_exposure_weight,
            n_miles=MILES_PER_BIN_PERIOD,
            n_events=int(c),
            rate_hat=float(p),
        )
        for b, c, p in zip(HB_BINS, counts0, point)
    ]
    con = duckdb.connect(":memory:")
    build_trials_table(con, records)
    fleet_rate = exposure_weighted_rate_sql(con)
    total_events, total_miles = total_events_and_miles_sql(con)

    numpy_ref = float(
        np.sum(np.array([r.rate_hat * r.fleet_exposure_weight for r in records]))
        / np.sum(np.array([r.fleet_exposure_weight for r in records]))
    )
    print(f"\nDuckDB SQL fleet rate: {fleet_rate:.6e}/mile; independent NumPy reference: {numpy_ref:.6e}/mile (match: {np.isclose(fleet_rate, numpy_ref)})")
    print(f"Total simulated events: {total_events} over {total_miles} total miles across all bins")
    print(f"Exposure-weighted fleet collision rate (posterior-point-estimate-based): {fleet_rate:.6e} per mile")
    print(f"  -> {miles_per_event(fleet_rate):,.0f} miles per event")
    print(f"  -> {rate_per_mile_to_per_100m_vmt(fleet_rate):.2f} events per 100M vehicle miles")
    print(f"\nHuman benchmark: {HUMAN_CRASHES_PER_100M_VMT:.2f} crashes per 100M VMT")
    print(f"  Source: {HUMAN_BENCHMARK_SOURCE}")
    ratio = benchmark_ratio(fleet_rate)
    print(f"  Simulated-fleet-rate / human-rate ratio: {ratio:.3f}x")


if __name__ == "__main__":
    main()
