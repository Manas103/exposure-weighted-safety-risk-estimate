"""End-to-end smoke test for the multiprocess coverage simulation harness
itself (risk.coverage_sim). Run at a tiny refit count and tiny draws/tune
purely to check the plumbing (worker pool init, set_data reuse, result
aggregation) is correct; the headline coverage numbers come from the much
larger production run recorded in docs/coverage_simulation_output.txt.
"""
from __future__ import annotations

from risk.coverage_sim import run_coverage_simulation


def test_coverage_simulation_runs_and_returns_bounded_fractions():
    summary = run_coverage_simulation(n_refits=4, draws=60, tune=60, n_workers=2)
    assert summary.n_refits == 4
    assert summary.n_bins == 12
    assert 0.0 <= summary.pooled_coverage <= 1.0
    assert 0.0 <= summary.unpooled_coverage <= 1.0
    assert summary.pooled_total_divergences >= 0
    assert summary.unpooled_total_divergences >= 0
    assert 0.0 <= summary.zero_count_frac <= 1.0
