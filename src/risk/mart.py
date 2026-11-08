"""DuckDB mart: every simulated per-mile trial, for every road-speed bin,
is loaded into one DuckDB table and the exposure-weighted fleet rate is
computed with a SQL aggregation query, not a pandas groupby. pandas is
still used elsewhere in this project (to assemble the per-bin trial arrays
before they are loaded), but the one number this repository calls its
output, the exposure-weighted miles-per-event rate, is produced by SQL
running inside DuckDB and cross-checked against an independent NumPy
computation in tests/test_mart.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd


@dataclass
class BinTrialRecord:
    bin_name: str
    fleet_exposure_weight: float
    n_miles: int
    n_events: int
    rate_hat: float


def build_trials_table(con: duckdb.DuckDBPyConnection, records: list[BinTrialRecord]) -> None:
    df = pd.DataFrame(
        {
            "bin_name": [r.bin_name for r in records],
            "fleet_exposure_weight": [r.fleet_exposure_weight for r in records],
            "n_miles": [r.n_miles for r in records],
            "n_events": [r.n_events for r in records],
            "rate_hat": [r.rate_hat for r in records],
        }
    )
    con.register("bin_trials_df", df)
    con.execute("CREATE OR REPLACE TABLE bin_trials AS SELECT * FROM bin_trials_df")


def exposure_weighted_rate_sql(con: duckdb.DuckDBPyConnection) -> float:
    """The fleet-level exposure-weighted collision rate per mile: each bin's
    rate_hat weighted by its fleet_exposure_weight, normalized by the sum of
    weights (defensive against weights that do not sum to exactly 1.0 due
    to floating point).
    """
    row = con.execute(
        """
        SELECT SUM(rate_hat * fleet_exposure_weight) / SUM(fleet_exposure_weight)
        FROM bin_trials
        """
    ).fetchone()
    return float(row[0])


def total_events_and_miles_sql(con: duckdb.DuckDBPyConnection) -> tuple[int, int]:
    row = con.execute("SELECT SUM(n_events), SUM(n_miles) FROM bin_trials").fetchone()
    return int(row[0]), int(row[1])


def exposure_weighted_rate_numpy_reference(records: list[BinTrialRecord]) -> float:
    """Independent NumPy computation of the same quantity, used as a
    reference oracle for exposure_weighted_rate_sql in tests.
    """
    weights = np.array([r.fleet_exposure_weight for r in records])
    rates = np.array([r.rate_hat for r in records])
    return float(np.sum(rates * weights) / np.sum(weights))
