import duckdb
import pytest

from risk.mart import (
    BinTrialRecord,
    build_trials_table,
    exposure_weighted_rate_numpy_reference,
    exposure_weighted_rate_sql,
    total_events_and_miles_sql,
)

RECORDS = [
    BinTrialRecord("urban_0_25mph", 0.38, 100_000, 3, 3e-5),
    BinTrialRecord("arterial_25_35mph", 0.27, 100_000, 1, 1e-5),
    BinTrialRecord("arterial_35_45mph", 0.16, 100_000, 2, 2e-5),
    BinTrialRecord("suburban_45_55mph", 0.11, 100_000, 2, 2e-5),
    BinTrialRecord("highway_55plus_mph", 0.08, 100_000, 3, 3e-5),
]


@pytest.fixture
def con():
    c = duckdb.connect(":memory:")
    build_trials_table(c, RECORDS)
    yield c
    c.close()


def test_sql_matches_independent_numpy_reference(con):
    sql_rate = exposure_weighted_rate_sql(con)
    numpy_rate = exposure_weighted_rate_numpy_reference(RECORDS)
    assert sql_rate == pytest.approx(numpy_rate, rel=1e-12)


def test_totals_match_input_records(con):
    events, miles = total_events_and_miles_sql(con)
    assert events == sum(r.n_events for r in RECORDS)
    assert miles == sum(r.n_miles for r in RECORDS)


def test_rate_is_weighted_average_bounded_by_bin_extremes(con):
    sql_rate = exposure_weighted_rate_sql(con)
    lo = min(r.rate_hat for r in RECORDS)
    hi = max(r.rate_hat for r in RECORDS)
    assert lo <= sql_rate <= hi
