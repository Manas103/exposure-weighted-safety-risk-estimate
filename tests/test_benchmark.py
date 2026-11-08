import pytest

from risk.benchmark import (
    HUMAN_CRASHES_PER_100M_VMT,
    benchmark_ratio,
    miles_per_event,
    rate_per_mile_to_per_100m_vmt,
)


def test_rate_conversion_round_trips_with_miles_per_event():
    rate = 2.0e-5
    per_100m = rate_per_mile_to_per_100m_vmt(rate)
    assert per_100m == pytest.approx(2000.0)
    assert miles_per_event(rate) == pytest.approx(1.0 / rate)


def test_miles_per_event_rejects_nonpositive_rate():
    with pytest.raises(ValueError):
        miles_per_event(0.0)


def test_benchmark_ratio_is_one_when_simulated_equals_human_rate():
    simulated_rate = HUMAN_CRASHES_PER_100M_VMT / 100_000_000.0
    assert benchmark_ratio(simulated_rate) == pytest.approx(1.0)


def test_benchmark_ratio_scales_linearly():
    r1 = benchmark_ratio(1e-6)
    r2 = benchmark_ratio(2e-6)
    assert r2 == pytest.approx(2 * r1)
