import pytest

from risk.fleet_dgp import FLEET_BINS, exposure_weighted_true_rate, validate_fleet_bins


def test_weights_sum_to_one():
    validate_fleet_bins()  # must not raise


def test_probabilities_in_range():
    for b in FLEET_BINS:
        assert 0.0 < b.true_collision_prob_per_mile < 1.0


def test_exposure_weighted_true_rate_between_bin_extremes():
    rate = exposure_weighted_true_rate()
    lo = min(b.true_collision_prob_per_mile for b in FLEET_BINS)
    hi = max(b.true_collision_prob_per_mile for b in FLEET_BINS)
    assert lo < rate < hi


def test_validate_rejects_bad_weights():
    from risk.fleet_dgp import RoadSpeedBin

    bad = (RoadSpeedBin("a", 0, 10, 1e-5, 0.5), RoadSpeedBin("b", 10, 20, 1e-5, 0.4))
    with pytest.raises(ValueError):
        validate_fleet_bins(bad)
