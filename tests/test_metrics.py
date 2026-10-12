import math

from f1qps.experiments.metrics import approx_ratio


def test_approx_ratio_at_optimum_is_one():
    assert approx_ratio(-3.5, -3.5) == 1.0


def test_approx_ratio_worse_than_optimum_is_between_zero_and_one():
    r = approx_ratio(-2.0, -4.0)
    assert 0.0 < r < 1.0


def test_approx_ratio_degenerate_zero_optimum_matches_zero():
    assert approx_ratio(0.0, 0.0) == 1.0


def test_approx_ratio_degenerate_zero_optimum_mismatch_is_nan():
    assert math.isnan(approx_ratio(-1.0, 0.0))
