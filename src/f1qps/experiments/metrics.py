"""Metrics for comparing solvers on the SC/VSC pit-decision QUBO.

The objective C(x) = -sum h_i x_i + sum J_ij x_i x_j is not a normalized
Max-Cut-style objective: x=0 ("nobody pits") always scores exactly 0, and
the true optimum is always <= 0. So approx_ratio = val/exact_val lands
naturally in [0, 1] (1.0 = optimal), as long as exact_val != 0.
"""
from __future__ import annotations

EXACT_VAL_ZERO_TOL = 1e-9


def approx_ratio(val: float, exact_val: float) -> float:
    """1.0 = matches the exact optimum. Degenerate case: if the exact optimum
    is ~0 (no benefit achievable at all), ratio is 1.0 if val is also ~0,
    else NaN (undefined -- the algorithm found an improvement where none
    should exist, which signals a bug rather than a quality difference).
    """
    if abs(exact_val) < EXACT_VAL_ZERO_TOL:
        return 1.0 if abs(val) < EXACT_VAL_ZERO_TOL else float("nan")
    return val / exact_val
