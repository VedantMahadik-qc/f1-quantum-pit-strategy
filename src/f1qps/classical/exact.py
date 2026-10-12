"""Exact brute-force solver — ground truth for correctness checks on small N."""
from __future__ import annotations

import itertools

import numpy as np

from f1qps.qubo.builder import QUBOInstance


def solve_exact(qubo: QUBOInstance) -> tuple[np.ndarray, float]:
    """Brute-force over all 2^n assignments. Only tractable for small n (<=~20)."""
    n = qubo.n
    best_x = None
    best_val = np.inf
    for bits in itertools.product((0, 1), repeat=n):
        x = np.array(bits, dtype=float)
        val = qubo.objective(x)
        if val < best_val:
            best_val = val
            best_x = x
    return best_x, best_val
