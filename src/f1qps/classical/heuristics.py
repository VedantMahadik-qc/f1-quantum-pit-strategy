"""Classical heuristic solvers used as baselines against QAOA/RQAOA."""
from __future__ import annotations

import numpy as np

from f1qps.qubo.builder import QUBOInstance


def solve_greedy(qubo: QUBOInstance) -> tuple[np.ndarray, float]:
    """Start from x=0 (nobody pits) and greedily flip whichever single bit
    most improves (decreases) the objective, until no flip helps.
    """
    n = qubo.n
    x = np.zeros(n)
    best_val = qubo.objective(x)
    improved = True
    while improved:
        improved = False
        for i in range(n):
            x[i] = 1 - x[i]
            val = qubo.objective(x)
            if val < best_val:
                best_val = val
                improved = True
            else:
                x[i] = 1 - x[i]  # revert
    return x, best_val
