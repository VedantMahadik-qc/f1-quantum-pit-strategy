"""Synthetic QUBO instance generators for testing/benchmarking, independent of real F1 data."""
from __future__ import annotations

import numpy as np

from f1qps.qubo.builder import QUBOInstance


def random_sparse_instance(
    n: int, edge_prob: float = 0.3, seed: int | None = None
) -> QUBOInstance:
    """Random instance with h_i ~ U(0,1) and J_ij ~ U(0,1) on each edge present
    independently with probability `edge_prob`. A dense random QUBO (the kind
    most QAOA benchmark papers use) corresponds to edge_prob close to 1.
    """
    rng = np.random.default_rng(seed)
    h = rng.uniform(0, 1, size=n)
    J = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < edge_prob:
                val = rng.uniform(0, 1)
                J[i, j] = val
                J[j, i] = val
    return QUBOInstance(h=h, J=J)


def path_like_instance(
    n: int, h_scale: float = 1.0, j_scale: float = 1.0, seed: int | None = None
) -> QUBOInstance:
    """Instance whose conflict graph is a path x_0 - x_1 - ... - x_{n-1}: each
    car only conflicts with its immediate track-position neighbor, modeling a
    pit-lane queue where only adjacent cars realistically compete for entry.
    """
    rng = np.random.default_rng(seed)
    h = rng.uniform(0, h_scale, size=n)
    J = np.zeros((n, n))
    for i in range(n - 1):
        val = rng.uniform(0, j_scale)
        J[i, i + 1] = val
        J[i + 1, i] = val
    return QUBOInstance(h=h, J=J)
