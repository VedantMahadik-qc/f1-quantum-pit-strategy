import numpy as np
import pytest

from f1qps.classical.exact import solve_exact
from f1qps.classical.heuristics import solve_greedy
from f1qps.qubo.synthetic import path_like_instance, random_sparse_instance


@pytest.mark.parametrize("seed", range(5))
def test_exact_matches_known_small_instance(seed):
    qubo = random_sparse_instance(n=3, edge_prob=1.0, seed=seed)
    x, val = solve_exact(qubo)
    assert val == pytest.approx(qubo.objective(x))
    # exact must be <= every other assignment's objective
    import itertools

    for bits in itertools.product((0, 1), repeat=3):
        assert val <= qubo.objective(np.array(bits, dtype=float)) + 1e-9


@pytest.mark.parametrize("seed", range(10))
def test_greedy_never_beats_exact(seed):
    qubo = path_like_instance(n=8, seed=seed)
    _, exact_val = solve_exact(qubo)
    _, greedy_val = solve_greedy(qubo)
    assert greedy_val >= exact_val - 1e-9
