import numpy as np
import pytest

from f1qps.classical.exact import solve_exact
from f1qps.classical.heuristics import solve_greedy
from f1qps.qubo.synthetic import path_like_instance, random_sparse_instance
from f1qps.quantum.qaoa import optimize_qaoa_angles, sample_qaoa_solution
from f1qps.quantum.rqaoa import solve_rqaoa


@pytest.mark.parametrize("seed", range(3))
def test_rqaoa_close_to_exact_on_path_like_instance(seed):
    qubo = path_like_instance(n=7, seed=seed)
    _, exact_val = solve_exact(qubo)

    _, rqaoa_val = solve_rqaoa(qubo, reps=2, min_size=3, maxiter=300, seed=seed)

    assert rqaoa_val <= exact_val + 0.5


@pytest.mark.parametrize("seed", range(3))
def test_rqaoa_close_to_exact_on_random_sparse_instance(seed):
    qubo = random_sparse_instance(n=7, edge_prob=0.4, seed=seed)
    _, exact_val = solve_exact(qubo)

    _, rqaoa_val = solve_rqaoa(qubo, reps=2, min_size=3, maxiter=300, seed=seed)

    assert rqaoa_val <= exact_val + 0.5


def test_rqaoa_with_shot_sampled_correlations_runs_and_is_reasonable():
    qubo = path_like_instance(n=7, seed=0)
    _, exact_val = solve_exact(qubo)

    _, rqaoa_val = solve_rqaoa(qubo, reps=2, min_size=3, maxiter=300, seed=0, shots=2000)

    assert rqaoa_val <= exact_val + 0.5


def test_rqaoa_ensemble_runs_and_is_reasonable():
    qubo = random_sparse_instance(n=7, edge_prob=0.4, seed=0)
    _, exact_val = solve_exact(qubo)

    _, rqaoa_val = solve_rqaoa(qubo, reps=2, min_size=3, maxiter=120, seed=0, ensemble_k=3)

    assert rqaoa_val <= exact_val + 0.5


def test_rqaoa_matches_exact_when_min_size_covers_whole_problem():
    # min_size >= n means RQAOA should skip straight to the exact base case.
    qubo = path_like_instance(n=5, seed=0)
    x, val = solve_rqaoa(qubo, min_size=5, seed=0)
    _, exact_val = solve_exact(qubo)
    assert val == pytest.approx(exact_val)


def test_standard_qaoa_sampling_returns_valid_bitstring_and_reasonable_objective():
    qubo = path_like_instance(n=5, seed=1)
    _, exact_val = solve_exact(qubo)
    _, greedy_val = solve_greedy(qubo)

    qaoa_result = optimize_qaoa_angles(qubo, reps=2, maxiter=300, seed=1)
    x, val = sample_qaoa_solution(qubo, qaoa_result, shots=4000, seed=1)

    assert x.shape == (5,)
    assert set(np.unique(x)).issubset({0.0, 1.0})
    # plain QAOA shouldn't be wildly worse than a simple greedy baseline here
    assert val <= max(exact_val, greedy_val) + 1.0
