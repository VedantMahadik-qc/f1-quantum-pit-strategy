"""Experiment harness: run all solvers on one instance, or sweep across many
synthetic instances to build a statistically meaningful comparison (rather
than reading anything into a single seed/instance -- see plans).
"""
from __future__ import annotations

import time

import pandas as pd

from f1qps.classical.exact import solve_exact
from f1qps.classical.heuristics import solve_greedy
from f1qps.experiments.metrics import approx_ratio
from f1qps.qubo.builder import QUBOInstance
from f1qps.qubo.synthetic import path_like_instance, random_sparse_instance
from f1qps.quantum.qaoa import optimize_qaoa_angles, sample_qaoa_solution
from f1qps.quantum.rqaoa import solve_rqaoa

INSTANCE_GENERATORS = {
    "path_like": path_like_instance,
    "random_sparse": lambda n, seed: random_sparse_instance(n, edge_prob=0.5, seed=seed),
}


def _timed(fn):
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


def run_instance(
    qubo: QUBOInstance,
    reps: int = 1,
    maxiter: int = 120,
    min_size: int = 3,
    shots: int = 2000,
    seed: int | None = None,
    methods: tuple[str, ...] = ("exact", "greedy", "qaoa", "rqaoa"),
) -> dict[str, dict]:
    """Run the requested methods on one QUBO instance.
    Returns {method: {"objective": float, "wall_clock": float}}.
    """
    results: dict[str, dict] = {}

    if "exact" in methods:
        (_, val), t = _timed(lambda: solve_exact(qubo))
        results["exact"] = {"objective": val, "wall_clock": t}

    if "greedy" in methods:
        (_, val), t = _timed(lambda: solve_greedy(qubo))
        results["greedy"] = {"objective": val, "wall_clock": t}

    if "qaoa" in methods:

        def _run_qaoa():
            qaoa_result = optimize_qaoa_angles(qubo, reps=reps, maxiter=maxiter, seed=seed)
            return sample_qaoa_solution(qubo, qaoa_result, shots=shots, seed=seed)

        (_, val), t = _timed(_run_qaoa)
        results["qaoa"] = {"objective": val, "wall_clock": t}

    if "rqaoa" in methods:
        (_, val), t = _timed(
            lambda: solve_rqaoa(qubo, reps=reps, min_size=min_size, maxiter=maxiter, seed=seed)
        )
        results["rqaoa"] = {"objective": val, "wall_clock": t}

    return results


def sweep_synthetic(
    n_values: list[int],
    seeds: list[int],
    instance_types: tuple[str, ...] = ("path_like", "random_sparse"),
    reps: int = 1,
    maxiter: int = 120,
    min_size: int = 3,
    shots: int = 2000,
    methods: tuple[str, ...] = ("exact", "greedy", "qaoa", "rqaoa"),
) -> pd.DataFrame:
    """Long-format DataFrame: instance_type, N, seed, method, objective,
    wall_clock, approx_ratio. "exact" is always run per instance (needed as
    the approx_ratio denominator) even if not in `methods`.
    """
    run_methods = tuple(set(methods) | {"exact"})
    rows = []

    for instance_type in instance_types:
        generator = INSTANCE_GENERATORS[instance_type]
        for n in n_values:
            for seed in seeds:
                qubo = generator(n, seed)
                results = run_instance(
                    qubo,
                    reps=reps,
                    maxiter=maxiter,
                    min_size=min_size,
                    shots=shots,
                    seed=seed,
                    methods=run_methods,
                )
                exact_val = results["exact"]["objective"]
                for method, data in results.items():
                    if method not in methods:
                        continue
                    rows.append(
                        {
                            "instance_type": instance_type,
                            "N": n,
                            "seed": seed,
                            "method": method,
                            "objective": data["objective"],
                            "wall_clock": data["wall_clock"],
                            "approx_ratio": approx_ratio(data["objective"], exact_val),
                        }
                    )

    return pd.DataFrame(rows)
