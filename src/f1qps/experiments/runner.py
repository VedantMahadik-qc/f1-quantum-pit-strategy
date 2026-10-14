"""Experiment harness: run all solvers on one instance, or sweep across many
synthetic instances to build a statistically meaningful comparison (rather
than reading anything into a single seed/instance -- see plans).
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from f1qps.classical.exact import solve_exact
from f1qps.classical.heuristics import solve_greedy
from f1qps.experiments.metrics import approx_ratio
from f1qps.qubo.builder import QUBOInstance
from f1qps.qubo.synthetic import path_like_instance, random_sparse_instance
from f1qps.quantum.qaoa import optimize_qaoa_angles, sample_qaoa_solution
from f1qps.quantum.rqaoa import solve_rqaoa

INSTANCE_GENERATORS = {
    # path_like_instance's signature is (n, h_scale, j_scale, seed=None) -- wrapped here (unlike
    # random_sparse below, whose seed is already its sole keyword-after-n) so positional
    # generator(n, seed) calls below bind `seed` to the actual seed param, not h_scale.
    "path_like": lambda n, seed: path_like_instance(n, seed=seed),
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
    rqaoa_shots: int | None = None,
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
            lambda: solve_rqaoa(
                qubo, reps=reps, min_size=min_size, maxiter=maxiter, seed=seed, shots=rqaoa_shots
            )
        )
        results["rqaoa"] = {"objective": val, "wall_clock": t}

    return results


def _sweep_one(
    instance_type: str,
    n: int,
    seed: int,
    reps: int,
    maxiter: int,
    min_size: int,
    shots: int,
    run_methods: tuple[str, ...],
    methods: tuple[str, ...],
    rqaoa_shots: int | None,
) -> list[dict]:
    """One (instance_type, N, seed) sweep cell. Module-level (rather than a
    closure inside sweep_synthetic) so it can be pickled and run in a worker
    process when sweep_synthetic's `max_workers` > 1.
    """
    generator = INSTANCE_GENERATORS[instance_type]
    qubo = generator(n, seed)
    results = run_instance(
        qubo,
        reps=reps,
        maxiter=maxiter,
        min_size=min_size,
        shots=shots,
        seed=seed,
        methods=run_methods,
        rqaoa_shots=rqaoa_shots,
    )
    exact_val = results["exact"]["objective"]
    rows = []
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
    return rows


def sweep_synthetic(
    n_values: list[int],
    seeds: list[int],
    instance_types: tuple[str, ...] = ("path_like", "random_sparse"),
    reps: int = 1,
    maxiter: int = 120,
    min_size: int = 3,
    shots: int = 2000,
    methods: tuple[str, ...] = ("exact", "greedy", "qaoa", "rqaoa"),
    rqaoa_shots: int | None = None,
    max_workers: int | None = None,
) -> pd.DataFrame:
    """Long-format DataFrame: instance_type, N, seed, method, objective,
    wall_clock, approx_ratio. "exact" is always run per instance (needed as
    the approx_ratio denominator) even if not in `methods`.

    `max_workers`: None (default) runs every (instance_type, N, seed) cell
    sequentially in-process, matching every prior result in this project. An
    int > 1 instead distributes the cells -- each one an independent,
    CPU-bound QAOA/RQAOA circuit simulation with no shared state -- across
    that many worker processes via ProcessPoolExecutor. Sweep wall-clock is
    almost entirely simulation time, not I/O, so this is expected to scale
    close to linearly with worker count on an idle multi-core machine.

    Each worker process independently imports numpy/qiskit-aer, both of
    which default to spinning up their OWN multi-threaded BLAS/OpenMP pool
    sized to every core on the machine -- harmless for one process, but
    `max_workers` of them simultaneously oversubscribes threads by
    worker_count x core_count and reliably crashes with OpenBLAS memory
    allocation errors (confirmed: 8 workers on a 16-core box). So whenever
    `max_workers` > 1, this caps each worker to a single BLAS/OpenMP thread
    (`setdefault` -- an explicit env var the caller already set wins) before
    spawning the pool, since the outer process-level parallelism is already
    the intended source of parallelism here, not any inner library threading.
    """
    run_methods = tuple(set(methods) | {"exact"})
    work_items = [
        (instance_type, n, seed)
        for instance_type in instance_types
        for n in n_values
        for seed in seeds
    ]

    rows: list[dict] = []
    if max_workers is None or max_workers <= 1:
        for instance_type, n, seed in work_items:
            rows.extend(
                _sweep_one(instance_type, n, seed, reps, maxiter, min_size, shots, run_methods, methods, rqaoa_shots)
            )
    else:
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ.setdefault(var, "1")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _sweep_one, instance_type, n, seed, reps, maxiter, min_size, shots,
                    run_methods, methods, rqaoa_shots,
                )
                for instance_type, n, seed in work_items
            ]
            for future in futures:
                rows.extend(future.result())

    return pd.DataFrame(rows)
