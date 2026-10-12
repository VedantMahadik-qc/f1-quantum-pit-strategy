"""Demo: build a small synthetic pit-lane-like QUBO and compare exact / greedy /
plain QAOA / RQAOA. This is the first tangible "does it work" artifact for the
project -- see plans/first implementation pass.

Usage:
    python scripts/demo_rqaoa_vs_exact.py
"""
from __future__ import annotations

import time

from f1qps.classical.exact import solve_exact
from f1qps.classical.heuristics import solve_greedy
from f1qps.qubo.synthetic import path_like_instance
from f1qps.quantum.qaoa import optimize_qaoa_angles, sample_qaoa_solution
from f1qps.quantum.rqaoa import solve_rqaoa

N = 8
SEED = 7
REPS = 1
MAXITER = 150


def timed(fn):
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


def main() -> None:
    qubo = path_like_instance(n=N, seed=SEED)
    print(f"Synthetic path-like pit-lane QUBO, N={N} cars (seed={SEED})\n")

    (_, exact_val), exact_t = timed(lambda: solve_exact(qubo))
    (_, greedy_val), greedy_t = timed(lambda: solve_greedy(qubo))

    def run_qaoa():
        result = optimize_qaoa_angles(qubo, reps=REPS, maxiter=MAXITER, seed=SEED)
        return sample_qaoa_solution(qubo, result, shots=4000, seed=SEED)

    (_, qaoa_val), qaoa_t = timed(run_qaoa)
    (_, rqaoa_val), rqaoa_t = timed(
        lambda: solve_rqaoa(qubo, reps=REPS, min_size=4, maxiter=MAXITER, seed=SEED)
    )

    def approx_ratio(val: float) -> str:
        if exact_val == 0:
            return "n/a"
        return f"{val / exact_val:.3f}"

    rows = [
        ("exact (brute force)", exact_val, exact_t),
        ("greedy", greedy_val, greedy_t),
        ("QAOA (reps=%d)" % REPS, qaoa_val, qaoa_t),
        ("RQAOA (reps=%d)" % REPS, rqaoa_val, rqaoa_t),
    ]

    print(f"{'method':<22} {'objective':>12} {'approx. ratio':>14} {'wall-clock (s)':>16}")
    for name, val, t in rows:
        print(f"{name:<22} {val:>12.4f} {approx_ratio(val):>14} {t:>16.2f}")


if __name__ == "__main__":
    main()
