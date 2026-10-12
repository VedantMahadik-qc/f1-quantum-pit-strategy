"""Scaling study: sweep RQAOA/QAOA/greedy against exact ground truth across
many synthetic seeds and N values, on both path_like (sparse, pit-lane-queue
shaped) and random_sparse instances. Replaces single-instance anecdotes with
an actual statistical comparison.

This is slow (QAOA/RQAOA dominate) -- the default config targets roughly
15-25 minutes. Shrink N_VALUES/SEEDS for a quick check, or grow them for a
more thorough run later.

Usage:
    python scripts/run_scaling_study.py
"""
from __future__ import annotations

import os

from f1qps.experiments.plotting import plot_approx_ratio_vs_n, plot_wallclock_vs_n
from f1qps.experiments.runner import sweep_synthetic

N_VALUES = [4, 6, 8]
SEEDS = list(range(5))
INSTANCE_TYPES = ("path_like", "random_sparse")
REPS = 1
MAXITER = 120
MIN_SIZE = 3

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(
        f"Sweeping N={N_VALUES}, {len(SEEDS)} seeds, instance_types={INSTANCE_TYPES} "
        f"-- this will take a while (QAOA/RQAOA dominate)."
    )
    df = sweep_synthetic(
        n_values=N_VALUES,
        seeds=SEEDS,
        instance_types=INSTANCE_TYPES,
        reps=REPS,
        maxiter=MAXITER,
        min_size=MIN_SIZE,
        methods=("greedy", "qaoa", "rqaoa"),
    )

    csv_path = os.path.join(RESULTS_DIR, "scaling_study.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved raw results to {csv_path}")

    print("\nMean approximation ratio by (instance_type, N, method):")
    summary = df.groupby(["instance_type", "N", "method"])["approx_ratio"].agg(["mean", "std"])
    print(summary.to_string())

    plot_approx_ratio_vs_n(df, os.path.join(RESULTS_DIR, "approx_ratio_vs_n.png"))
    plot_wallclock_vs_n(df, os.path.join(RESULTS_DIR, "wallclock_vs_n.png"))
    print(f"\nSaved plots to {RESULTS_DIR}/approx_ratio_vs_n.png and wallclock_vs_n.png")


if __name__ == "__main__":
    main()
