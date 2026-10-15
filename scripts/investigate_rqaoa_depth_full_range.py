"""Follow-up to run_scaling_study.py's N=4-16 result, where RQAOA (reps=1)
got steadily worse relative to QAOA as N grew (path_like 1.00->0.84,
random_sparse 1.00->0.69). A separate test at N=4-8 only showed deeper
inner-QAOA circuits (reps=3) substantially closing that gap.

This tests whether that fix generalizes: does reps=3 rescue RQAOA across the
FULL N=4-16 range, or does the downward trend persist regardless of depth
(which would be a stronger, more conclusive negative result for RQAOA here)?

Usage:
    python scripts/investigate_rqaoa_depth_full_range.py
"""
from __future__ import annotations

import os

from f1qps.experiments.plotting import plot_approx_ratio_vs_n, plot_wallclock_vs_n
from f1qps.experiments.runner import sweep_synthetic

MAX_WORKERS = os.cpu_count() or 1
N_VALUES = [4, 6, 8, 10, 12, 14, 16]
SEEDS = list(range(5))
INSTANCE_TYPES = ("path_like", "random_sparse")
REPS = 3
MAXITER = 120
MIN_SIZE = 3

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Sweeping N={N_VALUES}, {len(SEEDS)} seeds, reps={REPS} (vs reps=1 in the original sweep)")

    df = sweep_synthetic(
        n_values=N_VALUES,
        seeds=SEEDS,
        instance_types=INSTANCE_TYPES,
        reps=REPS,
        maxiter=MAXITER,
        min_size=MIN_SIZE,
        methods=("greedy", "qaoa", "rqaoa"),
        max_workers=MAX_WORKERS,
    )

    csv_path = os.path.join(RESULTS_DIR, "scaling_study_reps3.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved raw results to {csv_path}")

    print("\nMean approximation ratio by (instance_type, N, method):")
    summary = df.groupby(["instance_type", "N", "method"])["approx_ratio"].agg(["mean", "std"])
    print(summary.to_string())

    plot_approx_ratio_vs_n(df, os.path.join(RESULTS_DIR, "approx_ratio_vs_n_reps3.png"))
    plot_wallclock_vs_n(df, os.path.join(RESULTS_DIR, "wallclock_vs_n_reps3.png"))
    print(f"\nSaved plots to {RESULTS_DIR}/approx_ratio_vs_n_reps3.png and wallclock_vs_n_reps3.png")


if __name__ == "__main__":
    main()
