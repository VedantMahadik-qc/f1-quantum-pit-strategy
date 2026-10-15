"""Follow-up to investigate_rqaoa_depth_full_range.py, where reps=3 clearly
rescued RQAOA on path_like instances across N=4-16 but random_sparse kept
degrading with N (and got WORSE, not better, between N=14 and N=16 relative
to reps=1 in one case) -- the depth fix did not generalize across structure.

A quick single-seed timing calibration at N=16 (not in this script) already
produced one surprising data point worth noting up front: reps=7 at the same
maxiter=120 used throughout the main sweeps did WORSE (val=-1.74) than
reps=3 (val=-2.92) or reps=5 (val=-2.25) on the identical instance. COBYLA's
iteration budget was held fixed while the parameter count (2*reps) grew, so
a naive "more reps" reading of the depth story is confounded with
"optimizer increasingly starved of iterations relative to its search space."
reps=7 is excluded from the multi-seed ablation below purely on cost grounds
(633s for a single run) -- that one calibration point is reported as
anecdotal, not statistical.

Two hypotheses, tested in isolation:

1. Compute-starved, not structurally broken: is random_sparse's failure at
   N=16 explained by COBYLA not converging, separate from raw circuit depth?
   -> rerun at N=16 (random_sparse only, 3 seeds) with reps=3/maxiter=300
   (more optimizer budget, same depth) and reps=5/maxiter=120 and
   reps=5/maxiter=200 (same depth, budget scaled to roughly match the
   reps=3/maxiter=120 iterations-per-parameter ratio). Compared against the
   existing 5-seed reps=3/maxiter=120 baseline in scaling_study_reps3.csv.
   If more budget (at matched depth) buys real accuracy, this is at least
   partly a resource problem.
2. Irrevocable bad elimination CHOICES/SIGNS: RQAOA commits to each
   elimination based on a single QAOA snapshot's <Z_i Z_j> correlations and
   can never revisit it. Using the known exact ground state (cheap for
   N<=16) as a referee, check at each elimination step whether the chosen
   sign actually matches the true ground-state relationship between the two
   variables, and record the correlation margin over the runner-up pair.
   Compare the resulting "sign error rate" and margins between path_like
   and random_sparse (same N, same reps) -- a much higher error rate /
   thinner margins on random_sparse would point at the greedy elimination
   rule itself being unreliable on denser/frustrated graphs, which no amount
   of added depth would be expected to fix.

Kept small (N=16 only, 3 seeds, random_sparse-only for hypothesis 1; N=14,
5 seeds for hypothesis 2) and reps=7 dropped from the multi-seed part on
cost grounds -- targeted diagnostic, not a replacement for the main scaling
study.

Usage:
    python scripts/investigate_random_sparse_rqaoa_failure.py
"""
from __future__ import annotations

import os

import pandas as pd

from f1qps.classical.exact import solve_exact
from f1qps.experiments.runner import sweep_synthetic
from f1qps.qubo.synthetic import path_like_instance, random_sparse_instance
from f1qps.quantum.rqaoa import solve_rqaoa

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
MIN_SIZE = 3
MAX_WORKERS = os.cpu_count() or 1


def hypothesis_1_compute_ablation() -> None:
    print("\n=== Hypothesis 1: is random_sparse compute-starved at matched depth, N=16? ===")
    seeds = [0, 1, 2]
    configs = [
        {"reps": 3, "maxiter": 300},
        {"reps": 5, "maxiter": 120},
        {"reps": 5, "maxiter": 200},
    ]
    rows = []
    for cfg in configs:
        print(f"  running reps={cfg['reps']}, maxiter={cfg['maxiter']} ...")
        df = sweep_synthetic(
            n_values=[16],
            seeds=seeds,
            instance_types=("random_sparse",),
            reps=cfg["reps"],
            maxiter=cfg["maxiter"],
            min_size=MIN_SIZE,
            methods=("rqaoa",),
            max_workers=MAX_WORKERS,
        )
        df["reps"] = cfg["reps"]
        df["maxiter"] = cfg["maxiter"]
        rows.append(df)
        print(df[["seed", "approx_ratio", "wall_clock"]].to_string(index=False))

    full = pd.concat(rows, ignore_index=True)
    csv_path = os.path.join(RESULTS_DIR, "investigate_random_sparse_compute_ablation.csv")
    full.to_csv(csv_path, index=False)

    summary = full.groupby(["reps", "maxiter"])["approx_ratio"].agg(["mean", "std"])
    print("\nSummary (seeds 0-2 only; compare against reps=3/maxiter=120, 5-seed baseline"
          " in scaling_study_reps3.csv, N=16, random_sparse, mean~0.779):")
    print(summary.to_string())
    print(f"saved -> {csv_path}")


def hypothesis_2_elimination_correctness() -> None:
    print("\n=== Hypothesis 2: are elimination choices/signs themselves unreliable? ===")
    generators = {
        "path_like": lambda n, seed: path_like_instance(n, seed=seed),
        "random_sparse": lambda n, seed: random_sparse_instance(n, edge_prob=0.5, seed=seed),
    }

    rows = []
    for instance_type, generator in generators.items():
        for seed in range(5):
            qubo = generator(14, seed)
            x_star, _ = solve_exact(qubo)
            truth = {label: int(val) for label, val in zip(qubo.labels, x_star)}

            history: list[dict] = []
            solve_rqaoa(qubo, reps=3, min_size=MIN_SIZE, maxiter=120, seed=seed, history_out=history)

            for step_idx, step in enumerate(history):
                predicted_sign = 1 if truth[step["eliminated_label"]] == truth[step["fixed_label"]] else -1
                rows.append(
                    {
                        "instance_type": instance_type,
                        "seed": seed,
                        "step_idx": step_idx,
                        "n_before": step["n_before"],
                        "margin": step["margin"],
                        "sign_correct": int(step["sign"] == predicted_sign),
                    }
                )

    full = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "investigate_elimination_correctness.csv")
    full.to_csv(csv_path, index=False)

    print("\nSign-error rate and margin by instance_type (reps=3, N=14):")
    summary = full.groupby("instance_type").agg(
        sign_error_rate=("sign_correct", lambda s: 1 - s.mean()),
        mean_margin=("margin", "mean"),
        n_steps=("sign_correct", "size"),
    )
    print(summary.to_string())

    print("\nSign-error rate by n_before (does it get worse deeper into the recursion?):")
    summary2 = full.groupby(["instance_type", "n_before"]).agg(
        sign_error_rate=("sign_correct", lambda s: 1 - s.mean()),
        mean_margin=("margin", "mean"),
    )
    print(summary2.to_string())
    print(f"saved -> {csv_path}")


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    hypothesis_1_compute_ablation()
    hypothesis_2_elimination_correctness()


if __name__ == "__main__":
    main()
