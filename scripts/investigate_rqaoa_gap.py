"""Follow-up to the N=4-8 scaling study, where plain QAOA beat RQAOA on both
instance types -- the opposite of the sparse-graph hypothesis. Two candidate
explanations, each tested in isolation:

1. RQAOA's literature advantage over vanilla QAOA is usually reported at much
   larger N than 4-8, where single-pass QAOA genuinely starts to degrade.
   At tiny N, plain QAOA is already near-exact, leaving little room for
   RQAOA to show an edge. -> rerun at larger N (same reps=1).
2. RQAOA commits irrevocably to each elimination choice based on a shallow
   (reps=1) inner QAOA's correlation estimate; a bad early elimination can't
   be corrected later, unlike plain QAOA which samples many shots and keeps
   the best. -> rerun at the original small N with a deeper inner QAOA
   (reps=3) and see whether RQAOA closes the gap while QAOA (already near
   its ceiling) has little room left to also improve.

Both configs are kept small (fewer seeds / N values than a full study) to
stay within a reasonable runtime -- this is a targeted diagnostic, not a
replacement for the main scaling study.

Usage:
    python scripts/investigate_rqaoa_gap.py
"""
from __future__ import annotations

import os

from f1qps.experiments.runner import sweep_synthetic

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def run_and_report(label: str, csv_name: str, **sweep_kwargs) -> None:
    print(f"\n=== {label} ===")
    print({k: v for k, v in sweep_kwargs.items() if k != "methods"})
    df = sweep_synthetic(methods=("greedy", "qaoa", "rqaoa"), **sweep_kwargs)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, csv_name)
    df.to_csv(csv_path, index=False)

    summary = df.groupby(["instance_type", "N", "method"])["approx_ratio"].agg(["mean", "std"])
    print(summary.to_string())
    print(f"saved -> {csv_path}")


def main() -> None:
    # Hypothesis 1: larger N
    run_and_report(
        "Hypothesis 1: larger N (reps=1, same depth as the original sweep)",
        "investigate_larger_n.csv",
        n_values=[10],
        seeds=[0, 1, 2],
        instance_types=("path_like", "random_sparse"),
        reps=1,
        maxiter=120,
        min_size=3,
    )

    # Hypothesis 2: deeper inner QAOA at the original small N
    run_and_report(
        "Hypothesis 2: deeper circuits (reps=3) at the original N=4,6,8",
        "investigate_deeper_reps.csv",
        n_values=[4, 6, 8],
        seeds=[0, 1, 2],
        instance_types=("path_like", "random_sparse"),
        reps=3,
        maxiter=120,
        min_size=3,
    )


if __name__ == "__main__":
    main()
