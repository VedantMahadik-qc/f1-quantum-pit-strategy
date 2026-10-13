"""The third framing from the README, previously untouched: everything so
far (scaling study, depth-rescue, elimination-correctness, real-race
validation) used EXACT analytic <Z_i Z_j> expectation values and shot-based
sampling only to extract a best-of-many bitstring for plain QAOA. Real
hardware can't give you an exact expectation value -- only noisy estimates
from a finite number of measurement shots. This asks: how much does RQAOA's
already-fragile elimination signal (see elimination-correctness diagnostic)
degrade under a realistic finite-shot budget, and does the same budget
erode plain QAOA's reliability the same way?

Two methods are affected very differently by finite shots, and the script
measures both:
  - RQAOA commits to ONE elimination per shot-estimate, irrevocably. Shot
    noise directly corrupts the signal it bases that irrevocable choice on.
  - Plain QAOA takes the BEST of many sampled bitstrings. Shot noise mostly
    just shrinks how many candidate draws you get, not the quality signal
    itself -- expected to be far more robust.

Methodological note on wall-clock: in noiseless SIMULATION, the dominant
cost is the exact-statevector COBYLA optimization loop (maxiter calls),
which is unaffected by `shots` -- only the final readout is shot-based. So
a shots-vs-wall-clock curve would be nearly flat here; that's a simulation
artifact (on real hardware every shot costs real time), not a finding, and
is why this script measures QUALITY degradation vs. shots instead of
reconstructing the README's "wall-clock budget" framing literally.

Kept to N=14 (matches the elimination-correctness diagnostic for direct
comparability), 5 seeds, both instance types -- targeted diagnostic, not a
replacement for the main scaling study.

Usage:
    python scripts/investigate_shot_noise.py
"""
from __future__ import annotations

import os

import pandas as pd

from f1qps.classical.exact import solve_exact
from f1qps.experiments.metrics import approx_ratio
from f1qps.qubo.synthetic import path_like_instance, random_sparse_instance
from f1qps.quantum.qaoa import optimize_qaoa_angles, sample_qaoa_solution
from f1qps.quantum.rqaoa import solve_rqaoa

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
N = 14
SEEDS = list(range(5))
SHOT_LEVELS = [None, 8000, 2000, 500, 100]  # None = exact baseline
REPS = 3
MAXITER = 120
MIN_SIZE = 3

GENERATORS = {
    "path_like": path_like_instance,
    "random_sparse": lambda n, seed: random_sparse_instance(n, edge_prob=0.5, seed=seed),
}


def rqaoa_pass() -> pd.DataFrame:
    print("\n=== RQAOA: sign-error rate + outcome quality vs. shots ===")
    step_rows = []
    outcome_rows = []

    for instance_type, generator in GENERATORS.items():
        for seed in SEEDS:
            qubo = generator(N, seed)
            x_star, exact_val = solve_exact(qubo)
            truth = {label: int(val) for label, val in zip(qubo.labels, x_star)}

            for shots in SHOT_LEVELS:
                history: list[dict] = []
                _, val = solve_rqaoa(
                    qubo, reps=REPS, min_size=MIN_SIZE, maxiter=MAXITER,
                    seed=seed, shots=shots, history_out=history,
                )
                outcome_rows.append(
                    {
                        "instance_type": instance_type, "seed": seed, "shots": shots,
                        "approx_ratio": approx_ratio(val, exact_val),
                    }
                )
                for step_idx, step in enumerate(history):
                    predicted_sign = (
                        1 if truth[step["eliminated_label"]] == truth[step["fixed_label"]] else -1
                    )
                    step_rows.append(
                        {
                            "instance_type": instance_type, "seed": seed, "shots": shots,
                            "step_idx": step_idx, "margin": step["margin"],
                            "sign_correct": int(step["sign"] == predicted_sign),
                        }
                    )
            print(f"  {instance_type} seed={seed} done")

    steps_df = pd.DataFrame(step_rows)
    outcomes_df = pd.DataFrame(outcome_rows)
    steps_df.to_csv(os.path.join(RESULTS_DIR, "investigate_shot_noise_rqaoa_steps.csv"), index=False)
    outcomes_df.to_csv(os.path.join(RESULTS_DIR, "investigate_shot_noise_rqaoa_outcomes.csv"), index=False)

    print("\nSign-error rate by (instance_type, shots):")
    print(
        steps_df.groupby(["instance_type", "shots"], dropna=False)
        .agg(sign_error_rate=("sign_correct", lambda s: 1 - s.mean()), mean_margin=("margin", "mean"))
        .to_string()
    )
    print("\nOutcome approx_ratio by (instance_type, shots):")
    print(
        outcomes_df.groupby(["instance_type", "shots"], dropna=False)["approx_ratio"]
        .agg(["mean", "std"]).to_string()
    )
    return outcomes_df


def qaoa_pass() -> pd.DataFrame:
    print("\n=== Plain QAOA: outcome quality vs. shots (for contrast) ===")
    rows = []
    for instance_type, generator in GENERATORS.items():
        for seed in SEEDS:
            qubo = generator(N, seed)
            _, exact_val = solve_exact(qubo)
            qaoa_result = optimize_qaoa_angles(qubo, reps=REPS, maxiter=MAXITER, seed=seed)

            for shots in SHOT_LEVELS:
                # QAOA has no "exact" readout mode (unlike RQAOA's analytic
                # correlations) -- shots=None here just reuses the largest
                # shot level as the ceiling baseline, labeled None to line
                # up with RQAOA's x-axis for comparison.
                effective_shots = shots if shots is not None else 8000
                _, val = sample_qaoa_solution(qubo, qaoa_result, shots=effective_shots, seed=seed)
                rows.append(
                    {
                        "instance_type": instance_type, "seed": seed, "shots": shots,
                        "approx_ratio": approx_ratio(val, exact_val),
                    }
                )
            print(f"  {instance_type} seed={seed} done")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "investigate_shot_noise_qaoa_outcomes.csv"), index=False)
    print("\nOutcome approx_ratio by (instance_type, shots):")
    print(df.groupby(["instance_type", "shots"], dropna=False)["approx_ratio"].agg(["mean", "std"]).to_string())
    return df


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rqaoa_pass()
    qaoa_pass()


if __name__ == "__main__":
    main()
