"""First CONSTRUCTIVE follow-up in this line of investigation, rather than
another diagnostic. Every earlier result pointed at the same root cause:
RQAOA commits irrevocably to each elimination based on a SINGLE QAOA
optimization's correlation estimate, and that estimate's reliability turned
out to be idiosyncratic -- not predictable from graph topology, weight
magnitude, or measurement shot count (see investigate_random_sparse_*,
investigate_weight_magnitude_hypothesis, investigate_shot_noise). If the
unreliability really is single-trajectory noise rather than a structural
property of the instance, averaging independent trajectories should reduce
it. solve_rqaoa() now supports `ensemble_k`: k independent QAOA
optimizations (distinct seeds) per elimination step, correlation matrices
averaged before picking the pair/sign. This tests whether ensemble_k=5
actually fixes (or just costs 5x for no benefit on) the worst known cases:

  A. Synthetic random_sparse, N=14, 5 seeds, reps=3/maxiter=120 (matches
     investigate_elimination_correctness.csv exactly) -- same sign-error-
     rate referee as before, now under ensembling, plus the resulting
     approx_ratio compared against the existing ensemble_k=1 baseline in
     scaling_study_reps3.csv (mean 0.716) and investigate_elimination_
     correctness.csv (41.8% sign-error rate).
  B. The 6 real SC/VSC periods that failed in run_real_race_experiment.py's
     full 17-period sweep -- rebuilt from the same cached FastF1 data,
     re-solved with ensemble_k=5, compared directly against their recorded
     ensemble_k=1 approx_ratio in real_race_solver_comparison.csv.

ensemble_k=5 costs ~5x a single RQAOA solve; kept to N=14 synthetic (not
the full N=4-16 range) and only the known-failing real periods (not all 17)
to stay within a reasonable runtime.

Usage:
    python scripts/investigate_rqaoa_ensemble_fix.py
"""
from __future__ import annotations

import os

import pandas as pd

from f1qps.classical.exact import solve_exact
from f1qps.data.degradation import compute_h, fit_degradation_curves
from f1qps.data.fastf1_loader import (
    enable_cache,
    find_sc_vsc_periods,
    green_flag_laps,
    laps_in_period,
    load_race,
)
from f1qps.data.pitlane_graph import build_real_qubo
from f1qps.experiments.metrics import approx_ratio
from f1qps.qubo.synthetic import path_like_instance, random_sparse_instance
from f1qps.quantum.rqaoa import solve_rqaoa

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
N = 14
SEEDS = list(range(5))
ENSEMBLE_K = 5
REPS = 3
MAXITER = 120
MIN_SIZE = 3

GENERATORS = {
    "path_like": path_like_instance,
    "random_sparse": lambda n, seed: random_sparse_instance(n, edge_prob=0.5, seed=seed),
}

# (year, gp, start_lap, previously recorded ensemble_k=1 approx_ratio)
FAILING_REAL_PERIODS = [
    (2023, "Singapore", 19, -3.245788),
    (2021, "Azerbaijan", 30, 0.551509),
    (2019, "Germany", 40, 0.816855),
    (2023, "Australia", 7, 0.825536),
    (2023, "Australia", 53, 0.840723),
    (2019, "Germany", 2, float("nan")),
]


def synthetic_pass() -> None:
    print(f"\n=== A: synthetic random_sparse/path_like, N={N}, ensemble_k={ENSEMBLE_K} ===")
    step_rows = []
    outcome_rows = []

    for instance_type, generator in GENERATORS.items():
        for seed in SEEDS:
            qubo = generator(N, seed)
            x_star, exact_val = solve_exact(qubo)
            truth = {label: int(val) for label, val in zip(qubo.labels, x_star)}

            history: list[dict] = []
            _, val = solve_rqaoa(
                qubo, reps=REPS, min_size=MIN_SIZE, maxiter=MAXITER, seed=seed,
                ensemble_k=ENSEMBLE_K, history_out=history,
            )
            outcome_rows.append(
                {"instance_type": instance_type, "seed": seed, "approx_ratio": approx_ratio(val, exact_val)}
            )
            for step_idx, step in enumerate(history):
                predicted_sign = 1 if truth[step["eliminated_label"]] == truth[step["fixed_label"]] else -1
                step_rows.append(
                    {
                        "instance_type": instance_type, "seed": seed, "step_idx": step_idx,
                        "sign_correct": int(step["sign"] == predicted_sign),
                    }
                )
            print(f"  {instance_type} seed={seed}: approx_ratio={approx_ratio(val, exact_val):.4f}")

    steps_df = pd.DataFrame(step_rows)
    outcomes_df = pd.DataFrame(outcome_rows)
    steps_df.to_csv(os.path.join(RESULTS_DIR, "investigate_ensemble_fix_synthetic_steps.csv"), index=False)
    outcomes_df.to_csv(os.path.join(RESULTS_DIR, "investigate_ensemble_fix_synthetic_outcomes.csv"), index=False)

    print("\nSign-error rate with ensemble_k=5 (compare: ensemble_k=1 was 9.1% path_like, 41.8% random_sparse):")
    print(steps_df.groupby("instance_type")["sign_correct"].agg(
        sign_error_rate=lambda s: 1 - s.mean(), n_steps="size"
    ).to_string())
    print("\napprox_ratio with ensemble_k=5 (compare: ensemble_k=1 was 0.956 path_like, 0.716 random_sparse):")
    print(outcomes_df.groupby("instance_type")["approx_ratio"].agg(["mean", "std"]).to_string())


def real_data_pass() -> None:
    print(f"\n=== B: known-failing real SC/VSC periods, ensemble_k={ENSEMBLE_K} ===")
    enable_cache()
    rows = []
    session_cache: dict[tuple[int, str], object] = {}

    for year, gp, start_lap, old_ratio in FAILING_REAL_PERIODS:
        if (year, gp) not in session_cache:
            session_cache[(year, gp)] = load_race(year, gp)
        session = session_cache[(year, gp)]

        (period,) = [p for p in find_sc_vsc_periods(session) if p.start_lap == start_lap]
        decision_laps = laps_in_period(session, period)
        curves = fit_degradation_curves(green_flag_laps(session))
        h = compute_h(decision_laps, curves)
        qubo = build_real_qubo(decision_laps, h)

        _, exact_val = solve_exact(qubo)
        _, val = solve_rqaoa(
            qubo, reps=REPS, min_size=MIN_SIZE, maxiter=MAXITER, seed=0, ensemble_k=ENSEMBLE_K
        )
        new_ratio = approx_ratio(val, exact_val)
        print(f"  {year} {gp} lap {start_lap}: ensemble_k=1 ratio={old_ratio:.4f} -> "
              f"ensemble_k=5 ratio={new_ratio:.4f}")
        rows.append(
            {
                "year": year, "gp": gp, "start_lap": start_lap,
                "ensemble_k1_ratio": old_ratio, "ensemble_k5_ratio": new_ratio,
            }
        )

    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "investigate_ensemble_fix_real_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved -> {csv_path}")
    print(df.to_string(index=False))


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    synthetic_pass()
    real_data_pass()


if __name__ == "__main__":
    main()
