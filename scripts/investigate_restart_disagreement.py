"""Follow-up to investigate_rqaoa_ensemble_fix.py's open question: can the two
failure modes it found -- "noisy" (ensembling fixes it) vs "confidently wrong"
(ensembling doesn't, e.g. 2019 Germany lap 40, bit-for-bit identical at
ensemble_k=1/5/10) -- be told apart WITHOUT already knowing the classical
answer? The only thing available at run time, before checking against a
baseline, is how much the ensemble_k independent restarts agree with each
other at each elimination step. solve_rqaoa() now exposes those pre-averaging
correlation matrices via `restart_corrs_out`. This script asks: does low
cross-restart agreement predict "noisy" (and hence fixable), and does high
agreement predict "confidently wrong" (a shared, consistent mistake that no
amount of ensembling will fix)?

Two agreement metrics computed per elimination step, from the k individual
(pre-averaging) correlation matrices, using only the SAME pair/sign the
averaged matrix actually elects (i.e. nothing ground-truth needs to be known):
  - pair_agreement: fraction of the k restarts whose OWN top-magnitude pair
    (computed independently on that restart's own matrix) is the same pair
    the average elects.
  - sign_agreement: fraction of the k restarts whose correlation at the
    elected pair has the same sign as the average's.
Both 1.0 means total consensus (a "confident" step, right or wrong); either
metric low means the restarts actively disagree (a "noisy" step).

A. Real data: the same 6 known-failing SC/VSC periods from
   investigate_rqaoa_ensemble_fix.py, re-solved at ensemble_k=5 with
   restart_corrs_out captured, per-step and per-period agreement summarized
   and lined up against each period's already-recorded k=1/5(/10) ratios.
B. Synthetic: same N=14, 5-seed, path_like/random_sparse sweep as the
   ensemble-fix script's part A, at ensemble_k=5, comparing per-step
   agreement against whether that step's elected sign matches the classical
   ground truth (same sign-correctness referee used throughout this project).

Usage:
    python scripts/investigate_restart_disagreement.py
"""
from __future__ import annotations

import os

import numpy as np
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
    "path_like": lambda n, seed: path_like_instance(n, seed=seed),
    "random_sparse": lambda n, seed: random_sparse_instance(n, edge_prob=0.5, seed=seed),
}

# (year, gp, start_lap, k1_ratio, k5_ratio) -- from results/investigate_ensemble_fix_real_data.csv
FAILING_REAL_PERIODS = [
    (2023, "Singapore", 19, -3.245788, 0.28293745107228946),
    (2021, "Azerbaijan", 30, 0.551509, 1.0),
    (2019, "Germany", 40, 0.816855, 0.8168545424647286),
    (2023, "Australia", 7, 0.825536, 1.0),
    (2023, "Australia", 53, 0.840723, 0.9480804888170267),
    (2019, "Germany", 2, float("nan"), 1.0),
]


def _own_argmax(corr: np.ndarray) -> tuple[int, int, int]:
    abs_corr = np.abs(corr)
    i, j = np.unravel_index(np.argmax(abs_corr), abs_corr.shape)
    i, j = int(i), int(j)
    sign = 1 if corr[i, j] >= 0 else -1
    return i, j, sign


def _step_agreement(step_corrs: list[np.ndarray]) -> dict:
    """pair_agreement/sign_agreement for one elimination step's k restart
    matrices, relative to the pair/sign the AVERAGE of those matrices elects
    (i.e. the decision solve_rqaoa actually acts on).
    """
    avg = np.mean(step_corrs, axis=0)
    i, j, sign = _own_argmax(avg)

    pair_matches = []
    sign_matches = []
    for corr in step_corrs:
        ri, rj, _ = _own_argmax(corr)
        pair_matches.append({ri, rj} == {i, j})
        sign_matches.append((1 if corr[i, j] >= 0 else -1) == sign)

    return {
        "pair_agreement": float(np.mean(pair_matches)),
        "sign_agreement": float(np.mean(sign_matches)),
        "elected_i": i,
        "elected_j": j,
        "elected_sign": sign,
    }


def synthetic_pass() -> None:
    print(f"\n=== B: synthetic random_sparse/path_like, N={N}, ensemble_k={ENSEMBLE_K} ===")
    rows = []

    for instance_type, generator in GENERATORS.items():
        for seed in SEEDS:
            qubo = generator(N, seed)
            x_star, _ = solve_exact(qubo)
            truth = {label: int(val) for label, val in zip(qubo.labels, x_star)}

            history: list[dict] = []
            restart_corrs: list[list[np.ndarray]] = []
            solve_rqaoa(
                qubo, reps=REPS, min_size=MIN_SIZE, maxiter=MAXITER, seed=seed,
                ensemble_k=ENSEMBLE_K, history_out=history, restart_corrs_out=restart_corrs,
            )

            for step_idx, (step, step_corrs) in enumerate(zip(history, restart_corrs)):
                predicted_sign = 1 if truth[step["eliminated_label"]] == truth[step["fixed_label"]] else -1
                agreement = _step_agreement(step_corrs)
                rows.append(
                    {
                        "instance_type": instance_type,
                        "seed": seed,
                        "step_idx": step_idx,
                        "sign_correct": int(step["sign"] == predicted_sign),
                        "pair_agreement": agreement["pair_agreement"],
                        "sign_agreement": agreement["sign_agreement"],
                    }
                )

    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "investigate_restart_disagreement_synthetic_steps.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved -> {csv_path}")

    print("\nMean agreement by instance_type x sign_correct (does agreement distinguish right from wrong?):")
    print(
        df.groupby(["instance_type", "sign_correct"])[["pair_agreement", "sign_agreement"]]
        .agg(["mean", "count"])
        .to_string()
    )


def real_data_pass() -> None:
    print(f"\n=== A: known-failing real SC/VSC periods, ensemble_k={ENSEMBLE_K} ===")
    enable_cache()
    rows = []
    session_cache: dict[tuple[int, str], object] = {}

    for year, gp, start_lap, k1_ratio, k5_ratio in FAILING_REAL_PERIODS:
        if (year, gp) not in session_cache:
            session_cache[(year, gp)] = load_race(year, gp)
        session = session_cache[(year, gp)]

        (period,) = [p for p in find_sc_vsc_periods(session) if p.start_lap == start_lap]
        decision_laps = laps_in_period(session, period)
        curves = fit_degradation_curves(green_flag_laps(session))
        h = compute_h(decision_laps, curves)
        qubo = build_real_qubo(decision_laps, h)

        restart_corrs: list[list[np.ndarray]] = []
        solve_rqaoa(
            qubo, reps=REPS, min_size=MIN_SIZE, maxiter=MAXITER, seed=0,
            ensemble_k=ENSEMBLE_K, restart_corrs_out=restart_corrs,
        )

        step_agreements = [_step_agreement(sc) for sc in restart_corrs]
        pair_vals = [a["pair_agreement"] for a in step_agreements]
        sign_vals = [a["sign_agreement"] for a in step_agreements]

        row = {
            "year": year, "gp": gp, "start_lap": start_lap,
            "ensemble_k1_ratio": k1_ratio, "ensemble_k5_ratio": k5_ratio,
            "n_steps": len(step_agreements),
            "mean_pair_agreement": float(np.mean(pair_vals)),
            "min_pair_agreement": float(np.min(pair_vals)),
            "mean_sign_agreement": float(np.mean(sign_vals)),
            "min_sign_agreement": float(np.min(sign_vals)),
        }
        rows.append(row)
        print(
            f"  {year} {gp} lap {start_lap}: k1={k1_ratio!s:>10} k5={k5_ratio:.4f}  "
            f"mean_pair_agree={row['mean_pair_agreement']:.3f} min_pair_agree={row['min_pair_agreement']:.3f} "
            f"mean_sign_agree={row['mean_sign_agreement']:.3f} min_sign_agree={row['min_sign_agreement']:.3f}"
        )

    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "investigate_restart_disagreement_real.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved -> {csv_path}")
    print(df.to_string(index=False))


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    real_data_pass()
    synthetic_pass()


if __name__ == "__main__":
    main()
