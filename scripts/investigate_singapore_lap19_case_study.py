"""Case study: why does Singapore 2023 lap 19 specifically get WORSE (not just
fail to improve) under more restarts -- at ensemble_k=10
(investigate_rqaoa_ensemble_fix.py: 0.283->0.113) and again under selective
ensembling (investigate_selective_ensembling.py: 0.283->-3.492)? README
"What's Still Open" item 2.

Two things ruled out first:
  - Not a small-exact-value ratio artifact: Azerbaijan lap 30's exact objective
    magnitude (|-0.230|) is even smaller than Singapore's (|-0.261|), yet
    Azerbaijan is rock-solid across every configuration tried.
  - Not sign-error COUNT: both periods tie exactly at 3/15 sign errors
    (against the classical referee) at ensemble_k=1.

What actually differs, traced step-by-step via history_out:
  1. Azerbaijan's wrong picks land late in the recursion among variables with
     EXACTLY ZERO correlation (top_corr=0.000 -- genuinely disconnected in the
     conflict graph), so getting the sign "wrong" there costs nothing.
     Singapore has no such free pass -- even its weakest steps have small but
     nonzero top_corr (0.047-0.055).
  2. A confident single restart does not predict what a 5-restart AVERAGE
     will do: at one Singapore step, the first restart shared (bit-identical,
     same seed) by both the selective and full-ensemble runs would alone
     have picked one pair, but averaging in 4 more restarts flips the full
     ensemble's actual choice to a completely different one -- the same fact
     already on record as `pair_agreement` saturating at 0.0 on every real
     period (investigate_restart_disagreement.py), caught here in the act at
     one concrete fork.
  3. Past that fork, the selective run's path happens to chain several
     CONSECUTIVE wrong picks through a small set of recurring hub variables,
     which compounds through back-substitution far worse than the same
     number of wrong picks scattered across unrelated variables.

This is a 2-period case study, not a validated general rule -- it explains
Singapore lap 19 in hindsight but hasn't been checked against the other 4
real periods or turned into a predictive signal.

Usage:
    python scripts/investigate_singapore_lap19_case_study.py
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
from f1qps.quantum.rqaoa import solve_rqaoa

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
REPS = 3
MAXITER = 120
MIN_SIZE = 3
THRESHOLD = 0.5

PERIODS = {
    "singapore_lap19": (2023, "Singapore", 19),
    "azerbaijan_lap30": (2021, "Azerbaijan", 30),
}


def _load_qubo(year: int, gp: str, start_lap: int):
    session = load_race(year, gp)
    (period,) = [p for p in find_sc_vsc_periods(session) if p.start_lap == start_lap]
    decision_laps = laps_in_period(session, period)
    curves = fit_degradation_curves(green_flag_laps(session))
    h = compute_h(decision_laps, curves)
    return build_real_qubo(decision_laps, h)


def _step_rows(period_name: str, run_name: str, history: list[dict], truth: dict[str, int]) -> list[dict]:
    rows = []
    for step in history:
        predicted_sign = 1 if truth[step["eliminated_label"]] == truth[step["fixed_label"]] else -1
        rows.append(
            {
                "period": period_name,
                "run": run_name,
                "n_before": step["n_before"],
                "eliminated": step["eliminated_label"],
                "fixed": step["fixed_label"],
                "sign": step["sign"],
                "top_corr": step["top_corr"],
                "restarts_used": step["restarts_used"],
                "sign_correct": int(predicted_sign == step["sign"]),
            }
        )
    return rows


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    enable_cache()
    rows: list[dict] = []
    summary_rows: list[dict] = []

    for period_name, (year, gp, start_lap) in PERIODS.items():
        qubo = _load_qubo(year, gp, start_lap)
        x_star, exact_val = solve_exact(qubo)
        truth = {label: int(v) for label, v in zip(qubo.labels, x_star)}
        print(f"\n=== {period_name}: N={qubo.n}, exact_val={exact_val:.4f} ===")

        configs = {"k1": dict(ensemble_k=1), "k5_full": dict(ensemble_k=5)}
        if period_name == "singapore_lap19":
            configs["k5_selective"] = dict(ensemble_k=5, confidence_threshold=THRESHOLD)

        for run_name, kwargs in configs.items():
            history: list[dict] = []
            _, val = solve_rqaoa(
                qubo, reps=REPS, min_size=MIN_SIZE, maxiter=MAXITER, seed=0, history_out=history, **kwargs
            )
            step_rows = _step_rows(period_name, run_name, history, truth)
            rows.extend(step_rows)
            n_wrong = sum(1 - r["sign_correct"] for r in step_rows)
            print(
                f"  {run_name}: objective={val:.4f} ratio={val / exact_val:.4f} "
                f"sign_errors={n_wrong}/{len(step_rows)}"
            )
            summary_rows.append(
                {
                    "period": period_name,
                    "run": run_name,
                    "objective": val,
                    "exact_val": exact_val,
                    "ratio": val / exact_val,
                    "n_steps": len(step_rows),
                    "sign_errors": n_wrong,
                }
            )

    steps_df = pd.DataFrame(rows)
    steps_path = os.path.join(RESULTS_DIR, "investigate_singapore_lap19_case_study_steps.csv")
    steps_df.to_csv(steps_path, index=False)
    print(f"\nSaved per-step detail -> {steps_path}")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(RESULTS_DIR, "investigate_singapore_lap19_case_study_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary -> {summary_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
