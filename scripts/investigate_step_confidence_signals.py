"""Hunt for an advance predictor of RQAOA's two failure modes (noisy vs
"confidently wrong"), continuing README "What's Still Open" item 1. Every
signal tried so far -- topology, QUBO weight magnitude, shot count, ensemble
size, and cross-restart agreement -- has failed to distinguish them ahead of
a classical baseline (see investigate_restart_disagreement.py, the most
recent). This tests a genuinely different, and strictly cheaper, signal:
solve_rqaoa()'s history_out already returns, for free, from a SINGLE k=1 run
(no ensembling or restarts needed) a per-step "margin" (top correlation
minus the runner-up pair's) and "top_corr" (the correlation value of the
pair it actually elects). If either is a reliable confidence score, it would
be the cheapest possible mitigation -- flag low-confidence steps instead of
trusting every elimination equally.

`margin` was already logged (but never tested as a predictor, only
group-averaged) in investigate_elimination_correctness.csv. Re-examined
first as a zero-cost sanity check before writing any new code: pooled
correlation between margin and per-step sign-correctness there is -0.02, and
the error rate is flat across margin quartiles (25%/30%/22%/25%) -- no
signal, on N=14/5-seed/reps=3 synthetic data. This script extends that check
two ways: `top_corr` magnitude (never logged before), and -- more
importantly -- the actual real-world "confidently wrong" case (2019 Germany
lap 40), which margin's own data has never been tested against at all.

A. Real data: the same 6 known-failing SC/VSC periods used throughout this
   investigation series, each run once at k=1 (reps=3, maxiter=120, seed=0,
   matching convention elsewhere), logging history_out's margin/top_corr for
   every elimination step. Run-level aggregates (min/mean of each, across
   all steps in that run) are lined up against the already-recorded
   k1_ratio/k5_ratio to see whether any of them flags Germany lap 40 (the
   one case ensembling never fixes) as different in kind from the other
   five.
B. Synthetic: same N=14 path_like/random_sparse setup as
   investigate_random_sparse_rqaoa_failure.py's hypothesis 2, but with 20
   seeds per instance_type (vs. 5) for more statistical power, and
   `top_corr` logged alongside `margin`, to pool enough steps to say
   whether EITHER predicts per-step sign-correctness.

Usage:
    python scripts/investigate_step_confidence_signals.py
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

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
from f1qps.qubo.synthetic import path_like_instance, random_sparse_instance
from f1qps.quantum.rqaoa import solve_rqaoa

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
N = 14
SEEDS = list(range(20))
REPS = 3
MAXITER = 120
MIN_SIZE = 3
MAX_WORKERS = os.cpu_count() or 1

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


def _synthetic_one(instance_type: str, seed: int) -> list[dict]:
    """One (instance_type, seed) cell. Module-level so it can be pickled and
    run in a worker process.
    """
    generator = GENERATORS[instance_type]
    qubo = generator(N, seed)
    x_star, _ = solve_exact(qubo)
    truth = {label: int(val) for label, val in zip(qubo.labels, x_star)}

    history: list[dict] = []
    solve_rqaoa(qubo, reps=REPS, min_size=MIN_SIZE, maxiter=MAXITER, seed=seed, history_out=history)

    rows = []
    for step_idx, step in enumerate(history):
        predicted_sign = 1 if truth[step["eliminated_label"]] == truth[step["fixed_label"]] else -1
        rows.append(
            {
                "instance_type": instance_type,
                "seed": seed,
                "step_idx": step_idx,
                "n_before": step["n_before"],
                "margin": step["margin"],
                "abs_top_corr": abs(step["top_corr"]),
                "sign_correct": int(step["sign"] == predicted_sign),
            }
        )
    return rows


def synthetic_pass() -> None:
    print(f"\n=== B: synthetic path_like/random_sparse, N={N}, {len(SEEDS)} seeds/type ===")
    work_items = [(instance_type, seed) for instance_type in GENERATORS for seed in SEEDS]

    rows: list[dict] = []
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(_synthetic_one, instance_type, seed) for instance_type, seed in work_items]
        for future in futures:
            rows.extend(future.result())

    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "investigate_step_confidence_signals_synthetic.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved -> {csv_path} ({len(df)} steps)")

    for col in ("margin", "abs_top_corr"):
        print(f"\n--- {col} vs sign_correct ---")
        print(f"pooled correlation: {df[col].corr(df['sign_correct']):.4f}")
        df["_bin"] = pd.qcut(df[col], 4, duplicates="drop")
        summary = df.groupby("_bin", observed=True).agg(
            sign_error_rate=("sign_correct", lambda s: 1 - s.mean()), n=("sign_correct", "size")
        )
        print(summary.to_string())
    df.drop(columns="_bin", inplace=True)

    print("\nBy instance_type x sign_correct (does either signal separate right from wrong?):")
    print(
        df.groupby(["instance_type", "sign_correct"])[["margin", "abs_top_corr"]]
        .agg(["mean", "count"])
        .to_string()
    )


def real_data_pass() -> None:
    print(f"\n=== A: known-failing real SC/VSC periods, single k=1 run each ===")
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

        history: list[dict] = []
        solve_rqaoa(qubo, reps=REPS, min_size=MIN_SIZE, maxiter=MAXITER, seed=0, history_out=history)

        margins = [step["margin"] for step in history]
        abs_corrs = [abs(step["top_corr"]) for step in history]

        row = {
            "year": year, "gp": gp, "start_lap": start_lap,
            "k1_ratio": k1_ratio, "k5_ratio": k5_ratio,
            "n_steps": len(history),
            "min_margin": float(np.min(margins)),
            "mean_margin": float(np.mean(margins)),
            "min_abs_top_corr": float(np.min(abs_corrs)),
            "mean_abs_top_corr": float(np.mean(abs_corrs)),
        }
        rows.append(row)
        print(
            f"  {year} {gp} lap {start_lap}: k1={k1_ratio!s:>10} k5={k5_ratio:.4f}  "
            f"min_margin={row['min_margin']:.4f} mean_margin={row['mean_margin']:.4f} "
            f"min_abs_top_corr={row['min_abs_top_corr']:.4f} mean_abs_top_corr={row['mean_abs_top_corr']:.4f}"
        )

    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "investigate_step_confidence_signals_real.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved -> {csv_path}")
    print(df.to_string(index=False))


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    real_data_pass()
    synthetic_pass()


if __name__ == "__main__":
    main()
