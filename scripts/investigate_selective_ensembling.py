"""Tests the untested idea written into README "What's Still Open" item 2:
can SELECTIVE ensembling -- only spending extra QAOA restarts on elimination
steps that a single restart's own `abs_top_corr` flags as risky, rather than
ensembling every step uniformly -- recover most of `ensemble_k=5`'s real-world
fix rate (investigate_rqaoa_ensemble_fix.py) at a fraction of its compute?

`abs_top_corr` (the magnitude of the correlation value RQAOA elects at a
step, from a SINGLE restart, no ensembling needed) was shown in
investigate_step_confidence_signals.py to be a real per-step confidence
signal: r=0.30, p=7.7e-11 against sign-correctness, with error rate 29%
below the median value (~0.5) vs 7% above it. `solve_rqaoa`'s new
`confidence_threshold` param (src/f1qps/quantum/rqaoa.py) acts on exactly
that: run one restart, and only spend the remaining `ensemble_k - 1`
restarts on a step if that first restart's own top correlation magnitude is
below the threshold. THRESHOLD=0.5 here is that same below/above-median
split point.

For each of the 6 known-failing real SC/VSC periods used throughout this
series, reruns k=1, k=5 (full ensembling), and k=5 (selective) FRESH in this
one pass -- rather than trusting the k1/k5 numbers already on record, since
this project has already found that library-version drift alone can shift
results (see README's path_like seed-bug note) -- and reports approx_ratio
plus total restart count (the compute-cost proxy) for all three.

Usage:
    python scripts/investigate_selective_ensembling.py
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
from f1qps.quantum.rqaoa import solve_rqaoa

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
REPS = 3
MAXITER = 120
MIN_SIZE = 3
ENSEMBLE_K = 5
THRESHOLD = 0.5

REAL_PERIODS = [
    (2023, "Singapore", 19),
    (2021, "Azerbaijan", 30),
    (2019, "Germany", 40),
    (2023, "Australia", 7),
    (2023, "Australia", 53),
    (2019, "Germany", 2),
]


def _run(qubo, exact_val: float, **kwargs) -> tuple[float, int, int]:
    history: list[dict] = []
    _, val = solve_rqaoa(qubo, reps=REPS, min_size=MIN_SIZE, maxiter=MAXITER, history_out=history, **kwargs)
    total_restarts = sum(step["restarts_used"] for step in history)
    return approx_ratio(val, exact_val), total_restarts, len(history)


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    enable_cache()
    session_cache: dict[tuple[int, str], object] = {}
    rows = []

    for year, gp, start_lap in REAL_PERIODS:
        if (year, gp) not in session_cache:
            session_cache[(year, gp)] = load_race(year, gp)
        session = session_cache[(year, gp)]

        (period,) = [p for p in find_sc_vsc_periods(session) if p.start_lap == start_lap]
        decision_laps = laps_in_period(session, period)
        curves = fit_degradation_curves(green_flag_laps(session))
        h = compute_h(decision_laps, curves)
        qubo = build_real_qubo(decision_laps, h)
        _, exact_val = solve_exact(qubo)

        k1_ratio, k1_restarts, n_steps = _run(qubo, exact_val, seed=0, ensemble_k=1)
        k5_ratio, k5_restarts, _ = _run(qubo, exact_val, seed=0, ensemble_k=ENSEMBLE_K)
        sel_ratio, sel_restarts, _ = _run(
            qubo, exact_val, seed=0, ensemble_k=ENSEMBLE_K, confidence_threshold=THRESHOLD
        )

        row = {
            "year": year, "gp": gp, "start_lap": start_lap, "n_steps": n_steps,
            "k1_ratio": k1_ratio, "k1_restarts": k1_restarts,
            "k5_ratio": k5_ratio, "k5_restarts": k5_restarts,
            "selective_ratio": sel_ratio, "selective_restarts": sel_restarts,
            "selective_pct_of_k5_cost": sel_restarts / k5_restarts,
        }
        rows.append(row)
        print(
            f"{year} {gp} lap {start_lap}: k1={k1_ratio:.3f} ({k1_restarts}r)  "
            f"k5_full={k5_ratio:.3f} ({k5_restarts}r)  "
            f"k5_selective={sel_ratio:.3f} ({sel_restarts}r, "
            f"{row['selective_pct_of_k5_cost'] * 100:.0f}% of full-k5 cost)"
        )

    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "investigate_selective_ensembling_real.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved -> {csv_path}")
    print(df.to_string(index=False))

    print(
        f"\nMean approx_ratio: k1={df['k1_ratio'].mean():.3f}  "
        f"k5_full={df['k5_ratio'].mean():.3f}  selective={df['selective_ratio'].mean():.3f}"
    )
    print(f"Selective's mean cost as a fraction of full k5: {df['selective_pct_of_k5_cost'].mean() * 100:.0f}%")


if __name__ == "__main__":
    main()
