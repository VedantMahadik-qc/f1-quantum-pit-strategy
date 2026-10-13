"""Follow-up to survey_real_races.py, which only looked at real SC/VSC
conflict-graph STRUCTURE (confirming it's sparse/path-like, N~15-19, density
0.02-0.15, treewidth 1-3 across 8 candidate races) without running any
quantum method on it. Every RQAOA result so far (the N=4-16 scaling study,
the depth-rescue study, the elimination-correctness diagnostic) used
SYNTHETIC path_like/random_sparse instances as a proxy for that structure --
this is the first run of the full exact/greedy/QAOA/RQAOA comparison on an
actual real-race-derived QUBO.

Two periods, chosen from survey_real_races.py's output:
  - 2023 Australian GP, lap 7: N=18, density=0.150, max_degree=6,
    approx_treewidth=3 -- the structurally "hardest" (densest, most
    path-like-violating) real period found in the survey. If RQAOA holds up
    here, that's the strongest available validation of the depth-rescue
    (reps=3) result on real data.
  - 2019 German GP, lap 56: N=15, density=0.114, nontrivial exact objective
    already on record (-0.328720) -- a second, more typical real instance
    for contrast.

reps=3, maxiter=120 matches the config established as reliable for
path-like/sparse structure in the depth-rescue + elimination-correctness
studies. Both races are already in .fastf1_cache (no live network fetch).

Usage:
    python scripts/run_real_race_experiment.py
"""
from __future__ import annotations

import os

import pandas as pd

from f1qps.analysis.graph_structure import analyze
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
from f1qps.experiments.runner import run_instance
from f1qps.qubo.builder import QUBOInstance

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
TARGETS = [
    (2023, "Australia", 7),
    (2019, "Germany", 56),
]
REPS = 3
MAXITER = 120
MIN_SIZE = 3
SEED = 0


def build_instance(year: int, gp: str, start_lap: int) -> QUBOInstance:
    session = load_race(year, gp)
    periods = find_sc_vsc_periods(session)
    (period,) = [p for p in periods if p.start_lap == start_lap]

    decision_laps = laps_in_period(session, period)
    curves = fit_degradation_curves(green_flag_laps(session))
    h = compute_h(decision_laps, curves)
    return build_real_qubo(decision_laps, h)


def main() -> None:
    enable_cache()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []

    for year, gp, start_lap in TARGETS:
        print(f"\n=== {year} {gp}, lap {start_lap} ===")
        qubo = build_instance(year, gp, start_lap)
        report = analyze(qubo)
        print(f"Structure: {report.summary()}")

        results = run_instance(
            qubo,
            reps=REPS,
            maxiter=MAXITER,
            min_size=MIN_SIZE,
            seed=SEED,
            methods=("exact", "greedy", "qaoa", "rqaoa"),
        )
        exact_val = results["exact"]["objective"]
        for method, data in results.items():
            ratio = approx_ratio(data["objective"], exact_val)
            print(f"  {method:8s} objective={data['objective']: .4f}  "
                  f"approx_ratio={ratio:.4f}  wall_clock={data['wall_clock']:.2f}s")
            rows.append(
                {
                    "year": year,
                    "gp": gp,
                    "start_lap": start_lap,
                    "n": qubo.n,
                    "density": report.density,
                    "method": method,
                    "objective": data["objective"],
                    "approx_ratio": ratio,
                    "wall_clock": data["wall_clock"],
                }
            )

    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "real_race_solver_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved -> {csv_path}")


if __name__ == "__main__":
    main()
