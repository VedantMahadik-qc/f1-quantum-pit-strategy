"""Follow-up to survey_real_races.py, which only looked at real SC/VSC
conflict-graph STRUCTURE (confirming it's sparse/path-like, N~15-19, density
0.02-0.15, treewidth 1-3 across 8 candidate races) without running any
quantum method on it. Every RQAOA result before this script (the N=4-16
scaling study, the depth-rescue study, the elimination-correctness
diagnostic) used SYNTHETIC path_like/random_sparse instances as a proxy for
that structure.

A first pass of this script (2 hand-picked periods) found a nuance worth
checking against more data: on 2019 German GP lap 56 (N=15, density 0.11)
every method matched the true optimum exactly, but on 2023 Australian GP lap
7 (N=18, density 0.15 -- the structurally "hardest" real period in the
survey) RQAOA underperformed (ratio 0.83) despite low *global*
density/treewidth, because that graph has a local hub node (degree 6 of 18).
That's one data point for "local hub degree predicts RQAOA failure better
than aggregate density/treewidth does" -- this run broadens it to every real
SC/VSC period found across all 8 candidate races, to see whether the
hub-degree pattern holds up or Australia lap 7 was a fluke.

reps=3, maxiter=120 matches the config established as reliable for
path-like/sparse structure in the depth-rescue + elimination-correctness
studies. All 8 races are already in .fastf1_cache (no live network fetch).

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

CANDIDATE_RACES = [
    (2023, "Singapore"),
    (2023, "Qatar"),
    (2023, "Australia"),
    (2022, "Japan"),
    (2021, "Azerbaijan"),
    (2021, "Hungary"),
    (2019, "Germany"),
    (2018, "Germany"),
]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
REPS = 3
MAXITER = 120
MIN_SIZE = 3
SEED = 0


def main() -> None:
    enable_cache()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []

    for year, gp in CANDIDATE_RACES:
        print(f"\n{year} {gp}...")
        try:
            session = load_race(year, gp)
        except Exception as exc:
            print(f"  skipped: {exc}")
            continue

        periods = find_sc_vsc_periods(session)
        if not periods:
            print("  no SC/VSC periods")
            continue

        curves = fit_degradation_curves(green_flag_laps(session))

        for period in periods:
            decision_laps = laps_in_period(session, period)
            h = compute_h(decision_laps, curves)
            qubo = build_real_qubo(decision_laps, h)
            if qubo.n < 2:
                continue

            report = analyze(qubo)
            print(f"  lap {period.start_lap}: {report.summary()}")

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
                print(f"    {method:8s} objective={data['objective']: .4f}  "
                      f"approx_ratio={ratio:.4f}  wall_clock={data['wall_clock']:.2f}s")
                rows.append(
                    {
                        "year": year,
                        "gp": gp,
                        "start_lap": period.start_lap,
                        "n": qubo.n,
                        "density": report.density,
                        "max_degree": max(report.degree_sequence) if report.degree_sequence else 0,
                        "approx_treewidth": report.approx_treewidth,
                        "method": method,
                        "objective": data["objective"],
                        "approx_ratio": ratio,
                        "wall_clock": data["wall_clock"],
                    }
                )

    if not rows:
        print("\nNo SC/VSC periods found in any candidate race.")
        return

    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "real_race_solver_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} (period, method) row(s) -> {csv_path}")

    rqaoa = df[df["method"] == "rqaoa"][
        ["year", "gp", "start_lap", "n", "density", "max_degree", "approx_treewidth", "approx_ratio"]
    ].sort_values("max_degree")
    print("\nRQAOA approx_ratio vs. structure, sorted by max_degree:")
    print(rqaoa.to_string(index=False))
    print(f"\ncorrelation(max_degree, rqaoa approx_ratio) = {rqaoa['max_degree'].corr(rqaoa['approx_ratio']):.3f}")
    print(f"correlation(density, rqaoa approx_ratio)     = {rqaoa['density'].corr(rqaoa['approx_ratio']):.3f}")
    print(f"correlation(treewidth, rqaoa approx_ratio)   = {rqaoa['approx_treewidth'].corr(rqaoa['approx_ratio']):.3f}")


if __name__ == "__main__":
    main()
