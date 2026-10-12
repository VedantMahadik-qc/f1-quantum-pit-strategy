"""Survey several real races for SC/VSC periods, to see whether the
sparse/path-like conflict-graph finding from the 2023 Singapore GP holds up
more broadly, or was a one-off. Network-dependent; not run by the test suite.

No quantum methods here on purpose -- real N can reach ~20 and there are
multiple races/periods, which would blow up runtime. This is the structural
survey; a real-data RQAOA run is a follow-up once this confirms the pattern.

Usage:
    python scripts/survey_real_races.py
"""
from __future__ import annotations

import os

import pandas as pd

from f1qps.analysis.graph_structure import analyze
from f1qps.classical.exact import solve_exact
from f1qps.classical.heuristics import solve_greedy
from f1qps.data.degradation import compute_h, fit_degradation_curves
from f1qps.data.fastf1_loader import (
    enable_cache,
    find_sc_vsc_periods,
    green_flag_laps,
    laps_in_period,
    load_race,
)
from f1qps.data.pitlane_graph import build_real_qubo

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

EXACT_N_CUTOFF = 16  # brute force stays fast up to here; skip beyond

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def main() -> None:
    enable_cache()
    rows = []

    for year, gp in CANDIDATE_RACES:
        print(f"\n{year} {gp}...")
        try:
            session = load_race(year, gp)
        except Exception as exc:  # network/data issues for a given race shouldn't kill the survey
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
            _, greedy_val = solve_greedy(qubo)
            exact_val = None
            if qubo.n <= EXACT_N_CUTOFF:
                _, exact_val = solve_exact(qubo)

            print(f"  lap {period.start_lap}: {report.summary()}")
            rows.append(
                {
                    "year": year,
                    "gp": gp,
                    "start_lap": period.start_lap,
                    "n": report.n_nodes,
                    "n_edges": report.n_edges,
                    "density": report.density,
                    "max_degree": max(report.degree_sequence) if report.degree_sequence else 0,
                    "approx_treewidth": report.approx_treewidth,
                    "greedy_objective": greedy_val,
                    "exact_objective": exact_val,
                }
            )

    if not rows:
        print("\nNo SC/VSC periods found in any candidate race.")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "real_race_graph_structure.csv")
    df.to_csv(csv_path, index=False)

    print(f"\nSaved {len(df)} real SC/VSC period(s) to {csv_path}")
    print("\nSummary:")
    print(df[["year", "gp", "start_lap", "n", "density", "max_degree", "approx_treewidth"]].to_string(index=False))


if __name__ == "__main__":
    main()
