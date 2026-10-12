"""One-off, network-dependent: fetch a real race via FastF1, find its SC/VSC
periods, build a real QUBO instance for the first one, and report its
conflict-graph structure. Not run by the test suite.

Usage:
    python scripts/fetch_sc_data.py 2023 Bahrain
"""
from __future__ import annotations

import sys

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


def main() -> None:
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2023
    gp = sys.argv[2] if len(sys.argv) > 2 else "Bahrain"

    enable_cache()
    print(f"Loading {year} {gp} race session...")
    session = load_race(year, gp)

    periods = find_sc_vsc_periods(session)
    print(f"Found {len(periods)} SC/VSC period(s):")
    for p in periods:
        print(f"  laps {p.start_lap}-{p.end_lap}")
    if not periods:
        print("No SC/VSC periods in this race -- try a different one.")
        return

    period = periods[0]
    decision_laps = laps_in_period(session, period)
    print(f"\nDecision point: lap {period.start_lap}, {len(decision_laps)} cars")

    curves = fit_degradation_curves(green_flag_laps(session))
    print(f"Fitted degradation curves for compounds: {list(curves.keys())}")

    h = compute_h(decision_laps, curves)
    qubo = build_real_qubo(decision_laps, h)
    print(f"\nBuilt real QUBO: N={qubo.n} cars")
    print("h_i (pit benefit):", {lbl: round(v, 3) for lbl, v in zip(qubo.labels, qubo.h)})

    report = analyze(qubo)
    print(f"\nConflict graph structure: {report.summary()}")


if __name__ == "__main__":
    main()
