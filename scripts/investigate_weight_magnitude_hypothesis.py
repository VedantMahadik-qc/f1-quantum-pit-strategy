"""Follow-up to run_real_race_experiment.py's full 17-period real-race sweep,
where RQAOA failed on 6/17 periods but that failure didn't correlate with
any topology statistic (max_degree, density, treewidth all ~0 correlation
with RQAOA's approx_ratio). Those stats only describe which J_ij are
NONZERO; this script tests the leftover hypothesis flagged in the README:
that the actual MAGNITUDES of J_ij and h_i (and the resulting energy
landscape's spectral gap) predict RQAOA failure where topology doesn't.

No new quantum solver runs here -- reuses the RQAOA approx_ratio already
computed and saved in results/real_race_solver_comparison.csv, and just
adds weight/landscape statistics per real QUBO instance (rebuilt from the
same cached FastF1 data, same pipeline as run_real_race_experiment.py) to
correlate against it.

Not tested on the synthetic path_like/random_sparse instances: both
generators draw h_i and J_ij from the same U(0,1) distribution regardless
of N/seed/instance_type (see qubo/synthetic.py) -- there's no weight-scale
variation to test there, only topology varies. Real data is the only place
this hypothesis can actually be probed.

Usage:
    python scripts/investigate_weight_magnitude_hypothesis.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from f1qps.data.degradation import compute_h, fit_degradation_curves
from f1qps.data.fastf1_loader import (
    enable_cache,
    find_sc_vsc_periods,
    green_flag_laps,
    laps_in_period,
    load_race,
)
from f1qps.data.pitlane_graph import build_real_qubo
from f1qps.qubo.builder import QUBOInstance

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
SOLVER_CSV = os.path.join(RESULTS_DIR, "real_race_solver_comparison.csv")
EPS = 1e-6


def full_spectrum(qubo: QUBOInstance) -> np.ndarray:
    """All 2^n objective values, vectorized (fine up to the n<=19 real range)."""
    n = qubo.n
    bits = ((np.arange(2**n)[:, None] >> np.arange(n)[None, :]) & 1).astype(float)
    linear = -bits @ qubo.h
    quadratic = 0.5 * np.einsum("ij,ij->i", bits @ qubo.J, bits)
    return linear + quadratic


def weight_stats(qubo: QUBOInstance) -> dict:
    h = qubo.h
    nonzero_J = qubo.J[np.triu_indices(qubo.n, k=1)]
    nonzero_J = nonzero_J[nonzero_J != 0.0]

    h_abs = np.abs(h)
    j_abs = np.abs(nonzero_J) if nonzero_J.size else np.array([0.0])

    spectrum = np.sort(full_spectrum(qubo))
    best = spectrum[0]
    distinct = spectrum[np.concatenate([[True], np.diff(spectrum) > EPS])]
    second_best = distinct[1] if len(distinct) > 1 else best
    gap_abs = second_best - best
    scale = max(abs(best), h_abs.mean() if h_abs.size else 0.0, j_abs.mean() if j_abs.size else 0.0, EPS)

    return {
        "h_mean_abs": h_abs.mean(),
        "h_cv": h.std() / h_abs.mean() if h_abs.mean() > EPS else np.nan,
        "j_mean_abs": j_abs.mean(),
        "j_cv": nonzero_J.std() / j_abs.mean() if nonzero_J.size and j_abs.mean() > EPS else np.nan,
        "j_over_h_scale": j_abs.mean() / h_abs.mean() if h_abs.mean() > EPS else np.nan,
        "spectral_gap_abs": gap_abs,
        "spectral_gap_rel": gap_abs / scale,
        "n_near_degenerate": int(np.sum(spectrum < best + EPS)),
    }


def main() -> None:
    enable_cache()
    rows = []

    for year, gp in CANDIDATE_RACES:
        try:
            session = load_race(year, gp)
        except Exception as exc:
            print(f"{year} {gp}: skipped ({exc})")
            continue

        periods = find_sc_vsc_periods(session)
        curves = fit_degradation_curves(green_flag_laps(session))

        for period in periods:
            decision_laps = laps_in_period(session, period)
            h = compute_h(decision_laps, curves)
            qubo = build_real_qubo(decision_laps, h)
            if qubo.n < 2:
                continue

            stats = weight_stats(qubo)
            stats.update({"year": year, "gp": gp, "start_lap": period.start_lap})
            rows.append(stats)
            print(f"{year} {gp} lap {period.start_lap}: {stats}")

    weight_df = pd.DataFrame(rows)
    solver_df = pd.read_csv(SOLVER_CSV)
    rqaoa_df = solver_df[solver_df["method"] == "rqaoa"][
        ["year", "gp", "start_lap", "approx_ratio"]
    ]

    merged = weight_df.merge(rqaoa_df, on=["year", "gp", "start_lap"])
    csv_path = os.path.join(RESULTS_DIR, "real_race_weight_stats.csv")
    merged.to_csv(csv_path, index=False)
    print(f"\nSaved -> {csv_path}")

    # Instances where h is all-zero (nobody benefits from pitting, exact optimum
    # is trivially 0) are a different, degenerate regime -- exclude from the
    # correlation so a handful of near-arbitrary huge/zero gap values don't
    # drown out the signal from genuinely contested instances.
    nontrivial = merged[merged["h_mean_abs"] > EPS]
    print(f"\n{len(merged) - len(nontrivial)} of {len(merged)} periods are degenerate "
          f"(h all-zero, nothing to gain from pitting) -- excluded from correlation below.")

    stat_cols = [
        "h_mean_abs", "h_cv", "j_mean_abs", "j_cv", "j_over_h_scale",
        "spectral_gap_abs", "spectral_gap_rel", "n_near_degenerate",
    ]
    print(f"\nCorrelation of RQAOA approx_ratio against weight/landscape statistics (n={len(nontrivial)}):")
    for col in stat_cols:
        corr = nontrivial[col].corr(nontrivial["approx_ratio"])
        print(f"  {col:20s} corr = {corr: .3f}")

    print("\nNon-trivial instances, sorted by approx_ratio:")
    print(nontrivial.sort_values("approx_ratio")[["year", "gp", "start_lap", "approx_ratio"] + stat_cols].to_string(index=False))


if __name__ == "__main__":
    main()
