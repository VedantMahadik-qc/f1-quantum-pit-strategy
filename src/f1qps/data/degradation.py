"""Fit per-compound tire degradation curves from green-flag laps, and use
them to estimate h_i: the lap-time cost avoided by pitting now rather than
continuing on an aging tire.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_SAMPLES_PER_COMPOUND = 5


@dataclass
class DegradationCurve:
    compound: str
    coeffs: np.ndarray  # quadratic fit: laptime(tyre_age) = coeffs[0]*age^2 + coeffs[1]*age + coeffs[2]
    n_samples: int

    def marginal_cost(self, tyre_age: float) -> float:
        """d(laptime)/d(age) at the given tyre age -- the extra time one more
        lap on this tire costs right now. Clipped at 0 (a fit that predicts
        improving pace with age reflects noise, not a real effect).
        """
        a, b, _c = self.coeffs
        derivative = 2 * a * tyre_age + b
        return max(derivative, 0.0)


def _laptime_seconds(laps: pd.DataFrame) -> pd.Series:
    return laps["LapTime"].dt.total_seconds()


def fit_degradation_curves(green_laps: pd.DataFrame) -> dict[str, DegradationCurve]:
    """One quadratic fit per tire compound, pooling all drivers' green-flag
    laps on that compound. Excludes the in/out laps implicitly via FastF1's
    own lap validity where present, and drops rows with missing data.
    """
    curves: dict[str, DegradationCurve] = {}
    df = green_laps.dropna(subset=["LapTime", "TyreLife", "Compound"])
    for compound, group in df.groupby("Compound"):
        if len(group) < MIN_SAMPLES_PER_COMPOUND:
            continue
        age = group["TyreLife"].to_numpy(dtype=float)
        laptime = _laptime_seconds(group).to_numpy(dtype=float)
        coeffs = np.polyfit(age, laptime, deg=2)
        curves[compound] = DegradationCurve(compound=compound, coeffs=coeffs, n_samples=len(group))
    return curves


def compute_h(
    decision_laps: pd.DataFrame, curves: dict[str, DegradationCurve]
) -> dict[str, float]:
    """h_i per driver in the SC/VSC decision lap, using each driver's current
    compound + tyre age against the fitted curve for that compound. Drivers
    on a compound with no fitted curve (too few green-flag samples) fall
    back to h_i = 0 (no information -> no bias either way).
    """
    h: dict[str, float] = {}
    for _, row in decision_laps.iterrows():
        driver = row["Driver"]
        compound = row.get("Compound")
        tyre_age = row.get("TyreLife")
        curve = curves.get(compound)
        if curve is None or pd.isna(tyre_age):
            h[driver] = 0.0
        else:
            h[driver] = curve.marginal_cost(float(tyre_age))
    return h
