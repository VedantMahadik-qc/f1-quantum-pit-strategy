"""Pit-lane throughput conflict graph: J_ij from track-position gaps at the
SC/VSC decision lap. Only drivers ADJACENT in running order can realistically
queue for pit entry at the same moment, so this only ever connects neighbors
in running order -- the mechanism that should produce a sparse/path-like
graph, rather than that shape being hardcoded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from f1qps.qubo.builder import QUBOInstance

DEFAULT_QUEUE_THRESHOLD_SECONDS = 3.0
MIN_GAP_SECONDS = 0.1


def build_pitlane_conflict_graph(
    decision_laps: pd.DataFrame,
    queue_threshold_seconds: float = DEFAULT_QUEUE_THRESHOLD_SECONDS,
    scale: float = 1.0,
) -> tuple[list[str], np.ndarray]:
    """Returns (drivers_in_running_order, J) where J[k, k+1] is nonzero only
    if the on-track gap between running-order neighbors k and k+1 is below
    `queue_threshold_seconds` (the realistic pit-entry queueing window).
    """
    df = decision_laps.dropna(subset=["Position", "Time"]).sort_values("Position")
    drivers = df["Driver"].tolist()
    n = len(drivers)
    times = df["Time"].dt.total_seconds().to_numpy()

    J = np.zeros((n, n))
    for k in range(n - 1):
        gap = abs(times[k + 1] - times[k])
        if gap < queue_threshold_seconds:
            conflict = scale / max(gap, MIN_GAP_SECONDS)
            J[k, k + 1] = conflict
            J[k + 1, k] = conflict
    return drivers, J


def build_real_qubo(
    decision_laps: pd.DataFrame,
    h_by_driver: dict[str, float],
    queue_threshold_seconds: float = DEFAULT_QUEUE_THRESHOLD_SECONDS,
    scale: float = 1.0,
) -> QUBOInstance:
    """Combine the degradation-fit h_i with the pit-lane conflict graph into
    one QUBOInstance, using running order (from the conflict graph) as the
    canonical driver ordering.
    """
    drivers, J = build_pitlane_conflict_graph(decision_laps, queue_threshold_seconds, scale)
    h = np.array([h_by_driver.get(d, 0.0) for d in drivers], dtype=float)
    return QUBOInstance(h=h, J=J, labels=drivers)
