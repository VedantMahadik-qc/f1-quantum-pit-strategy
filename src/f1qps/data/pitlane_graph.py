"""Pit-lane throughput conflict graph: J_ij from track-position gaps at the
SC/VSC decision lap.

Every pair of drivers is a CANDIDATE edge, scored by their on-track time gap
at the decision lap; only pairs under the queueing threshold actually get a
nonzero J_ij. This is deliberate: if edges were restricted to running-order
neighbors by construction, a sparse/path-like result would be guaranteed by
construction rather than a discoverable fact about SC/VSC periods -- under a
safety car the whole field can bunch up, so two cars far apart in finishing
order could in principle still end up close enough on-track to conflict.
Whether the result ends up sparse/path-like is therefore a real empirical
question, not a tautology.
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
    """Returns (drivers_in_running_order, J). Every pair (i, j) is scored by
    their on-track time gap at the decision lap; J[i, j] is nonzero only if
    that gap is below `queue_threshold_seconds` (the realistic pit-entry
    queueing window). Running order is used only to label/order the output,
    not to restrict which pairs can conflict.
    """
    df = decision_laps.dropna(subset=["Position", "Time"]).sort_values("Position")
    drivers = df["Driver"].tolist()
    n = len(drivers)
    times = df["Time"].dt.total_seconds().to_numpy()

    J = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            gap = abs(times[j] - times[i])
            if gap < queue_threshold_seconds:
                conflict = scale / max(gap, MIN_GAP_SECONDS)
                J[i, j] = conflict
                J[j, i] = conflict
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
