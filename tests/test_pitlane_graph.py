import pandas as pd
import pytest

from f1qps.data.pitlane_graph import build_pitlane_conflict_graph, build_real_qubo


def _decision_laps(gaps_seconds: list[float]) -> pd.DataFrame:
    """n drivers in running order with given consecutive gaps (seconds)."""
    n = len(gaps_seconds) + 1
    times = [0.0]
    for gap in gaps_seconds:
        times.append(times[-1] + gap)
    return pd.DataFrame(
        {
            "Driver": [f"D{i}" for i in range(n)],
            "Position": list(range(1, n + 1)),
            "Time": pd.to_timedelta(times, unit="s"),
        }
    )


def test_close_gap_creates_edge_far_gap_does_not():
    laps = _decision_laps([1.0, 10.0])  # D0-D1 close (queue risk), D1-D2 far apart
    drivers, J = build_pitlane_conflict_graph(laps, queue_threshold_seconds=3.0)

    assert drivers == ["D0", "D1", "D2"]
    assert J[0, 1] > 0
    assert J[1, 2] == 0


def test_graph_is_sparse_path_like_for_realistic_gaps():
    laps = _decision_laps([0.5, 0.8, 1.2, 15.0, 0.9])
    drivers, J = build_pitlane_conflict_graph(laps, queue_threshold_seconds=3.0)
    # only adjacent-in-running-order pairs can ever be nonzero
    for i in range(len(drivers)):
        for j in range(len(drivers)):
            if abs(i - j) > 1:
                assert J[i, j] == 0


def test_build_real_qubo_combines_h_and_conflict_graph():
    laps = _decision_laps([1.0, 1.0])
    h_by_driver = {"D0": 0.5, "D1": 0.2, "D2": 0.9}
    qubo = build_real_qubo(laps, h_by_driver, queue_threshold_seconds=3.0)

    assert qubo.labels == ["D0", "D1", "D2"]
    assert qubo.h.tolist() == pytest.approx([0.5, 0.2, 0.9])
    assert qubo.J[0, 1] > 0
    assert qubo.J[1, 2] > 0
