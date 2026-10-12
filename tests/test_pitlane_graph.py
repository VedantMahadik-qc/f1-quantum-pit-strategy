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


def test_non_adjacent_pairs_can_still_conflict_if_their_gap_is_small():
    # D0-D1 gap 0.1s, D1-D2 gap 0.1s -> D0-D2 cumulative gap is 0.2s, still
    # well under the threshold. A bunched-up field (as under a safety car)
    # can put non-adjacent cars in conflict too -- this must NOT be zeroed
    # out by construction, or the "is the graph sparse" question becomes a
    # tautology instead of something the data can actually answer.
    laps = _decision_laps([0.1, 0.1])
    drivers, J = build_pitlane_conflict_graph(laps, queue_threshold_seconds=3.0)

    assert drivers == ["D0", "D1", "D2"]
    assert J[0, 2] > 0


def test_distant_pairs_get_no_edge():
    # cumulative times: D0=0, D1=0.5, D2=1.3, D3=2.5, D4=17.5, D5=18.4
    laps = _decision_laps([0.5, 0.8, 1.2, 15.0, 0.9])
    drivers, J = build_pitlane_conflict_graph(laps, queue_threshold_seconds=3.0)
    # D4 (after the 15s gap) is >= 15s from D0-D3, well over the threshold
    d4 = drivers.index("D4")
    for other in ("D0", "D1", "D2", "D3"):
        assert J[d4, drivers.index(other)] == 0
    # but still close to D5 (0.9s gap)
    assert J[d4, drivers.index("D5")] > 0


def test_build_real_qubo_combines_h_and_conflict_graph():
    laps = _decision_laps([1.0, 1.0])
    h_by_driver = {"D0": 0.5, "D1": 0.2, "D2": 0.9}
    qubo = build_real_qubo(laps, h_by_driver, queue_threshold_seconds=3.0)

    assert qubo.labels == ["D0", "D1", "D2"]
    assert qubo.h.tolist() == pytest.approx([0.5, 0.2, 0.9])
    assert qubo.J[0, 1] > 0
    assert qubo.J[1, 2] > 0
