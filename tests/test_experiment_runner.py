import pandas as pd

from f1qps.experiments.runner import run_instance, sweep_synthetic
from f1qps.qubo.synthetic import path_like_instance


def test_run_instance_exact_and_greedy_only():
    qubo = path_like_instance(n=6, seed=0)
    results = run_instance(qubo, methods=("exact", "greedy"))

    assert set(results.keys()) == {"exact", "greedy"}
    for data in results.values():
        assert "objective" in data and "wall_clock" in data
        assert data["wall_clock"] >= 0


def test_sweep_synthetic_shape_and_columns_exact_greedy_only():
    df = sweep_synthetic(
        n_values=[4, 6],
        seeds=[0, 1],
        instance_types=("path_like",),
        methods=("exact", "greedy"),
    )

    expected_cols = {
        "instance_type",
        "N",
        "seed",
        "method",
        "objective",
        "wall_clock",
        "approx_ratio",
    }
    assert expected_cols.issubset(df.columns)
    # 2 N values * 2 seeds * 1 instance type * 2 methods
    assert len(df) == 8
    assert set(df["method"].unique()) == {"exact", "greedy"}

    # exact should always match itself -> approx_ratio 1.0
    exact_rows = df[df["method"] == "exact"]
    assert (exact_rows["approx_ratio"] == 1.0).all()

    # greedy should never beat exact (ratio <= 1 given both are <= 0)
    greedy_rows = df[df["method"] == "greedy"]
    assert (greedy_rows["approx_ratio"] <= 1.0 + 1e-9).all()


def test_sweep_synthetic_max_workers_matches_sequential_result():
    kwargs = dict(n_values=[4, 6], seeds=[0, 1], instance_types=("path_like",), methods=("exact", "greedy"))

    sequential = sweep_synthetic(**kwargs)
    parallel = sweep_synthetic(**kwargs, max_workers=2)

    sort_cols = ["instance_type", "N", "seed", "method"]
    compare_cols = ["instance_type", "N", "seed", "method", "objective", "approx_ratio"]
    pd.testing.assert_frame_equal(
        sequential.sort_values(sort_cols).reset_index(drop=True)[compare_cols],
        parallel.sort_values(sort_cols).reset_index(drop=True)[compare_cols],
    )
