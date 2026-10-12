"""Plots for the scaling study: approximation ratio and wall-clock vs N,
split by instance type (path_like vs random_sparse) so the sparse-structure
hypothesis is directly visible rather than averaged away.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

# Colors are the dataviz skill's validated categorical slots 1/2/3
# (blue/orange/aqua) in their documented fixed order -- this exact triple
# passes CVD + normal-vision separation checks on all pairs, not just
# adjacent ones. Marker shapes carry identity too, so color is never the
# only distinguishing channel between lines.
METHOD_STYLE = {
    "qaoa": {"color": "#2a78d6", "marker": "o"},
    "greedy": {"color": "#eb6834", "marker": "s"},
    "rqaoa": {"color": "#1baf7a", "marker": "^"},
}


def plot_approx_ratio_vs_n(df: pd.DataFrame, out_path: str) -> None:
    instance_types = sorted(df["instance_type"].unique())
    fig, axes = plt.subplots(1, len(instance_types), figsize=(6 * len(instance_types), 4.5), sharey=True)
    if len(instance_types) == 1:
        axes = [axes]

    for ax, instance_type in zip(axes, instance_types):
        sub = df[df["instance_type"] == instance_type]
        for method in [m for m in METHOD_STYLE if m in sub["method"].unique()]:
            agg = sub[sub["method"] == method].groupby("N")["approx_ratio"].agg(["mean", "std"])
            style = METHOD_STYLE[method]
            ax.errorbar(
                agg.index, agg["mean"], yerr=agg["std"], label=method, capsize=3, **style
            )
        ax.set_title(instance_type)
        ax.set_xlabel("N (cars)")
        ax.axhline(1.0, color="black", linewidth=0.5, linestyle="--")
        ax.legend()
    axes[0].set_ylabel("approximation ratio (mean +/- std over seeds)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_wallclock_vs_n(df: pd.DataFrame, out_path: str) -> None:
    instance_types = sorted(df["instance_type"].unique())
    fig, axes = plt.subplots(1, len(instance_types), figsize=(6 * len(instance_types), 4.5), sharey=True)
    if len(instance_types) == 1:
        axes = [axes]

    for ax, instance_type in zip(axes, instance_types):
        sub = df[df["instance_type"] == instance_type]
        for method in [m for m in METHOD_STYLE if m in sub["method"].unique()]:
            agg = sub[sub["method"] == method].groupby("N")["wall_clock"].agg(["mean", "std"])
            style = METHOD_STYLE[method]
            ax.errorbar(
                agg.index, agg["mean"], yerr=agg["std"], label=method, capsize=3, **style
            )
        ax.set_title(instance_type)
        ax.set_xlabel("N (cars)")
        ax.set_yscale("log")
        ax.legend()
    axes[0].set_ylabel("wall-clock seconds (mean +/- std over seeds, log scale)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
