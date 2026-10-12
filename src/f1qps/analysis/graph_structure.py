"""Sparsity / treewidth check on a QUBO's conflict graph.

This is the cheap early test of the "sparse/path-like" hypothesis the whole
RQAOA-over-vanilla-QAOA choice rests on. Run this against real SC/VSC-derived
instances before investing further in later milestones.
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from f1qps.qubo.builder import QUBOInstance


@dataclass
class GraphStructureReport:
    n_nodes: int
    n_edges: int
    max_possible_edges: int
    density: float
    degree_sequence: list[int]
    approx_treewidth: int

    def summary(self) -> str:
        return (
            f"n={self.n_nodes}, edges={self.n_edges}/{self.max_possible_edges} "
            f"(density={self.density:.3f}), degrees={self.degree_sequence}, "
            f"approx_treewidth<={self.approx_treewidth}"
        )


def to_networkx(qubo: QUBOInstance) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(qubo.n))
    for i in range(qubo.n):
        for j in range(i + 1, qubo.n):
            if qubo.J[i, j] != 0.0:
                g.add_edge(i, j, weight=qubo.J[i, j])
    return g


def analyze(qubo: QUBOInstance) -> GraphStructureReport:
    g = to_networkx(qubo)
    n = qubo.n
    max_edges = n * (n - 1) // 2
    n_edges = g.number_of_edges()
    degrees = sorted((d for _, d in g.degree()), reverse=True)

    # min_fill_in heuristic gives an upper bound on treewidth; exact treewidth
    # is NP-hard, this is the standard practical approximation.
    _, decomp = nx.algorithms.approximation.treewidth_min_fill_in(g)
    approx_tw = max((len(bag) - 1 for bag in decomp.nodes()), default=0)

    return GraphStructureReport(
        n_nodes=n,
        n_edges=n_edges,
        max_possible_edges=max_edges,
        density=n_edges / max_edges if max_edges else 0.0,
        degree_sequence=degrees,
        approx_treewidth=approx_tw,
    )
