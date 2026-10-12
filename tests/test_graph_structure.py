import numpy as np

from f1qps.analysis.graph_structure import analyze
from f1qps.qubo.builder import QUBOInstance
from f1qps.qubo.synthetic import path_like_instance


def test_path_like_instance_has_treewidth_one():
    qubo = path_like_instance(n=6, seed=0)
    report = analyze(qubo)
    assert report.n_nodes == 6
    assert report.n_edges == 5  # path on 6 nodes has 5 edges
    assert max(report.degree_sequence) <= 2
    assert report.approx_treewidth <= 1


def test_fully_connected_instance_has_higher_treewidth():
    n = 6
    h = np.ones(n)
    J = np.ones((n, n)) - np.eye(n)
    qubo = QUBOInstance(h=h, J=J)
    report = analyze(qubo)
    assert report.n_edges == n * (n - 1) // 2
    assert report.approx_treewidth == n - 1
