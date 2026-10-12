"""QUBO instance representation for the SC/VSC pit-decision problem.

Decision variables x_i in {0,1}, x_i = 1 means car i pits this lap.

    C(x) = -sum_i h_i * x_i + sum_{i<j} J_ij * x_i * x_j      (minimize)

h_i >= 0: benefit of pitting now (degradation cost avoided).
J_ij >= 0: pit-lane queue conflict penalty if i and j both pit this lap.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class QUBOInstance:
    """A QUBO instance: minimize x^T Q x over x in {0,1}^n.

    Q is stored as a symmetric matrix split so that x^T Q x reproduces
    C(x) above: Q[i,i] = -h_i, and J_ij is split evenly onto Q[i,j]
    and Q[j,i] (Q[i,j] = Q[j,i] = J_ij / 2) so that
    x^T Q x = sum_i Q_ii x_i + 2 * sum_{i<j} Q_ij x_i x_j = C(x).
    """

    h: np.ndarray  # shape (n,)
    J: np.ndarray  # shape (n, n), symmetric, zero diagonal, J[i,j] = J[j,i] >= 0
    labels: list[str] | None = None

    def __post_init__(self) -> None:
        h = np.asarray(self.h, dtype=float)
        J = np.asarray(self.J, dtype=float)
        n = h.shape[0]
        if J.shape != (n, n):
            raise ValueError(f"J must be ({n},{n}), got {J.shape}")
        if not np.allclose(J, J.T):
            raise ValueError("J must be symmetric")
        if not np.allclose(np.diag(J), 0):
            raise ValueError("J must have zero diagonal")
        self.h = h
        self.J = J
        if self.labels is None:
            self.labels = [f"x{i}" for i in range(n)]
        elif len(self.labels) != n:
            raise ValueError("labels length must match n")

    @property
    def n(self) -> int:
        return self.h.shape[0]

    @property
    def Q(self) -> np.ndarray:
        """Symmetric matrix Q such that x^T Q x == objective(x)."""
        return np.diag(-self.h) + self.J / 2.0

    def objective(self, x: np.ndarray) -> float:
        """C(x) = -sum_i h_i x_i + sum_{i<j} J_ij x_i x_j."""
        x = np.asarray(x, dtype=float)
        linear = -np.dot(self.h, x)
        quadratic = 0.5 * x @ self.J @ x  # J symmetric, zero diag -> sum_{i<j} J_ij x_i x_j * 2 / 2
        return float(linear + quadratic)

    def to_ising(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Convert to Ising form via x_i = (1 - z_i) / 2, z_i in {-1,+1}.

        Returns (field, coupling, offset) such that

            C(x) == offset + sum_i field_i * z_i + sum_{i<j} coupling_ij * z_i * z_j
        """
        n = self.n
        h, J = self.h, self.J

        # x_i = (1 - z_i)/2
        # -h_i x_i = -h_i/2 + (h_i/2) z_i
        field = h.copy() / 2.0
        offset = -0.5 * np.sum(h)

        # J_ij x_i x_j = J_ij/4 * (1 - z_i)(1 - z_j)
        #              = J_ij/4 * (1 - z_i - z_j + z_i z_j)
        coupling = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                Jij = J[i, j]
                if Jij == 0.0:
                    continue
                offset += Jij / 4.0
                field[i] -= Jij / 4.0
                field[j] -= Jij / 4.0
                coupling[i, j] = Jij / 4.0
                coupling[j, i] = Jij / 4.0

        return field, coupling, offset

    def reduce(self, eliminate: int, fixed_to: int, sign: int) -> "QUBOInstance":
        """Eliminate variable `eliminate` by substituting x_eliminate = x_fixed_to
        if sign == +1, or x_eliminate = 1 - x_fixed_to if sign == -1.

        Used by RQAOA's variable-elimination step. Returns a new, smaller
        QUBOInstance over the remaining n-1 variables (original ordering
        preserved, `eliminate` removed).
        """
        if sign not in (1, -1):
            raise ValueError("sign must be +1 or -1")
        n = self.n
        keep = [i for i in range(n) if i != eliminate]
        h_new = self.h[keep].copy()
        J_new = self.J[np.ix_(keep, keep)].copy()
        labels_new = [self.labels[i] for i in keep]

        fixed_pos = keep.index(fixed_to)
        J_row = self.J[eliminate, keep]  # coupling from eliminated var to each remaining var

        if sign == 1:
            # x_elim = x_fixed. The J_elim,fixed term becomes linear (x_fixed^2 = x_fixed),
            # so it folds into h_new[fixed], not into J_new.
            h_new[fixed_pos] += self.h[eliminate] - J_row[fixed_pos]
            for pos, k in enumerate(keep):
                if k == fixed_to:
                    continue
                J_new[fixed_pos, pos] += J_row[pos]
                J_new[pos, fixed_pos] += J_row[pos]
        else:
            # x_elim = 1 - x_fixed
            # (constant term -h_elim dropped; objective differs by a constant, fine for argmin)
            h_new[fixed_pos] -= self.h[eliminate]
            # J_elim,k * x_elim * x_k = J_elim,k * (1 - x_fixed) * x_k
            #   = J_elim,k * x_k - J_elim,k * x_fixed * x_k
            for pos, k in enumerate(keep):
                if k == fixed_to:
                    continue
                h_new[pos] -= J_row[pos]
                J_new[fixed_pos, pos] -= J_row[pos]
                J_new[pos, fixed_pos] -= J_row[pos]

        return QUBOInstance(h=h_new, J=J_new, labels=labels_new)
