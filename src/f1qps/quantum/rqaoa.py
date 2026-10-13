"""Recursive QAOA (RQAOA): repeatedly run QAOA, find the most-correlated
variable pair <Z_i Z_j>, and eliminate one by substituting it in terms of the
other -- shrinking the QUBO until it's small enough to solve exactly, then
back-substituting. Chosen over vanilla QAOA because it empirically performs
better on sparse/structured graphs (Bravyi, Kahanamoku-Meyer, Campbell, Chen
2019), which is the expected shape of the pit-lane conflict graph.
"""
from __future__ import annotations

import numpy as np

from f1qps.classical.exact import solve_exact
from f1qps.qubo.builder import QUBOInstance
from f1qps.quantum.qaoa import compute_correlations, optimize_qaoa_angles


def solve_rqaoa(
    qubo: QUBOInstance,
    reps: int = 1,
    min_size: int = 4,
    maxiter: int = 200,
    seed: int | None = None,
    history_out: list[dict] | None = None,
) -> tuple[np.ndarray, float]:
    """Returns (x, objective_value) over the ORIGINAL variable ordering.

    If `history_out` is passed, each elimination step's diagnostics
    (eliminated/fixed labels, sign, correlation margin between the chosen
    pair and the runner-up) are appended to it -- used by the investigation
    scripts to tell a wrong elimination CHOICE apart from a wrong sign on
    the right choice.
    """
    if min_size < 1:
        raise ValueError("min_size must be >= 1")

    current = qubo
    history: list[tuple[str, str, int]] = []  # (eliminated_label, fixed_to_label, sign)
    rng_seed = seed

    while current.n > min_size and current.n > 1:
        qaoa_result = optimize_qaoa_angles(current, reps=reps, maxiter=maxiter, seed=rng_seed)
        corr = compute_correlations(qaoa_result)

        abs_corr = np.abs(corr)
        i, j = np.unravel_index(np.argmax(abs_corr), abs_corr.shape)
        i, j = int(i), int(j)
        sign = 1 if corr[i, j] >= 0 else -1

        eliminated_label = current.labels[i]
        fixed_label = current.labels[j]
        history.append((eliminated_label, fixed_label, sign))

        if history_out is not None:
            top_val = abs_corr[i, j]
            flat = abs_corr.copy()
            flat[i, j] = flat[j, i] = -1.0
            runner_up = float(flat.max()) if current.n > 2 else 0.0
            history_out.append(
                {
                    "n_before": current.n,
                    "eliminated_label": eliminated_label,
                    "fixed_label": fixed_label,
                    "sign": sign,
                    "top_corr": float(corr[i, j]),
                    "margin": float(top_val - runner_up),
                }
            )

        current = current.reduce(eliminate=i, fixed_to=j, sign=sign)

    x_small, _ = solve_exact(current)
    values = {label: float(val) for label, val in zip(current.labels, x_small)}

    for eliminated_label, fixed_label, sign in reversed(history):
        fixed_val = values[fixed_label]
        values[eliminated_label] = fixed_val if sign == 1 else 1 - fixed_val

    x_full = np.array([values[label] for label in qubo.labels])
    return x_full, qubo.objective(x_full)
