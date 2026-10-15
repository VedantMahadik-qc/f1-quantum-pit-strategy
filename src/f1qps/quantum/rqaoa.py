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
from f1qps.quantum.qaoa import compute_correlations, optimize_qaoa_angles, sample_correlations


def solve_rqaoa(
    qubo: QUBOInstance,
    reps: int = 1,
    min_size: int = 4,
    maxiter: int = 200,
    seed: int | None = None,
    history_out: list[dict] | None = None,
    shots: int | None = None,
    ensemble_k: int = 1,
    restart_corrs_out: list[list[np.ndarray]] | None = None,
    confidence_threshold: float | None = None,
) -> tuple[np.ndarray, float]:
    """Returns (x, objective_value) over the ORIGINAL variable ordering.

    If `history_out` is passed, each elimination step's diagnostics
    (eliminated/fixed labels, sign, correlation margin between the chosen
    pair and the runner-up) are appended to it -- used by the investigation
    scripts to tell a wrong elimination CHOICE apart from a wrong sign on
    the right choice.

    If `restart_corrs_out` is passed (only meaningful with `ensemble_k` > 1),
    each elimination step's `ensemble_k` individual pre-averaging correlation
    matrices are appended to it as a list -- used to measure how much the
    independent restarts agree with each other before they get collapsed into
    the single averaged matrix RQAOA actually acts on.

    `shots`: None (default) uses the exact analytic <Z_i Z_j> correlation,
    matching every prior result in this project. An int switches the
    elimination signal to a finite-shot measurement estimate instead --
    the realistic way this would have to be read off real hardware.

    `ensemble_k`: 1 (default) matches every prior result -- a single QAOA
    optimization per elimination step. k>1 runs k independent optimizations
    (distinct seeds) per step and averages their correlation matrices before
    picking the pair/sign, on the theory that RQAOA's failures trace to
    idiosyncratic single-trajectory noise (see investigate_rqaoa_* scripts)
    that averaging across independent restarts should cancel out. Costs k
    times the wall-clock of a single run.

    `confidence_threshold`: None (default) always runs all `ensemble_k`
    restarts per step, matching every prior result. A float switches to
    SELECTIVE ensembling: run one restart first, and only spend the
    remaining `ensemble_k - 1` restarts on THAT step if its own top
    correlation magnitude is below this threshold -- per
    investigate_step_confidence_signals.py, that magnitude (`abs_top_corr`)
    is a real predictor of per-step sign-correctness (r=0.30, p=7.7e-11 on
    synthetic data), so a strong first-restart pick is trusted as-is while a
    weak one gets the full ensembling treatment. Meant to recover most of
    `ensemble_k`'s accuracy at a fraction of its compute. No-op (always runs
    the full `ensemble_k`) when `ensemble_k == 1`. `history_out` records how
    many restarts each step actually used (`restarts_used`).
    """
    if min_size < 1:
        raise ValueError("min_size must be >= 1")
    if ensemble_k < 1:
        raise ValueError("ensemble_k must be >= 1")

    current = qubo
    history: list[tuple[str, str, int]] = []  # (eliminated_label, fixed_to_label, sign)
    rng_seed = seed
    step = 0

    while current.n > min_size and current.n > 1:
        corr = np.zeros((current.n, current.n))
        step_corrs: list[np.ndarray] = []
        restarts_run = 0
        for k in range(ensemble_k):
            restart_seed = None if rng_seed is None else rng_seed + k
            qaoa_result = optimize_qaoa_angles(current, reps=reps, maxiter=maxiter, seed=restart_seed)
            if shots is None:
                restart_corr = compute_correlations(qaoa_result)
            else:
                sampler_seed = None if seed is None else seed + step * ensemble_k + k
                restart_corr = sample_correlations(qaoa_result, shots=shots, seed=sampler_seed)
            corr += restart_corr
            restarts_run += 1
            if restart_corrs_out is not None:
                step_corrs.append(restart_corr.copy())
            if (
                confidence_threshold is not None
                and k == 0
                and ensemble_k > 1
                and np.abs(restart_corr).max() >= confidence_threshold
            ):
                break
        corr /= restarts_run
        if restart_corrs_out is not None:
            restart_corrs_out.append(step_corrs)
        step += 1

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
                    "restarts_used": restarts_run,
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
