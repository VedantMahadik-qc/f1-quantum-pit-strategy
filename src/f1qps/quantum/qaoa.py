"""Standard QAOA on a local noiseless simulator (Qiskit reference primitives).

No IonQ/hardware dependency here on purpose -- this is the correctness/sanity
layer. Swapping the Estimator/Sampler for IonQ-backed primitives later is a
drop-in change at the call sites in this module only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit.circuit.library import QAOAAnsatz
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize

from f1qps.qubo.builder import QUBOInstance


def _cost_hamiltonian(qubo: QUBOInstance) -> SparsePauliOp:
    """Build the Ising cost Hamiltonian as a SparsePauliOp over n qubits."""
    n = qubo.n
    field, coupling, offset = qubo.to_ising()

    terms: list[tuple[str, float]] = []
    identity = "I" * n
    if offset != 0.0:
        terms.append((identity, offset))
    for i in range(n):
        if field[i] == 0.0:
            continue
        label = "".join("Z" if k == i else "I" for k in range(n))[::-1]
        terms.append((label, field[i]))
    for i in range(n):
        for j in range(i + 1, n):
            if coupling[i, j] == 0.0:
                continue
            label = "".join("Z" if k in (i, j) else "I" for k in range(n))[::-1]
            terms.append((label, coupling[i, j]))

    if not terms:
        terms.append((identity, 0.0))
    return SparsePauliOp.from_list(terms)


@dataclass
class QAOAResult:
    params: np.ndarray
    reps: int
    cost_hamiltonian: SparsePauliOp
    ansatz: QAOAAnsatz
    optimized_energy: float


def optimize_qaoa_angles(
    qubo: QUBOInstance, reps: int = 1, maxiter: int = 200, seed: int | None = None
) -> QAOAResult:
    """Variational loop: minimize <H_C> over QAOA (gamma, beta) angles using
    an exact noiseless statevector Estimator (fast and deterministic for the
    small qubit counts in this project's correctness-testing regime).
    """
    cost_op = _cost_hamiltonian(qubo)
    ansatz = QAOAAnsatz(cost_operator=cost_op, reps=reps)
    estimator = StatevectorEstimator()

    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0, np.pi, size=ansatz.num_parameters)

    def energy(params: np.ndarray) -> float:
        job = estimator.run([(ansatz, cost_op, params)])
        return float(job.result()[0].data.evs)

    result = minimize(energy, x0, method="COBYLA", options={"maxiter": maxiter})
    return QAOAResult(
        params=result.x,
        reps=reps,
        cost_hamiltonian=cost_op,
        ansatz=ansatz,
        optimized_energy=float(result.fun),
    )


def compute_correlations(qaoa_result: QAOAResult) -> np.ndarray:
    """<Z_i Z_j> for every pair i<j, evaluated on the optimized QAOA state.
    Used by RQAOA to pick the next variable to eliminate.
    """
    ansatz = qaoa_result.ansatz
    n = ansatz.num_qubits
    estimator = StatevectorEstimator()

    observables = []
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            label = "".join("Z" if k in (i, j) else "I" for k in range(n))[::-1]
            observables.append(SparsePauliOp.from_list([(label, 1.0)]))
            pairs.append((i, j))

    corr = np.zeros((n, n))
    if observables:
        job = estimator.run([(ansatz, observables, qaoa_result.params)])
        values = job.result()[0].data.evs
        for (i, j), val in zip(pairs, values):
            corr[i, j] = val
            corr[j, i] = val
    return corr


def sample_qaoa_solution(
    qubo: QUBOInstance, qaoa_result: QAOAResult, shots: int = 2000, seed: int | None = None
) -> tuple[np.ndarray, float]:
    """Sample the optimized QAOA circuit and return the best (lowest-objective)
    bitstring observed -- the standard way plain QAOA extracts a candidate
    solution.
    """
    ansatz = qaoa_result.ansatz.copy()
    ansatz.measure_all()
    sampler = StatevectorSampler(seed=seed)
    job = sampler.run([(ansatz, qaoa_result.params)], shots=shots)
    counts = job.result()[0].data.meas.get_counts()

    best_x = None
    best_val = np.inf
    for bitstring, _count in counts.items():
        x = np.array([int(b) for b in bitstring[::-1]], dtype=float)
        val = qubo.objective(x)
        if val < best_val:
            best_val = val
            best_x = x
    return best_x, best_val
