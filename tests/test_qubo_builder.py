import itertools

import numpy as np
import pytest

from f1qps.qubo.builder import QUBOInstance


def test_objective_matches_hand_computed_small_case():
    # 2 cars: h = [1, 2], J_01 = 3
    # C(x) = -1*x0 - 2*x1 + 3*x0*x1
    h = np.array([1.0, 2.0])
    J = np.array([[0.0, 3.0], [3.0, 0.0]])
    qubo = QUBOInstance(h=h, J=J)

    assert qubo.objective([0, 0]) == pytest.approx(0.0)
    assert qubo.objective([1, 0]) == pytest.approx(-1.0)
    assert qubo.objective([0, 1]) == pytest.approx(-2.0)
    assert qubo.objective([1, 1]) == pytest.approx(-1 - 2 + 3)


def test_objective_matches_Q_matrix_quadratic_form():
    rng = np.random.default_rng(0)
    n = 5
    h = rng.uniform(0, 1, n)
    J = rng.uniform(0, 1, (n, n))
    J = (J + J.T) / 2
    np.fill_diagonal(J, 0)
    qubo = QUBOInstance(h=h, J=J)

    for bits in itertools.product((0, 1), repeat=n):
        x = np.array(bits, dtype=float)
        assert qubo.objective(x) == pytest.approx(x @ qubo.Q @ x)


def test_to_ising_roundtrip_matches_objective():
    rng = np.random.default_rng(1)
    n = 4
    h = rng.uniform(0, 1, n)
    J = rng.uniform(0, 1, (n, n))
    J = (J + J.T) / 2
    np.fill_diagonal(J, 0)
    qubo = QUBOInstance(h=h, J=J)
    field, coupling, offset = qubo.to_ising()

    for bits in itertools.product((0, 1), repeat=n):
        x = np.array(bits, dtype=float)
        z = 1 - 2 * x  # x_i = (1-z_i)/2  =>  z_i = 1-2x_i
        ising_val = offset + field @ z + sum(
            coupling[i, j] * z[i] * z[j] for i in range(n) for j in range(i + 1, n)
        )
        assert ising_val == pytest.approx(qubo.objective(x))


def _random_qubo(n, seed):
    rng = np.random.default_rng(seed)
    h = rng.uniform(0, 1, n)
    J = rng.uniform(0, 1, (n, n))
    J = (J + J.T) / 2
    np.fill_diagonal(J, 0)
    return QUBOInstance(h=h, J=J)


@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("seed", range(5))
def test_reduce_preserves_objective_up_to_constant(sign, seed):
    """reduce() substitutes x_eliminate as a function of x_fixed_to. For any
    fixed values of the OTHER variables, the original objective (with x_eliminate
    and x_fixed_to set according to the substitution rule) must equal the
    reduced objective plus a constant that does NOT depend on those other values.
    This is exactly the algebraic identity reduce() is supposed to implement.
    """
    n = 5
    qubo = _random_qubo(n, seed)
    eliminate, fixed_to = 1, 3
    reduced = qubo.reduce(eliminate=eliminate, fixed_to=fixed_to, sign=sign)

    remaining = [i for i in range(n) if i != eliminate]
    offsets = []
    for bits in itertools.product((0, 1), repeat=len(remaining)):
        x_small = np.array(bits, dtype=float)

        x_full = np.zeros(n)
        for pos, idx in enumerate(remaining):
            x_full[idx] = x_small[pos]
        x_fixed_val = x_full[fixed_to]
        x_full[eliminate] = x_fixed_val if sign == 1 else 1 - x_fixed_val

        original_val = qubo.objective(x_full)
        reduced_val = reduced.objective(x_small)
        offsets.append(original_val - reduced_val)

    # All offsets must be equal (the dropped constant is independent of the
    # other variables' values) -- i.e. reduce() is a valid substitution, not
    # just correct at one point.
    assert np.allclose(offsets, offsets[0])


def test_reduce_drops_eliminated_variable_and_keeps_others():
    qubo = _random_qubo(4, seed=42)
    reduced = qubo.reduce(eliminate=2, fixed_to=0, sign=1)
    assert reduced.n == 3
    assert reduced.labels == ["x0", "x1", "x3"]
