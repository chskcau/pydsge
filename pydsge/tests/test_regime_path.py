"""Path checks with known violations away from the five switching dates."""

import pickle

import numpy as np
import pytest

from pydsge.engine import _constraint_grid
from pydsge.exceptions import UnresolvedRegimeError
from pydsge.regime_path import RegimePaths


def shift_model(**options):
    # q is a shift register. The shadow value at date t is exactly initial[t],
    # followed by zeros once the register is empty. Both regimes obey these
    # same equations, so feasible intervals can be read directly from the input.
    d = 8
    lam = np.eye(d, k=1)
    omega = np.zeros((1, d))
    a = np.diag([1.0] * d + [0.0])
    b = np.eye(d + 1)
    b[:d, :d] = lam
    lag = np.eye(d)[0]
    constraint = np.zeros(d), np.zeros(1), lag
    grid = (4, 5)
    policy = (
        np.tile(omega, (*grid, 1, 1)),
        np.tile(lam, (*grid, 1, 1)),
        np.zeros((*grid, 1)),
        np.zeros((*grid, d)),
    )
    matrices = policy + _constraint_grid(policy, *constraint)
    paths = RegimePaths(
        (a, b, a, b, np.zeros(d + 1)),
        (omega, lam, -1.0),
        constraint,
        np.zeros(grid, dtype=bool),
        **options,
    )
    return paths, matrices


@pytest.mark.parametrize(
    ("state", "old_choice"),
    [
        ([-2, -2, 0, -2, 0, 0, 0, 0], (0, 2)),
        ([0, 0, 0, 0, 0, -2, 0, 0], (0, 0)),
    ],
)
def test_full_check_detects_a_second_binding_spell(state, old_choice):
    paths, matrices = shift_model()
    state = np.array(state, dtype=float)
    assert paths.select(matrices, None, state, False) == (*old_choice, 0)
    assert paths.select(matrices, None, state, True)[2] != 0
    if old_choice == (0, 2):
        # (0,4) passes the endpoint checks too, but date 2 is free.
        assert not paths._valid(0, 4, state, matrices, True)


@pytest.mark.parametrize("full", [False, True])
def test_repair_precedes_endpoint_rejection_and_survives_serialization(full):
    paths, matrices = shift_model(regime_fallback_decimal_digits=83)
    state = np.array([-2.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # A spurious positive shadow value would eliminate the true candidate.
    matrices[-1][2, 0, 2] = 100
    paths.pending[0, 2] = True
    assert paths.select(matrices, None, state, full) == (0, 2, 0)
    assert paths.repaired[0, 2]
    np.testing.assert_array_equal(
        paths.transition(0, 2, state)[:-1], np.r_[state[1:], 0]
    )
    copy, matrices = pickle.loads(pickle.dumps((paths, matrices)))
    assert copy.select(matrices, None, state, full) == (0, 2, 0)
    np.testing.assert_array_equal(
        copy.transition(0, 2, state), paths.transition(0, 2, state)
    )
    assert copy._precision[0].dps == paths._precision[0].dps == 83


def test_disabled_recovery_still_checks_the_path_and_leaves_clear_signs_usable():
    paths, matrices = shift_model(regime_fallback_decimal_digits=None)
    state = np.array([-2.0, -2.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0])
    assert paths.select(matrices, None, state, False) == (0, 2, 0)
    assert paths.select(matrices, None, state, True)[2] != 0
    # The free path touches the bound beyond the original free-regime test dates.
    state = np.array([0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0])
    with pytest.raises(UnresolvedRegimeError, match="L=0, K=0"):
        paths.select(matrices, None, state, True)
    assert paths._precision is None


def test_earlier_unresolved_cell_cannot_be_skipped():
    paths, matrices = shift_model()
    paths.pending[0, 1] = True
    a, b, h = paths.system[0]
    paths.system = ((a * 0, b * 0, h), paths.system[1])
    with pytest.raises(UnresolvedRegimeError, match="L=0, K=1"):
        paths.select(matrices, None, np.full(8, -2.0), True)


def test_roundoff_at_an_endpoint_does_not_eliminate_a_feasible_candidate():
    paths, matrices = shift_model()
    state = np.zeros(8)
    state[0] = -1
    matrices[-1][2, 0, 1] = np.finfo(float).eps
    assert not paths.pending.any()
    assert paths.select(matrices, None, state, False) == (0, 1, 0)
    assert paths.repaired[0, 1]


def test_equality_at_an_interior_date_and_release_is_feasible():
    paths, matrices = shift_model()
    state = np.array([-2.0, -1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 0.0])
    assert paths.select(matrices, None, state, True) == (0, 3, 0)
