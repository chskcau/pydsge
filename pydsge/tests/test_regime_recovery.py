"""Regime selection and likelihood when part of the ELB grid is unavailable."""

from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats.qmc import LatinHypercube
from threadpoolctl import threadpool_limits

from pydsge import DSGE, engine
from pydsge.exceptions import UnresolvedRegimeError


# Problematic DIME proposal.
NK_PROPOSAL = np.array([
    0.2391041290453905, 0.1394423375453827, 0.23869682778582516,
    0.17702194361739315, 0.8998082195797835, 0.11243400269663996,
    1.389469429645767, 0.6508553459207632, 2.038299208580448,
    0.8130353874184681, 0.07787885938876356, 0.7980609993381669,
    0.1615529783771566, 0.43777517486638245,
])


def nk_likelihood(k_max, proposal=NK_PROPOSAL):
    mod = DSGE.read(str(Path(__file__).parent / "resources" / "nk_small.yaml"))
    mod.load_data(pd.DataFrame([
        [-0.31251786705022155, 0.50641760988492379, 0.99749999999999994],
        [0.44976430402177137, 0.95404762605411086, 0.97333333333333327],
        [0.29750705487576745, 1.1082325343331867, 1.0433333333333332],
    ], columns=["dy", "pinfobs", "robs"]))
    mod.set_par("calib", l_max=32, k_max=k_max)
    mod.prep_estim(linear=False, N=32, R=np.eye(3) * 1e-12, seed=42,
                   qmc_engine=partial(LatinHypercube, seed=42), l_max=32,
                   k_max=k_max, ncores=0, debug=True, dry_run=False, verbose=False)
    transition = mod.t_func
    regimes = []

    def record(*args, return_k=False, **kwargs):
        result, lk, flag = transition(*args, **kwargs, return_k=True)
        regimes.append((lk, flag))
        return (result, lk, flag) if return_k else (result, flag)

    mod.t_func = record
    mod.filter.seed = 42
    posterior = mod.lprob(proposal)
    return mod, posterior, regimes


def test_real_nk_proposal_preserves_likelihood_and_all_selected_regimes():
    with threadpool_limits(limits=1, user_api="blas"):
        short, reference, expected_regimes = nk_likelihood(32)
        full, posterior, regimes = nk_likelihood(64)
    assert np.isfinite(reference) and posterior == reference
    assert full.ll == short.ll
    assert regimes == expected_regimes == [((0, 0), 0)] * (3 * 32)
    assert full.precalc_mat[0].shape[1] == 65


def test_partial_grid_matches_complete_grid_and_rejects_required_hole(monkeypatch, caplog):
    complete, expected, regimes = nk_likelihood(8)
    original_cell = engine._fill_cell

    def singular(*args):
        raise np.linalg.LinAlgError("Matrix is singular to machine precision.")

    def one_missing_cell(system, policy, l, k, prev_l, prev_k):
        if (l, k) == (1, 3):
            singular()
        original_cell(system, policy, l, k, prev_l, prev_k)

    monkeypatch.setattr(engine, "preprocess_jit", singular)
    monkeypatch.setattr(engine, "_fill_cell", one_missing_cell)
    partial_model, actual, selected = nk_likelihood(8)
    assert np.isfinite(expected) and actual == expected
    assert partial_model.ll == complete.ll and selected == regimes
    available = partial_model.regime_available
    assert available[0].all() and not available[1:, 3].any()
    assert available[1:, 4:].all()  # These cells have independent predecessors.
    state = np.zeros(partial_model.dimq - partial_model.dimeps)
    shock = np.array([0., 0., -4.])
    reference_state, lk, flag = complete.t_func(state, shock, return_k=True)
    recovered_state, recovered_lk, recovered_flag = partial_model.t_func(state, shock, return_k=True)
    assert (lk, flag) == (recovered_lk, recovered_flag) == ((0, 1), 0)
    np.testing.assert_array_equal(recovered_state, reference_state)
    with pytest.raises(UnresolvedRegimeError, match=r"L=1, K=3"):
        partial_model.t_func(state, set_k=(1, 3))

    # Drive the real likelihood through the unavailable regime.
    def need_missing_regime(*args, **kwargs):
        return recorded_transition(*args, **kwargs, set_k=(1, 3))

    recorded_transition = partial_model.t_func
    partial_model.t_func = need_missing_regime
    assert np.isneginf(partial_model.lprob(NK_PROPOSAL))
    assert "Rejecting proposal. Regime (L=1, K=3)" in caplog.text
    caplog.clear()
    with pytest.raises(UnresolvedRegimeError, match=r"L=1, K=3"):
        partial_model.lprob(NK_PROPOSAL, raise_on_unresolved=True)
    assert not caplog.records


def test_selection_keeps_lexicographic_priority_and_checks_only_visited_cells():
    bmat = np.zeros((5, 3, 4, 1))
    bterm = np.zeros((5, 3, 4))
    bterm[2, 0, 0] = -1  # Enter the search for a binding regime.
    bterm[4, 0, 1:3] = -1  # (0,3) and (1,1) are both admissible.
    state = np.zeros(1)
    available = np.ones((3, 4), dtype=bool)
    assert engine.find_lk(bmat, bterm, 0., state) == (0, 3, 0)
    available[1, 1] = False
    assert engine.find_lk(bmat, bterm, 0., state, available) == (0, 3, 0)
    available[1, 1] = True
    available[0, 3] = False
    with pytest.raises(UnresolvedRegimeError, match=r"L=0, K=3"):
        engine.find_lk(bmat, bterm, 0., state, available)

    available[0, 3] = True
    bterm[4, 0, 3] = np.nan
    with pytest.raises(UnresolvedRegimeError, match=r"L=0, K=3"):
        engine.find_lk(bmat, bterm, 0., state, available)

    bterm[2, :, 0] = 1
    assert engine.find_lk(bmat, bterm, 0., state, available) == (0, 0, 0)
    available[2, 0] = False
    with pytest.raises(UnresolvedRegimeError, match=r"L=2, K=0"):
        engine.find_lk(bmat, bterm, 0., state, available)
