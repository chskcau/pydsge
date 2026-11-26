"""Regime selection and likelihood when part of the ELB grid is unavailable."""

from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats.qmc import LatinHypercube
from threadpoolctl import threadpool_limits

from pydsge import DSGE, engine, gen_sys_from_dict
from pydsge.exceptions import UnresolvedRegimeError


# Problematic DIME proposal.
NK_PROPOSAL = np.array([
    0.2391041290453905, 0.1394423375453827, 0.23869682778582516,
    0.17702194361739315, 0.8998082195797835, 0.11243400269663996,
    1.389469429645767, 0.6508553459207632, 2.038299208580448,
    0.8130353874184681, 0.07787885938876356, 0.7980609993381669,
    0.1615529783771566, 0.43777517486638245,
])


@pytest.mark.parametrize("value", [0, -1, False, np.bool_(True), 50.5, "200"])
def test_invalid_regime_precision_fails_before_model_construction(value):
    with pytest.raises(ValueError, match="regime_fallback_decimal_digits"):
        gen_sys_from_dict({}, regime_fallback_decimal_digits=value)


def test_dict_model_cache_distinguishes_precision():
    # x = .5*x(-1) + .25*(r - x) + u, with shadow rate r = x.
    data = dict(vars=np.array(["x", "r"]), const_var="r", shocks=["u"],
                AA=np.zeros((1, 2)), BB=np.array([[1.25, -0.25]]), CC=np.array([[-0.5, 0.]]),
                DD=np.array([[-1.]]), fb=np.array([1., -1.]), fc=np.zeros(2),
                ZZ0=np.empty(0), x_bar=-1.)
    options = dict(l_max=2, k_max=3, parallel=False, verbose=False)
    default = gen_sys_from_dict(data, **options)
    disabled = gen_sys_from_dict(data, **options, regime_fallback_decimal_digits=None)
    assert default is not disabled
    assert gen_sys_from_dict(data, **options, regime_fallback_decimal_digits=200) is default
    assert gen_sys_from_dict(data, **options, regime_fallback_decimal_digits=None) is disabled
    assert disabled.regime_paths.regime_fallback_decimal_digits is None


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


def test_nk_2000q3_filter_uses_unrounded_forecasts_with_common_draws():
    """Step 99/walker 0. Freeze the archived forecast inputs, before solving the model."""
    resources = Path(__file__).parent / "resources"
    mod = DSGE.read(str(resources / "nk_small.yaml"), nocache=True)
    mod.load_data(pd.DataFrame([[-0.21643063688796538, 0.5829637584435332, 1.63]],
                               columns=["dy", "pinfobs", "robs"]))
    proposal = {
        "sigetab": .1650194432493293, "sigetap": .1524811395850479, "sigetar": .2428044315548397,
        "rhob": .6609736881851435, "rhop": .9886587864762852, "rhor": .30746053531430423,
        "sigc": .16069348182851056, "xip": .6382114564317779, "rpi": 2.5134262462838146,
        "rho": .6442236265723346, "ry": .21586312300752392, "pibar": .37244957134329076,
        "timepref": .34932935274631405, "gambar": .3283284123092366,
    }
    par = [proposal.get(p.name, v) for p, v in zip(mod.parameters, mod.par_fix, strict=True)]
    states, shocks, noise = np.split(np.loadtxt(resources / "nk_2000q3_forecast_inputs.txt"), [5, 8], axis=1)
    with threadpool_limits(1):
        mod.gen_sys(par=par, l_max=32, k_max=64, verbose=False)
        filt = mod.create_filter(N=len(states), R=np.eye(3) * 1e-12, reduced_form=True)
        def draws(mean, cov, size, **kwargs):
            if cov is filt.P:
                return states
            return shocks[None] if cov is filt.Q else noise[None]
        filt.multivariate = draws
        ll = mod.run_filter(smoother=False, get_ll=True)
    # Independent 200-digit solution of the three NK equations and one full
    # ensemble update on these inputs. Scoring the old float64 forecasts gave -66.19076808.
    assert ll == pytest.approx(-67.74428367480442, rel=0, abs=1e-7)
    assert filt.high_precision_completed == 1


@pytest.mark.parametrize("options", [{}, {"regime_fallback_decimal_digits": 80},
                                     {"regime_fallback_decimal_digits": None}],
                         ids=["default", "80-digits", "disabled"])
def test_nk_transition_selects_one_delayed_elb_period_on_the_full_grid(options):
    """2009Q4, step 60/walker 23, upstream forecast member 30."""
    model = DSGE.read(str(Path(__file__).parent / "resources" / "nk_small.yaml"))
    proposal = {
        "sigetab": 0.15666569101694547, "sigetap": 0.14419405337917843, "sigetar": 0.12548556701344102,
        "rhob": 0.9518516488260561, "rhop": 0.9948800002095105, "rhor": 0.12923233486950358,
        "sigc": 1.3154836743967901, "xip": 0.6427315100043057, "rpi": 1.9290546144994951,
        "rho": 0.919289091149836, "ry": 0.0935956933505108, "pibar": 0.8277086533254504,
        "timepref": 0.33994589080876175, "gambar": 0.3781579509093228,
    }
    par = [proposal.get(p.name, value) for p, value in zip(model.parameters, model.par_fix, strict=True)]
    state = np.array([-1.6348862459408866, 2.461626217010558, 7.652043946434822,
                      0.16543393613277552, -12.888139901294837])
    shock = np.array([0.1982358678621746, -0.31351688927160654, 0.24782299953087153])
    with threadpool_limits(limits=1, user_api="blas"):
        model.gen_sys(par=par, l_max=32, k_max=64, verbose=False, **options)
        assert model.regime_paths.pending[0, 58]
        digits = options.get("regime_fallback_decimal_digits", 200)
        if digits is None:
            with pytest.raises(UnresolvedRegimeError, match=r"L=0, K=58"):
                model.t_func(state, shock, set_k=(0, 58))
            for full in (True, False):
                model.verify_regime_path = full
                with pytest.raises(UnresolvedRegimeError):
                    model.t_func(state, shock)
            assert model.regime_paths._precision is None
            return
        (_, forecast), regime, flag = model.t_func(state, shock, return_k=True, get_obs=True)
        assert model.regime_paths._precision[0].dps == digits
        initial = np.r_[state, shock]
        assert model.regime_paths.repaired[0, 58]
        false_path = model.traj(initial, l=0, k=58, verbose=False) - model.x_bar
        selected_path = model.traj(initial, verbose=False) - model.x_bar
        model.verify_regime_path = False
        _, guarded_regime, guarded_flag = model.t_func(state, shock, return_k=True)

    # Independent backward solution of the IS, Phillips and policy equations,
    # using the stable free continuation after one delayed ELB period. The
    # shadow-rate margins at entry, binding and release are +.00624, -.0290,
    # +.04392. For 58 immediate ELB periods the first margin is +2.75e17.
    assert (regime, flag) == ((1, 1), 0)
    assert (guarded_regime, guarded_flag) == (regime, flag)
    assert false_path[0] == pytest.approx(2.7536266941088144e17, rel=1e-10)
    np.testing.assert_allclose(selected_path[:3], [.00624, -.0290, .04392], atol=5e-5)
    assert np.all(selected_path[2:] >= 0)
    np.testing.assert_allclose(forecast, [-0.19025563980219676, -1.307393577631679,
                                         0.006241059715952433], rtol=0, atol=1e-9)


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
    evaluation = partial_model.lprob(NK_PROPOSAL, return_evaluation=True)
    assert evaluation.outcome == "unresolved_regime"
    assert (evaluation.regime_l, evaluation.regime_k) == (1, 3)
    assert np.isneginf(evaluation.log_posterior) and np.isfinite(evaluation.log_prior)


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
