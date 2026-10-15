"""Tests for processing and saving estimation results."""

import numpy as np
import pandas as pd
import pytest

from pydsge import DSGE, meta_data


@pytest.fixture
def stored_model():
    mod = DSGE.load(meta_data)
    try:
        yield mod
    finally:
        if getattr(mod, "pool", None) is not None:
            mod.pool.clear()


def test_processing(stored_model):
    mod = stored_model
    info = mod.info()
    assert mod.name in info

    chain = mod.get_chain()
    assert chain.ndim == 3
    assert chain.shape[-1] == len(mod.prior)
    assert np.isfinite(chain).all()

    summary = mod.mcmc_summary()
    posterior = chain[-mod.get_tune:]
    assert isinstance(summary, pd.DataFrame)
    assert list(summary.index) == list(mod.prior)
    assert np.isfinite(summary[["mean", "sd", "mode", "error"]]).all().all()
    np.testing.assert_allclose(summary["mean"], posterior.mean(axis=(0, 1)))
    np.testing.assert_allclose(summary["sd"], posterior.std(axis=(0, 1)))

    nsamples, T = 100, 30
    pars = mod.get_par("posterior", nsamples=nsamples, full=True)
    assert pars.shape == (nsamples, len(mod.par_names))
    assert np.isfinite(pars).all()
    posterior_vectors = posterior.reshape(-1, posterior.shape[-1])
    for draw in pars[:, mod.prior_arg]:
        assert np.any(np.all(posterior_vectors == draw, axis=1))
    fixed_indices = np.setdiff1d(np.arange(len(mod.par_names)), mod.prior_arg)
    np.testing.assert_array_equal(
        pars[:, fixed_indices],
        np.broadcast_to(mod.par_fix[fixed_indices], (nsamples, len(fixed_indices))),
    )

    irfs, regimes, flags = mod.irfs(("e_r", 1, 0), pars, T=T, verbose=False)
    assert irfs.shape == (nsamples, T, len(mod.vv))
    assert regimes.shape == (2 * nsamples, T)
    assert np.shape(flags) == (nsamples,)
    assert np.isfinite(irfs).all()
    assert np.isfinite(regimes).all()
    assert (regimes >= 0).all()
    assert (regimes == np.floor(regimes)).all()
    assert not np.any(flags), f"IRF solver reported errors: {flags}"
    floor = (
        -1 / pars[:, mod.par_names.index("beta")]
        + pars[:, mod.par_names.index("elb_level")]
    )
    np.testing.assert_allclose(
        irfs[:, :, mod.vix("r")[0]],
        np.maximum(irfs[:, :, mod.vix("rn")[0]], floor[:, None]),
        rtol=1e-9,
        atol=1e-10,
    )

    original_pars = pars.copy()
    pars_sig1 = np.array([mod.set_par("sigma", 1, p) for p in pars])
    sigma_index = mod.par_names.index("sigma")
    other_indices = np.arange(len(mod.par_names)) != sigma_index
    assert pars_sig1.shape == pars.shape
    np.testing.assert_array_equal(pars_sig1[:, sigma_index], np.ones(nsamples))
    np.testing.assert_array_equal(
        pars_sig1[:, other_indices], original_pars[:, other_indices]
    )
    np.testing.assert_array_equal(pars, original_pars)

    mod.load_estim()
    epsd = mod.extract(pars, nsamples=1, bound_sigma=4)

    assert set(epsd) == {"pars", "init", "resid", "flags"}
    assert epsd["init"].shape == (nsamples, len(mod.vv))
    assert epsd["resid"].shape == (nsamples, len(mod.data) - 1, len(mod.shocks))
    assert np.shape(epsd["flags"]) == (nsamples,)
    np.testing.assert_array_equal(epsd["pars"], pars)
    assert np.isfinite(epsd["init"]).all()
    assert np.isfinite(epsd["resid"]).all()
    assert not np.any(epsd["flags"]), f"Shock extraction reported errors: {epsd['flags']}"


def test_model_roundtrip(stored_model, tmp_path):
    """Saved metadata preserves calibration, data and impulse responses."""
    mod = stored_model
    mod.name = "roundtrip_test"
    mod.load_data(mod.data.iloc[:8].copy())
    parameters = mod.set_par("calib")
    expected_irfs, expected_regimes, flags = mod.irfs(
        ("e_r", 1, 0), T=10, verbose=False
    )
    assert not np.any(flags), f"IRF solver reported errors: {flags}"
    assert np.isfinite(expected_irfs.to_numpy()).all()

    filename = tmp_path / "roundtrip_meta.npz"
    mod.save(str(filename), verbose=False)
    restored = DSGE.load(str(filename))

    assert restored.name == mod.name
    assert restored.par_names == mod.par_names
    assert restored.shocks == mod.shocks
    assert restored.observables == mod.observables
    pd.testing.assert_frame_equal(restored.data, mod.data)
    np.testing.assert_array_equal(restored.set_par("calib"), parameters)
    np.testing.assert_allclose(restored.pcompile(parameters), mod.pcompile(parameters))

    actual_irfs, actual_regimes, flags = restored.irfs(
        ("e_r", 1, 0), T=10, verbose=False
    )
    assert not np.any(flags), f"Restored IRF solver reported errors: {flags}"
    pd.testing.assert_frame_equal(
        actual_irfs, expected_irfs, check_exact=False, rtol=1e-9, atol=1e-10
    )
    np.testing.assert_array_equal(actual_regimes, expected_regimes)
