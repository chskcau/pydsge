"""This file contains tests for the parsing of the model."""
import numpy as np

from pydsge import DSGE
from pydsge import example


def test_parsing_yaml():
    """Check parameter ordering, calibration and derived parameters."""
    yaml_file, _ = example
    mod = DSGE.read(yaml_file)
    par = mod.set_par("calib")

    expected = {
        "beta": 0.99,
        "theta": 0.66,
        "sigma": 1.5,
        "phi_pi": 1.7,
        "phi_y": 0.125,
        "rho": 0.8,
        "rho_u": 0.8,
        "rho_z": 0.9,
        "rho_r": 0.7,
        "sig_u": 0.5,
        "sig_z": 0.3,
        "sig_r": 0.3,
        "psi": 0.3,
        "nub": 0.1,
        "y_mean": 0.35,
        "pi_mean": 0.5,
        "elb_level": 0.07,
    }

    assert mod.par_names == list(expected)
    assert isinstance(par, np.ndarray)
    assert par.shape == (len(expected),)
    np.testing.assert_allclose(par, list(expected.values()), rtol=1e-12, atol=1e-12)

    calibrated, derived = mod.get_par(asdict=True, roundto=15)
    assert set(calibrated) == set(expected)
    np.testing.assert_allclose(
        [calibrated[name] for name in expected],
        list(expected.values()),
        rtol=1e-12,
        atol=1e-12,
    )
    expected_derived = {
        "kappa": (1 - 0.66) * (1 - 0.99 * 0.66) / 0.66,
        "eta": (1.5 + 0.3 + 0.1) / (1 - 0.1),
        "nu": 0.1 / (1 - 0.1),
        "x_bar": -1 / 0.99 + 0.07,
    }
    assert set(derived) == set(expected_derived)
    np.testing.assert_allclose(
        [derived[name] for name in expected_derived],
        list(expected_derived.values()),
        rtol=1e-12,
        atol=1e-12,
    )
    assert mod.shocks == ["e_z", "e_u", "e_r"]
    assert mod.observables == ["GDP", "Infl", "FFR"]
