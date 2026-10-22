"""This file contains tests for the parsing of the model."""
from pathlib import Path

import numpy as np
import pytest

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


@pytest.mark.parametrize(
    "dates",
    [(-1,), (-2,), (1,), (2,), (-1, 1)],
    ids=["lag", "second-lag", "lead", "second-lead", "lag-and-lead"],
)
def test_observation_equation_rejects_leads_and_lags(dates):
    filename = Path(__file__).parents[1] / "examples" / "minimal_nk.yaml"
    source = filename.read_text(encoding="utf-8")
    expression = "Pi + pi_mean" + "".join(f" + Pi({date})" for date in dates)
    source = source.replace("Infl: Pi + pi_mean", f"Infl: {expression}")

    with pytest.raises(
        NotImplementedError, match="Leads and lags.*observation equation 'Infl'"
    ) as error:
        DSGE.parse(source, "", verbose=False)

    for date in dates:
        assert f"Pi({date})" in str(error.value)
    assert "auxiliary variables" in str(error.value)
