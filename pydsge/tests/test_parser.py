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


@pytest.fixture
def gaussian_source():
    filename = Path(__file__).parent / "resources" / "linear_gaussian.yaml"
    return filename.read_text(encoding="utf-8")


@pytest.mark.parametrize("section", ["covariances_sqrt", "covariances"])
def test_shock_scale_section(gaussian_source, section, recwarn):
    source = gaussian_source.replace("covariances_sqrt:", f"{section}:")
    mod = DSGE.parse(source, "", verbose=False)
    scales = mod.QQ(mod.pcompile(mod.p0()))

    assert mod.shocks == ["e_x", "e_z"]
    np.testing.assert_array_equal(scales, np.diag([0.4, 0.7]))
    if section == "covariances":
        warning = recwarn.pop(UserWarning)
        assert "misleading" in str(warning.message)
        assert "covariances_sqrt" in str(warning.message)
    assert not recwarn


def test_conflicting_shock_scale_sections(gaussian_source):
    source = gaussian_source.replace(
        "  covariances_sqrt:\n",
        "  covariances:\n    e_x: sig_x\n    e_z: sig_z\n  covariances_sqrt:\n",
    )
    with pytest.raises(ValueError, match="only one.*not both"):
        DSGE.parse(source, "", verbose=False)


@pytest.mark.parametrize("section", ["covariances_sqrt", "covariances"])
@pytest.mark.parametrize("value", [-0.1, 0.0, 0.1])
def test_off_diagonal_shock_entries(gaussian_source, section, value, recwarn):
    source = gaussian_source.replace("covariances_sqrt:", f"{section}:")
    source = source.replace("    e_z: sig_z", f"    e_z: sig_z\n    e_x,e_z: {value}")
    with pytest.raises(NotImplementedError, match="Off-diagonal.*e_x,e_z"):
        DSGE.parse(source, "", verbose=False)


@pytest.mark.parametrize("expression", ["-0.4", "-sig_x"])
def test_negative_shock_standard_deviation(gaussian_source, expression):
    source = gaussian_source.replace("e_x: sig_x", f"e_x: {expression}")
    with pytest.raises(ValueError, match="non-negative.*e_x"):
        DSGE.parse(source, "", verbose=False)


@pytest.mark.parametrize("value", ["-0.4", ".nan", ".inf"])
def test_invalid_calibrated_shock_scale(gaussian_source, value):
    source = gaussian_source.replace("sig_x: 0.4", f"sig_x: {value}")
    with pytest.raises(ValueError, match="finite and non-negative.*e_x"):
        DSGE.parse(source, "", verbose=False)


def test_zero_shock_standard_deviation(gaussian_source):
    source = gaussian_source.replace("sig_x: 0.4", "sig_x: 0.0")
    mod = DSGE.parse(source, "", verbose=False)
    scales = mod.QQ(mod.pcompile(mod.p0()))
    np.testing.assert_array_equal(scales, np.diag([0.0, 0.7]))


def test_negative_shock_scale_after_parameter_change(gaussian_source):
    mod = DSGE.parse(gaussian_source, "", verbose=False)
    parameters = np.array(mod.p0())
    parameters[mod.par_names.index("sig_x")] = -0.4
    with pytest.raises(ValueError, match="non-negative.*e_x"):
        mod.set_par(parameters, l_max=2, k_max=4, verbose=False)


def test_legacy_shock_scale_warning_with_cached_model(gaussian_source, tmp_path):
    filename = tmp_path / "legacy.yaml"
    filename.write_text(
        gaussian_source.replace("covariances_sqrt:", "covariances:"), encoding="utf-8"
    )
    for _ in range(2):
        with pytest.warns(UserWarning, match="misleading.*standard deviations"):
            DSGE.read(str(filename))
