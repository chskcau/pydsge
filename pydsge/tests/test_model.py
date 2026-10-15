"""This file contains tests for the solving of the model."""
import numpy as np
import pytest

from pydsge import DSGE


@pytest.fixture(
    scope="module",
    params=[
        pytest.param({}, id="yaml-calibration"),
        pytest.param({"rho_u": 0.9}, id="persistent-demand"),
        pytest.param({"phi_pi": 5}, id="stronger-policy-response"),
        pytest.param({"theta": 0.8}, id="stickier-prices"),
        pytest.param(
            {"beta": 0.97, "elb_level": 0.15}, id="alternative-discount-and-floor"
        ),
    ],
)
def parametrized_minimal_nk(request):
    """Parametrize minimal nk model."""
    # Retrieve the example data
    yaml_file = "pydsge/examples/minimal_nk.yaml"

    # Read-in the yaml_file
    mod = DSGE.read(yaml_file)

    # Apply each set of overrides to a fresh copy of the YAML calibration.
    parameters = mod.get_par("calib", k_max=32)
    for name, value in request.param.items():
        parameters[mod.par_names.index(name)] = value
    mod.set_par(parameters, k_max=32)
    # Preserve precision in derived parameters such as kappa for the reference.
    par1, par2 = mod.get_par(asdict=True, roundto=15)

    return mod, par1, par2


def test_minimal_nkmodel_linear(parametrized_minimal_nk):
    """Compare linear inflation and output impacts with the analytical solution."""
    # Get model
    mod, par1, par2 = parametrized_minimal_nk

    # The analytical solution assumes r = phi_pi * Pi without a binding floor.
    # This large demand shock would bind the floor in a constrained simulation.
    shock_list = ("e_u", 4.0, 0)  # (name, size, period)
    x1, _, flags = mod.irfs(shock_list, linear=True, verbose=False)
    assert not np.any(flags), f"Linear solver reported errors: {flags}"

    # Get the numerical impact responses.
    infl_mod_imp, outgap_mod_imp = x1["Pi"].iloc[0], x1["y"].iloc[0]

    # With Pi = theta_pi * u and y = theta_y * u, the YAML equations imply:
    # (1 - beta * rho_u) * theta_pi = kappa * theta_y
    # (rho_u - 1) * theta_y = (phi_pi - rho_u) * theta_pi + 1
    Delta = (
        (1 - par1["rho_u"]) * (1 - par1["beta"] * par1["rho_u"])
        + par2["kappa"] * (par1["phi_pi"] - par1["rho_u"])
    )
    theta_pi = -par2["kappa"] / Delta
    theta_y = -(1 - par1["beta"] * par1["rho_u"]) / Delta

    infl_analy_imp = theta_pi * shock_list[1]  # inflation as linear response to shock
    outgap_analy_imp = theta_y * shock_list[1]  # output gap as linear response to shock

    # Allow small floating-point differences across supported platforms.
    np.testing.assert_allclose(
        [infl_mod_imp, outgap_mod_imp],
        [infl_analy_imp, outgap_analy_imp],
        rtol=1e-9,
        atol=1e-10,
    )


@pytest.mark.parametrize(
    "crossing_time, expected_K",
    [
        pytest.param(None, 0, id="no-shock"),
        pytest.param(-1.0, 0, id="small-unconstrained-shock"),
        pytest.param(-1e-6, 0, id="just-below-first-binding"),
        pytest.param(1e-6, 1, id="just-above-first-binding"),
        pytest.param(0.5, 1, id="K=1"),
        pytest.param(4.5, 5, id="K=5"),
        pytest.param(9.5, 10, id="K=10"),
        pytest.param(10 - 1e-6, 10, id="just-before-K=11"),
        pytest.param(10 + 1e-6, 11, id="just-after-K=11"),
        pytest.param(30.5, 31, id="K=31-beyond-output-horizon"),
    ],
)
def test_minimal_nkmodel_nonlinear(
    parametrized_minimal_nk, crossing_time, expected_K
):
    mod, par1, par2 = parametrized_minimal_nk
    beta, rho_u, phi_pi = par1["beta"], par1["rho_u"], par1["phi_pi"]
    kappa, x_bar = par2["kappa"], par2["x_bar"]

    # Model solution for linear model
    Delta = (1 - rho_u) * (1 - beta * rho_u) + kappa * (phi_pi - rho_u)
    theta_pi = -kappa / Delta
    theta_y = -(1 - beta * rho_u) / Delta
    # crossing_time is when the unbounded shadow rate reaches x_bar.
    u_0 = (
        0.0
        if crossing_time is None
        else x_bar / (phi_pi * theta_pi) * rho_u ** (-crossing_time)
    )
    q = phi_pi * theta_pi * u_0

    # K is first period in linear model where shadow rate is above x_bar. It can be
    # shown analytically that in the non-linear model, the constraint binds precisely
    # from t=0 to t=K-1. For t>=K, the solutions of the linear and non-linear models
    # coincide.
    K = int(np.ceil(np.log(x_bar / q) / np.log(rho_u))) if q < x_bar else 0
    assert K == expected_K

    T = 30
    # Backward recursion needs the terminal values at period K, even when K >= T.
    T_analytical = max(T, K + 1)
    analytical_periods = np.arange(T_analytical)
    u = u_0 * rho_u**analytical_periods
    Pi_u = theta_pi * u
    Y_u = theta_y * u
    rn_ub = phi_pi * Pi_u

    delta_y = np.zeros(T_analytical)
    delta_pi = np.zeros(T_analytical)
    for t in range(K - 1, -1, -1):
        delta_y[t] = delta_y[t + 1] + delta_pi[t + 1] + rn_ub[t] - x_bar
        delta_pi[t] = beta * delta_pi[t + 1] + kappa * delta_y[t]

    Pi_c = Pi_u + delta_pi
    Y_c = Y_u + delta_y
    rn_c = phi_pi * Pi_c
    r_c = np.where(analytical_periods < K, x_bar, rn_c)
    analytical = np.column_stack((Pi_c, Y_c, r_c, rn_c, u))

    x1, regimes, flags = mod.irfs(("e_u", u_0, 0), T=T, linear=False, verbose=False)
    assert not np.any(flags), f"Nonlinear solver reported errors: {flags}"
    np.testing.assert_array_equal(regimes[0], np.zeros(T))
    np.testing.assert_array_equal(regimes[1], np.maximum(K - np.arange(T), 0))
    np.testing.assert_allclose(
        x1[["Pi", "y", "r", "rn", "u"]].to_numpy(),
        analytical[:T],
        rtol=1e-9,
        atol=1e-10,
    )
