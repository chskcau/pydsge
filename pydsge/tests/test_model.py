"""This file contains tests for the solving of the model."""
import pytest
import numpy as np

from pydsge import DSGE


@pytest.fixture(scope="module")
def parametrized_minimal_nk():
    """Parametrize minimal nk model."""
    # Retrieve the example data
    yaml_file = "pydsge/examples/minimal_nk.yaml"

    # Read-in the yaml_file
    mod = DSGE.read(yaml_file)

    # Load in parameters i.e. calibrate
    _ = mod.set_par("calib")
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


def test_minimal_nkmodel_nonlinear(parametrized_minimal_nk):
    mod, par1, par2 = parametrized_minimal_nk
    beta, rho_u, phi_pi = par1["beta"], par1["rho_u"], par1["phi_pi"]
    kappa, x_bar = par2["kappa"], par2["x_bar"]
    u_0 = 4.0

    # Model solution for linear model
    Delta = (1 - rho_u) * (1 - beta * rho_u) + kappa * (phi_pi - rho_u)
    theta_pi = -kappa / Delta
    theta_y = -(1 - beta * rho_u) / Delta
    q = phi_pi * theta_pi * u_0

    # K is first period in linear model where shadow rate is above x_bar. It can be
    # shown analytically that in the non-linear model, the constraint binds precisely
    # from t=0 to t=K-1. For t>=K, the solutions of the linear and non-linear models
    # coincide.
    K = int(np.ceil(np.log(x_bar / q) / np.log(rho_u))) if q < x_bar else 0

    T = max(30, K + 1)
    periods = np.arange(T)
    u = u_0 * rho_u**periods
    Pi_u = theta_pi * u
    Y_u = theta_y * u
    rn_ub = phi_pi * Pi_u

    delta_y = np.zeros(T)
    delta_pi = np.zeros(T)
    for t in range(K - 1, -1, -1):
        delta_y[t] = delta_y[t + 1] + delta_pi[t + 1] + rn_ub[t] - x_bar
        delta_pi[t] = beta * delta_pi[t + 1] + kappa * delta_y[t]

    Pi_c = Pi_u + delta_pi
    Y_c = Y_u + delta_y
    rn_c = phi_pi * Pi_c
    r_c = np.where(periods < K, x_bar, rn_c)
    analytical = np.column_stack((Pi_c, Y_c, r_c, rn_c, u))

    x1, _, flags = mod.irfs(("e_u", u_0, 0), T=T, linear=False, verbose=False)
    assert not np.any(flags), f"Nonlinear solver reported errors: {flags}"
    np.testing.assert_allclose(
        x1[["Pi", "y", "r", "rn", "u"]].to_numpy(),
        analytical,
        rtol=1e-9,
        atol=1e-10,
    )
