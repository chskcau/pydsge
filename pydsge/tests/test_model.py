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


@pytest.fixture(
    scope="module",
    params=[
        pytest.param(({}, 0, 2), id="yaml-calibration"),
        pytest.param(({"rho_r": 0.0}, 0, 5), id="no-smoothing"),
        pytest.param(({"rho_r": 0.5}, 0, 4), id="moderate-smoothing"),
        pytest.param(({"rho_r": 0.95}, 0, 0), id="strong-smoothing-unbound"),
        pytest.param(
            ({"theta": 0.8, "rho_r": 0.9, "rho_u": 0.85}, 2, 5),
            id="delayed-L=2-K=5",
        ),
        pytest.param(
            ({"theta": 0.9, "rho_r": 0.97, "rho_u": 0.95}, 3, 15),
            id="delayed-L=3-K=15",
        ),
        pytest.param(
            ({"theta": 0.95, "rho_r": 0.98, "rho_u": 0.97}, 10, 24),
            id="delayed-L=10-K=24",
        ),
        pytest.param(
            ({"theta": 0.95, "rho_r": 0.97, "rho_u": 0.97}, 3, 32),
            id="delayed-L=3-K=32-beyond-output-horizon",
        ),
    ],
)
def parametrized_minimal_nk_smoothing(request):
    overrides, L, K = request.param
    mod = DSGE.read("pydsge/examples/minimal_nk_smoothing.yaml")
    parameters = mod.get_par("calib")
    for name, value in overrides.items():
        parameters[mod.par_names.index(name)] = value
    mod.set_par(parameters, l_max=12, k_max=50)
    par1, par2 = mod.get_par(asdict=True, roundto=15)
    return mod, par1, par2, L, K


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


def _minimal_nk_smoothing_coefficients(par1, par2):
    beta, rho_r, rho_u = par1["beta"], par1["rho_r"], par1["rho_u"]
    kappa = par2["kappa"]
    phi_eff = (1 - rho_r) * par1["phi_pi"]

    # With Pi=a_pi*r(-1)+b_pi*u and y=a_y*r(-1)+b_y*u,
    # the stable coefficient lambda_r in r=lambda_r*r(-1)+phi_eff*b_pi*u
    # solves (lambda_r-rho_r)*((1-lambda_r)*(1-beta*lambda_r)
    #                          -kappa*lambda_r) + kappa*phi_eff*lambda_r = 0.
    roots = np.roots(
        [
            beta,
            -(1 + beta + kappa + beta * rho_r),
            1 + rho_r * (1 + beta + kappa) + kappa * phi_eff,
            -rho_r,
        ]
    )
    stable_roots = roots[(np.abs(roots.imag) < 1e-12) & (np.abs(roots) < 1)]
    assert stable_roots.size == 1, f"Expected one stable real root, got {roots}"
    lambda_r = stable_roots[0].real
    a_pi = (lambda_r - rho_r) / phi_eff
    a_y = a_pi * (1 - beta * lambda_r) / kappa
    B = 1 - beta * rho_u - beta * a_pi * phi_eff
    Delta = (1 - rho_u) * B - kappa * (rho_u + (a_y + a_pi - 1) * phi_eff)
    b_pi = -kappa / Delta
    b_y = -B / Delta
    return np.array([a_pi, a_y]), np.array([b_pi, b_y]), lambda_r


def test_minimal_nkmodel_smoothing_linear(parametrized_minimal_nk_smoothing):
    """Compare full linear IRFs with independently derived policy coefficients."""
    mod, par1, par2, _, _ = parametrized_minimal_nk_smoothing
    rho_r, rho_u = par1["rho_r"], par1["rho_u"]
    phi_eff = (1 - rho_r) * par1["phi_pi"]
    a, b, _ = _minimal_nk_smoothing_coefficients(par1, par2)

    T, u_0 = 30, 4.0
    analytical = np.empty((T, 5))
    r_previous = 0.0
    for t in range(T):
        u = u_0 * rho_u**t
        Pi, y = a * r_previous + b * u
        r = rho_r * r_previous + phi_eff * Pi
        analytical[t] = Pi, y, r, r, u
        r_previous = r

    x1, _, flags = mod.irfs(("e_u", u_0, 0), T=T, linear=True, verbose=False)
    assert not np.any(flags), f"Linear solver reported errors: {flags}"
    np.testing.assert_allclose(
        x1[["Pi", "y", "r", "rn", "u"]].to_numpy(),
        analytical,
        rtol=1e-9,
        atol=1e-10,
    )


def _matrix_geometric_sum(matrix, n):
    """Evaluate S_n(X) without requiring I-X to be invertible."""
    return sum(
        (np.linalg.matrix_power(matrix, j) for j in range(n)),
        start=np.zeros_like(matrix),
    )


def test_minimal_nkmodel_smoothing_nonlinear(parametrized_minimal_nk_smoothing):
    """Compare constrained IRFs with the closed-form binding and continuation paths."""
    mod, par1, par2, L, expected_K = parametrized_minimal_nk_smoothing
    beta, rho_r, rho_u = par1["beta"], par1["rho_r"], par1["rho_u"]
    kappa, x_bar = par2["kappa"], par2["x_bar"]
    phi_eff = (1 - rho_r) * par1["phi_pi"]
    a, b, lambda_r = _minimal_nk_smoothing_coefficients(par1, par2)
    nu = phi_eff * b[0]

    T, u_0 = 30, 4.0
    u_b = (1 - lambda_r) * x_bar / nu
    K = max(0, int(np.ceil(np.log(u_b / u_0) / np.log(rho_u))))
    assert K == expected_K
    periods = np.arange(T)
    # Include period K for the terminal condition, even if K exceeds T.
    T_analytical = max(T, K + 1)
    analytical_periods = np.arange(T_analytical)
    u = u_0 * rho_u**analytical_periods
    M = np.array([[beta + kappa, kappa], [1.0, 1.0]])
    d = np.array([kappa, 1.0])
    terminal = a * x_bar + b * u[K]
    Pi_y = np.empty((T_analytical, 2))
    r = np.full(T_analytical, x_bar)

    # Closed-form binding path from minimal_nk_smoothing_solution.tex.
    for t in range(L, K):
        n = K - t
        M_n = np.linalg.matrix_power(M, n)
        S_M = _matrix_geometric_sum(M, n)
        S_rho_M = _matrix_geometric_sum(rho_u * M, n)
        Pi_y[t] = M_n @ terminal - x_bar * (S_M @ d) - u[t] * (S_rho_M @ d)

    if L:
        A = np.array(
            [
                [beta + kappa, kappa, -kappa],
                [1.0, 1.0, -1.0],
                [-phi_eff * (beta + kappa) / rho_r, -phi_eff * kappa / rho_r,
                 (1 + phi_eff * kappa) / rho_r],
            ]
        )
        g = np.array([-kappa, -1.0, phi_eff * kappa / rho_r])
        A_L = np.linalg.matrix_power(A, L)
        R = (
            -(A_L @ np.r_[Pi_y[L], 0.0])[2]
            - u_0 * (_matrix_geometric_sum(rho_u * A, L) @ g)[2]
        ) / A_L[2, 2]
        w_L = np.r_[Pi_y[L], R]
        for t in range(L):
            w_t = (
                np.linalg.matrix_power(A, L - t) @ w_L
                + u[t] * _matrix_geometric_sum(rho_u * A, L - t) @ g
            )
            Pi_y[t] = w_t[:2]
            r[t] = rho_r * w_t[2] + phi_eff * w_t[0]

    n = np.arange(T_analytical - K)
    if lambda_r == rho_u:
        geometric_sum = np.zeros_like(n, dtype=float)
        geometric_sum[1:] = n[1:] * rho_u ** (n[1:] - 1)
    else:
        geometric_sum = (lambda_r**n - rho_u**n) / (lambda_r - rho_u)
    initial_rate = x_bar if K else 0.0
    r_previous = lambda_r**n * initial_rate + nu * u[K] * geometric_sum
    Pi_y[K:] = r_previous[:, None] * a + u[K:, None] * b
    r[K:] = lambda_r * r_previous + nu * u[K:]
    rn = rho_r * np.r_[0.0, r[:-1]] + phi_eff * Pi_y[:, 0]

    # Verify the candidate binds precisely from t=L to K-1.
    assert np.all(rn[:L] >= x_bar)
    assert np.all(rn[L:K] < x_bar)
    assert np.all(rn[K:] >= x_bar)
    analytical = np.column_stack((Pi_y, r, rn, u))
    x1, regimes, flags = mod.irfs(("e_u", u_0, 0), T=T, linear=False, verbose=False)
    assert not np.any(flags), f"Nonlinear solver reported errors: {flags}"
    np.testing.assert_array_equal(regimes[0], np.maximum(L - periods, 0))
    np.testing.assert_array_equal(
        regimes[1], np.where(periods < L, K - L, np.maximum(K - periods, 0))
    )
    np.testing.assert_allclose(
        x1[["Pi", "y", "r", "rn", "u"]].to_numpy(),
        analytical[:T],
        rtol=1e-9,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        x1["Pi"].to_numpy()[:-1],
        beta * x1["Pi"].to_numpy()[1:] + kappa * x1["y"].to_numpy()[:-1],
        rtol=1e-9,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        x1["y"].to_numpy()[:-1],
        x1["y"].to_numpy()[1:] - x1["r"].to_numpy()[:-1]
        + x1["Pi"].to_numpy()[1:] - x1["u"].to_numpy()[:-1],
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
