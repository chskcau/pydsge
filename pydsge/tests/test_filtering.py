"""Numerical Kalman references and integration tests for ensemble filtering."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pydsge import DSGE


def _state_mapping(mod, reduced_form=False):
    loadings = {"x": [1.0, 0.0], "z": [0.0, 1.0], "r": [0.0, 1.0]}
    names = mod.svv if reduced_form else mod.vv
    return np.array([loadings[name] for name in names])


def _load_gaussian_model():
    filename = Path(__file__).parent / "resources" / "linear_gaussian.yaml"
    mod = DSGE.read(str(filename))
    observations = np.array(
        [
            [0.4, -0.2],
            [-0.3, 0.5],
            [0.2, 0.1],
            [0.7, -0.4],
            [-0.1, 0.3],
            [0.1, -0.6],
        ]
    ) + np.array([1.25, -0.8])
    mod.load_data(pd.DataFrame(observations, columns=["X", "Z"]))
    return mod


@pytest.fixture(params=["stationary", "specified"])
def linear_gaussian_model(request):
    mod = _load_gaussian_model()
    mod.prep_estim(
        linear=True,
        ncores=0,
        seed=0,
        dry_run=False,
        verbose=False,
        R=np.array([[0.09, 0.025], [0.025, 0.16]]),
    )
    initial_cov = None
    if request.param == "specified":
        initial_cov = np.array([[0.8, 0.15], [0.15, 0.5]])
        mapping = _state_mapping(mod)
        mod.filter.init_cov = mapping @ initial_cov @ mapping.T
    return mod, initial_cov


def _gaussian_reference(data, rho_x, sig_x, initial_cov, smoother=False):
    """Condition the explicit joint distribution, without a Kalman recursion."""
    transition = np.diag([rho_x, 0.3])
    shock_cov = np.diag([sig_x**2, 0.7**2])
    observation_matrix = np.array([[1.0, 0.4], [-0.3, 1.2]])
    observation_cov = np.array([[0.09, 0.025], [0.025, 0.16]])
    if initial_cov is None:
        initial_cov = np.diag(np.diag(shock_cov) / (1 - np.diag(transition)**2))

    T = len(data)
    # x[-1] and all innovations are independent Gaussian vectors.
    input_cov = np.kron(np.eye(T + 1), shock_cov)
    input_cov[:2, :2] = initial_cov
    loadings = np.zeros((2 * T, 2 * (T + 1)))
    for t in range(T):
        rows = slice(2 * t, 2 * (t + 1))
        loadings[rows, :2] = np.linalg.matrix_power(transition, t + 1)
        for j in range(t + 1):
            loadings[rows, 2 * (j + 1):2 * (j + 2)] = np.linalg.matrix_power(
                transition, t - j
            )
    state_cov = loadings @ input_cov @ loadings.T
    measurement = np.kron(np.eye(T), observation_matrix)
    data_cov = measurement @ state_cov @ measurement.T + np.kron(
        np.eye(T), observation_cov
    )
    residual = (np.asarray(data) - np.array([1.25, -0.8])).ravel()
    sign, logdet = np.linalg.slogdet(data_cov)
    assert sign > 0
    log_likelihood = -0.5 * (
        residual.size * np.log(2 * np.pi)
        + logdet
        + residual @ np.linalg.solve(data_cov, residual)
    )

    means = np.empty((T, 2))
    covariances = np.empty((T, 2, 2))
    for t in range(T):
        rows = slice(2 * t, 2 * (t + 1))
        end = 2 * T if smoother else 2 * (t + 1)
        cross_cov = (state_cov[rows] @ measurement.T)[:, :end]
        conditioned_cov = data_cov[:end, :end]
        means[t] = cross_cov @ np.linalg.solve(conditioned_cov, residual[:end])
        covariances[t] = state_cov[rows, rows] - cross_cov @ np.linalg.solve(
            conditioned_cov, cross_cov.T
        )
    return means, covariances, log_likelihood


@pytest.mark.parametrize("smoother", [False, True], ids=["filtered", "smoothed"])
def test_linear_gaussian_filter(linear_gaussian_model, smoother):
    mod, initial_cov = linear_gaussian_model
    expected_means, expected_covs, _ = _gaussian_reference(
        mod.data, 0.6, 0.4, initial_cov, smoother=smoother
    )
    mapping = _state_mapping(mod)
    means, covariances = mod.run_filter(smoother=smoother)
    np.testing.assert_allclose(
        means, expected_means @ mapping.T, rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        covariances, mapping @ expected_covs @ mapping.T, rtol=1e-10, atol=1e-12
    )


@pytest.mark.parametrize(
    "parameters",
    [
        pytest.param([0.6, 0.4], id="calibration"),
        pytest.param([-0.35, 0.65], id="changed-persistence-and-shock-scale"),
    ],
)
def test_linear_gaussian_likelihood(linear_gaussian_model, parameters):
    mod, initial_cov = linear_gaussian_model
    _, _, expected_ll = _gaussian_reference(mod.data, *parameters, initial_cov)
    parameters = np.array(parameters)
    log_prior = mod.lprob(parameters, temp=0)
    log_posterior = mod.lprob(parameters)
    assert np.isfinite(log_prior)
    np.testing.assert_allclose(
        log_posterior - log_prior, expected_ll, rtol=1e-10, atol=1e-12
    )
    for _ in range(2):
        np.testing.assert_allclose(mod.get_ll(), expected_ll, rtol=1e-10, atol=1e-12)


@pytest.fixture(params=[True, False], ids=["reduced", "full"])
def ensemble_model(request):
    mod = _load_gaussian_model()
    mod.prep_estim(
        linear=False,
        reduced_form=request.param,
        N=16,
        ncores=0,
        seed=0,
        dry_run=False,
        verbose=False,
        R=np.array([[0.09, 0.025], [0.025, 0.16]]),
    )
    assert mod.filter.name == "TEnKF"
    assert mod.filter.reduced_form == request.param
    assert mod.filter.dim_x == (2 if request.param else 3)
    assert mod.filter.dim_z == 2
    assert mod.filter.N == 16
    return mod


def _record_filter_calls(mod, monkeypatch, likelihood=-12.5):
    calls = []

    def batch_filter(data, **kwargs):
        calls.append(
            {
                "data": data.copy(),
                "Q": mod.filter.Q.copy(),
                "R": mod.filter.R.copy(),
                "P": mod.filter.P.copy(),
                **kwargs,
            }
        )
        return likelihood

    monkeypatch.setattr(mod.filter, "batch_filter", batch_filter)
    return calls


@pytest.mark.parametrize("shock_z", [0.3, -6.0], ids=["unbound", "binding"])
def test_ensemble_filter_model_functions(ensemble_model, monkeypatch, shock_z):
    mod = ensemble_model
    _record_filter_calls(mod, monkeypatch)
    mod.get_ll()

    previous = {"x": 0.2, "z": -0.4, "r": -0.4}
    names = mod.svv if mod.filter.reduced_form else mod.vv
    state = np.array([previous[name] for name in names])
    shocks = {"e_x": 0.1, "e_z": shock_z}
    shocks = np.array([shocks[name] for name in mod.shocks])
    result, flag = mod.filter.t_func(state, shocks)
    assert flag == 0

    z = 0.3 * previous["z"] + shock_z
    r = max(z, -5.0)
    x = 0.6 * previous["x"] + r - z + 0.1
    expected = {"x": x, "z": z, "r": r}
    expected_observations = np.array(
        [x + 0.4 * z + 1.25, -0.3 * x + 1.2 * z - 0.8]
    )
    if mod.filter.reduced_form:
        actual_state, observations = result
        assert mod.filter.o_func is None
    else:
        actual_state = result
        observations = mod.filter.o_func(actual_state[None, :])[0]
    np.testing.assert_allclose(
        actual_state, [expected[name] for name in names], rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        observations, expected_observations, rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize(
    "specified_cov", [False, True], ids=["stationary", "specified"]
)
def test_ensemble_filter_covariances(ensemble_model, monkeypatch, specified_cov):
    mod = ensemble_model
    calls = _record_filter_calls(mod, monkeypatch)
    mapping = _state_mapping(mod, mod.filter.reduced_form)
    latent_cov = (
        np.array([[0.8, 0.15], [0.15, 0.5]])
        if specified_cov
        else np.diag([0.4**2 / (1 - 0.6**2), 0.7**2 / (1 - 0.3**2)])
    )
    expected_cov = mapping @ latent_cov @ mapping.T
    kwargs = {"init_cov": expected_cov.copy()} if specified_cov else {}
    assert mod.get_ll(seed=17, **kwargs) == -12.5
    assert mod.ll == -12.5
    assert len(calls) == 1
    call = calls[0]
    np.testing.assert_array_equal(call["data"], mod.data.to_numpy())
    np.testing.assert_allclose(call["Q"], np.diag([0.4**2, 0.7**2]))
    np.testing.assert_allclose(call["R"], [[0.09, 0.025], [0.025, 0.16]])
    np.testing.assert_allclose(call["P"], expected_cov, rtol=1e-12, atol=1e-12)
    assert call["calc_ll"] is True
    assert not call["store"]
    assert call["seed"] == 17
    assert not call["verbose"]


def test_ensemble_filter_likelihood_updates(ensemble_model, monkeypatch):
    mod = ensemble_model
    calls = _record_filter_calls(mod, monkeypatch)
    mapping = _state_mapping(mod, mod.filter.reduced_form)
    names = list(mod.svv if mod.filter.reduced_form else mod.vv)
    previous = {"x": 0.2, "z": -0.4, "r": -0.4}
    state = np.array([previous[name] for name in names])
    for rho_x, sig_x in [(0.6, 0.4), (-0.35, 0.65)]:
        parameters = np.array([rho_x, sig_x])
        log_prior = mod.lprob(parameters, temp=0)
        log_posterior = mod.lprob(parameters, temp=0.4)
        assert np.isfinite(log_prior)
        np.testing.assert_allclose(log_posterior, log_prior + 0.4 * (-12.5))
        np.testing.assert_allclose(calls[-1]["Q"], np.diag([sig_x**2, 0.7**2]))
        np.testing.assert_allclose(calls[-1]["R"], [[0.09, 0.025], [0.025, 0.16]])
        latent_cov = np.diag([sig_x**2 / (1 - rho_x**2), 0.7**2 / (1 - 0.3**2)])
        np.testing.assert_allclose(
            calls[-1]["P"], mapping @ latent_cov @ mapping.T, rtol=1e-12, atol=1e-12
        )
        result, flag = mod.filter.t_func(state, np.array([0.1, 0.3]))
        assert flag == 0
        actual_state = result[0] if mod.filter.reduced_form else result
        np.testing.assert_allclose(
            actual_state[names.index("x")], rho_x * 0.2 + 0.1,
            rtol=1e-12, atol=1e-12,
        )
    assert len(calls) == 2


def test_ensemble_filter_invalid_likelihood(ensemble_model, monkeypatch):
    mod = ensemble_model
    _record_filter_calls(mod, monkeypatch, likelihood=np.nan)
    assert np.isneginf(mod.get_ll())
    assert np.isneginf(mod.ll)
    assert np.isneginf(mod.lprob(np.array([0.6, 0.4])))


def test_ensemble_filter_run(ensemble_model):
    mod = ensemble_model
    # The full stationary covariance is singular because r=z.
    samples = mod.run_filter(
        smoother=False, seed=17, init_cov=np.eye(mod.filter.dim_x)
    )
    assert samples.shape == (16, len(mod.data), mod.filter.dim_x)
    assert np.isfinite(samples).all()
    assert mod.X is samples
    assert np.isfinite(mod.get_ll(seed=17))
