# -*- coding: utf-8 -*-

import time
import dill
import econsieve
import numpy as np
import scipy.linalg as sl
import pandas as pd
from collections import deque
from econsieve import KalmanFilter, TEnKF
from grgrlib import timeprint
from grgrlib.multiprocessing import serializer
from econsieve.stats import logpdf
from .engine import require_regime
from .exceptions import UnresolvedForecastError
from .precision import current_precision
from .tools import o_func as model_observation, t_func as model_transition

# Older EconSieve versions do not offer the explicit precision-rejection policy.
_precision_required = getattr(econsieve, "ForecastPrecisionRequired", ())


def create_obs_cov(self, scale_obs=0.1):

    self.Z = np.array(self.data)
    sig_obs = np.var(self.Z, axis=0) * scale_obs ** 2
    obs_cov = np.diagflat(sig_obs)

    return obs_cov


def _stationary_initial_root(transition, impact, shock_cov, covariance):
    """Factor the stationary shock response directly when rounding P can inflate its weak directions.

    sqrtm on an already rounded, singular P can turn an O(eps) eigenvalue
    error into an O(sqrt(eps)) initial draw. Factoring the impulse responses
    before squaring also retains genuine small innovations.
    """
    eig = sl.eigvalsh((covariance + covariance.T) * 0.5)
    eps = np.finfo(float).eps
    if eig[-1] == 0 and not np.any(shock_cov):
        return None
    if eig[0] > 64 * len(eig) * eps * max(abs(eig)):
        return None  # The existing principal-root calculation is well conditioned.

    noise, vectors = sl.eigh(shock_cov)
    if noise[0] < -len(noise) * eps * max(abs(noise)):
        raise ValueError("The shock covariance must be positive semidefinite.")
    factor = impact @ (vectors * np.sqrt(np.maximum(noise, 0)))
    power = transition
    # If B B' sums 2^j terms of the stationary Gramian, [B, F^(2^j) B]
    # factors the sum for 2^(j+1) terms. QR keeps at most dim_x columns.
    for _ in range(64):
        if sl.norm(power, np.inf) <= eps:
            break
        block = np.column_stack((factor, power @ factor))
        _, triangular = sl.qr(block.T, mode="economic")
        factor = triangular.T
        power = power @ power
    else:
        raise FloatingPointError("No stationary initial covariance factor could be established.")
    # A triangular factor has the right covariance but rotates the seeded draws.
    # Preserve the principal, symmetric root used by the existing QMC convention.
    u, singular, _ = sl.svd(factor, full_matrices=False)
    return (u * singular) @ u.T


def _stationary_initialization(self, shocks, *, reduced, sampling_root):
    require_regime(getattr(self, "regime_available", None), 1, 0)
    pmat, qmat = self.precalc_mat[:2]
    block = qmat[1, 0][:-self.neps]
    F, E = block[:, :-self.neps], block[:, -self.neps:]
    if not reduced:
        F = np.vstack((pmat[1, 0][:, :-self.neps], F))
        E = np.vstack((pmat[1, 0][:, -self.neps:], E))
    noise = E @ shocks @ E.T
    if reduced:
        transition = F
        covariance = sl.solve_discrete_lyapunov(F, noise)
    else:
        p4 = sl.solve_discrete_lyapunov(F[self.dimp:, :], noise[self.dimp:, self.dimp:])
        covariance = F @ p4 @ F.T + noise
        transition = np.pad(F, ((0, 0), (self.dimp, 0)))
    root = _stationary_initial_root(transition, E, shocks, covariance) if sampling_root else None
    return covariance, root


def get_p_init_lyapunov(self, Q):
    return _stationary_initialization(self, Q, reduced=False, sampling_root=False)[0]


def get_eps_lin(self, x, xp, rcond=1e-14):
    """Get filter-implied (smoothed) shocks for linear model"""

    require_regime(getattr(self, "regime_available", None), 1, 0)
    qmat = self.precalc_mat[1]

    if self.filter.name == "KalmanFilter":
        pmat = self.precalc_mat[0]
        F = self.filter.F
        E = np.vstack(
            (pmat[1, 0][:, -self.neps:], qmat[1, 0][: -self.neps, -self.neps:])
        )

    else:
        F = qmat[1, 0][:, : -self.neps]
        E = qmat[1, 0][:, -self.neps:]

    return np.linalg.pinv(E, rcond) @ (x - F @ xp)


def create_filter(
    self,
    R=None,
    N=None,
    ftype=None,
    seed=None,
    incl_obs=False,
    reduced_form=False,
    **fargs
):

    self.Z = np.array(self.data)

    if ftype == "KalmanFilter":
        ftype = "KF"
    elif ftype == "ParticleFilter":
        ftype = "PF"
    elif ftype == "AuxiliaryParticleFilter":
        ftype = "APF"
    elif ftype == "InversionFilter":
        ftype = "IF"

    if ftype == "KF":

        f = KalmanFilter(dim_x=self.dimx, dim_z=self.nobs, **fargs)

    elif ftype == "IF":

        from grgrwip.inversion_filter import InversionFilter

        omitted_obs_ind = self.observables.index(fargs['omitted_obs'])
        omitted_shock_ind = self.shocks.index(fargs['omitted_shock'])

        f = InversionFilter(self.neps, omitted_shock_ind=omitted_shock_ind,
                            omitted_obs_ind=omitted_obs_ind, x_bar=fargs['x_bar'], **fargs)

    elif ftype in ("PF", "APF"):

        print(
            "Warning: Particle filter is experimental and currently not under development."
        )
        from .pfilter import ParticleFilter

        if N is None:
            N = 10000

        aux_bs = ftype == "APF"
        f = ParticleFilter(
            N=N, dim_x=self.dimx, dim_z=self.nobs, auxiliary_bootstrap=aux_bs, **fargs
        )

    else:
        ftype = "TEnKF"

        if N is None:
            N = int((self.dimq - self.dimeps) ** 2 / 2) + 1

        dimx = self.dimq - self.dimeps if reduced_form else self.dimx
        f = TEnKF(N=N, dim_x=dimx, dim_z=self.nobs, seed=seed, **fargs)
        f.reduced_form = reduced_form

    if R is not None:
        f.R = R

    # use lyapunov equation as default. Otherwise to be defined manually via `*.filter.p`
    f.init_cov = None

    # QQ currently contains diagonal standard deviations. Keep matrix squaring;
    # see parser._get_shock_scales for the unresolved off-diagonal semantics.
    scales = self.QQ(self.ppar) if hasattr(self, "QQ") else self.fdict["QQ"]
    f.Q = scales @ scales
    self.filter = f

    return f


def get_ll(self, **args):
    try:
        return run_filter(self, smoother=False, get_ll=True, **args)
    except np.linalg.LinAlgError as exc:
        raise FloatingPointError(
            f"{self.filter.name} could not evaluate the log likelihood: {exc}"
        ) from exc


def run_filter(
    self,
    smoother=True,
    get_ll=False,
    init_cov=None,
    dispatch=None,
    rcond=1e-14,
    seed=None,
    verbose=False,
    ll_contributions=None,
):

    if verbose:
        st = time.time()

    self.Z = np.array(self.data)
    if ll_contributions is not None and (self.filter.name != "TEnKF" or not get_ll):
        raise ValueError("Per-observation likelihood recording requires the TEnKF likelihood.")

    if self.Z.shape[1] == 1:
        raise NotImplementedError(
            "Filtering with just one single time series is not implemented.")

    if self.Z.shape[1] != self.neps:
        raise NotImplementedError(
            f"The number of shocks must equal the number of observables, but is {self.neps} and {self.Z.shape[1]}.")

    dimp = self.dimp

    if init_cov is not None:
        self.filter.init_cov = init_cov

    # assign current transition & observation functions (of parameters)
    batch_options = {}
    if self.filter.name == "KalmanFilter":

        require_regime(getattr(self, "regime_available", None), 1, 0)
        pmat = self.precalc_mat[0]
        qmat = self.precalc_mat[1]

        F = np.vstack(
            (pmat[1, 0][:, : -self.neps], qmat[1, 0]
             [: -self.neps, : -self.neps])
        )
        F = np.pad(F, ((0, 0), (dimp, 0)))

        self.filter.F = F
        self.filter.H = np.hstack((self.hx[0], self.hx[1])), self.hx[2]

        if self.filter.Q.shape[0] == self.neps:
            E = np.vstack(
                (pmat[1, 0][:, -self.neps:], qmat[1, 0]
                 [: -self.neps, -self.neps:])
            )
            self.filter.Q = E @ self.filter.Q @ E.T

        if self.filter.init_cov is None:
            p4 = sl.solve_discrete_lyapunov(
                F[dimp:, dimp:], self.filter.Q[dimp:, dimp:]
            )
            self.filter.P = F[:, dimp:] @ p4 @ F.T[dimp:] + self.filter.Q
        else:
            self.filter.P = self.filter.init_cov

    elif self.filter.name == "InversionFilter":

        self.filter.t_func = self.t_func
        self.filter.pmat, self.filter.qmat = self.precalc_mat[:2]
        self.filter.hx = self.hx

    elif dispatch or self.filter.name == "ParticleFilter":
        from .engine import func_dispatch

        t_func_jit, o_func_jit, get_eps_jit = func_dispatch(self, full=True)
        self.filter.t_func = t_func_jit
        self.filter.o_func = o_func_jit
        self.filter.get_eps = get_eps_jit

    else:
        if self.filter.reduced_form:
            self.filter.t_func = lambda *x: self.t_func(*x, get_obs=True)
            self.filter.o_func = None
        else:
            self.filter.t_func = self.t_func
            self.filter.o_func = self.o_func
        if self.filter.init_cov is None:
            self.filter.P, root = _stationary_initialization(
                self, self.filter.Q, reduced=self.filter.reduced_form, sampling_root=self.filter.name == "TEnKF")
            if root is not None:
                batch_options["initial_cov_root"] = root
        else:
            self.filter.P = self.filter.init_cov

    self.filter.get_eps = self.get_eps_lin

    if (self.filter.name == "TEnKF" and not dispatch and hasattr(self, "regime_paths")
            and getattr(self.filter, "precise_forecast", ...) is None):
        reduced = self.filter.reduced_form
        predict_state = self.t_func
        # A custom transition can change X or Y even when decorated with
        # functools.wraps. An observer may explicitly opt in to native replay.
        native_forecast = (getattr(predict_state, "__func__", predict_state) is model_transition
                           or getattr(predict_state, "preserves_model_forecast", False))
        if (native_forecast
                and (reduced or getattr(self.o_func, "__func__", None) is model_observation)):
            # Keep the actual member choices so a retry repeats neither the regime search nor the draws.
            regimes = deque(maxlen=self.filter.N)

            def predict(state, shocks):
                result, pair, flag = self.t_func(state, shocks, get_obs=reduced, return_k=True)
                regimes.append(pair)
                return result, flag

            self.filter.t_func = predict
            batch_options["precise_forecast"] = lambda states, shocks: self.regime_paths.forecast_ensemble(
                regimes, states, shocks, self.hx, reduced)

    if self.filter.name == "KalmanFilter":

        means, covs, ll = self.filter.batch_filter(self.Z)

        if smoother:
            means, covs, _, _ = self.filter.rts_smoother(
                means, covs, inv=lambda x: np.linalg.pinv(x, rcond=rcond)
            )

        if get_ll:
            res = ll
        else:
            means = means
            res = (means, covs)

    elif self.filter.name == "InversionFilter":

        if smoother:
            raise NotImplementedError()

        if not get_ll:
            res = self.filter.run(self.Z)
        else:
            res = self.filter.get_lprob(self.Z)

    elif self.filter.name == "ParticleFilter":

        res = self.filter.batch_filter(self.Z)

        if smoother:

            if verbose > 0:
                print(
                    "[run_filter:]".ljust(15, " ")
                    + " Filtering done after %s seconds, starting smoothing..."
                    % np.round(time.time() - st, 3)
                )

            if isinstance(smoother, bool):
                smoother = 10
            res = self.filter.smoother(smoother)

    else:

        smoother &= not get_ll

        try:
            res = self.filter.batch_filter(
                self.Z, calc_ll=get_ll, store=smoother, seed=seed, verbose=verbose > 0,
                **batch_options,
                **({"ll_contributions": ll_contributions} if ll_contributions is not None else {})
            )
        except _precision_required as exc:
            raise UnresolvedForecastError(str(exc)) from exc
        finally:
            stats = current_precision.get()
            if stats is not None and self.filter.name == "TEnKF":
                work = stats.filter
                work.requests += self.filter.precision_requests
                work.attempts += self.filter.high_precision_updates
                work.completed += self.filter.high_precision_completed
                work.seconds += self.filter.high_precision_seconds

        if smoother:
            res = self.filter.rts_smoother(res, rcond=rcond)

    if get_ll:
        if np.isnan(res) or np.isposinf(res):
            raise FloatingPointError(f"Filter returned an invalid log likelihood: {res}.")
        self.ll = res
    else:
        self.X = res

    if verbose > 0:
        mess = "[run_filter:]".ljust(15, " ") + " Filtering done in %s." % timeprint(
            time.time() - st, 3
        )
        if get_ll:
            mess += "Likelihood is %s." % res
        print(mess)

    return res


def extract(
    self,
    sample=None,
    nsamples=1,
    init_cov=None,
    precalc=True,
    seed=0,
    nattemps=4,
    accept_failure=False,
    verbose=True,
    debug=False,
    l_max=None,
    k_max=None,
    **npasargs
):
    """Extract the timeseries of (smoothed) shocks.

    Parameters
    ----------
    sample : array, optional
        Provide one or several parameter vectors used for which the smoothed shocks are calculated (default is the current `self.par`)
    nsamples : int, optional
        Number of `npas`-draws for each element in `sample`. Defaults to 1
    nattemps : int, optional
        Number of attemps per sample to crunch the sample with a different seed. Defaults to 4

    Returns
    -------
    tuple
        The result(s)
    """

    import tqdm
    import os
    from grgrlib import map2arr

    if sample is None:
        if type(self).__name__ == "DSGE_DUMMY":
            sample = None
        else:
            sample = self.par

    if np.ndim(sample) <= 1:
        sample = [sample]

    np.random.seed(seed)

    fname = self.filter.name
    verbose = max(verbose, debug)

    if getattr(self, "pool", None) is not None:
        from .estimation import create_pool

        create_pool(self)

    run_filter = serializer(self.run_filter)

    if fname == "ParticleFilter":
        raise NotImplementedError

    elif fname == "KalmanFilter":
        if nsamples > 1:
            print(
                "[extract:]".ljust(15, " ")
                + " Setting `nsamples` to 1 as the linear filter does not rely on sampling."
            )
        nsamples = 1
        debug = not hasattr(self, "debug") or self.debug
        self.debug = True

    else:
        if self.filter.reduced_form:
            self.create_filter(
                R=self.filter.R, N=self.filter.N, reduced_form=False)

            print(
                "[extract:]".ljust(15, " ")
                + " Extraction requires filter in non-reduced form. Recreating filter instance."
            )

        run_filter, npas = serializer(self.run_filter, self.filter.npas)

    self.debug |= debug

    if sample[0] is not None:
        set_par = serializer(self.set_par)

    t_func = serializer(self.t_func)
    edim = len(self.shocks)
    xdim = len(self.vv)
    odim = len(self.observables)

    obs_func = serializer(self.obs)
    filter_get_eps = serializer(self.get_eps_lin)

    dimeps = self.dimeps
    dimp = self.dimp

    # win explodes with 2**32
    seeds = np.random.randint(2 ** 31, size=nsamples)
    sample = [(x, y) for x in sample for y in seeds]

    def runner(arg):

        par, seed_loc = arg

        if par is not None:
            set_par(par, l_max=l_max, k_max=k_max)

        res = run_filter(verbose=verbose > 2, seed=seed_loc, init_cov=init_cov)

        if fname == "KalmanFilter":
            means, covs = res
            res = means.copy()
            resid = np.empty((means.shape[0] - 1, dimeps))

            for t, x in enumerate(means[1:]):
                resid[t] = filter_get_eps(x, res[t])
                res[t + 1] = t_func(res[t], resid[t], linear=True)[0]

            return par, res[0], resid, 0

        sample = np.dstack((obs_func(res), res[..., dimp:]))
        # Pair each target with its own initial state. Choose the member from
        # a local stream so the path does not depend on worker scheduling.
        members = np.random.default_rng(seed_loc).permutation(len(res))

        def t_func_loc(states, eps):

            (q, pobs), flag = t_func(states, eps, get_obs=True)

            return np.hstack((pobs, q)), flag

        for natt in range(nattemps):
            member = members[natt % len(members)]
            try:
                init, resid, flags = npas(
                    func=t_func_loc,
                    X=np.roll(sample, -int(member), axis=0),
                    init_states=res[[member], 0, :],
                    verbose=max(len(sample) == 1, verbose - 1),
                    seed=(int(seed_loc) + natt) % 2**32,
                    nsamples=1,
                    **npasargs
                )

                return par, init, resid[0], flags

            except ValueError as e:
                # CMA-ES can draw an entire population of infeasible shocks.
                if str(e) != "Sample returns infs only.":
                    raise
                raised_error = e

        if accept_failure:
            print(
                "[extract:]".ljust(15, " ")
                + "got an error: '%s' (after %s unsuccessful attemps)."
                % (raised_error, natt + 1)
            )
            return None
        else:
            raise ValueError(
                f"No feasible shock path found after {natt + 1} attempts: {raised_error}"
            ) from raised_error

    wrap = tqdm.tqdm if (verbose and len(sample) >
                         1) else (lambda x, **kwarg: x)
    res = wrap(
        self.mapper(runner, sample),
        unit=" sample(s)",
        total=len(sample),
        dynamic_ncols=True,
    )
    pars, init, resid, flags = map2arr(res)

    if hasattr(self, "pool") and self.pool:
        self.pool.close()

    if fname == "KalmanFilter":
        self.debug = debug

    if resid.shape[0] == 1:
        resid[0] = pd.DataFrame(
            resid[0], index=self.data.index[:-1], columns=self.shocks
        )

    edict = {"pars": pars, "init": init, "resid": resid, "flags": flags}

    return edict
