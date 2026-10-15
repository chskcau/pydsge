"""This file contains the test for the estimation of the model."""

import os
import numpy as np
import pandas as pd
from pydsge import DSGE, example


def test_estimation(tmp_path, create=False):

    yaml, data = example
    mod = DSGE.read(yaml)

    d0 = pd.read_csv(data, index_col='date', parse_dates=True)

    # adjust elb
    zlb = mod.get_par('elb_level')
    rate = d0['FFR']
    d0['FFR'] = np.maximum(rate, zlb)

    mod.load_data(d0, start='1998Q1')

    mod.prep_estim(N=150, seed=0, verbose=True)

    mod.filter.R = mod.create_obs_cov(1e-1)
    ind = mod.observables.index('FFR')
    mod.filter.R[ind, ind] /= 1e1

    # sample pars from prior
    p0 = mod.prior_sampler(22, check_likelihood=False, verbose=True)

    backend_file = tmp_path / "dfi_sampler.h5"
    try:
        mod.mcmc(
            p0, nsteps=20, tune=5, update_freq=0, backend=str(backend_file)
        )
    finally:
        mod.pool.clear()
    # mod.save()

    mod.mcmc_summary()
    chain = mod.sampler.get_chain()
    log_prob = mod.sampler.get_log_prob()
    assert backend_file.is_file()
    assert chain.shape == (20, 22, len(mod.prior))
    assert log_prob.shape == chain.shape[:2]
    assert np.isfinite(chain).all()
    # -inf is allowed for walkers with zero posterior density.
    assert not np.isnan(log_prob).any()
    assert not np.isposinf(log_prob).any()
    assert np.isfinite(log_prob).any(axis=1).all()
    np.testing.assert_array_equal(
        mod.get_chain(backend_file=str(backend_file)), chain
    )
    this_sample = chain[0, 0]

    pth = os.path.dirname(__file__)
    save_path = os.path.join(pth, "resources", "estimation.npz")

    if create:
        with open(save_path, "wb") as f:
            np.save(f, this_sample)

    else:
        with open(save_path, "rb") as f:

            test_sample = np.load(f, allow_pickle=False)
            np.testing.assert_allclose(this_sample, test_sample)
            print(this_sample)
            print(test_sample)
