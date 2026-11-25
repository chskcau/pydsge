
pydsge
======

.. |badge0| image:: https://badge.fury.io/py/pydsge.svg
    :target: https://badge.fury.io/py/pydsge

.. |badge1| image:: https://github.com/gboehl/pydsge/actions/workflows/continuous-integration.yml/badge.svg
    :target: https://github.com/gboehl/pydsge/actions/workflows/continuous-integration.yml


|badge0| |badge1| 

----

A package for solving, filtering and estimating **linear** DSGE models with the ZLB (or other occasionally binding constraints).

The set of methods is introduced in the paper `Estimation of DSGE Models with the Effective Lower Bound <https://gregorboehl.com/live/bczlb_BS.pdf>`_ *(Gregor Boehl & Felix Strobel, 2023, JEDC)*, where we also estimate the medium-scale New Keynesian model to post-2008 US data.

Check out my `Econpizza <https://github.com/gboehl/econpizza>`_ package if you are interested in simulating **nonlinear** DSGE models with (or without) **heterogeneous agents**.

A collection of models that can be (and were) used with this package can be found in `another repo <https://github.com/gboehl/projectlib/tree/master/yamls>`_.

Installation
-------------

Installing the stable version is as simple as typing

.. code-block:: bash

   pip install pydsge

in your terminal (Linux/MacOS) or Anaconda Prompt (Win). 

For development on this fork, keep the EconSieve and grgrlib forks in sibling
directories named ``../econsieve`` and ``../grgrlib``. From the pydsge checkout,
run ``uv sync --locked`` to install both in editable mode. Run the tests with
``uv run pytest`` or select this checkout's ``.venv`` interpreter in VS Code.

Documentation
-------------

Documentation can be found on `ReadTheDocs <https://pydsge.readthedocs.io/en/latest/index.html>`_:

- `Installation Guide <https://pydsge.readthedocs.io/en/latest/installation_guide.html>`_
- `Getting Started <https://pydsge.readthedocs.io/en/latest/getting_started.html>`_
- `Module Documentation <https://pydsge.readthedocs.io/en/latest/modules.html>`_

Checking the constraint path
----------------------------

For a given state and current innovations, the solver sets future innovations
to zero. A candidate ``(L, K)`` has ``L`` initially free periods followed by
``K`` binding periods and then the free decision rule. ``(0, 0)`` denotes
the entirely free continuation. A new innovation next period can change
the selected regime.

By default, an endogenous candidate that passes the switching-date tests must
also satisfy the constraint in **every period on the grid**, through the latest
release date represented by the grid. This includes the free continuation
after an early release. For binding candidates the solver checks the path
against the model equations before their RQ transformation. An interior
violation makes the search continue in its original order. Use
``model.gen_sys(..., verify_regime_path=False)`` or
``model.prep_estim(..., verify_regime_path=False)`` to request the
switching-date checks alone. Explicitly imposing a regime with ``set_k``
still permits counterfactual paths.

Both settings monitor the conditioning of the backward recursion. When the
estimated condition number exceeds the inverse square root of machine
epsilon, a candidate or a successor depending on it is recomputed on demand
from the original equations. The default precision is 200 significant decimal
digits. Set ``regime_fallback_decimal_digits`` to a positive integer in
``gen_sys``, ``set_par`` or ``prep_estim`` to change it. Later calls on the
model inherit the setting, including after saving and loading model metadata.
With ``None``, the solver still checks conditioning and follows
``verify_regime_path``. If a required regime needs a high-precision check,
it raises ``UnresolvedRegimeError``. In the likelihood this follows the
usual unresolved-proposal handling.

For example, with a TEnKF that supports configurable precision, the following
call uses 300 digits for the regime solver and 100 digits for the filter.

.. code-block:: python

   model.prep_estim(regime_fallback_decimal_digits=300, fallback_decimal_digits=100)

The free terminal policy and model inputs retain their original Float64 precision.

To measure one likelihood evaluation, use
``with pydsge.precision.collect_precision() as stats:`` around the call to
``model.lprob(...)`` (import ``pydsge.precision`` first). The attributes
``stats.regime`` and ``stats.filter`` count requests, attempts, completed
operations and elapsed seconds. Measurements belong to the calling execution
context. They include work up to a rejection and preserve the usual return type.
With the local EconSieve fork, ``fallback_decimal_digits=None`` keeps forecast
checks active. A required retry then yields the proposal outcome
``unresolved_forecast`` when ``return_evaluation=True``.

Citation
--------

**pydsge** is developed by Gregor Boehl to simulate, filter, and estimate DSGE models with the zero lower bound on nominal interest rates in various applications (see `my website <https://gregorboehl.com>`_ for research papers using the package). Please cite it with

.. code-block:: latex

    @TechReport{boehl2022meth,
      title = {{Estimation of DSGE Models with the Effective Lower Bound}},
      author = {Boehl, Gregor and Strobel, Felix},
      journal = {Journal of Economic Dynamics and Control},
      volume = {158},
      year = {2022},
      publisher = {Elsevier}
    }

.. code-block:: latex

    @techreport{boehl2022obc,
      title = Efficient solution and computation of models with occasionally binding constraints},
      author = {Boehl, Gregor},
      journal = {Journal of Economic Dynamics and Control},
      volume = {143},
      year = {2022},
      publisher = {Elsevier}
    }


We appreciate citations for **pydsge** because it helps us to find out how people have
been using the package and it motivates further work.


Parser
------

The parser originally was a fork of Ed Herbst's fork from Pablo Winant's (excellent) package **dolo**. 

See https://github.com/EconForge/dolo and https://github.com/eph.

Specify shock standard deviations under ``calibration.covariances_sqrt``:

.. code-block:: yaml

   calibration:
     covariances_sqrt:
       e_u: sig_u
       e_z: sig_z

Only independent innovations are currently supported: each entry is a finite,
non-negative standard deviation, not a variance. Off-diagonal entries raise
``NotImplementedError``. The legacy name ``covariances`` retains the same
numerical meaning but emits a warning because the name is misleading. Specifying
both section names raises ``ValueError``.


References
----------

Boehl, Gregor (2022). `Efficient Solution and Computation of Models with Occasionally Binding Constraints <http://gregorboehl.com/live/obc_boehl.pdf>`_. *Journal of Economic Dynamics and Control*
