# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as nl
import time
import sys
from numba import njit, prange
from numba.extending import register_jitable

from .exceptions import UnresolvedRegimeError

aca = np.ascontiguousarray
si_eps = sys.float_info.epsilon


@njit(cache=True, nogil=True)
def _get_lam_and_condition(omg, psi, S, T, V, W, h, l):

    dimp, dimq = omg.shape

    A = S if l else V
    B = T if l else W
    c = np.zeros(dimq) if l else h[:dimq]

    block = A[:dimq, :dimq] + aca(A[:dimq, dimq:]) @ omg
    inv = nl.inv(block)
    lam = inv @ aca(B[:dimq, :dimq])
    xi = inv @ (c - aca(A[:dimq, dimq:]) @ psi)

    # The inverse is already available. Estimate amplification of Float64
    # perturbations, including those introduced by the preceding RQ rotation.
    condition = nl.norm(block, np.inf) * nl.norm(inv, np.inf)
    return lam, xi, condition


@njit(cache=True, nogil=True)
def get_lam(omg, psi, S, T, V, W, h, l):
    lam, xi, _ = _get_lam_and_condition(omg, psi, S, T, V, W, h, l)
    return lam, xi


@njit(cache=True, nogil=True)
def get_omg(omg, psi, lam, xi, S, T, V, W, h, l):

    dimp, dimq = omg.shape

    A = S if l else V
    B = T if l else W
    c = np.zeros(dimp) if l else h[dimq:]

    dum = A[dimq:, :dimq] + aca(A[dimq:, dimq:]) @ omg
    psi = dum @ xi + aca(A[dimq:, dimq:]) @ psi - c
    omg = dum @ lam - aca(B[dimq:, :dimq])

    return omg, psi


@njit(cache=True)
def _initialize_grid(S, T, V, W, h, omg, lam, l_max, k_max):

    dimp, dimq = omg.shape
    l_max += 1
    k_max += 1

    T22 = T[dimq:, dimq:]
    if nl.cond(T22) > 1 / si_eps:
        print(
            "[preprocess:]".ljust(15, " ")
            + " WARNING: at least one control indetermined"
        )

    T22i = nl.inv(T22)
    T[dimq:] = T22i @ aca(T[dimq:])
    S[dimq:] = T22i @ aca(S[dimq:])

    W22 = W[dimq:, dimq:]
    W22i = nl.inv(W22)
    W[dimq:] = W22i @ aca(W[dimq:])
    V[dimq:] = W22i @ aca(V[dimq:])
    h[dimq:] = W22i @ aca(h[dimq:])

    pmat = np.empty((l_max, k_max, dimp, dimq))
    qmat = np.empty((l_max, k_max, dimq, dimq))
    pterm = np.empty((l_max, k_max, dimp))
    qterm = np.empty((l_max, k_max, dimq))

    pmat[0, 0] = omg
    pterm[0, 0] = np.zeros(dimp)
    qmat[0, 0] = lam
    qterm[0, 0] = np.zeros(dimq)
    return pmat, qmat, pterm, qterm


@njit(cache=True)
def _fill_cell(system, policy, l, k, prev_l, prev_k):
    S, T, V, W, h = system
    pmat, qmat, pterm, qterm = policy
    omg, psi = pmat[prev_l, prev_k], pterm[prev_l, prev_k]
    qmat[l, k], qterm[l, k], condition = _get_lam_and_condition(omg, psi, S, T, V, W, h, l)
    pmat[l, k], pterm[l, k] = get_omg(
        omg, psi, qmat[l, k], qterm[l, k], S, T, V, W, h, l
    )
    return not np.isfinite(condition) or condition > 1 / np.sqrt(si_eps)


@njit(cache=True)
def _fill_grid(system, policy):
    suspect = np.zeros(policy[0].shape[:2], dtype=np.bool_)
    for l in range(policy[0].shape[0]):
        for k in range(policy[0].shape[1]):
            if l or k:
                prev_l, prev_k = max(l - 1, 0), k if l else k - 1
                ill_conditioned = _fill_cell(system, policy, l, k, prev_l, prev_k)
                suspect[l, k] = ill_conditioned or suspect[prev_l, prev_k]
    return suspect


def _recover_grid(system, policy):
    """Recompute independent cells after a singular regime inversion."""
    available = np.ones(policy[0].shape[:2], dtype=bool)
    suspect = np.zeros_like(available)
    for l, k in np.ndindex(available.shape):
        if l == k == 0:
            continue
        prev_l, prev_k = max(l - 1, 0), k if l else k - 1
        if available[prev_l, prev_k]:
            try:
                ill_conditioned = _fill_cell(system, policy, l, k, prev_l, prev_k)
            except nl.LinAlgError as exc:
                if not _singular_inversion(exc):
                    raise
            else:
                if all(np.isfinite(matrix[l, k]).all() for matrix in policy):
                    suspect[l, k] = bool(ill_conditioned) or suspect[prev_l, prev_k]
                    continue
        available[l, k] = False
        for matrix in policy:
            matrix[l, k] = np.nan
    return available, suspect


def _singular_inversion(exc):
    return str(exc) in ("Matrix is singular to machine precision.", "Singular matrix")


@register_jitable(inline="always")
def _constraint_grid(policy, fq1, fp1, fq0):
    pmat, qmat, pterm, qterm = policy
    l_max, k_max, dimq = qterm.shape
    bmat = np.empty((5, l_max, k_max, dimq))
    bterm = np.empty((5, l_max, k_max))

    for l in range(0, l_max):
        for k in prange(0, k_max):
            # initialize local lam, xi to iterate upon
            lam = np.eye(dimq)
            xi = np.zeros(dimq)

            for s in range(l + k + 1):

                l_loc = max(l - s, 0)
                k_loc = max(min(k, k + l - s), 0)

                y2r = fp1 @ pmat[l_loc, k_loc] + fq1 @ qmat[l_loc, k_loc] + fq0
                cr = fp1 @ pterm[l_loc, k_loc] + fq1 @ qterm[l_loc, k_loc]

                if s == 0:
                    bmat[0, l, k] = y2r @ lam
                    bterm[0, l, k] = cr + y2r @ xi
                if s == l - 1:
                    bmat[1, l, k] = y2r @ lam
                    bterm[1, l, k] = cr + y2r @ xi
                if s == l:
                    bmat[2, l, k] = y2r @ lam
                    bterm[2, l, k] = cr + y2r @ xi
                if s == l + k - 1:
                    bmat[3, l, k] = y2r @ lam
                    bterm[3, l, k] = cr + y2r @ xi
                if s == l + k:
                    bmat[4, l, k] = y2r @ lam
                    bterm[4, l, k] = cr + y2r @ xi

                lam = qmat[l_loc, k_loc] @ lam
                xi = qmat[l_loc, k_loc] @ xi + qterm[l_loc, k_loc]

    return bmat, bterm


def preprocess_jittable(S, T, V, W, h, fq1, fp1, fq0, omg, lam, x_bar, l_max, k_max, suspect=None):
    policy = _initialize_grid(S, T, V, W, h, omg, lam, l_max, k_max)
    sensitivity = _fill_grid((S, T, V, W, h), policy)
    if suspect is not None:
        suspect[:] = sensitivity
    return policy + _constraint_grid(policy, fq1, fp1, fq0)


preprocess_jit = njit(preprocess_jittable, cache=True, nogil=True)
preprocess_jit_parallel = njit(
    preprocess_jittable, cache=True, nogil=True, parallel=True
)
_partial_bounds = njit(_constraint_grid, cache=True, nogil=True)
_partial_bounds_parallel = njit(_constraint_grid, cache=True, nogil=True, parallel=True)


@njit(cache=True, nogil=True, parallel=True)
def preprocess_tmats_jit(pmat, pterm, qmat, qterm, fq1, fp1, fq0, omg, l_max, k_max):
    """jitted preprocessing of system matrices until (l_max, k_max)"""

    dimp, dimq = omg.shape
    l_max += 1
    k_max += 1

    tmat = np.empty((l_max + k_max, l_max, k_max, dimq))
    tterm = np.empty((l_max + k_max, l_max, k_max))

    for l in range(0, l_max):
        for k in prange(0, k_max):

            # initialize local lam, xi to iterate upon
            lam = np.eye(dimq)
            xi = np.zeros(dimq)

            for s in range(l_max + k_max):

                l_loc = max(l - s, 0)
                k_loc = max(min(k, k + l - s), 0)

                y2r = fp1 @ pmat[l_loc, k_loc] + fq1 @ qmat[l_loc, k_loc] + fq0
                cr = fp1 @ pterm[l_loc, k_loc] + fq1 @ qterm[l_loc, k_loc]
                tmat[s, l, k] = y2r @ lam
                tterm[s, l, k] = cr + y2r @ xi

                lam = qmat[l_loc, k_loc] @ lam
                xi = qmat[l_loc, k_loc] @ xi + qterm[l_loc, k_loc]

    return tmat, tterm


def preprocess(self, PU, MU, PR, MR, gg, fq1, fp1, fq0, parallel=False, verbose=False):
    """dispatcher to jitted preprocessing"""

    l_max, k_max = self.lks
    omg, lam, x_bar = self.sys

    st = time.time()
    preprocess_jit_loc = preprocess_jit_parallel if parallel else preprocess_jit
    system = PU, MU, PR, MR, gg
    available = None
    suspect = np.empty((l_max + 1, k_max + 1), dtype=bool)
    try:
        # Normalization in the compiled kernel is in-place. Retain the inputs
        # so that a failed recursion can be replayed cell by cell.
        matrices = preprocess_jit_loc(
            *(a.copy(order="K") for a in system), fq1, fp1, fq0, omg, lam, x_bar, l_max, k_max, suspect
        )
    except nl.LinAlgError as exc:
        if not _singular_inversion(exc):
            raise
        # Singular common blocks still fail here, before any regime can be used.
        policy = _initialize_grid(*system, omg, lam, l_max, k_max)
        available, suspect = _recover_grid(system, policy)
        bounds = _partial_bounds_parallel if parallel else _partial_bounds
        matrices = policy + bounds(policy, fq1, fp1, fq0)
    if hasattr(self, "precalc_tmat"):
        del self.precalc_tmat
    self.precalc_mat = matrices
    self.regime_available = available

    if verbose:
        print(
            "[preprocess:]".ljust(15, " ")
            + " Preprocessing finished within %ss." % np.round((time.time() - st), 5)
        )

    return suspect


def preprocess_tmats(self, fq1, fp1, fq0, verbose):
    """dispatcher to jitted preprocessing"""

    l_max, k_max = self.lks
    omg, lam, x_bar = self.sys

    st = time.time()
    pmat, qmat, pterm, qterm, bmat, bterm = self.precalc_mat
    self.precalc_tmat = preprocess_tmats_jit(
        pmat, pterm, qmat, qterm, fq1, fp1, fq0, omg, l_max, k_max
    )

    if verbose:
        print(
            "[preprocess_tmats:]".ljust(15, " ")
            + " Preprocessing finished within %ss." % np.round((time.time() - st), 5)
        )

    return


@njit(cache=True, nogil=True)
def t_func_jit(
    pmat,
    pterm,
    qmat,
    qterm,
    bmat,
    bterm,
    x_bar,
    hxp,
    hxq,
    hxc,
    state,
    shocks,
    set_l,
    set_k,
    x_space,
    available=None,
):
    """jitted transitiona function"""

    s = np.hstack((state, shocks))

    if set_k == -1:
        # find (l,k) if requested
        l, k, flag = find_lk(bmat, bterm, x_bar, s, available)
    else:
        l, k = set_l, set_k
        flag = 0
        require_regime(available, l, k)

    p = pmat[l, k] @ s + pterm[l, k]
    q = aca(qmat[l, k]) @ s + qterm[l, k]

    # either return p or obs
    if x_space:
        p_or_obs = hxp @ p + hxq @ q + hxc
    else:
        p_or_obs = p

    return p_or_obs, q, l, k, flag


@njit(nogil=True, cache=True)
def require_regime(available, l, k):
    if available is None:
        return
    if not available[l, k]:
        raise UnresolvedRegimeError(l, k)


@njit(nogil=True, cache=True)
def find_lk(bmat, bterm, x_bar, q, available=None, start=0, pending=None, repaired=None):
    """iteration loop to find (l,k) given state q"""

    _, l_max, k_max = bterm.shape

    flag = 0
    l, k = 0, 0

    # check if (0,0) is a solution
    while start == 0 and check_cnst(bmat, bterm, 2, l, 0, q, available) - x_bar > 0:
        l += 1
        if l == l_max:
            break

    # check if (0,k) is a solution
    if l < l_max:
        # needs to be wrapped so that both loops can be exited at once
        l, k = first_binding_regime(bmat, bterm, x_bar, q, available, start, pending, repaired)

        # if still no solution, use approximation
        if l == 999:
            flag = 1
            l, k = 0, 0
            loop_type = False

            if k_max == 1:
                return 0, 0, 2
            while True:
                k += 1
                if not loop_type:
                    # first iterate until r < r_bar
                    loop_type = check_cnst(bmat, bterm, 4, 0, k, q, available) - x_bar < 0
                elif check_cnst(bmat, bterm, 4, 0, k, q, available) - x_bar > 0:
                    # then iterate until r > r_bar again
                    break
                if k == k_max - 1:
                    # set error flag 'no solution + k_max reached'
                    flag = 2
                    break
    else:
        l = 0

    return l, k, flag


@njit(nogil=True, cache=True)
def first_binding_regime(bmat, bterm, x_bar, q, available=None, start=0, pending=None, repaired=None):
    """iterate over (l,k) until (l_max, k_max) and check if RE equilibrium"""
    _, l_max, k_max = bterm.shape

    for index in range(max(start - 1, 0), l_max * (k_max - 1)):
        l, offset = divmod(index, k_max - 1)
        k = offset + 1
        require_regime(available, l, k)
        if pending is not None and pending[l, k]:
            return l, k
        for slot in (0, 1, 4, 2, 3):
            if (slot == 0 and l == 0) or (slot == 1 and l <= 1) or (slot == 3 and k == 1):
                continue
            row = bmat[slot, l, k]
            margin = (row @ q + bterm[slot, l, k]) - x_bar
            if slot == 2 or slot == 3:
                margin = -margin
            if np.isfinite(margin):
                if margin >= 0:
                    continue
                if pending is None:
                    break
                # Even a well-conditioned cell can have a dot product whose
                # sign is lost to cancellation for a particular initial state.
                scale = abs(bterm[slot, l, k]) + abs(x_bar)
                for j in range(q.size):
                    scale += abs(row[j]) * abs(q[j])
                if -margin > (q.size + 3) * si_eps * scale:
                    break
            elif pending is None:
                raise UnresolvedRegimeError(l, k)
            # Let the caller resolve this ambiguous test before advancing.
            # On revisiting a repaired cell the other cheap tests still apply.
            if pending is not None and (repaired is None or not repaired[l, k]):
                pending[l, k] = True
                return l, k
        else:
            return l, k

    return 999, 999


bruite_wrapper = first_binding_regime  # Previous public spelling.


@njit(nogil=True, cache=True)
def check_cnst(bmat, bterm, s, l, k, q0, available=None):
    """constraint value in period s given CDR-state q0 under the assumptions (l,k)"""
    require_regime(available, l, k)
    value = bmat[s, l, k] @ q0 + bterm[s, l, k]
    if not np.isfinite(value):
        raise UnresolvedRegimeError(l, k)
    return value
