"""Check a conditional path against the model equations before the RQ rotation."""

import numpy as np
from mpmath import mp
from numba import njit
from scipy.sparse import bsr_matrix
from scipy.sparse.linalg import splu

from .engine import find_lk, require_regime
from .exceptions import UnresolvedRegimeError
from .precision import timed_regime_operation

DEFAULT_REGIME_FALLBACK_DIGITS = 200


def resolve_regime_fallback_digits(value, metadata):
    """Use the saved setting when omitted and disable recovery on explicit None."""
    if value is ...:
        value = metadata.get(
            "regime_fallback_decimal_digits", DEFAULT_REGIME_FALLBACK_DIGITS
        )
        if isinstance(value, np.ndarray) and value.ndim == 0:
            value = value.item()  # A scalar read from a saved model's .npz metadata.
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(
            "regime_fallback_decimal_digits must be a positive integer or None."
        )
    return int(value)


@njit(cache=True, nogil=True)
def _path_status(rows, initial, bound, l, k):
    """Return -1 for a violation, 1 for feasibility, or 0 for uncertain signs."""
    status = 1
    eps = (len(initial) + 2) * np.finfo(np.float64).eps
    for t in range(len(rows)):
        value, scale = 0.0, abs(bound)
        for j in range(len(initial)):
            term = rows[t, j] * initial[j]
            value += term
            scale += abs(term)
        margin = bound - value if l <= t < l + k else value - bound
        if (
            not np.isfinite(margin)
            or not np.isfinite(scale)
            or abs(margin) <= eps * scale
        ):
            status = 0
        elif margin < 0:
            return -1
    return status


class RegimePaths:
    """Independent conditional solutions for one parameter vector."""

    # Also supplies the original precision for older pickled solvers.
    regime_fallback_decimal_digits = DEFAULT_REGIME_FALLBACK_DIGITS

    def __init__(
        self,
        system,
        terminal,
        constraint,
        suspect,
        *,
        regime_fallback_decimal_digits=DEFAULT_REGIME_FALLBACK_DIGITS,
    ):
        self.regime_fallback_decimal_digits = regime_fallback_decimal_digits
        pu, mu, pr, mr, h = system
        # Indexed by whether the current date is free (t < L).
        self.system = ((pr, mr, h), (pu, mu, np.zeros_like(h)))
        omega, lam, self.bound = terminal
        self.d = lam.shape[0]
        self.terminal = np.column_stack((np.vstack((lam, omega)), np.zeros(len(h))))
        fq, fp, self.lag = constraint
        self.shadow = np.r_[fq, fp]
        self.pending = suspect.copy()
        self.repaired = np.zeros_like(suspect)
        self.paths = {}
        # All candidates share the finite horizon through the latest possible release.
        self.horizon = sum(suspect.shape) - 2
        # traj() also exposes one date beyond the search horizon.
        self.free = np.empty((self.horizon + 2, self.d))
        self.free[0] = self.shadow @ self.terminal[:, : self.d] + self.lag
        for i in range(1, len(self.free)):
            self.free[i] = self.free[i - 1] @ lam
        self._precision = None

    def __getstate__(self):
        # A cloned mpmath context is process-local. Rebuild its memoized rules
        # from the model equations if an estimation worker needs them again.
        return dict(self.__dict__, _precision=None)

    def restore_free_cells(self, matrices):
        # With K=0 every date follows the same terminal policy. No backward
        # inversion is needed to recover an ill-conditioned redundant cell.
        p, q, pc, qc, b, bc = matrices
        for l in np.flatnonzero(self.pending[:, 0]):
            p[l, 0], q[l, 0] = (
                self.terminal[self.d :, :-1],
                self.terminal[: self.d, :-1],
            )
            pc[l, 0], qc[l, 0] = 0.0, 0.0
            b[:, l, 0] = self.free[[0, max(l - 1, 0), l, max(l - 1, 0), l]]
            bc[:, l, 0] = 0.0
            self.pending[l, 0] = False

    def _solve_path(self, l, k):
        """Solve all dates together, with p_T = omega q_(T-1) at release."""
        d, periods = self.d, l + k
        n = self.terminal.shape[0]
        blocks, columns, starts = [], [], [0]
        rhs = np.zeros((periods, n, d + 1))
        for t in range(periods):
            a, b, h = self.system[t < l]
            # A [q_t; p_(t+1)] = B [q_(t-1); p_t] + h
            diag = np.column_stack((a[:, :d], -b[:, d:]))
            if t:
                lower = np.zeros_like(a)
                lower[:, :d] = -b[:, :d]
                blocks.append(lower)
                columns.append(t - 1)
            else:
                rhs[t, :, :d] = b[:, :d]
            rhs[t, :, -1] = h
            if t == periods - 1:
                diag[:, :d] += a[:, d:] @ self.terminal[d:, :d]
            blocks.append(diag)
            columns.append(t)
            if t < periods - 1:
                upper = np.zeros_like(a)
                upper[:, d:] = a[:, d:]
                blocks.append(upper)
                columns.append(t + 1)
            starts.append(len(columns))
        system = bsr_matrix(
            (blocks, columns, starts), shape=(periods * n, periods * n)
        ).tocsc()
        system.eliminate_zeros()
        rhs = rhs.reshape(-1, d + 1)
        try:
            solution = splu(system).solve(rhs)
        except RuntimeError as exc:
            if "singular" in str(exc).lower():
                return None
            raise
        with np.errstate(over="ignore", invalid="ignore"):
            residual = abs(system @ solution - rhs)
            scale = abs(system) @ abs(solution) + abs(rhs)
        if not (
            np.isfinite(scale).all()
            and np.isfinite(residual).all()
            and np.all(residual <= 32 * n * np.finfo(float).eps * scale)
        ):
            return None
        solution = solution.reshape(periods, n, d + 1)
        rows = self.shadow @ solution
        rows[0, :d] += self.lag
        rows[1:] += self.lag @ solution[:-1, :d]
        tail = self.free[: len(self.free) - periods] @ solution[-1, :d]
        return solution[0].copy(), np.vstack((rows, tail))

    def _precise_rules(self, l, k):
        """Backward substitution in original coordinates at the configured precision."""
        if self.regime_fallback_decimal_digits is None:
            raise UnresolvedRegimeError(l, k)
        if self._precision is None:
            ctx = mp.clone()
            ctx.dps = self.regime_fallback_decimal_digits
            systems = [
                tuple(ctx.matrix(x.tolist()) for x in system) for system in self.system
            ]
            terminal = ctx.matrix(self.terminal.tolist())
            shadow = ctx.matrix(self.shadow.tolist()).T
            lag = ctx.matrix([*self.lag.tolist(), 0.0]).T
            free = shadow * terminal + lag
            endpoints = ctx.matrix(5, terminal.cols)
            for slot in range(5):
                endpoints[slot, :] = free
            rules = {(0, 0): (terminal, endpoints)}
            self._precision = ctx, systems, rules, shadow, lag
        ctx, systems, rules, shadow, lag = self._precision
        d = self.d
        # Only the (0,j) prefix and the (i,k) column are predecessors.
        for i, j in [(0, j) for j in range(1, k + 1)] + [
            (i, k) for i in range(1, l + 1)
        ]:
            if (i, j) in rules:
                continue
            previous, previous_bounds = rules[(i - 1, j) if i else (0, j - 1)]
            successor = previous[d:, :]
            a, b, h = systems[bool(i)]
            lhs = ctx.matrix(a.rows)
            lhs[:, :d] = a[:, :d] + a[:, d:] * successor[:, :d]
            lhs[:, d:] = -b[:, d:]
            rhs = ctx.matrix(a.rows, d + 1)
            rhs[:, :d] = b[:, :d]
            rhs[:, d:] = h - a[:, d:] * successor[:, d:]
            rule = lhs**-1 * rhs
            # An endpoint one or more dates ahead is the successor's endpoint
            # evaluated at this cell's next state. Reuse it across the grid.
            next_state = ctx.eye(d + 1)
            next_state[:d, :] = rule[:d, :]
            endpoints = previous_bounds * next_state
            now = shadow * rule + lag
            endpoints[0, :] = now
            if i <= 1:
                endpoints[1, :] = now
            if i == 0:
                endpoints[2, :] = now
                if j == 1:
                    endpoints[3, :] = now
            rules[i, j] = rule, endpoints
        return ctx, rules

    @timed_regime_operation
    def _precise_path(self, l, k, initial=None, length=None):
        """Propagate affine maps, or a particular state, without intermediate rounding."""
        ctx, rules = self._precise_rules(l, k)
        d = self.d
        q = ctx.eye(d + 1) if initial is None else ctx.matrix(initial.tolist())
        shadow, lag = (
            ctx.matrix(self.shadow.tolist()).T,
            ctx.matrix(self.lag.tolist()).T,
        )
        rows = ctx.matrix(self.horizon + 1 if length is None else length, q.cols)
        for t in range(rows.rows):
            i = max(l - t, 0)
            j = min(k, max(l + k - t, 0))
            current = rules[i, j][0] * q
            rows[t, :] = shadow * current + lag * q[:d, :]
            q[:d, :] = current[:d, :]
        return rows

    @timed_regime_operation
    def _repair(self, l, k, matrices):
        try:
            _, rules = self._precise_rules(l, k)
            first, bounds = rules[l, k]
        except ZeroDivisionError as exc:
            raise UnresolvedRegimeError(l, k) from exc
        first = np.asarray(first.tolist(), dtype=float)
        bounds = np.asarray(bounds.tolist(), dtype=float)
        if not (np.isfinite(first).all() and np.isfinite(bounds).all()):
            raise UnresolvedRegimeError(l, k)
        p, q, pc, qc, b, bc = matrices
        p[l, k], pc[l, k] = first[self.d :, :-1], first[self.d :, -1]
        q[l, k], qc[l, k] = first[: self.d, :-1], first[: self.d, -1]
        b[:, l, k], bc[:, l, k] = bounds[:, :-1], bounds[:, -1]
        self.paths.pop((l, k), None)
        self.repaired[l, k] = True
        self.pending[l, k] = False

    def _prepare_path(self, l, k, matrices):
        if (l, k) in self.paths or self.repaired[l, k]:
            return
        solution = self._solve_path(l, k)
        if solution is not None:
            first, rows = solution
            p, q, pc, qc, b, bc = matrices
            grid_first = np.column_stack(
                (np.vstack((q[l, k], p[l, k])), np.r_[qc[l, k], pc[l, k]])
            )
            dates = (0, l - 1, l, l + k - 1, l + k)
            active = np.array([l > 0, l > 1, True, k > 1, True])
            grid_rows = np.column_stack((b[active, l, k], bc[active, l, k]))
            if (
                np.isfinite(rows).all()
                and np.allclose(first, grid_first, rtol=1e-8, atol=1e-12)
                and np.allclose(
                    rows[np.asarray(dates)[active]], grid_rows, rtol=1e-8, atol=1e-12
                )
            ):
                self.paths[l, k] = rows
                return
        self._repair(l, k, matrices)

    @timed_regime_operation
    def _precise_endpoints_valid(self, l, k, x):
        ctx, rules = self._precise_rules(l, k)
        values = rules[l, k][1] * ctx.matrix(x.tolist())
        return (
            (not l or values[0] >= self.bound)
            and (l <= 1 or values[1] >= self.bound)
            and values[2] <= self.bound
            and (k <= 1 or values[3] <= self.bound)
            and values[4] >= self.bound
        )

    def _valid(self, l, k, state, matrices, full):
        x = np.r_[state, 1.0]
        if not full:
            return self._precise_endpoints_valid(l, k, x)
        self._prepare_path(l, k, matrices)
        if (l, k) not in self.paths:
            rows = self._precise_path(l, k, length=len(self.free))
            self.paths[l, k] = np.asarray(rows.tolist(), dtype=float)
        rows = self.paths[l, k][: self.horizon + 1]
        status = _path_status(rows, x, self.bound, l, k)
        if status:
            return status > 0
        if not self.repaired[l, k]:
            self._repair(l, k, matrices)
        path = self._precise_path(l, k, x)
        return all(
            value <= self.bound if l <= t < l + k else value >= self.bound
            for t, value in enumerate(path)
        )

    def _free_valid(self, state):
        rows = self.free[: self.horizon + 1]
        status = _path_status(rows, state, self.bound, 0, 0)
        if status:
            return status > 0
        path = self._precise_path(0, 0, np.r_[state, 1.0])
        return all(value >= self.bound for value in path)

    def select(self, matrices, available, state, full):
        b, bc = matrices[4:]
        start = 0
        while True:
            l, k, flag = find_lk(
                b, bc, self.bound, state, available, start, self.pending, self.repaired
            )
            if flag:
                return l, k, flag
            if not k:
                if full and not self._free_valid(state):
                    start = 1  # The free path re-crosses the bound later in the grid.
                    continue
                return l, k, flag
            index = l * (bc.shape[2] - 1) + k
            if self.pending[l, k]:
                self._repair(l, k, matrices)
                start = index
            elif (not full and not self.repaired[l, k]) or self._valid(
                l, k, state, matrices, full
            ):
                return l, k, flag
            else:
                start = index + 1

    def imposed(self, l, k, matrices, available):
        require_regime(available, l, k)
        if self.pending[l, k]:
            self._repair(l, k, matrices)

    @timed_regime_operation
    def forecast_ensemble(self, regimes, states, shocks, hx, reduced=True):
        """Apply the selected conditional rules and measurement equations before rounding."""
        n = states.shape[1]
        nq = self.d - shocks.shape[1]
        inputs = np.vstack((states[states.shape[0] - nq:], shocks.T, np.ones(n)))
        x = np.empty(states.shape, dtype=object)
        y = np.empty((len(hx[2]), n), dtype=object)
        groups = {}
        for member, pair in enumerate(regimes):
            groups.setdefault(pair, []).append(member)
        for (l, k), members in groups.items():
            try:
                ctx, rules = self._precise_rules(l, k)
                current = rules[l, k][0] * ctx.matrix(inputs[:, members].tolist())
            except ZeroDivisionError as exc:
                raise UnresolvedRegimeError(l, k) from exc
            hxp, hxq, hxc = (ctx.matrix(a.tolist()) for a in hx)
            q = current[:nq, :]
            obs = hxp * current[self.d:, :] + hxq * q + hxc * ctx.ones(1, len(members))
            x[:, members] = q.tolist() if reduced else current[self.d:, :].tolist() + q.tolist()
            y[:, members] = obs.tolist()
        return x, y

    @timed_regime_operation
    def transition(self, l, k, state):
        ctx, rules = self._precise_rules(l, k)
        current = rules[l, k][0] * ctx.matrix([*state.tolist(), 1.0])
        value = np.array(list(current), dtype=float)
        if not np.isfinite(value).all():
            raise UnresolvedRegimeError(l, k)
        return value

    def trajectory(self, l, k, state, matrices):
        if not k:
            return self.free @ state
        self._prepare_path(l, k, matrices)
        if self.repaired[l, k]:
            path = self._precise_path(l, k, np.r_[state, 1.0], len(self.free))
            return np.array(list(path), dtype=float)
        return self.paths[l, k] @ np.r_[state, 1.0]

    @timed_regime_operation
    def release(self, l, k, state):
        ctx, rules = self._precise_rules(l, k)
        return float((rules[l, k][1][4, :] * ctx.matrix([*state.tolist(), 1.0]))[0])
