"""Optional, evaluation-local diagnostics for high-precision calculations."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from time import perf_counter


@dataclass
class PrecisionWork:
    requests: int = 0
    attempts: int = 0
    completed: int = 0
    seconds: float = 0.0


@dataclass
class PrecisionDiagnostics:
    regime: PrecisionWork = field(default_factory=PrecisionWork)
    filter: PrecisionWork = field(default_factory=PrecisionWork)


current_precision = ContextVar("pydsge_precision", default=None)


@contextmanager
def collect_precision():
    """Collect work in the current evaluation, including work before a rejection."""
    stats = PrecisionDiagnostics()
    token = current_precision.set(stats)
    try:
        yield stats
    finally:
        current_precision.reset(token)


def timed_regime_operation(method):
    """Count outermost high-precision solver operations, including failed attempts."""
    @wraps(method)
    def measured(self, *args, **kwargs):
        stats = current_precision.get()
        if stats is None:
            return method(self, *args, **kwargs)
        work = stats.regime
        work.requests += 1
        # Each measured method includes any work it does in the shared rule builder.
        if self.regime_fallback_decimal_digits is None:
            return method(self, *args, **kwargs)
        started = perf_counter()
        work.attempts += 1
        try:
            result = method(self, *args, **kwargs)
            work.completed += 1
            return result
        finally:
            work.seconds += perf_counter() - started
    return measured
