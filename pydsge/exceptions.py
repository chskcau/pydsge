class InvalidProposalError(ValueError):
    """The proposed parameters violate a known model or solution condition."""


class UnresolvedForecastError(InvalidProposalError):
    """A required high-precision forecast update was disabled."""


class UnresolvedRegimeError(InvalidProposalError):
    """A required regime could not be resolved numerically."""

    def __str__(self):
        l, k = self.args
        return f"Regime (L={l}, K={k}) is numerically unresolved."
