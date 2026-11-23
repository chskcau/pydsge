class InvalidProposalError(ValueError):
    """The proposed parameters violate a known model or solution condition."""


class UnresolvedRegimeError(InvalidProposalError):
    """A singular policy recursion prevents evaluation of a required regime."""

    def __str__(self):
        l, k = self.args
        return f"Regime (L={l}, K={k}) is numerically unresolved after a singular regime inversion."
