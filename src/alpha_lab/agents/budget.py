"""A shared spend limit for one run. Checked before every LLM call."""

from __future__ import annotations


class BudgetExceeded(RuntimeError):
    """Raised when the next call could take the run over its budget."""


class Budget:
    def __init__(self, limit_usd: float) -> None:
        if limit_usd <= 0:
            raise ValueError("budget must be positive")
        self.limit_usd = limit_usd
        self.spent_usd = 0.0
        self.reserved_usd = 0.0

    @property
    def remaining_usd(self) -> float:
        return self.limit_usd - self.spent_usd - self.reserved_usd

    def reserve(self, estimate_usd: float) -> None:
        """Hold back the worst-case cost of a call. Raises before anything is sent."""
        if self.spent_usd + self.reserved_usd + estimate_usd > self.limit_usd:
            raise BudgetExceeded(
                f"next call estimated at ${estimate_usd:.4f} but only "
                f"${self.remaining_usd:.4f} remains of ${self.limit_usd:.2f}"
            )
        self.reserved_usd += estimate_usd

    def settle(self, estimate_usd: float, actual_usd: float) -> None:
        """Release the reservation and record what the call really cost."""
        self.reserved_usd -= estimate_usd
        self.spent_usd += actual_usd
