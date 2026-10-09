"""Validation gate: the final signal runs once on the validation window.

This happens after the revision loop has finished, so no agent ever receives a
validation result. The ledger keeps full metrics. The only thing that leaves
this module as an event payload is the pass/fail boolean.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from alpha_lab.backtest.engine import run_backtest
from alpha_lab.backtest.holding import hold_weights
from alpha_lab.checks.sandbox import SandboxedSignal
from alpha_lab.config import Settings
from alpha_lab.data.splits import split_window


@dataclass(frozen=True)
class ValidationOutcome:
    passed: bool
    metrics: dict | None
    returns: dict[str, float] | None
    error: str | None


def validate_signal(
    code: str,
    validation_prices: pd.DataFrame,
    settings: Settings,
    holding_days: int = 1,
) -> ValidationOutcome:
    """Run the signal on history up to validation_end, then score only the validation window."""
    prices = validation_prices.loc[: pd.Timestamp(settings.dates.validation_end)]
    sandboxed = SandboxedSignal(
        code, settings.sandbox.timeout_seconds, settings.sandbox.allowed_imports
    )
    outcome = sandboxed.run(prices)
    if not outcome.ok:
        return ValidationOutcome(False, None, None, f"sandbox {outcome.status}: {outcome.message}")

    lower, upper = split_window(settings.dates, "validation")
    try:
        held = hold_weights(outcome.weights, holding_days)
        result = run_backtest(held, prices, lower, upper, settings.backtest)
    except ValueError as exc:  # for example, the validation window has no data
        return ValidationOutcome(False, None, None, f"validation could not be scored: {exc}")
    metrics = {k: float(v) for k, v in asdict(result.metrics).items()}
    returns = {str(day.date()): float(v) for day, v in result.net_returns.items()}

    gates = settings.gates
    passed = (
        metrics["sharpe"] >= gates.min_validation_sharpe
        and metrics["avg_daily_turnover"] <= gates.max_turnover_daily
    )
    return ValidationOutcome(passed, metrics, returns, None)
