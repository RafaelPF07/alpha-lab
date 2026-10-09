"""Validation gate: the final signal runs once on the validation window.

This happens after the revision loop has finished, so no agent ever receives a
validation result. The ledger keeps full metrics. The event stream carries the pass/fail boolean
and, when the window cannot be scored, the data error text. That text names a validation-window
ticker and date and never a holdout value (DECISION-1). Only BacktestDataError is a data error;
any other exception is a bug in the harness and propagates, so the pipeline records failed_error.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import pandas as pd

from alpha_lab.backtest.engine import BacktestDataError, run_backtest
from alpha_lab.backtest.holding import hold_weights
from alpha_lab.checks.sandbox import SandboxedSignal, SandboxFailure, SandboxOutcome
from alpha_lab.config import Settings
from alpha_lab.data.splits import split_window


def run_signal(sandboxed: SandboxedSignal, prices: pd.DataFrame) -> SandboxOutcome:
    """The one place the pipeline calls SandboxedSignal.run (round-2 SEC-05).

    An error that escapes run(), such as a Windows cleanup error or a malformed child result, says
    nothing about the signal. It is raised as SandboxFailure, so the attempt records aborted_error
    and the pipeline failed_error. Nothing is written as a validation verdict.
    """
    try:
        return sandboxed.run(prices)
    except Exception as exc:  # noqa: BLE001 - re-raised as a harness failure, never a verdict
        raise SandboxFailure(
            SandboxOutcome(
                status="exception",
                error_type="SandboxHarnessError",
                message=f"the sandbox failed after its child process ran: {type(exc).__name__}",
            )
        ) from exc


@dataclass(frozen=True)
class ValidationOutcome:
    passed: bool
    metrics: dict | None
    returns: dict[str, float] | None
    error: str | None
    # The attempt status to record when the window gives no verdict. The signal is neither passed nor
    # failed, so the attempt must not count as passing the critic (R4-WHOLE-03):
    #   failed_sandbox   the sandbox did not return weights (timeout, crash, bad output, all zero)
    #   failed_backtest  the window could not be scored (NaN inside it, or the data stops)
    #   failed_gate      the validation Sharpe or turnover is not a number (the training gate does the same)
    # None when the signal received a verdict, pass or fail.
    no_verdict_status: str | None = None


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
    outcome = run_signal(sandboxed, prices)
    if not outcome.ok:
        # Every non-ok status is the same as on the training window (failed_sandbox), so the
        # validation window cannot turn a sandbox failure into a verdict (R4-AGT-08).
        return ValidationOutcome(
            False,
            None,
            None,
            f"sandbox {outcome.status}: {outcome.message}",
            no_verdict_status="failed_sandbox",
        )

    lower, upper = split_window(settings.dates, "validation")
    try:
        held = hold_weights(outcome.weights, holding_days)
        result = run_backtest(held, prices, lower, upper, settings.backtest)
    except BacktestDataError as exc:  # NaN inside the window, or the window runs past the data
        return ValidationOutcome(
            False,
            None,
            None,
            f"validation could not be scored: {exc}",
            no_verdict_status="failed_backtest",
        )
    metrics = {k: float(v) for k, v in asdict(result.metrics).items()}
    returns = {str(day.date()): float(v) for day, v in result.net_returns.items()}

    if not (math.isfinite(metrics["sharpe"]) and math.isfinite(metrics["avg_daily_turnover"])):
        # A NaN compares False with every limit, so it would be recorded as validation_fail.
        return ValidationOutcome(
            False,
            None,
            None,
            "validation gate cannot be applied: the Sharpe ratio or the turnover is not a finite "
            "number, because the scoring window has no variance",
            no_verdict_status="failed_gate",
        )

    gates = settings.gates
    passed = (
        metrics["sharpe"] >= gates.min_validation_sharpe
        and metrics["avg_daily_turnover"] <= gates.max_turnover_daily
    )
    return ValidationOutcome(passed, metrics, returns, None)
