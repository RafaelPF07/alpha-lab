"""The only module allowed to read holdout-period data. Run by a human, once per signal.

`lab holdout` is the only entry point. It needs an explicit flag, lists the
signals, and asks for confirmation. Each signal is recorded as "running" in the
ledger before any holdout price is read, so a crash cannot leave a signal
untested-on-record while its holdout data has been touched.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from alpha_lab.backtest.engine import run_backtest
from alpha_lab.backtest.holding import hold_weights
from alpha_lab.checks.sandbox import SandboxedSignal
from alpha_lab.config import DATA_DIR, Settings
from alpha_lab.data import loader
from alpha_lab.data.splits import split_window
from alpha_lab.ledger.db import Ledger


class HoldoutRefused(RuntimeError):
    """Raised when a holdout run is not allowed. Nothing has been read or recorded."""


@dataclass(frozen=True)
class HoldoutCandidate:
    attempt_id: str
    name: str
    code: str
    holding_days: int = 1


@dataclass(frozen=True)
class HoldoutResult:
    attempt_id: str
    name: str
    status: str
    metrics: dict | None
    error: str | None


def holdout_slice(settings: Settings, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Holdout-period prices only (validation_end + 1 day through holdout_end)."""
    return loader.read_holdout_cache(data_dir).loc[: pd.Timestamp(settings.dates.holdout_end)]


def check_not_already_run(ledger: Ledger, candidates: list[HoldoutCandidate]) -> None:
    already = [c.name for c in candidates if ledger.get_holdout(c.attempt_id) is not None]
    if already:
        raise HoldoutRefused(
            "holdout has already been run for: "
            + ", ".join(already)
            + ". Each signal gets exactly one holdout run."
        )


def run_holdout(
    *,
    ledger: Ledger,
    run_id: str,
    settings: Settings,
    candidates: list[HoldoutCandidate],
    i_understand_this_is_final: bool,
    confirm: Callable[[list[HoldoutCandidate]], bool],
    data_dir: Path = DATA_DIR,
) -> list[HoldoutResult]:
    """Run the holdout once for each candidate. Refuses unless every check passes."""
    if not i_understand_this_is_final:
        raise HoldoutRefused("refusing to run: pass --i-understand-this-is-final")
    if not candidates:
        raise HoldoutRefused("no validated signals were selected")
    check_not_already_run(ledger, candidates)
    if not confirm(candidates):
        raise HoldoutRefused("not confirmed, nothing was run")

    # Holdout prices are read here, inside this module, and nowhere else.
    pre = loader.read_pre_holdout(data_dir)
    holdout = loader.read_holdout_cache(data_dir)
    full = pd.concat([pre, holdout]).sort_index()
    full = full.loc[: pd.Timestamp(settings.dates.holdout_end)]
    lower, upper = split_window(settings.dates, "holdout")

    results: list[HoldoutResult] = []
    for candidate in candidates:
        holdout_id = uuid.uuid4().hex
        ledger.add_holdout(holdout_id, candidate.attempt_id, run_id, "running", None)
        try:
            sandboxed = SandboxedSignal(
                candidate.code, settings.sandbox.timeout_seconds, settings.sandbox.allowed_imports
            )
            outcome = sandboxed.run(full)
            if not outcome.ok:
                raise RuntimeError(f"sandbox {outcome.status}: {outcome.message}")
            held = hold_weights(outcome.weights, candidate.holding_days)
            result = run_backtest(held, full, lower, upper, settings.backtest)
            metrics = {k: float(v) for k, v in asdict(result.metrics).items()}
            ledger.update_holdout(candidate.attempt_id, "complete", metrics)
            results.append(
                HoldoutResult(candidate.attempt_id, candidate.name, "complete", metrics, None)
            )
        except Exception as exc:  # noqa: BLE001 - recorded, and the attempt stays blocked
            ledger.update_holdout(candidate.attempt_id, "error", None)
            results.append(
                HoldoutResult(candidate.attempt_id, candidate.name, "error", None, str(exc))
            )
    return results
