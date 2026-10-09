"""The only module that reads holdout-period prices. Run by a human, once per idea.

`lab holdout` is the only entry point. This module does not cache the holdout window on disk. It
is downloaded here, held in memory, and its SHA-256 is recorded with each run. Candidate code
receives the window in memory. The in-process refusals (checks/static.py and the runner) stop the
spellings the reviewers found, including the import route that passed a caller's globals (R4-SEC-01).
A candidate can still write the window to a file by a route no list names, and the runner is not
an operating-system boundary (R3-SEC-03, DECISION-4). The residual risk is documented in the README
sandbox section, and it is the reason every candidate's code should be read before `lab holdout`
runs. Nothing here closes that residual route.

Order of operations. Each step is a precondition for the next:
  1. Flag and candidates. An idea that already has a holdout row is skipped (one run per idea).
     The current static filter runs on each candidate's code. A refusal names the candidates and
     claims nothing (R5-F3).
  2. The pre-holdout cache must match the current settings. No holdout data is read here.
  3. Confirmation from the caller. Nothing is written if it declines.
  4. Claim: every eligible candidate gets a 'running' row, in one transaction. The check runs
     again inside that transaction, so a second process cannot slip in between.
  5. Download the holdout window in memory, hash it, run each candidate, and record the result.
  6. Every claimed row ends as 'complete' or 'error'. A failure or an interrupt marks the
     rows 'error', so the idea stays used (DECISION-14). A hard kill (taskkill /F, a power cut)
     runs no cleanup, so its rows stay 'running'. They still block a second run, and nothing
     marks them stale yet (R3-SEC-11).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import sqlite3
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path

import pandas as pd

from alpha_lab.backtest.engine import run_backtest
from alpha_lab.backtest.holding import hold_weights
from alpha_lab.checks.sandbox import SandboxedSignal
from alpha_lab.checks.static import static_check
from alpha_lab.config import DATA_DIR, Settings, Universe
from alpha_lab.data import loader
from alpha_lab.data.loader import PriceSource
from alpha_lab.data.splits import split_window
from alpha_lab.ledger.db import HoldoutConflict, Ledger

# Calendar days fetched before validation_end so forward-fill sees the last pre-holdout prices.
# The forward-fill limit is a few trading days, so two weeks is ample.
CONTEXT_CALENDAR_DAYS = 14


class HoldoutRefused(RuntimeError):
    """Raised when a holdout run is not allowed. Nothing has been requested or recorded."""


@dataclass(frozen=True)
class HoldoutCandidate:
    attempt_id: str
    idea_id: str
    name: str
    code: str
    holding_days: int = 1


@dataclass(frozen=True)
class HoldoutResult:
    attempt_id: str
    name: str
    status: str  # complete | error | skipped
    metrics: dict | None
    error: str | None


def holdout_sha256(frame: pd.DataFrame) -> str:
    """SHA-256 of a canonical JSON form: column names, ISO dates and float values.

    Keys are sorted and separators fixed. Floats use Python's shortest round-trip repr, and NaN
    is written as NaN. The same numbers therefore give the same digest on every platform.
    """
    payload = {
        "columns": [str(c) for c in frame.columns],
        "dates": [d.isoformat() for d in frame.index.date],
        "values": frame.to_numpy(dtype="float64").tolist(),
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def download_holdout(settings: Settings, tickers: list[str], source: PriceSource) -> pd.DataFrame:
    """Holdout-period adjusted closes for `tickers`, cleaned like the cache, held in memory only."""
    dates = settings.dates
    context_start = dates.validation_end - timedelta(days=CONTEXT_CALENDAR_DAYS)
    raw = source.fetch_adjusted_close(
        list(tickers), context_start, dates.holdout_end + timedelta(days=1)
    )
    raw = raw.sort_index()
    raw = raw[~raw.index.duplicated(keep="first")].reindex(columns=list(tickers))
    raw = raw.loc[: pd.Timestamp(dates.holdout_end)]
    filled = raw.ffill(limit=settings.data.ffill_limit_days).astype("float64")
    return filled[filled.index > pd.Timestamp(dates.validation_end)]


def _score(
    candidate: HoldoutCandidate,
    full: pd.DataFrame,
    lower,
    upper,
    settings: Settings,
) -> dict:
    sandboxed = SandboxedSignal(
        candidate.code, settings.sandbox.timeout_seconds, settings.sandbox.allowed_imports
    )
    outcome = sandboxed.run(full)
    if not outcome.ok:
        raise RuntimeError(f"sandbox {outcome.status}: {outcome.message}")
    held = hold_weights(outcome.weights, candidate.holding_days)
    result = run_backtest(held, full, lower, upper, settings.backtest)
    return {k: float(v) for k, v in asdict(result.metrics).items()}


def _claim(
    ledger: Ledger,
    run_id: str,
    candidates: list[HoldoutCandidate],
    invocation: dict | None,
) -> None:
    rows = [
        {
            "holdout_id": uuid.uuid4().hex,
            "attempt_id": c.attempt_id,
            "idea_id": c.idea_id,
            "run_id": run_id,
        }
        for c in candidates
    ]
    try:
        ledger.claim_holdouts(rows, invocation=invocation)
    except HoldoutConflict as exc:
        raise HoldoutRefused(str(exc)) from exc
    except sqlite3.IntegrityError as exc:
        raise HoldoutRefused(f"the ledger rejected the holdout claim: {exc}") from exc


def _mark_pending_error(
    ledger: Ledger, pending: dict[str, HoldoutCandidate], sha: str | None
) -> None:
    for attempt_id in list(pending):
        # If this also fails, the row stays 'running'. Any row still blocks a second run.
        with contextlib.suppress(Exception):
            ledger.finish_holdout(attempt_id, "error", None, sha)


def run_holdout(
    *,
    ledger: Ledger,
    run_id: str,
    settings: Settings,
    universe: Universe,
    candidates: list[HoldoutCandidate],
    i_understand_this_is_final: bool,
    confirm: Callable[[list[HoldoutCandidate]], bool],
    data_dir: Path = DATA_DIR,
    source: PriceSource | None = None,
    invocation: dict | None = None,
) -> list[HoldoutResult]:
    """Run the holdout once for each eligible candidate. Refuses unless every check passes."""
    if not i_understand_this_is_final:
        raise HoldoutRefused("refusing to run: pass --i-understand-this-is-final")
    unique = list({c.attempt_id: c for c in candidates}.values())
    if not unique:
        raise HoldoutRefused("no validated signals were selected")

    skipped = [c for c in unique if ledger.holdout_claimed(c.idea_id, c.attempt_id)]
    skipped_ids = {c.attempt_id for c in skipped}
    eligible = [c for c in unique if c.attempt_id not in skipped_ids]
    if not eligible:
        names = ", ".join(c.name for c in skipped)
        raise HoldoutRefused(
            f"holdout has already been run for: {names}. Each idea gets exactly one holdout run."
        )
    ideas = [c.idea_id for c in eligible]
    if len(set(ideas)) != len(ideas):
        raise HoldoutRefused(
            "two selected attempts belong to the same idea. Each idea gets one holdout run, "
            "so select one attempt (--attempt)."
        )
    # Attribute reads have no runtime layer (R4-SEC-02). A candidate validated before a rule was
    # added passed an older filter, so the current filter runs again here, before any claim.
    refused = [
        c for c in eligible if not static_check(c.code, settings.sandbox.allowed_imports).ok
    ]
    if refused:
        names = ", ".join(c.name for c in refused)
        raise HoldoutRefused(
            f"the static filter refuses the code of: {names}. Nothing was claimed. "
            "Select other attempts with --attempt."
        )

    pre = loader.read_checked_pre_holdout(settings, universe, data_dir)
    if not confirm(eligible):
        raise HoldoutRefused("not confirmed, nothing was run")

    _claim(ledger, run_id, eligible, invocation)  # nothing has been requested yet
    results = [
        HoldoutResult(c.attempt_id, c.name, "skipped", None, "this idea already has a holdout run")
        for c in skipped
    ]
    pending = {c.attempt_id: c for c in eligible}
    sha: str | None = None
    try:
        # The only holdout request in the codebase. It happens after the claim above.
        holdout = download_holdout(settings, list(pre.columns), source or loader.YFinanceSource())
        sha = holdout_sha256(holdout)
        full = pd.concat([pre, holdout.reindex(columns=pre.columns)]).sort_index()
        full = full.loc[: pd.Timestamp(settings.dates.holdout_end)]
        lower, upper = split_window(settings.dates, "holdout")
        for candidate in eligible:
            try:
                metrics = _score(candidate, full, lower, upper, settings)
                status, error = "complete", None
            except Exception as exc:  # noqa: BLE001 - recorded; the idea stays used
                metrics, status, error = None, "error", str(exc)
            try:
                ledger.finish_holdout(candidate.attempt_id, status, metrics, sha)
            except ValueError as exc:  # no claimed row to write to (finish_holdout, R3-AGT-15)
                metrics, status = None, "error"
                error = f"the outcome could not be recorded: {exc}"
            results.append(
                HoldoutResult(candidate.attempt_id, candidate.name, status, metrics, error)
            )
            pending.pop(candidate.attempt_id, None)
    except Exception as exc:  # noqa: BLE001 - the download or the setup failed
        _mark_pending_error(ledger, pending, sha)
        results.extend(
            HoldoutResult(c.attempt_id, c.name, "error", None, str(exc)) for c in pending.values()
        )
    except BaseException:  # Ctrl+C and similar: record the claimed rows as used, then stop
        _mark_pending_error(ledger, pending, sha)
        raise
    return results
