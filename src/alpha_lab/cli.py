"""Command line interface: `lab data`, `lab backtest`, `lab run`, `lab ledger`, `lab holdout`."""

from __future__ import annotations

import asyncio
import getpass
import json
import logging
import math
import os
import re
import secrets
import sqlite3
import sys
import uuid

import anthropic
import pandas as pd
import typer

from alpha_lab.backtest.engine import run_backtest
from alpha_lab.config import DATA_DIR, PROJECT_ROOT, load_env_file, load_settings, load_universe
from alpha_lab.data.loader import (
    PRE_HOLDOUT_FILE,
    CacheMismatch,
    YFinanceSource,
    check_fingerprint,
    fetch_and_cache,
    format_quality_summary,
    load_prices,
    read_report,
)
from alpha_lab.data.splits import split_window
from alpha_lab.events.bus import EventBus, PipelineEvent
from alpha_lab.ledger.db import LEDGER_PATH, MIGRATIONS, Ledger
from alpha_lab.pipeline.run import execute_run
from alpha_lab.pipeline.swarm import SwarmProgress
from alpha_lab.signals.baselines import SIGNALS

app = typer.Typer(help="Alpha Lab command line.", no_args_is_help=True)
data_app = typer.Typer(help="Market data: fetch, clean, summarise.", no_args_is_help=True)
ledger_app = typer.Typer(help="Inspect the run ledger.", no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(ledger_app, name="ledger")

BACKTEST_SPLITS = ("train", "validation")
GRACEFUL_SHUTDOWN_SECONDS = 3
# Tests point these at temporary paths. Every command reads them at call time.
RUNS_DIR = PROJECT_ROOT / "runs"
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


OLDER_LEDGER_MESSAGE = (
    "The ledger at {path} was written by an older version, so it is not read until it is "
    "migrated. Run `lab ledger migrate` once, then retry."
)


def _open_ledger() -> Ledger:
    """The read-write ledger. Only the commands that record something use it (run, holdout, migrate)."""
    return Ledger(LEDGER_PATH)


def _schema_is_current(ledger: Ledger) -> bool:
    """False for a ledger an older version wrote: a column from MIGRATIONS is missing, or events has no seq."""
    for table, columns in MIGRATIONS.items():
        present = {row["name"] for row in ledger.conn.execute(f"PRAGMA table_info({table})")}
        if not set(columns) <= present:
            return False
    events = {row["name"] for row in ledger.conn.execute("PRAGMA table_info(events)")}
    return "seq" in events


def _open_ledger_read_only() -> Ledger | None:
    """A read-only ledger, or None if there is no ledger file. Never creates or migrates the file.

    An older schema is refused with a message and left as it is (R4-WHOLE-08, R4-AGT-13).
    """
    if not LEDGER_PATH.exists():
        return None
    ledger = Ledger(LEDGER_PATH, read_only=True)
    if not _schema_is_current(ledger):
        ledger.close()
        typer.echo(OLDER_LEDGER_MESSAGE.format(path=LEDGER_PATH), err=True)
        raise typer.Exit(1)
    return ledger


def _plain(text: str | None) -> str:
    """Remove control characters from agent-written text before it reaches a terminal."""
    return _CONTROL_CHARS.sub("", text or "")


def _stdin_is_tty() -> bool:
    """True only for an interactive console. On Windows the null device also reports isatty() (R3-SEC-06)."""
    try:
        if not sys.stdin.isatty():
            return False
        fd = sys.stdin.fileno()
        if os.name == "nt":
            return _windows_console(fd)
        null, here = os.stat(os.devnull), os.fstat(fd)
    except (AttributeError, ValueError, OSError):
        return False
    return (here.st_dev, here.st_ino) != (null.st_dev, null.st_ino)


def _windows_console(fd: int) -> bool:
    """GetConsoleMode succeeds only for a console handle, and it fails for the NUL device."""
    import ctypes
    import msvcrt
    from ctypes import wintypes

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetConsoleMode.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel32.GetConsoleMode.restype = wintypes.BOOL
        mode = wintypes.DWORD()
        return bool(kernel32.GetConsoleMode(msvcrt.get_osfhandle(fd), ctypes.byref(mode)))
    except (AttributeError, OSError, ValueError):
        return False


def _metric(value: float | None, spec: str) -> str:
    """A metric for display. A missing or non-finite value (a flat scoring window) prints as n/a."""
    if value is None or not math.isfinite(value):
        return "n/a"
    return format(value, spec)


def _current_user() -> str:
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001 - the audit field is best effort
        return "unknown"


def _confirmation_code() -> str:
    return secrets.token_hex(2).upper()


@data_app.command("fetch")
def data_fetch(
    refresh: bool = typer.Option(False, "--refresh", help="Re-download even if cached."),
) -> None:
    """Download adjusted closes, clean them, and write the Parquet cache."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = load_settings()
    universe = load_universe()
    if refresh or not (DATA_DIR / PRE_HOLDOUT_FILE).exists():
        report = fetch_and_cache(settings, universe, YFinanceSource(), DATA_DIR)
    else:
        try:
            check_fingerprint(settings, universe, DATA_DIR)
        except CacheMismatch as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(1) from None
        typer.echo("Cache found. Use --refresh to re-download.")
        report = read_report(DATA_DIR)
    typer.echo(format_quality_summary(report))


@data_app.command("summary")
def data_summary() -> None:
    """Print the data-quality summary from the last fetch."""
    try:
        report = read_report(DATA_DIR)
    except FileNotFoundError:
        typer.echo("No cache yet. Run `lab data fetch` first.", err=True)
        raise typer.Exit(1) from None
    typer.echo(format_quality_summary(report))


@app.command("backtest")
def backtest(
    signal_name: str = typer.Argument(..., help=f"One of: {', '.join(SIGNALS)}"),
    split: str = typer.Option("train", "--split", help="train or validation"),
) -> None:
    """Run a signal over one split and print the metrics table."""
    if split not in BACKTEST_SPLITS:
        typer.echo(
            f"--split must be one of {BACKTEST_SPLITS}. The holdout only runs through "
            "`lab holdout` (Phase 4).",
            err=True,
        )
        raise typer.Exit(2)
    if signal_name not in SIGNALS:
        typer.echo(f"unknown signal {signal_name!r}. Choose from: {', '.join(SIGNALS)}", err=True)
        raise typer.Exit(2)

    settings = load_settings()
    universe = load_universe()
    try:
        prices = load_prices(settings, universe)
    except CacheMismatch as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None
    lower, upper = split_window(settings.dates, split)

    visible = prices.loc[: pd.Timestamp(upper)]  # the signal may use history up to the window end
    weights = SIGNALS[signal_name](visible)
    result = run_backtest(weights, visible, lower, upper, settings.backtest)

    m = result.metrics
    typer.echo(f"{signal_name} on {split} ({lower} to {upper})")
    rows = [
        ("trading days", f"{m.n_days}"),
        ("annualised return", f"{m.ann_return:+.2%}"),
        ("annualised volatility", f"{m.ann_volatility:.2%}"),
        ("Sharpe (rf=0)", f"{m.sharpe:.3f}"),
        ("max drawdown", f"{m.max_drawdown:.2%}"),
        ("avg daily turnover", f"{m.avg_daily_turnover:.4f}"),
        ("hit rate", f"{m.hit_rate:.2%}"),
    ]
    for label, value in rows:
        typer.echo(f"  {label:<22}{value:>12}")


async def _print_events(queue: asyncio.Queue[PipelineEvent]) -> None:
    while True:
        event = await queue.get()
        stamp = event.timestamp.strftime("%H:%M:%S.%f")[:-3]
        who = event.agent or event.stage
        pipeline = (event.pipeline_id or "-")[:6]
        detail = json.dumps(event.payload, default=str)
        if len(detail) > 140:
            detail = detail[:137] + "..."
        typer.echo(f"{stamp}  p:{pipeline}  {event.type:<18} {who:<9} {detail}")


@app.command("run")
def run_command(
    pipelines: int = typer.Option(1, "--pipelines", min=1, help="How many pipelines to run."),
    budget: float | None = typer.Option(None, "--budget", help="Spend limit in USD."),
    verbose: bool = typer.Option(False, "--verbose", help="Print every event, not just progress."),
) -> None:
    """Run the agent swarm against the Anthropic API, within a budget."""
    try:
        load_env_file()
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from None
    if not os.environ.get("ANTHROPIC_API_KEY"):
        typer.echo(
            "ANTHROPIC_API_KEY is not set. Add it to .env (never commit that file).", err=True
        )
        raise typer.Exit(2)

    settings = load_settings()
    universe = load_universe()
    budget_usd = settings.run.default_budget_usd if budget is None else budget
    try:
        prices = load_prices(settings, universe)
    except CacheMismatch as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None

    ledger = _open_ledger()
    bus = EventBus(ledger)
    # Only --verbose reads the event stream. A queue nobody drains would grow for the whole run.
    queue = bus.subscribe() if verbose else None
    run_id = uuid.uuid4().hex[:12]
    client = anthropic.AsyncAnthropic(max_retries=0)  # retries are handled by LLMClient
    holder: dict[str, SwarmProgress] = {}

    async def heartbeat() -> None:
        while True:
            await asyncio.sleep(5)
            p = holder.get("progress")
            if p is not None:
                typer.echo(
                    f"  running {p.running:<3} done {p.done:<3}/{p.total:<3}"
                    f" passed critic {p.passed_critic:<3} passed validation {p.passed_validation:<3}"
                    f" spend ${p.spent_usd:.4f} of ${budget_usd:.2f}"
                    f" ({p.spend_rate_per_min:.4f}/min)"
                )

    async def main():
        tasks = [asyncio.create_task(heartbeat())]
        if queue is not None:
            tasks.append(asyncio.create_task(_print_events(queue)))
        try:
            return await execute_run(
                run_id=run_id,
                settings=settings,
                prices=prices,
                n_pipelines=pipelines,
                budget_usd=budget_usd,
                ledger=ledger,
                bus=bus,
                client=client,
                progress_sink=holder,
            )
        finally:
            for task in tasks:
                task.cancel()

    try:
        summary = asyncio.run(main())
    except anthropic.APIError as exc:
        typer.echo(f"Anthropic API error, run {run_id} marked failed: {exc.message}", err=True)
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        typer.echo(f"Interrupted. Run {run_id} is recorded as interrupted in the ledger.", err=True)
        raise typer.Exit(130) from None
    finally:
        if queue is not None:
            bus.unsubscribe(queue)  # stop queueing events for a finished run

    typer.echo("")
    typer.echo(f"Run {summary.run_id}: {summary.status}")
    typer.echo(
        f"  pipelines finished: {len(summary.results)} of {pipelines}"
        f"   passed critic: {summary.passed_critic}   passed validation: {summary.passed_validation}"
    )
    for status, count in sorted(summary.pipeline_statuses.items()):
        typer.echo(f"    {status:<20}{count}")
    typer.echo(f"  cost: ${summary.total_cost_usd:.4f} of ${summary.budget_usd:.2f} budget")


@ledger_app.command("show")
def ledger_show(
    run: str | None = typer.Option(None, "--run", help="Run ID. Omit to list recent runs."),
    code: bool = typer.Option(False, "--code", help="Print each attempt's code."),
) -> None:
    """List runs, or show one run's ideas, checks, metrics and LLM cost. Read-only: it never creates
    or migrates the ledger."""
    ledger = _open_ledger_read_only()
    if ledger is None:
        if run is None:
            typer.echo("No runs yet. Try `lab run --pipelines 1 --budget 0.50`.")
            return
        typer.echo(f"No ledger at {LEDGER_PATH}. Run `lab run` first.", err=True)
        raise typer.Exit(1)
    if run is None:
        rows = ledger.list_runs()
        if not rows:
            typer.echo("No runs yet. Try `lab run --pipelines 1 --budget 0.50`.")
            return
        typer.echo(f"{'run_id':<14}{'status':<16}{'cost':>10}{'budget':>9}  started")
        for row in rows:
            typer.echo(
                f"{row['run_id']:<14}{row['status']:<16}{row['total_cost_usd']:>10.4f}"
                f"{row['budget_usd']:>9.2f}  {row['started_at']}"
            )
        return

    row = ledger.get_run(run)
    if row is None:
        typer.echo(f"no run with id {run!r}", err=True)
        raise typer.Exit(1)
    typer.echo(
        f"Run {run}: {row['status']}   cost ${row['total_cost_usd']:.4f} of ${row['budget_usd']:.2f}"
    )

    ideas = {i["idea_id"]: i for i in ledger.ideas_for_run(run)}
    for idea_id, idea in ideas.items():
        typer.echo("")
        typer.echo(f"Idea: {_plain(idea['name'])}")
        typer.echo(f"  hypothesis: {_plain(idea['hypothesis'])}")
        typer.echo(f"  description: {_plain(idea['description'])}")
        typer.echo(f"  holding period: {idea['expected_holding_period_days']} days")
        for attempt in ledger.attempts_for_run(run):
            if attempt["idea_id"] != idea_id:
                continue
            typer.echo(f"  attempt {attempt['attempt_number']}: {attempt['status']}")
            if attempt["error"]:
                first = attempt["error"].splitlines()[0][:200]
                typer.echo(f"    check: {_plain(first)}")
            if attempt["train_metrics_json"]:
                m = json.loads(attempt["train_metrics_json"])
                typer.echo(
                    f"    train: Sharpe {_metric(m.get('sharpe'), '.3f')}, "
                    f"ann. return {_metric(m.get('ann_return'), '+.2%')}, "
                    f"max DD {_metric(m.get('max_drawdown'), '.2%')}, "
                    f"turnover {_metric(m.get('avg_daily_turnover'), '.4f')}"
                )
            if code and attempt["code"]:
                typer.echo("    code:")
                for line in attempt["code"].splitlines():
                    typer.echo(f"      {_plain(line)}")

    typer.echo("")
    typer.echo("LLM cost by agent:")
    for r in ledger.llm_cost_by_agent(run):
        typer.echo(
            f"  {r['agent']:<8}{r['model']:<20}calls {r['calls']:<4}"
            f"in {r['input_tokens']:<8}out {r['output_tokens']:<8}${r['cost_usd']:.4f}"
        )


@ledger_app.command("migrate")
def ledger_migrate() -> None:
    """Upgrade the ledger file to the current schema. An explicit write: no API call is made, and the
    file is created if it is missing. The read commands never migrate."""
    ledger = _open_ledger()
    ledger.close()
    typer.echo(f"Ledger at {LEDGER_PATH} is at the current schema.")


@app.command("report")
def report_command(
    run: str | None = typer.Option(None, "--run", help="Run ID. Default: the latest run."),
) -> None:
    """Write the Markdown and HTML report for a run, with research figures. Read-only on the ledger."""
    from alpha_lab.ledger.report import write_report

    ledger = _open_ledger_read_only()
    if ledger is None:
        typer.echo(f"No ledger at {LEDGER_PATH}. Run `lab run` first.", err=True)
        raise typer.Exit(1)
    if run is None:
        rows = ledger.list_runs(1)
        if not rows:
            typer.echo("No runs yet.", err=True)
            raise typer.Exit(1)
        run = rows[0]["run_id"]
    if ledger.get_run(run) is None:
        typer.echo(f"no run with id {run!r}", err=True)
        raise typer.Exit(1)
    path = write_report(ledger, run, out_dir=RUNS_DIR / run)
    typer.echo(f"Report written to {path} (and report.html, figures/ next to it).")


@app.command("holdout")
def holdout_command(
    run: str = typer.Option(..., "--run", help="Run ID whose validated signals are tested."),
    top: int = typer.Option(
        5, "--top", min=1, help="How many signals, best deflated Sharpe first."
    ),
    attempt: list[str] | None = typer.Option(  # noqa: B008 - typer's documented call-style default
        None,
        "--attempt",
        help="Test only these attempt IDs (full IDs, repeatable). Ignores --top.",
    ),
    i_understand_this_is_final: bool = typer.Option(
        False,
        "--i-understand-this-is-final",
        help="Required. Each idea gets exactly one holdout run, ever.",
    ),
) -> None:
    """Run the one-time holdout test on validated signals. Human only, in a terminal."""
    from alpha_lab.ledger.report import MIN_TRIALS_FOR_VARIANCE, build_report_data
    from alpha_lab.pipeline.holdout import HoldoutCandidate, HoldoutRefused, run_holdout

    if not i_understand_this_is_final:
        typer.echo(
            "Refusing to run. The holdout is final and can be run once per idea.\n"
            "Re-run with --i-understand-this-is-final if you mean it.",
            err=True,
        )
        raise typer.Exit(2)
    if not _stdin_is_tty():
        typer.echo(
            "Refusing to run: `lab holdout` needs an interactive terminal. It does not read a "
            "piped or scripted confirmation.",
            err=True,
        )
        raise typer.Exit(2)

    ledger = _open_ledger()
    if ledger.get_run(run) is None:
        typer.echo(f"no run with id {run!r}", err=True)
        raise typer.Exit(1)
    data = build_report_data(ledger, run)
    passed = [c for c in data.candidates if c.passed]
    if attempt:
        wanted = set(attempt)
        survivors = [c for c in passed if c.attempt_id in wanted]
        unknown = sorted(wanted - {c.attempt_id for c in survivors})
        if unknown:
            typer.echo(
                f"no validated signal in run {run!r} has attempt id(s): {', '.join(unknown)}",
                err=True,
            )
            raise typer.Exit(1)
    else:
        if data.dsr_reason:
            # The ranking behind --top is the deflated Sharpe. Without it, the order is not a
            # correction for the number of trials, so --top refuses. --attempt is an explicit choice.
            typer.echo(
                f"Refusing --top: the deflated Sharpe is not defined for run {run!r} "
                f"({data.dsr_reason}). Select attempts explicitly with --attempt.",
                err=True,
            )
            raise typer.Exit(1)
        if data.n_train_trials < MIN_TRIALS_FOR_VARIANCE:
            # R4-STAT-09: the trial variance behind the ranking rests on too few trials.
            typer.echo(
                f"Refusing --top: run {run!r} has {data.n_train_trials} trial(s) with a train "
                f"Sharpe, and at least {MIN_TRIALS_FOR_VARIANCE} are needed to estimate the "
                f"trial variance (ledger trials: {data.n_trials}). Select attempts explicitly "
                "with --attempt.",
                err=True,
            )
            raise typer.Exit(1)
        survivors = passed[:top]
    if not survivors:
        typer.echo("No validated signals in this run. Nothing to test.", err=True)
        raise typer.Exit(1)

    candidates = []
    for c in survivors:
        row = ledger.conn.execute(
            "SELECT a.code, a.idea_id, i.expected_holding_period_days AS hold FROM attempts a "
            "JOIN ideas i ON i.idea_id = a.idea_id WHERE a.attempt_id = ?",
            (c.attempt_id,),
        ).fetchone()
        candidates.append(
            HoldoutCandidate(
                attempt_id=c.attempt_id,
                idea_id=row["idea_id"],
                name=c.name,
                code=row["code"],
                holding_days=row["hold"],
            )
        )

    def confirm(items: list[HoldoutCandidate]) -> bool:
        typer.echo("About to run the holdout, once, on these signals:")
        for item in items:
            typer.echo(f"  - {_plain(item.name)}  attempt {item.attempt_id}")
        code = _confirmation_code()
        typed = typer.prompt(
            f"This downloads the holdout period and cannot be undone. Type {code} to continue"
        )
        return typed.strip().upper() == code

    settings = load_settings()
    universe = load_universe()
    # The TTY check above refuses a piped or scripted run. It is not recorded: a recorded True
    # proves nothing about a human, and it was always True in a written row (R4-SEC-07).
    invocation = {
        "argv": sys.argv,
        "user": _current_user(),
        "pid": os.getpid(),
        "run_id": run,
    }
    try:
        results = run_holdout(
            ledger=ledger,
            run_id=run,
            settings=settings,
            universe=universe,
            candidates=candidates,
            i_understand_this_is_final=i_understand_this_is_final,
            confirm=confirm,
            data_dir=DATA_DIR,
            invocation=invocation,
        )
    except (HoldoutRefused, CacheMismatch) as exc:
        typer.echo(f"Refused: {exc}", err=True)
        raise typer.Exit(1) from None
    except sqlite3.IntegrityError as exc:
        typer.echo(f"Refused: the ledger rejected the holdout record ({exc}).", err=True)
        raise typer.Exit(1) from None

    by_id = {c.attempt_id: c for c in survivors}
    failed = False
    for r in results:
        c = by_id[r.attempt_id]
        typer.echo(_plain(r.name))
        typer.echo(
            f"  train SR {_metric(c.train_sharpe, '.3f')}   "
            f"validation SR {_metric(c.validation_sharpe, '.3f')}   "
            f"deflated SR {_metric(c.deflated_sharpe, '.3f')}"
        )
        if r.status == "complete" and r.metrics:
            m = r.metrics
            typer.echo(
                f"  holdout SR {m['sharpe']:.3f}   ann. return {m['ann_return']:+.2%}   "
                f"max DD {m['max_drawdown']:.2%}"
            )
        else:
            failed = failed or r.status == "error"
            typer.echo(f"  holdout status: {r.status} ({_plain(r.error)})")
    if failed:
        raise typer.Exit(1)


@app.command("dashboard")
def dashboard_command(
    run: str | None = typer.Option(None, "--run", help="Open this run on mission control."),
    demo: bool = typer.Option(False, "--demo", help="Stream a synthetic demo run. No API calls."),
    port: int = typer.Option(8000, "--port", help="Port to serve on."),
) -> None:
    """Serve the dashboard at http://localhost:8000. Read-only: it never creates or migrates the ledger,
    and it refuses a ledger written by an older version."""
    import uvicorn

    from alpha_lab.dashboard.server import create_app

    ledger_path = PROJECT_ROOT / "runs" / "demo.db" if demo else LEDGER_PATH
    if not demo and ledger_path.exists():
        # The dashboard only reads. An older schema is refused here, not migrated (R4-AGT-13).
        probe = Ledger(ledger_path, read_only=True)
        try:
            current = _schema_is_current(probe)
        finally:
            probe.close()
        if not current:
            typer.echo(OLDER_LEDGER_MESSAGE.format(path=ledger_path), err=True)
            raise typer.Exit(1)
    if run is not None and not demo:
        # The page for an unknown run would wait forever (R3-WHOLE-14). Check it here, read-only.
        if not ledger_path.exists():
            typer.echo(f"No ledger at {ledger_path}. Run `lab run` first.", err=True)
            raise typer.Exit(1)
        ledger = Ledger(ledger_path, read_only=True)
        try:
            found = ledger.get_run(run) is not None
        finally:
            ledger.close()
        if not found:
            typer.echo(f"no run with id {run!r}", err=True)
            raise typer.Exit(1)
    if demo:
        typer.echo("Demo mode: streaming a synthetic run into runs/demo.db. No API calls are made.")
    typer.echo(f"Dashboard at http://localhost:{port}" + (f"/?run={run}" if run else "/"))
    uvicorn.run(
        create_app(ledger_path=ledger_path, demo=demo),
        host="127.0.0.1",
        port=port,
        log_level="warning",
        # An open event stream would otherwise hold the server, and the demo, as running (R3-BE-02).
        timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_SECONDS,
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
