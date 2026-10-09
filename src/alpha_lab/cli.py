"""Command line interface: `lab data`, `lab backtest`, `lab run`, `lab ledger`."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid

import anthropic
import pandas as pd
import typer

from alpha_lab.backtest.engine import run_backtest
from alpha_lab.config import DATA_DIR, load_env_file, load_settings, load_universe
from alpha_lab.data.loader import (
    PRE_HOLDOUT_FILE,
    YFinanceSource,
    fetch_and_cache,
    format_quality_summary,
    load_prices,
    read_report,
)
from alpha_lab.data.splits import split_window
from alpha_lab.events.bus import EventBus, PipelineEvent
from alpha_lab.ledger.db import LEDGER_PATH, Ledger
from alpha_lab.pipeline.run import execute_run
from alpha_lab.pipeline.swarm import SwarmProgress
from alpha_lab.signals.baselines import SIGNALS

app = typer.Typer(help="Alpha Lab command line.", no_args_is_help=True)
data_app = typer.Typer(help="Market data: fetch, clean, summarise.", no_args_is_help=True)
ledger_app = typer.Typer(help="Inspect the run ledger.", no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(ledger_app, name="ledger")

BACKTEST_SPLITS = ("train", "validation")


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
    prices = load_prices(settings, universe)
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
    load_env_file()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        typer.echo(
            "ANTHROPIC_API_KEY is not set. Add it to .env (never commit that file).", err=True
        )
        raise typer.Exit(2)

    settings = load_settings()
    universe = load_universe()
    budget_usd = settings.run.default_budget_usd if budget is None else budget
    prices = load_prices(settings, universe)

    ledger = Ledger()
    bus = EventBus(ledger)
    queue = bus.subscribe()
    run_id = uuid.uuid4().hex[:12]
    client = anthropic.AsyncAnthropic(max_retries=0)  # retries are handled by LLMClient
    holder: dict[str, SwarmProgress] = {}

    async def heartbeat() -> None:
        while True:
            await asyncio.sleep(5)
            p = holder.get("progress")
            if p is not None:
                typer.echo(
                    f"  running {p.running:<3} done {p.done:<3}/{p.total:<3} passed {p.passed:<3}"
                    f" spend ${p.spent_usd:.4f} of ${budget_usd:.2f}"
                    f" ({p.spend_rate_per_min:.4f}/min)"
                )

    async def main():
        tasks = [asyncio.create_task(heartbeat())]
        if verbose:
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

    typer.echo("")
    typer.echo(f"Run {summary.run_id}: {summary.status}")
    typer.echo(
        f"  pipelines finished: {len(summary.results)} of {pipelines}   critic passes: {summary.train_passes}"
    )
    for status, count in sorted(summary.pipeline_statuses.items()):
        typer.echo(f"    {status:<20}{count}")
    typer.echo(f"  cost: ${summary.total_cost_usd:.4f} of ${summary.budget_usd:.2f} budget")


@ledger_app.command("show")
def ledger_show(
    run: str | None = typer.Option(None, "--run", help="Run ID. Omit to list recent runs."),
    code: bool = typer.Option(False, "--code", help="Print each attempt's code."),
) -> None:
    """List runs, or show one run's ideas, checks, metrics and LLM cost."""
    ledger = Ledger()
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
        typer.echo(f"Idea: {idea['name']}")
        typer.echo(f"  hypothesis: {idea['hypothesis']}")
        typer.echo(f"  description: {idea['description']}")
        typer.echo(f"  holding period: {idea['expected_holding_period_days']} days")
        for attempt in ledger.attempts_for_run(run):
            if attempt["idea_id"] != idea_id:
                continue
            typer.echo(f"  attempt {attempt['attempt_number']}: {attempt['status']}")
            if attempt["error"]:
                typer.echo(f"    check: {attempt['error'].splitlines()[0][:200]}")
            if attempt["train_metrics_json"]:
                m = json.loads(attempt["train_metrics_json"])
                typer.echo(
                    f"    train: Sharpe {m['sharpe']:.3f}, ann. return {m['ann_return']:+.2%}, "
                    f"max DD {m['max_drawdown']:.2%}, turnover {m['avg_daily_turnover']:.4f}"
                )
            if code and attempt["code"]:
                typer.echo("    code:")
                for line in attempt["code"].splitlines():
                    typer.echo(f"      {line}")

    typer.echo("")
    typer.echo("LLM cost by agent:")
    for r in ledger.llm_cost_by_agent(run):
        typer.echo(
            f"  {r['agent']:<8}{r['model']:<20}calls {r['calls']:<4}"
            f"in {r['input_tokens']:<8}out {r['output_tokens']:<8}${r['cost_usd']:.4f}"
        )


@app.command("report")
def report_command(
    run: str | None = typer.Option(None, "--run", help="Run ID. Default: the latest run."),
) -> None:
    """Write the Markdown and HTML report for a run, with research figures."""
    from alpha_lab.ledger.report import write_report

    ledger = Ledger()
    if run is None:
        rows = ledger.list_runs(1)
        if not rows:
            typer.echo("No runs yet.", err=True)
            raise typer.Exit(1)
        run = rows[0]["run_id"]
    if ledger.get_run(run) is None:
        typer.echo(f"no run with id {run!r}", err=True)
        raise typer.Exit(1)
    path = write_report(ledger, run)
    typer.echo(f"Report written to {path} (and report.html, figures/ next to it).")


@app.command("holdout")
def holdout_command(
    run: str = typer.Option(..., "--run", help="Run ID whose validated signals are tested."),
    top: int = typer.Option(
        5, "--top", min=1, help="How many signals, best deflated Sharpe first."
    ),
    i_understand_this_is_final: bool = typer.Option(
        False,
        "--i-understand-this-is-final",
        help="Required. Each signal gets exactly one holdout run, ever.",
    ),
) -> None:
    """Run the one-time holdout test on the top validated signals. Human only."""
    from alpha_lab.ledger.report import build_report_data
    from alpha_lab.pipeline.holdout import HoldoutCandidate, HoldoutRefused, run_holdout

    if not i_understand_this_is_final:
        typer.echo(
            "Refusing to run. The holdout is final and can be run once per signal.\n"
            "Re-run with --i-understand-this-is-final if you mean it.",
            err=True,
        )
        raise typer.Exit(2)

    ledger = Ledger()
    if ledger.get_run(run) is None:
        typer.echo(f"no run with id {run!r}", err=True)
        raise typer.Exit(1)
    data = build_report_data(ledger, run)
    survivors = [c for c in data.candidates if c.passed][:top]
    if not survivors:
        typer.echo("No validated signals in this run. Nothing to test.", err=True)
        raise typer.Exit(1)

    candidates = []
    for c in survivors:
        row = ledger.conn.execute(
            "SELECT a.code, i.expected_holding_period_days AS hold FROM attempts a "
            "JOIN ideas i ON i.idea_id = a.idea_id WHERE a.attempt_id = ?",
            (c.attempt_id,),
        ).fetchone()
        candidates.append(
            HoldoutCandidate(
                attempt_id=c.attempt_id, name=c.name, code=row["code"], holding_days=row["hold"]
            )
        )

    def confirm(items: list) -> bool:
        typer.echo("About to run the holdout, once, on these signals:")
        for item in items:
            typer.echo(f"  - {item.name} ({item.attempt_id[:8]})")
        return typer.confirm(
            "This uses the holdout period and cannot be undone. Continue?", default=False
        )

    settings = load_settings()
    try:
        results = run_holdout(
            ledger=ledger,
            run_id=run,
            settings=settings,
            candidates=candidates,
            i_understand_this_is_final=True,
            confirm=confirm,
        )
    except HoldoutRefused as exc:
        typer.echo(f"Refused: {exc}", err=True)
        raise typer.Exit(1) from None

    by_id = {c.attempt_id: c for c in survivors}
    for r in results:
        c = by_id[r.attempt_id]
        typer.echo(f"{r.name}")
        typer.echo(
            f"  train SR {c.train_sharpe:.3f}   validation SR {c.validation_sharpe:.3f}   deflated SR {c.deflated_sharpe:.3f}"
        )
        if r.metrics:
            m = r.metrics
            typer.echo(
                f"  holdout SR {m['sharpe']:.3f}   ann. return {m['ann_return']:+.2%}   max DD {m['max_drawdown']:.2%}"
            )
        else:
            typer.echo(f"  holdout status: {r.status} ({r.error})")


@app.command("dashboard")
def dashboard_command(
    run: str | None = typer.Option(None, "--run", help="Open this run on mission control."),
    demo: bool = typer.Option(False, "--demo", help="Stream a synthetic demo run. No API calls."),
    port: int = typer.Option(8000, "--port", help="Port to serve on."),
) -> None:
    """Serve the dashboard at http://localhost:8000."""
    import uvicorn

    from alpha_lab.config import PROJECT_ROOT
    from alpha_lab.dashboard.server import create_app

    ledger_path = PROJECT_ROOT / "runs" / "demo.db" if demo else LEDGER_PATH
    if demo:
        typer.echo("Demo mode: streaming a synthetic run into runs/demo.db. No API calls are made.")
    typer.echo(f"Dashboard at http://localhost:{port}" + (f"/?run={run}" if run else "/"))
    uvicorn.run(
        create_app(ledger_path=ledger_path, demo=demo),
        host="127.0.0.1",
        port=port,
        log_level="warning",
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
