"""Command line interface: `lab data fetch|summary`, `lab backtest`."""

from __future__ import annotations

import logging

import pandas as pd
import typer

from alpha_lab.backtest.engine import run_backtest
from alpha_lab.config import DATA_DIR, load_settings, load_universe
from alpha_lab.data.loader import (
    PRE_HOLDOUT_FILE,
    YFinanceSource,
    fetch_and_cache,
    format_quality_summary,
    load_prices,
    read_report,
)
from alpha_lab.data.splits import split_window
from alpha_lab.signals.baselines import SIGNALS

app = typer.Typer(help="Alpha Lab command line.", no_args_is_help=True)
data_app = typer.Typer(help="Market data: fetch, clean, summarise.", no_args_is_help=True)
app.add_typer(data_app, name="data")

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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
