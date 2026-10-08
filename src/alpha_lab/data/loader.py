"""Fetch, clean and cache daily adjusted closes.

Holdout rows go to their own Parquet file. The default reader never opens it,
so ordinary code cannot see holdout prices by accident.

Cleaning decisions that look at missingness use only the pre-holdout period,
so a holdout-period gap cannot change which tickers are kept.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Protocol

import pandas as pd

from alpha_lab.config import DATA_DIR, DataConfig, Dates, Settings, Universe
from alpha_lab.data.splits import check_holdout_caller

log = logging.getLogger(__name__)

PRE_HOLDOUT_FILE = "prices_pre_holdout.parquet"
HOLDOUT_FILE = "prices_holdout.parquet"
REPORT_FILE = "cleaning_report.json"


class PriceSource(Protocol):
    """Anything that can return daily adjusted closes. Swap implementations here."""

    def fetch_adjusted_close(
        self, tickers: Sequence[str], start: date, end_exclusive: date
    ) -> pd.DataFrame: ...


class YFinanceSource:
    """Adjusted closes from Yahoo Finance via yfinance."""

    def fetch_adjusted_close(
        self, tickers: Sequence[str], start: date, end_exclusive: date
    ) -> pd.DataFrame:
        import yfinance as yf

        raw = yf.download(
            list(tickers),
            start=start.isoformat(),
            end=end_exclusive.isoformat(),
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        close = raw["Close"].reindex(columns=list(tickers))
        index = pd.DatetimeIndex(close.index)
        if index.tz is not None:
            index = index.tz_localize(None)
        close.index = index
        close.index.name = "date"
        return close.astype("float64").sort_index()


@dataclass
class CleaningReport:
    fetched_at: str
    first_date: str
    last_date: str
    tickers_requested: list[str]
    tickers_kept: list[str]
    dropped: dict[str, str]
    missing_after_fill: dict[str, int]
    return_flags: list[dict]


def clean_prices(
    raw: pd.DataFrame, dates: Dates, data_cfg: DataConfig
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Drop badly-missing tickers, then forward-fill short gaps. Returns (cleaned, dropped)."""
    prices = raw.sort_index()
    prices = prices[~prices.index.duplicated(keep="first")]
    prices = prices.loc[: pd.Timestamp(dates.holdout_end)]

    pre_holdout = prices[prices.index <= pd.Timestamp(dates.validation_end)]
    missing = pre_holdout.isna().mean()

    dropped: dict[str, str] = {}
    for ticker in prices.columns:
        frac = float(missing[ticker])
        if frac > data_cfg.max_missing_fraction:
            reason = (
                f"{frac:.2%} of pre-holdout rows missing "
                f"(limit {data_cfg.max_missing_fraction:.0%})"
            )
            dropped[ticker] = reason
            log.warning("dropping %s: %s", ticker, reason)

    kept = [t for t in prices.columns if t not in dropped]
    cleaned = prices[kept].ffill(limit=data_cfg.ffill_limit_days).astype("float64")
    return cleaned, dropped


def find_return_flags(pre_holdout: pd.DataFrame, data_cfg: DataConfig) -> list[dict]:
    """Daily moves beyond the flag thresholds. Flagged, never removed."""
    returns = pre_holdout.pct_change(fill_method=None)
    flags: list[dict] = []
    for ticker in returns.columns:
        series = returns[ticker]
        big = series[series.abs() > data_cfg.flag_return_large].dropna()
        for day, value in big.items():
            severity = "extreme" if abs(value) > data_cfg.flag_return_extreme else "large"
            flags.append(
                {
                    "date": day.date().isoformat(),
                    "ticker": ticker,
                    "daily_return": float(value),
                    "severity": severity,
                }
            )
    return sorted(flags, key=lambda f: (f["date"], f["ticker"]))


def write_cache(prices: pd.DataFrame, dates: Dates, data_dir: Path) -> None:
    validation_end = pd.Timestamp(dates.validation_end)
    data_dir.mkdir(parents=True, exist_ok=True)
    prices[prices.index <= validation_end].to_parquet(data_dir / PRE_HOLDOUT_FILE)
    prices[prices.index > validation_end].to_parquet(data_dir / HOLDOUT_FILE)


def read_pre_holdout(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Prices from start through validation_end. Holdout is not included."""
    return pd.read_parquet(data_dir / PRE_HOLDOUT_FILE)


def read_holdout_cache(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Holdout-period prices. Only pipeline/holdout.py may call this."""
    check_holdout_caller()
    return pd.read_parquet(data_dir / HOLDOUT_FILE)


def fetch_and_cache(
    settings: Settings,
    universe: Universe,
    source: PriceSource,
    data_dir: Path = DATA_DIR,
) -> CleaningReport:
    dates = settings.dates
    raw = source.fetch_adjusted_close(
        universe.tickers,
        dates.start,
        dates.holdout_end + timedelta(days=1),
    )
    cleaned, dropped = clean_prices(raw, dates, settings.data)
    write_cache(cleaned, dates, data_dir)

    pre = cleaned[cleaned.index <= pd.Timestamp(dates.validation_end)]
    report = CleaningReport(
        fetched_at=datetime.now(UTC).isoformat(timespec="seconds"),
        first_date=pre.index.min().date().isoformat(),
        last_date=pre.index.max().date().isoformat(),
        tickers_requested=list(universe.tickers),
        tickers_kept=list(cleaned.columns),
        dropped=dropped,
        missing_after_fill={t: int(pre[t].isna().sum()) for t in cleaned.columns},
        return_flags=find_return_flags(pre, settings.data),
    )
    (data_dir / REPORT_FILE).write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


def read_report(data_dir: Path = DATA_DIR) -> CleaningReport:
    return CleaningReport(**json.loads((data_dir / REPORT_FILE).read_text(encoding="utf-8")))


def load_prices(
    settings: Settings,
    universe: Universe,
    refresh: bool = False,
    source: PriceSource | None = None,
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """Pre-holdout prices, fetching first if the cache is missing or `refresh` is set."""
    if refresh or not (data_dir / PRE_HOLDOUT_FILE).exists():
        fetch_and_cache(settings, universe, source or YFinanceSource(), data_dir)
    return read_pre_holdout(data_dir)


def format_quality_summary(report: CleaningReport) -> str:
    lines = [
        f"Fetched at: {report.fetched_at}",
        f"Pre-holdout date range: {report.first_date} to {report.last_date}",
        f"Tickers requested: {len(report.tickers_requested)}, kept: {len(report.tickers_kept)}",
    ]
    if report.dropped:
        lines.append("Dropped tickers:")
        lines += [f"  {t}: {why}" for t, why in report.dropped.items()]
    else:
        lines.append("Dropped tickers: none")

    lines.append("Missing values after forward-fill (pre-holdout):")
    for ticker, count in report.missing_after_fill.items():
        if count:
            lines.append(f"  {ticker}: {count}")

    flags = report.return_flags
    lines.append(f"Daily moves beyond the flag threshold: {len(flags)} (not removed)")
    for flag in flags:
        lines.append(
            f"  {flag['date']}  {flag['ticker']:<6} {flag['daily_return']:+.2%}  {flag['severity']}"
        )
    return "\n".join(lines)
