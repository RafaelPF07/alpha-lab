"""Fetch, clean and cache daily adjusted closes. The cache holds pre-holdout data only.

The holdout window is never written to disk. pipeline/holdout.py downloads it in memory during
`lab holdout`, and its SHA-256 goes into the ledger.

A fingerprint file beside the cache records the tickers, dates and cleaning settings the cache
was built with. load_prices refuses a cache whose fingerprint does not match the current settings.

Cleaning decisions that look at missingness use only the pre-holdout period, so a holdout-period
gap cannot change which tickers are kept.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from alpha_lab.config import DATA_DIR, DataConfig, Dates, Settings, Universe

log = logging.getLogger(__name__)

PRE_HOLDOUT_FILE = "prices_pre_holdout.parquet"
FINGERPRINT_FILE = "cache_fingerprint.json"
REPORT_FILE = "cleaning_report.json"


class CacheMismatch(RuntimeError):
    """The price cache has no fingerprint, or was built for different settings."""


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
    """Drop badly-missing tickers, then forward-fill short gaps. Returns (cleaned, dropped).

    Rows after validation_end are discarded here, so no holdout row can reach the cache.
    """
    prices = raw.sort_index()
    prices = prices[~prices.index.duplicated(keep="first")]
    prices = prices.loc[: pd.Timestamp(dates.validation_end)]

    missing = prices.isna().mean()

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
    """Write the pre-holdout rows (up to validation_end). Later rows are dropped, never written."""
    validation_end = pd.Timestamp(dates.validation_end)
    data_dir.mkdir(parents=True, exist_ok=True)
    prices[prices.index <= validation_end].to_parquet(data_dir / PRE_HOLDOUT_FILE)


def read_pre_holdout(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Prices from start through validation_end. No fingerprint check; see read_checked_pre_holdout."""
    return pd.read_parquet(data_dir / PRE_HOLDOUT_FILE)


def cache_fingerprint(settings: Settings, universe: Universe) -> dict[str, Any]:
    """What the cache depends on: the tickers requested, the dates and the cleaning settings."""
    return {
        "tickers": list(universe.tickers),
        "dates": settings.dates.model_dump(mode="json"),
        "data": settings.data.model_dump(mode="json"),
    }


def _digest(fingerprint: dict[str, Any]) -> str:
    text = json.dumps(fingerprint, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_fingerprint(settings: Settings, universe: Universe, data_dir: Path = DATA_DIR) -> None:
    fingerprint = cache_fingerprint(settings, universe)
    data_dir.mkdir(parents=True, exist_ok=True)
    body = {"sha256": _digest(fingerprint), "fingerprint": fingerprint}
    (data_dir / FINGERPRINT_FILE).write_text(json.dumps(body, indent=2), encoding="utf-8")


def check_fingerprint(settings: Settings, universe: Universe, data_dir: Path = DATA_DIR) -> None:
    """Raise CacheMismatch unless the cache was built for exactly these settings."""
    expected = _digest(cache_fingerprint(settings, universe))
    try:
        stored = json.loads((data_dir / FINGERPRINT_FILE).read_text(encoding="utf-8"))["sha256"]
    except (OSError, ValueError, KeyError, TypeError):
        stored = None
    if stored != expected:
        raise CacheMismatch(
            f"The price cache in {data_dir} was built for different settings (tickers, dates "
            "or cleaning rules), or it has no fingerprint. Run `lab data fetch --refresh` "
            "to rebuild it."
        )


def read_checked_pre_holdout(
    settings: Settings, universe: Universe, data_dir: Path = DATA_DIR
) -> pd.DataFrame:
    """Pre-holdout prices, but only if the cache matches the current settings."""
    check_fingerprint(settings, universe, data_dir)
    return read_pre_holdout(data_dir)


def fetch_and_cache(
    settings: Settings,
    universe: Universe,
    source: PriceSource,
    data_dir: Path = DATA_DIR,
) -> CleaningReport:
    dates = settings.dates
    # Requested only up to validation_end. The holdout window is never fetched here.
    raw = source.fetch_adjusted_close(
        universe.tickers,
        dates.start,
        dates.validation_end + timedelta(days=1),
    )
    cleaned, dropped = clean_prices(raw, dates, settings.data)
    write_cache(cleaned, dates, data_dir)
    write_fingerprint(settings, universe, data_dir)

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
    """Pre-holdout prices, fetching first if the cache is missing or `refresh` is set.

    Raises CacheMismatch if the cache was built for other settings.
    """
    if refresh or not (data_dir / PRE_HOLDOUT_FILE).exists():
        fetch_and_cache(settings, universe, source or YFinanceSource(), data_dir)
    return read_checked_pre_holdout(settings, universe, data_dir)


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
