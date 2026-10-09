"""Fetch, clean and cache daily adjusted closes. The cache holds pre-holdout data only.

This module never writes the holdout window. pipeline/holdout.py downloads it in memory during
`lab holdout`, and its SHA-256 goes into the ledger. That module hands the window to candidate
code. The in-process refusals in checks/sandbox_runner.py cover the spellings the reviewers found,
but a candidate can still write the window by a route they do not cover. Nothing in this module
prevents that (DECISION-4; see the README sandbox section).

A fingerprint file beside the cache records the tickers, dates and cleaning settings the cache
was built with. Its digest and its body must both match the current settings, or the cache is
refused. This is a consistency check on honest use. Anyone who can write the data directory can
write a matching fingerprint too.

Which tickers are kept is decided with the training window only (start to train_end), so neither
validation nor holdout prices can change the universe (DECISION-15).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
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
# Version of the cleaning rule that decides which tickers are kept. 1 decided on all rows (before
# DECISION-15). 2 decides on the training window only. A cache with any other value is refused, so
# a cache built under the old universe rule cannot pass for the new one (R4-SEC-05).
CLEANING_RULE = 2


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
    """Drop tickers that are too incomplete in training, then forward-fill. Returns (cleaned, dropped).

    Every decision about which tickers are kept uses the training window only (DECISION-15):
    the missing-fraction limit and the check for gaps that forward-fill cannot close. Validation
    rows are kept in the cache, but they cannot change the universe. Rows after validation_end are
    discarded here, so no holdout row can reach the cache.
    """
    prices = raw.sort_index()
    prices = prices[~prices.index.duplicated(keep="first")]
    prices = prices.loc[: pd.Timestamp(dates.validation_end)]
    train = prices.loc[: pd.Timestamp(dates.train_end)]

    missing = train.isna().mean()

    dropped: dict[str, str] = {}
    for ticker in prices.columns:
        frac = float(missing[ticker])
        if frac > data_cfg.max_missing_fraction:
            reason = (
                f"{frac:.2%} of training-window rows missing "
                f"(limit {data_cfg.max_missing_fraction:.0%})"
            )
            dropped[ticker] = reason
            log.warning("dropping %s: %s", ticker, reason)

    kept = [t for t in prices.columns if t not in dropped]
    filled = prices[kept].ffill(limit=data_cfg.ffill_limit_days).astype("float64")

    # A gap longer than the fill limit leaves NaN. Any backtest whose window covers a training
    # day with NaN fails, so the ticker is dropped here, not left to fail every run (R3-STAT-02).
    residual = filled.loc[: pd.Timestamp(dates.train_end)].isna().sum()
    for ticker in kept:
        count = int(residual[ticker])
        if count:
            reason = (
                f"{count} training-window day(s) still missing after forward fill "
                f"(limit {data_cfg.ffill_limit_days} days)"
            )
            dropped[ticker] = reason
            log.warning("dropping %s: %s", ticker, reason)

    survivors = [t for t in kept if t not in dropped]
    return filled[survivors], dropped


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


def read_pre_holdout(data_dir: Path = DATA_DIR, validation_end: date | None = None) -> pd.DataFrame:
    """Prices from start through validation_end. No fingerprint check; see read_checked_pre_holdout.

    With validation_end given, a cache that holds any later row is refused (CacheMismatch). The
    cut is enforced on read, not only on write, because the file can be replaced by hand.
    """
    prices = pd.read_parquet(data_dir / PRE_HOLDOUT_FILE)
    if validation_end is not None and len(prices.index):
        latest = prices.index.max()
        if latest > pd.Timestamp(validation_end):
            raise CacheMismatch(
                f"The price cache in {data_dir} has rows after validation_end ({validation_end}). "
                "Run `lab data fetch --refresh` to rebuild it."
            )
    return prices


def cache_fingerprint(settings: Settings, universe: Universe) -> dict[str, Any]:
    """What the cache depends on: the tickers requested, the dates, the cleaning settings and the rule."""
    return {
        "tickers": list(universe.tickers),
        "dates": settings.dates.model_dump(mode="json"),
        "data": settings.data.model_dump(mode="json"),
        "cleaning_rule": CLEANING_RULE,
    }


def _digest(fingerprint: dict[str, Any]) -> str:
    text = json.dumps(fingerprint, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _replace_text(path: Path, text: str) -> None:
    """Write through a temporary file, so a reader never sees a half-written file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_fingerprint(settings: Settings, universe: Universe, data_dir: Path = DATA_DIR) -> None:
    fingerprint = cache_fingerprint(settings, universe)
    data_dir.mkdir(parents=True, exist_ok=True)
    body = {"sha256": _digest(fingerprint), "fingerprint": fingerprint}
    _replace_text(data_dir / FINGERPRINT_FILE, json.dumps(body, indent=2))


def check_fingerprint(settings: Settings, universe: Universe, data_dir: Path = DATA_DIR) -> None:
    """Raise CacheMismatch unless the stored fingerprint equals the current settings in full.

    Two things must hold: the stored body (tickers, dates and cleaning settings) equals what the
    current settings produce, and the stored digest is the digest of that body. Comparing only
    the digest, as before, accepted a file whose body had been edited.
    """
    current = cache_fingerprint(settings, universe)
    try:
        body = json.loads((data_dir / FINGERPRINT_FILE).read_text(encoding="utf-8"))
        stored, digest = body["fingerprint"], body["sha256"]
    except (OSError, ValueError, KeyError, TypeError):
        stored, digest = None, None
    if stored != current or digest != _digest(current):
        raise CacheMismatch(
            f"The price cache in {data_dir} was built for different settings (tickers, dates "
            "or cleaning rules), or it has no fingerprint. Run `lab data fetch --refresh` "
            "to rebuild it."
        )


def read_checked_pre_holdout(
    settings: Settings, universe: Universe, data_dir: Path = DATA_DIR
) -> pd.DataFrame:
    """Pre-holdout prices, but only if the cache matches the current settings and has no later rows."""
    check_fingerprint(settings, universe, data_dir)
    return read_pre_holdout(data_dir, validation_end=settings.dates.validation_end)


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
    # The old fingerprint goes first and the new one is written last. A crash in between leaves
    # no fingerprint, so the cache is refused, and it never passes for settings it was not built for.
    (data_dir / FINGERPRINT_FILE).unlink(missing_ok=True)
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
    """Pre-holdout prices from the cache. Downloads only when `refresh` is set.

    A missing cache is an error, not a silent download (R3-WHOLE-17): the user runs `lab data fetch`
    and sees the dropped-ticker report. Raises CacheMismatch if the cache is missing or was built
    for other settings.
    """
    if refresh:
        fetch_and_cache(settings, universe, source or YFinanceSource(), data_dir)
    elif not (data_dir / PRE_HOLDOUT_FILE).exists():
        raise CacheMismatch(f"There is no price cache in {data_dir}. Run `lab data fetch` first.")
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
