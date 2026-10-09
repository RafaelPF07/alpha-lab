"""Weights -> daily returns -> transaction costs -> net returns -> metrics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from alpha_lab.backtest.metrics import Metrics, compute_metrics
from alpha_lab.config import BacktestConfig


@dataclass(frozen=True)
class BacktestResult:
    gross_returns: pd.Series
    net_returns: pd.Series
    turnover: pd.Series
    metrics: Metrics


class BacktestDataError(ValueError):
    """The prices cannot score this window: NaN or infinite values inside it, fewer than two
    scored days, or a window that runs past the start or end of the prices.

    Subclasses ValueError, so existing handlers still catch it. Pipelines catch this type to
    fail one attempt (or one holdout) without failing the whole run.
    """


def scale_to_gross(weights: pd.DataFrame, max_gross: float) -> pd.DataFrame:
    """Clean NaN/inf to 0, then shrink any day whose gross exposure exceeds max_gross."""
    w = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float64")
    gross = w.abs().sum(axis=1)
    scale = (max_gross / gross).where(gross > max_gross, 1.0)
    return w.mul(scale, axis=0)


# Window edges can fall on a weekend or a market holiday, so a window may end up to this many
# calendar days after the last price (or start that long before the first) and still be scored.
WINDOW_TOLERANCE_DAYS = 7


def _check_coverage(index: pd.DatetimeIndex, lower: date, upper: date) -> None:
    if len(index) == 0:
        raise BacktestDataError("prices are empty")
    tolerance = pd.Timedelta(days=WINDOW_TOLERANCE_DAYS)
    if index[0] > pd.Timestamp(lower) + tolerance:
        raise BacktestDataError(
            f"prices start on {index[0].date()}, after the window starts on {lower}"
        )
    if index[-1] < pd.Timestamp(upper) - tolerance:
        raise BacktestDataError(
            f"prices end on {index[-1].date()}, before the window ends on {upper}"
        )


def run_backtest(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    lower: date,
    upper: date,
    cfg: BacktestConfig,
) -> BacktestResult:
    """Score `weights` on returns realised between `lower` and `upper` inclusive.

    Raises BacktestDataError when the prices do not cover the window, when a scored return is
    NaN or infinite (this includes a NaN price on the day before the window), or when fewer than
    two days are scored.
    """
    if not weights.index.equals(prices.index) or not weights.columns.equals(prices.columns):
        raise ValueError("weights must have the same index and columns as prices")
    _check_coverage(prices.index, lower, upper)

    w = scale_to_gross(weights, cfg.max_gross_leverage)
    returns = prices.pct_change(fill_method=None)

    # The most important line in the project. Weights decided at the close of day t
    # are held through day t+1, so they earn the return from close t to close t+1.
    # Row t+1 of returns holds that return, and shifting the weights down one row
    # lines up the weight decided at t with it. Any other alignment is lookahead.
    positions = w.shift(1).fillna(0.0)

    gross = (positions * returns).sum(axis=1)  # the first row has no return, so it sums to 0
    turnover = positions.diff().abs().sum(axis=1)
    cost = turnover * cfg.cost_bps_per_side / 10_000.0
    net = gross - cost

    idx = prices.index
    positional = np.arange(len(idx)) >= max(1, cfg.min_history_days)
    in_window = np.asarray((idx >= pd.Timestamp(lower)) & (idx <= pd.Timestamp(upper)))
    keep = in_window & positional

    scored_days = np.flatnonzero(keep)
    if scored_days.size < 2:
        raise BacktestDataError(
            f"the window {lower}..{upper} has {scored_days.size} scored day(s); "
            "at least 2 are needed"
        )
    # A return on day t needs yesterday's price too, so checking the scored returns also covers a
    # NaN on the day before the window. Infinite values are bad data as well.
    scored_returns = returns.iloc[scored_days]
    bad = ~np.isfinite(scored_returns.to_numpy(dtype=float))
    if bad.any():
        row, col = np.argwhere(bad)[0]
        raise BacktestDataError(
            "prices contain NaN or infinite values inside the scoring window "
            f"({scored_returns.columns[col]} on {scored_returns.index[row].date()}); "
            "clean the data first"
        )

    gross, net, turnover = gross[keep], net[keep], turnover[keep]
    metrics = compute_metrics(net, turnover, cfg.annualization_days)
    return BacktestResult(gross_returns=gross, net_returns=net, turnover=turnover, metrics=metrics)
