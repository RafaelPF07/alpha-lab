"""Weights -> daily returns -> transaction costs -> net returns -> metrics."""

from __future__ import annotations

from dataclasses import dataclass, replace
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


# A return across a gap is booked as one day's return, so a long gap hides several days of moves
# (R3-STAT-05). Gaps of up to this many weekdays pass. Without a trading calendar a holiday looks the
# same as a missing day, so a single missing day is still spanned by one return. That is a residual.
MAX_INTERIOR_GAP_WEEKDAYS = 4


def _check_interior_gaps(index: pd.DatetimeIndex, lower: date, upper: date) -> None:
    """Refuse a gap of more than MAX_INTERIOR_GAP_WEEKDAYS weekdays that a scored return crosses.

    The return on a row uses the row before it. So only a gap whose later row lies inside the window
    can change a scored return. Gaps before the window and after it are not checked (R4-STAT-06).
    """
    if len(index) < 2:
        return
    days = index.values.astype("datetime64[D]")
    # Weekdays strictly between each pair of consecutive price rows.
    missing = np.busday_count(days[:-1] + np.timedelta64(1, "D"), days[1:])
    later = index[1:]
    in_window = np.asarray((later >= pd.Timestamp(lower)) & (later <= pd.Timestamp(upper)))
    missing = np.where(in_window, missing, 0)
    worst = int(missing.max())
    if worst > MAX_INTERIOR_GAP_WEEKDAYS:
        gap_ends = index[int(missing.argmax()) + 1].date()
        raise BacktestDataError(
            f"prices have a gap of {worst} weekdays ending on {gap_ends}; at most "
            f"{MAX_INTERIOR_GAP_WEEKDAYS} are allowed, because a return across a longer gap "
            "would hide several days of moves"
        )


def _check_coverage(index: pd.DatetimeIndex, lower: date, upper: date) -> None:
    if len(index) == 0:
        raise BacktestDataError("prices are empty")
    _check_interior_gaps(index, lower, upper)
    tolerance = pd.Timedelta(days=WINDOW_TOLERANCE_DAYS)
    if index[0] > pd.Timestamp(lower) + tolerance:
        raise BacktestDataError(
            f"prices start on {index[0].date()}, after the window starts on {lower}"
        )
    if index[-1] < pd.Timestamp(upper) - tolerance:
        raise BacktestDataError(
            f"prices end on {index[-1].date()}, before the window ends on {upper}"
        )


def _execute(
    w: pd.DataFrame, returns: pd.DataFrame, scored_days: np.ndarray, cost_bps: float, lag: int
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Gross return, net return and turnover on every row, for weights filled `lag` bars late.

    Raises BacktestDataError when a NaN or infinite return on a scored day is held or traded across.
    """
    # The most important line in the project. Weights decided at the close of day t
    # are held through day t+1, so they earn the return from close t to close t+1.
    # Row t+1 of returns holds that return, and shifting the weights down one row
    # lines up the weight decided at t with it. Any other alignment is lookahead.
    # Each extra bar of lag shifts the fills one row further (execution_lag, STAT-07).
    positions = w.shift(1 + lag).fillna(0.0)

    turnover = positions.diff().abs().sum(axis=1)
    cost = turnover * cost_bps / 10_000.0

    # A return on day t needs yesterday's price too, so checking the scored returns also covers a
    # NaN on the day before the window. Infinite values are bad data as well.
    bad = ~np.isfinite(returns.iloc[scored_days].to_numpy(dtype=float))
    if bad.any():
        # A NaN return on day t is harmless only when the column is flat on day t and on day t-1:
        # then it earns no exposure and no trade is booked at its price. Anything held or traded
        # across the NaN is refused. scored_days starts at 1 or later, so day t-1 exists.
        held = positions.to_numpy(dtype=float)
        flat = (held[scored_days] == 0.0) & (held[scored_days - 1] == 0.0)
        harmful = bad & ~flat
        if harmful.any():
            row, col = np.argwhere(harmful)[0]
            raise BacktestDataError(
                "prices contain NaN or infinite values inside the scoring window "
                f"({returns.columns[col]} on {returns.index[scored_days[row]].date()}); "
                "clean the data first"
            )
    # Every non-finite return is zeroed here. The flat check above has already refused any that
    # mattered, so a zeroed return is one on a column with no position, and it earns nothing.
    # Zeroing first keeps 0 * inf out of the products, so no skipna is needed.
    clean = returns.where(np.isfinite(returns), 0.0)
    gross = (positions * clean).sum(axis=1)  # the first row has no return: sums to 0
    return gross, gross - cost, turnover


def _lag_one_sharpe(
    w: pd.DataFrame,
    returns: pd.DataFrame,
    scored_days: np.ndarray,
    keep: np.ndarray,
    cfg: BacktestConfig,
    execution_lag: int,
    sharpe: float,
) -> float:
    """The Sharpe of the same weights filled one bar after the decision, or NaN if the data refuse."""
    if execution_lag == 1:
        return sharpe
    try:
        _, net, turnover = _execute(w, returns, scored_days, cfg.cost_bps_per_side, 1)
    except BacktestDataError:
        return float("nan")
    return compute_metrics(net[keep], turnover[keep], cfg.annualization_days).sharpe


def run_backtest(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    lower: date,
    upper: date,
    cfg: BacktestConfig,
    execution_lag: int = 0,
) -> BacktestResult:
    """Score `weights` on returns realised between `lower` and `upper` inclusive.

    execution_lag is the number of bars between the decision and the fill. The default, 0, is the
    convention the whole harness uses: weights decided at the close of day t earn the return from t to
    t+1. 1 fills one bar later, and so on. The Sharpe at lag 1 is always recorded in
    metrics.sharpe_lag1, whatever lag the result itself uses.

    Raises BacktestDataError when the prices do not cover the window, when the prices have a gap of
    more than MAX_INTERIOR_GAP_WEEKDAYS weekdays inside the window, when a scored return is NaN or
    infinite on a column that is held or traded across it (this includes a NaN price on the day before
    the window), or when fewer than two days are scored. A NaN return on a column that is flat on that
    day and the day before scores as no return.
    """
    if isinstance(execution_lag, bool) or not isinstance(execution_lag, int) or execution_lag < 0:
        raise ValueError(
            f"execution_lag must be a whole number of bars, 0 or more; got {execution_lag!r}"
        )
    if not weights.index.equals(prices.index) or not weights.columns.equals(prices.columns):
        raise ValueError("weights must have the same index and columns as prices")
    _check_coverage(prices.index, lower, upper)

    w = scale_to_gross(weights, cfg.max_gross_leverage)
    returns = prices.pct_change(fill_method=None)

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

    gross, net, turnover = _execute(w, returns, scored_days, cfg.cost_bps_per_side, execution_lag)
    gross, net, turnover = gross[keep], net[keep], turnover[keep]
    metrics = compute_metrics(net, turnover, cfg.annualization_days)
    lag_one = _lag_one_sharpe(w, returns, scored_days, keep, cfg, execution_lag, metrics.sharpe)
    metrics = replace(metrics, sharpe_lag1=lag_one)
    return BacktestResult(gross_returns=gross, net_returns=net, turnover=turnover, metrics=metrics)
