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


def scale_to_gross(weights: pd.DataFrame, max_gross: float) -> pd.DataFrame:
    """Clean NaN/inf to 0, then shrink any day whose gross exposure exceeds max_gross."""
    w = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float64")
    gross = w.abs().sum(axis=1)
    scale = (max_gross / gross).where(gross > max_gross, 1.0)
    return w.mul(scale, axis=0)


def run_backtest(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    lower: date,
    upper: date,
    cfg: BacktestConfig,
) -> BacktestResult:
    """Score `weights` on returns realised between `lower` and `upper` inclusive."""
    if not weights.index.equals(prices.index) or not weights.columns.equals(prices.columns):
        raise ValueError("weights must have the same index and columns as prices")

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

    scored = prices.loc[keep]
    if scored.isna().any().any():
        raise ValueError("prices contain NaN inside the scoring window; clean the data first")

    gross, net, turnover = gross[keep], net[keep], turnover[keep]
    metrics = compute_metrics(net, turnover, cfg.annualization_days)
    return BacktestResult(gross_returns=gross, net_returns=net, turnover=turnover, metrics=metrics)
