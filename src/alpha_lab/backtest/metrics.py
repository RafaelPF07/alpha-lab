"""Performance metrics for a daily net-return series. Risk-free rate is zero."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Metrics:
    n_days: int
    ann_return: float  # compound annual growth rate
    ann_volatility: float
    sharpe: float  # mean / std of daily returns, annualised by sqrt(annualization_days)
    max_drawdown: float  # most negative peak-to-trough fall of the equity curve, <= 0
    avg_daily_turnover: float
    hit_rate: float  # fraction of days with a positive net return


def compute_metrics(
    net_returns: pd.Series, turnover: pd.Series, annualization_days: int
) -> Metrics:
    r = net_returns.astype("float64")
    n = len(r)
    if n < 2:
        raise ValueError("need at least 2 daily returns to compute metrics")

    total = float((1.0 + r).prod())
    ann_return = total ** (annualization_days / n) - 1.0 if total > 0 else -1.0

    std = float(r.std(ddof=1))
    ann_vol = std * np.sqrt(annualization_days)
    sharpe = float(r.mean()) / std * np.sqrt(annualization_days) if std > 0 else float("nan")

    equity = (1.0 + r).cumprod()
    peak = np.maximum(equity.cummax(), 1.0)  # the curve starts at 1.0 before day one
    max_dd = float((equity / peak - 1.0).min())

    return Metrics(
        n_days=n,
        ann_return=float(ann_return),
        ann_volatility=float(ann_vol),
        sharpe=float(sharpe),
        max_drawdown=max_dd,
        avg_daily_turnover=float(turnover.mean()),
        hit_rate=float((r > 0).mean()),
    )


def deflated_sharpe_ratio(*args: object, **kwargs: object) -> float:
    """Deflated Sharpe ratio (Bailey and López de Prado, 2014). Implemented in Phase 3."""
    raise NotImplementedError("deflated Sharpe ratio is implemented in Phase 3")
