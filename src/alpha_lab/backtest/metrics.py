"""Performance metrics and the deflated Sharpe ratio. Risk-free rate is zero.

Units: annualised_* functions return annual figures. The deflated Sharpe ratio
works in per-period (daily) units, because the paper's formula is stated that way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd

EULER_GAMMA = 0.5772156649015329
_NORMAL = NormalDist()


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
    ann_vol = std * math.sqrt(annualization_days)
    sharpe = float(r.mean()) / std * math.sqrt(annualization_days) if std > 0 else float("nan")

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


def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """SR0: the Sharpe ratio the best of N pure-noise trials is expected to reach.

    SR0 = sqrt(V) * [ (1 - g) * Z^-1(1 - 1/N) + g * Z^-1(1 - 1/(N e)) ]

    V is the variance of the trials' Sharpe ratios, g is the Euler-Mascheroni
    constant, Z^-1 is the inverse standard normal CDF and e is Euler's number.
    This is a Gumbel approximation to the expected maximum of N independent normal draws,
    not an exact value. Against Monte Carlo it is about 0.04 low at N = 2 (0.520 against the
    exact 1/sqrt(pi) = 0.564) and within about 0.04 for N from 10 to 500.
    """
    if n_trials < 2:
        return 0.0
    first = _NORMAL.inv_cdf(1.0 - 1.0 / n_trials)
    second = _NORMAL.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(max(sharpe_variance, 0.0)) * (
        (1.0 - EULER_GAMMA) * first + EULER_GAMMA * second
    )


def deflated_sharpe_ratio(
    sharpe: float,
    n_obs: int,
    skew: float,
    kurtosis: float,
    n_trials: int,
    sharpe_variance: float,
) -> float:
    """Deflated Sharpe ratio (Bailey and Lopez de Prado, 2014). Returns a probability in [0, 1].

    DSR = Z[ (SR - SR0) * sqrt(T - 1) / sqrt(1 - g3 * SR + (g4 - 1) / 4 * SR^2) ]

    Terms, all per-period (daily) and not annualised:
      SR      the candidate's Sharpe ratio on its own returns
      SR0     the best Sharpe expected from N noise trials (see expected_max_sharpe)
      T       number of return observations
      g3      skewness of the returns (0 for a normal distribution)
      g4      kurtosis of the returns, NON-excess (3 for a normal distribution)
      Z       the standard normal CDF

    sharpe_variance must be measured on samples of length n_obs. The sampling variance of a
    per-period Sharpe under noise is about 1 / (T - 1), so a variance measured on a longer
    training sample has to be rescaled first. The report does this (report.length_matched_variance).

    A DSR near 1 means the Sharpe is unlikely to be a lucky maximum over N trials.
    A DSR near 0.5 means the candidate is indistinguishable from the best noise trial.
    """
    if n_obs < 2:
        raise ValueError("need at least 2 observations")
    sr0 = expected_max_sharpe(n_trials, sharpe_variance)
    variance_term = 1.0 - skew * sharpe + (kurtosis - 1.0) / 4.0 * sharpe**2
    if variance_term <= 0:
        return float("nan")  # the normal approximation breaks down for extreme skew/kurtosis
    z = (sharpe - sr0) * math.sqrt(n_obs - 1) / math.sqrt(variance_term)
    return _NORMAL.cdf(z)


def per_period_sharpe(returns: pd.Series) -> float:
    std = float(returns.std(ddof=1))
    return float(returns.mean()) / std if std > 0 else float("nan")


def moments(returns: pd.Series) -> tuple[float, float]:
    """Skewness and non-excess kurtosis (3 for a normal distribution)."""
    r = returns.astype("float64")
    centred = r - r.mean()
    sigma = float(r.std(ddof=0))
    if sigma == 0:
        return 0.0, 3.0
    skew = float((centred**3).mean() / sigma**3)
    kurt = float((centred**4).mean() / sigma**4)
    return skew, kurt
