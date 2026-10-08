"""Hand-written reference signals. Each one maps prices -> weights (same shape, index and columns).

`cheater` is a test fixture. It reads tomorrow's return, so the lookahead tests must reject it.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

Signal = Callable[[pd.DataFrame], pd.DataFrame]


def _long_short(score: pd.DataFrame, bucket: float = 0.2) -> pd.DataFrame:
    """Equal-weight the top `bucket` (long) and bottom `bucket` (short) of each day's scores.

    Dollar-neutral: long weights sum to +1 and short weights to -1 before the engine
    scales gross exposure down to the leverage limit.
    """
    ranks = score.rank(axis=1, pct=True)
    long = (ranks > 1.0 - bucket).astype("float64")
    short = (ranks <= bucket).astype("float64")
    n_long = long.sum(axis=1).replace(0.0, np.nan)
    n_short = short.sum(axis=1).replace(0.0, np.nan)
    return long.div(n_long, axis=0).fillna(0.0) - short.div(n_short, axis=0).fillna(0.0)


def momentum_12_1(prices: pd.DataFrame) -> pd.DataFrame:
    """Winners over the last 12 months, skipping the most recent month (about 252 and 21 trading days)."""
    score = prices.shift(21) / prices.shift(252) - 1.0
    return _long_short(score)


def reversal_5d(prices: pd.DataFrame) -> pd.DataFrame:
    """Losers over the last 5 days are bought, winners sold."""
    score = -(prices / prices.shift(5) - 1.0)
    return _long_short(score)


def equal_weight_long(prices: pd.DataFrame) -> pd.DataFrame:
    """Market benchmark: hold every ticker at 1/N."""
    n = prices.shape[1]
    return pd.DataFrame(1.0 / n, index=prices.index, columns=prices.columns)


def cheater(prices: pd.DataFrame) -> pd.DataFrame:
    """Deliberately broken: ranks on tomorrow's return via shift(-1). Must fail the lookahead tests."""
    next_day_return = prices.pct_change(fill_method=None).shift(-1)
    return _long_short(next_day_return)


SIGNALS: dict[str, Signal] = {
    "momentum_12_1": momentum_12_1,
    "reversal_5d": reversal_5d,
    "equal_weight_long": equal_weight_long,
    "cheater": cheater,
}
