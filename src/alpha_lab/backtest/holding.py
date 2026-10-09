"""Holding period: the harness takes weights on a rebalance schedule and holds them in between.

Weights are sampled every `holding_days` rows and forward-filled. A weight taken on day t
only uses prices up to day t, so this does not introduce lookahead. It lowers turnover,
and with it the cost drag that daily-rebalanced signals suffer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def hold_weights(weights: pd.DataFrame, holding_days: int) -> pd.DataFrame:
    if holding_days <= 1:
        return weights
    held = weights.copy()
    on_rebalance = np.arange(len(held)) % holding_days == 0
    held.iloc[~on_rebalance] = np.nan
    return held.ffill().fillna(0.0)
