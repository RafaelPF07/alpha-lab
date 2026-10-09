"""Holding period: the harness takes weights on a rebalance schedule and holds them in between.

Weights are sampled every `holding_days` rows and forward-filled. A weight taken on day t
only uses prices up to day t, so this does not introduce lookahead. It lowers turnover,
and with it the cost drag that daily-rebalanced signals suffer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def hold_weights(weights: pd.DataFrame, holding_days: int) -> pd.DataFrame:
    # NaN and infinite weights count as flat, the same convention scale_to_gross uses in the
    # engine. Cleaning first stops a NaN on a rebalance day from carrying an old weight forward.
    clean = weights.astype("float64").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if holding_days <= 1:
        return clean
    on_rebalance = np.arange(len(clean)) % holding_days == 0
    # clean is a new frame, so the caller's weights are untouched.
    clean.iloc[~on_rebalance] = np.nan
    return clean.ffill().fillna(0.0)
