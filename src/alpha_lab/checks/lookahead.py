"""Dynamic lookahead tests. A signal must pass both to be accepted.

1. Truncation: signal(prices[:T]) must match signal(prices) on every row up to T.
   If it doesn't, the signal used data from after T.
2. Perturbation: scramble prices after T and check the rows up to T are unchanged.
   This catches cases where the truncated run happened to hide the future.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha_lab.signals.baselines import Signal

TOLERANCE = 1e-9


@dataclass(frozen=True)
class LookaheadResult:
    test: str
    passed: bool
    cutoff: pd.Timestamp | None = None
    first_offending_date: pd.Timestamp | None = None
    ticker: str | None = None


def _cutoffs(index: pd.DatetimeIndex, n: int) -> list[pd.Timestamp]:
    """n cutoff dates spread evenly from 15% to 95% of the rows.

    The span reaches both tails, so a leak confined to late rows is compared. The first cutoff
    stays past the 252-day lookback on the training window (about 350 rows of about 2,350).
    """
    positions = np.linspace(int(len(index) * 0.15), int(len(index) * 0.95), n).astype(int)
    return [index[p] for p in positions]


def _checked(weights: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    if not weights.index.equals(prices.index) or not weights.columns.equals(prices.columns):
        raise ValueError("signal must return a frame with the same index and columns as prices")
    return weights


def _first_mismatch(
    a: pd.DataFrame, b: pd.DataFrame, tol: float
) -> tuple[pd.Timestamp, str] | None:
    """First (date, ticker) where a and b differ. NaN is treated as equal to NaN."""
    close = np.isclose(a.to_numpy(float), b.to_numpy(float), rtol=0.0, atol=tol, equal_nan=True)
    if close.all():
        return None
    rows, cols = np.nonzero(~close)
    first_row = rows.min()
    first_col = cols[rows == first_row][0]
    return a.index[first_row], a.columns[first_col]


def truncation_test(
    signal: Signal, prices: pd.DataFrame, n_cutoffs: int = 5, tol: float = TOLERANCE
) -> LookaheadResult:
    full = _checked(signal(prices), prices)
    for cutoff in _cutoffs(prices.index, n_cutoffs):
        visible = prices.loc[:cutoff]
        partial = _checked(signal(visible), visible)
        mismatch = _first_mismatch(partial, full.loc[:cutoff], tol)
        if mismatch is not None:
            return LookaheadResult("truncation", False, cutoff, mismatch[0], mismatch[1])
    return LookaheadResult("truncation", True)


def perturbation_test(
    signal: Signal,
    prices: pd.DataFrame,
    n_cutoffs: int = 5,
    seed: int = 42,
    noise_scale: float = 0.05,
    tol: float = TOLERANCE,
) -> LookaheadResult:
    base = _checked(signal(prices), prices)
    rng = np.random.default_rng(seed)
    for cutoff in _cutoffs(prices.index, n_cutoffs):
        perturbed = prices.copy()
        after = perturbed.index > cutoff
        noise = rng.normal(0.0, noise_scale, size=(int(after.sum()), prices.shape[1]))
        perturbed.loc[after] = perturbed.loc[after].to_numpy() * (1.0 + noise)
        changed = _checked(signal(perturbed), perturbed)
        mismatch = _first_mismatch(changed.loc[:cutoff], base.loc[:cutoff], tol)
        if mismatch is not None:
            return LookaheadResult("perturbation", False, cutoff, mismatch[0], mismatch[1])
    return LookaheadResult("perturbation", True)


def run_lookahead_tests(signal: Signal, prices: pd.DataFrame) -> list[LookaheadResult]:
    return [truncation_test(signal, prices), perturbation_test(signal, prices)]


def passes_lookahead(results: list[LookaheadResult]) -> bool:
    return all(r.passed for r in results)
