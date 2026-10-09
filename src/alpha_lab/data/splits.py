"""Train / validation / holdout windows.

A window is a range of dates on which a daily return is *realised*. A weight
set on the last train day earns the return of the first validation day, and
that return counts toward validation. The windows therefore do not overlap.

The holdout window is defined here, but its prices are never read from disk. The only code
that requests them is pipeline/holdout.py, and tests/test_holdout_guard.py enforces that with an
AST scan. This module has no runtime check, because a runtime check on the caller's name can be
bypassed by any code that can run.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Literal

import pandas as pd

from alpha_lab.config import Dates

Split = Literal["train", "validation", "holdout"]


def split_window(dates: Dates, split: Split) -> tuple[date, date]:
    """Inclusive (first, last) calendar dates on which returns count for this split."""
    day_after_train = dates.train_end + timedelta(days=1)
    day_after_validation = dates.validation_end + timedelta(days=1)
    windows = {
        "train": (dates.start, dates.train_end),
        "validation": (day_after_train, dates.validation_end),
        "holdout": (day_after_validation, dates.holdout_end),
    }
    return windows[split]


def holdout_dates(dates: Dates) -> tuple[date, date]:
    """First and last date of the holdout window. Dates only: no prices are read or requested.

    Dashboard code uses this for the shaded band on the equity chart (R3-WHOLE-05). It is defined
    here, and not called with the literal "holdout" elsewhere, so the holdout tripwire in
    tests/test_holdout_guard.py stays meaningful.
    """
    return split_window(dates, "holdout")


def get_split(prices: pd.DataFrame, split: Split, dates: Dates) -> pd.DataFrame:
    """Return the rows of `prices` that fall inside the split's window."""
    lower, upper = split_window(dates, split)
    in_window = (prices.index >= pd.Timestamp(lower)) & (prices.index <= pd.Timestamp(upper))
    return prices.loc[in_window]
