"""Train / validation / holdout windows and the holdout guard.

A window is a range of dates on which a daily return is *realised*. A weight
set on the last train day earns the return of the first validation day, and
that return counts toward validation. The windows therefore do not overlap.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from typing import Literal

import pandas as pd

from alpha_lab.config import Dates

Split = Literal["train", "validation", "holdout"]

HOLDOUT_MODULE = "alpha_lab.pipeline.holdout"


class HoldoutAccessError(PermissionError):
    """Raised when holdout data is requested from outside pipeline/holdout.py."""


def check_holdout_caller() -> None:
    """Raise unless the function that called us lives in pipeline/holdout.py.

    This is a convention guard, not a security boundary. It stops accidental
    access, and a test checks that nothing else calls the holdout readers.
    Frame 0 is this function, frame 1 is the function that called it (get_split
    or read_holdout_cache), and frame 2 is that function's caller.
    """
    caller = sys._getframe(2).f_globals.get("__name__", "")
    if caller != HOLDOUT_MODULE:
        raise HoldoutAccessError(
            f"holdout data can only be read from {HOLDOUT_MODULE}, not from {caller!r}"
        )


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


def get_split(prices: pd.DataFrame, split: Split, dates: Dates) -> pd.DataFrame:
    """Return the rows of `prices` that fall inside the split's window."""
    if split == "holdout":
        check_holdout_caller()
    lower, upper = split_window(dates, split)
    in_window = (prices.index >= pd.Timestamp(lower)) & (prices.index <= pd.Timestamp(upper))
    return prices.loc[in_window]
