"""Holdout protection. Holdout rows must not be reachable from ordinary code."""

import re
from pathlib import Path

import pandas as pd
import pytest

from alpha_lab.config import load_settings
from alpha_lab.data.loader import read_holdout_cache
from alpha_lab.data.splits import HoldoutAccessError, get_split

SRC = Path(__file__).resolve().parents[1] / "src" / "alpha_lab"


@pytest.fixture
def dates():
    return load_settings().dates


def test_get_split_refuses_holdout_outside_holdout_module(dates):
    prices = pd.DataFrame({"A": [1.0]}, index=pd.DatetimeIndex([dates.holdout_end]))
    with pytest.raises(HoldoutAccessError):
        get_split(prices, "holdout", dates)


def test_read_holdout_cache_refuses_outside_holdout_module(tmp_path):
    with pytest.raises(HoldoutAccessError):
        read_holdout_cache(tmp_path)


def test_train_and_validation_splits_are_allowed(dates):
    prices = pd.DataFrame(
        {"A": 1.0},
        index=pd.bdate_range(dates.start, dates.validation_end),
    )
    assert len(get_split(prices, "train", dates)) > 0
    assert len(get_split(prices, "validation", dates)) > 0


def test_only_holdout_module_calls_the_holdout_reader():
    callers = []
    for path in SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if re.search(r"read_holdout_cache\(", text) and not re.search(
            r"def read_holdout_cache\(", text
        ):
            callers.append(path.relative_to(SRC).as_posix())
    assert callers == ["pipeline/holdout.py"]
