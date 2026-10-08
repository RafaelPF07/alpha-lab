"""Cleaning rules, tested on synthetic data so no network is needed."""

import numpy as np
import pandas as pd
import pytest

from alpha_lab.config import load_settings
from alpha_lab.data.loader import clean_prices, find_return_flags, read_pre_holdout, write_cache


@pytest.fixture
def settings():
    return load_settings()


@pytest.fixture
def synthetic_raw(settings):
    index = pd.bdate_range(settings.dates.start, settings.dates.holdout_end)
    n = len(index)
    raw = pd.DataFrame(index=index)
    raw.index.name = "date"
    raw["GOOD"] = 100.0 * 1.0001 ** np.arange(n)

    hole = raw["GOOD"].copy()
    hole.iloc[::33] = np.nan  # about 3% of rows missing: over the 2% limit
    raw["HOLE"] = hole

    gap = raw["GOOD"].copy()
    gap.iloc[500:504] = np.nan  # a 4-day gap: forward-fill fills 3, leaves 1
    raw["GAP"] = gap

    spike = raw["GOOD"].copy()
    spike.iloc[100:] *= 1.6  # a single +60% day
    raw["SPIKE"] = spike

    move = raw["GOOD"].copy()
    move.iloc[200:] *= 1.2  # a single +20% day
    raw["MOVE"] = move
    return raw


def test_drops_tickers_over_the_missing_limit(settings, synthetic_raw):
    cleaned, dropped = clean_prices(synthetic_raw, settings.dates, settings.data)

    assert "HOLE" in dropped
    assert "missing" in dropped["HOLE"]
    assert "HOLE" not in cleaned.columns
    assert {"GOOD", "GAP", "SPIKE", "MOVE"} <= set(cleaned.columns)


def test_forward_fill_is_limited_to_three_days(settings, synthetic_raw):
    cleaned, _ = clean_prices(synthetic_raw, settings.dates, settings.data)
    # Rows 500-503 are NaN. Rows 500-502 get filled, row 503 stays NaN.
    assert cleaned["GAP"].isna().sum() == 1
    assert pd.isna(cleaned["GAP"].iloc[503])
    assert not pd.isna(cleaned["GAP"].iloc[502])


def test_large_moves_are_flagged_not_deleted(settings, synthetic_raw):
    cleaned, _ = clean_prices(synthetic_raw, settings.dates, settings.data)
    pre = cleaned[cleaned.index <= pd.Timestamp(settings.dates.validation_end)]
    flags = find_return_flags(pre, settings.data)

    by_ticker = {(f["ticker"], f["severity"]) for f in flags}
    assert ("SPIKE", "extreme") in by_ticker  # +60% beyond the 50% threshold
    assert ("MOVE", "large") in by_ticker  # +20% beyond the 15% threshold
    assert "GOOD" not in {f["ticker"] for f in flags}
    # The SPIKE values are still in the data.
    assert cleaned["SPIKE"].iloc[100] == pytest.approx(synthetic_raw["SPIKE"].iloc[100])


def test_cache_splits_holdout_away_from_default_reader(settings, synthetic_raw, tmp_path):
    cleaned, _ = clean_prices(synthetic_raw, settings.dates, settings.data)
    write_cache(cleaned, settings.dates, tmp_path)

    pre = read_pre_holdout(tmp_path)
    assert pre.index.max() <= pd.Timestamp(settings.dates.validation_end)
    assert len(pre) < len(cleaned)
