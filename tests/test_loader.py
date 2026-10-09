"""Cleaning rules, the cache and its fingerprint. Synthetic data only, so no network is needed."""

import json

import numpy as np
import pandas as pd
import pytest

from alpha_lab.config import load_settings, load_universe
from alpha_lab.data import loader
from alpha_lab.data.loader import (
    FINGERPRINT_FILE,
    PRE_HOLDOUT_FILE,
    CacheMismatch,
    clean_prices,
    fetch_and_cache,
    find_return_flags,
    load_prices,
    read_checked_pre_holdout,
    read_pre_holdout,
    write_cache,
    write_fingerprint,
)


@pytest.fixture
def settings():
    return load_settings()


@pytest.fixture
def universe():
    return load_universe()


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
    assert {"GOOD", "SPIKE", "MOVE"} <= set(cleaned.columns)  # GAP has a training gap, see below


def test_a_residual_gap_in_training_drops_the_ticker(settings, synthetic_raw):
    # Rows 500-503 are NaN and fall in training. Forward-fill closes 500-502 and leaves 503, so
    # every train backtest covering the day would fail. The ticker is dropped instead (R3-STAT-02).
    cleaned, dropped = clean_prices(synthetic_raw, settings.dates, settings.data)
    assert "GAP" in dropped
    assert "training-window" in dropped["GAP"]
    assert "GAP" not in cleaned.columns


def test_a_gap_outside_training_is_kept_and_left_for_the_engine(settings, synthetic_raw):
    # DECISION-15: validation rows cannot change the universe. A 4-day gap in validation is kept,
    # and its one residual NaN is left for the engine (F1) to report as a data error.
    raw = synthetic_raw[["GOOD"]].copy()
    raw["VALGAP"] = raw["GOOD"]
    start = raw.index.get_loc(pd.Timestamp("2020-03-02"))
    raw.iloc[start : start + 4, raw.columns.get_loc("VALGAP")] = np.nan
    cleaned, dropped = clean_prices(raw, settings.dates, settings.data)
    assert "VALGAP" not in dropped
    assert cleaned["VALGAP"].isna().sum() == 1


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


def test_cleaning_discards_every_row_after_validation_end(settings, synthetic_raw):
    cleaned, _ = clean_prices(synthetic_raw, settings.dates, settings.data)
    assert cleaned.index.max() <= pd.Timestamp(settings.dates.validation_end)
    assert len(cleaned) < len(synthetic_raw)


def test_validation_only_gaps_cannot_drop_a_ticker(settings, synthetic_raw):
    # DECISION-15 (R3-STAT-11): half the validation rows are missing. Over the full pre-holdout
    # period that is above the 2% limit, but the universe is decided on training rows, so it is kept.
    raw = synthetic_raw[["GOOD"]].copy()
    raw["VALMISS"] = raw["GOOD"]
    validation = (raw.index > pd.Timestamp(settings.dates.train_end)) & (
        raw.index <= pd.Timestamp(settings.dates.validation_end)
    )
    rows = np.flatnonzero(validation)
    raw.iloc[rows[::2], raw.columns.get_loc("VALMISS")] = np.nan  # every other validation row
    cleaned, dropped = clean_prices(raw, settings.dates, settings.data)
    assert "VALMISS" not in dropped
    assert "VALMISS" in cleaned.columns


def test_holdout_only_gaps_cannot_change_the_universe_or_the_cleaned_values(settings, synthetic_raw):
    # M3 guard (R3-STAT-04): a ticker missing 15% of its holdout rows, and nothing earlier, is kept,
    # and its cleaned values equal those from a raw frame that has no holdout rows at all.
    raw = synthetic_raw[["GOOD"]].copy()
    raw["HOLDGAP"] = raw["GOOD"]
    holdout = raw.index > pd.Timestamp(settings.dates.validation_end)
    holdout_values = raw.loc[holdout, "HOLDGAP"].to_numpy().copy()
    holdout_values[np.arange(len(holdout_values)) % 7 == 0] = np.nan  # about 15% of holdout rows
    raw.loc[holdout, "HOLDGAP"] = holdout_values
    cleaned, dropped = clean_prices(raw, settings.dates, settings.data)
    reference, _ = clean_prices(raw.loc[~holdout], settings.dates, settings.data)
    assert "HOLDGAP" not in dropped
    pd.testing.assert_frame_equal(cleaned, reference)


def test_write_cache_keeps_holdout_rows_off_disk(settings, synthetic_raw, tmp_path):
    # Pass the full frame on purpose. write_cache must still drop the holdout rows.
    write_cache(synthetic_raw.loc[:, ["GOOD"]], settings.dates, tmp_path)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["prices_pre_holdout.parquet"]
    pre = read_pre_holdout(tmp_path)
    assert pre.index.max() <= pd.Timestamp(settings.dates.validation_end)


def test_load_prices_refuses_a_cache_built_for_other_settings(settings, universe, tmp_path):
    index = pd.bdate_range(settings.dates.start, settings.dates.validation_end)
    write_cache(pd.DataFrame(1.0, index=index, columns=["AAA"]), settings.dates, tmp_path)
    other = universe.model_copy(update={"stocks": [*universe.stocks, "ZZZZ"]})
    write_fingerprint(settings, other, tmp_path)

    with pytest.raises(CacheMismatch, match="lab data fetch --refresh"):
        load_prices(settings, universe, data_dir=tmp_path)


def test_load_prices_refuses_a_cache_with_no_fingerprint(settings, universe, tmp_path):
    index = pd.bdate_range(settings.dates.start, settings.dates.validation_end)
    write_cache(pd.DataFrame(1.0, index=index, columns=["AAA"]), settings.dates, tmp_path)

    with pytest.raises(CacheMismatch, match="lab data fetch --refresh"):
        load_prices(settings, universe, data_dir=tmp_path)


def test_load_prices_accepts_a_cache_that_matches_the_settings(settings, universe, tmp_path):
    index = pd.bdate_range(settings.dates.start, settings.dates.validation_end)
    write_cache(pd.DataFrame(1.0, index=index, columns=["AAA"]), settings.dates, tmp_path)
    write_fingerprint(settings, universe, tmp_path)

    loaded = load_prices(settings, universe, data_dir=tmp_path)
    assert list(loaded.columns) == ["AAA"]


def test_a_fingerprint_whose_body_was_edited_is_refused(settings, universe, tmp_path):
    # R3-SEC-09: the digest matches the real settings, but the stored body names other tickers.
    index = pd.bdate_range(settings.dates.start, settings.dates.validation_end)
    write_cache(pd.DataFrame(1.0, index=index, columns=["AAA"]), settings.dates, tmp_path)
    write_fingerprint(settings, universe, tmp_path)
    path = tmp_path / FINGERPRINT_FILE
    body = json.loads(path.read_text(encoding="utf-8"))
    body["fingerprint"]["tickers"] = ["WRONG"]
    path.write_text(json.dumps(body), encoding="utf-8")

    with pytest.raises(CacheMismatch):
        load_prices(settings, universe, data_dir=tmp_path)


def test_a_cache_with_rows_after_validation_end_is_refused_on_read(settings, universe, tmp_path):
    # R3-SEC-09: the cut is enforced on read. A hand-made file with a valid fingerprint still fails.
    index = pd.bdate_range(settings.dates.start, settings.dates.holdout_end)
    pd.DataFrame(1.0, index=index, columns=["AAA"]).to_parquet(tmp_path / PRE_HOLDOUT_FILE)
    write_fingerprint(settings, universe, tmp_path)

    with pytest.raises(CacheMismatch, match="after validation_end"):
        read_checked_pre_holdout(settings, universe, tmp_path)


def test_a_failed_refresh_leaves_no_fingerprint_for_the_old_cache(settings, universe, tmp_path, monkeypatch):
    # R3-STAT-07: the old fingerprint is removed before the parquet is replaced, so a crash in
    # between cannot leave old settings vouching for new data.
    index = pd.bdate_range(settings.dates.start, settings.dates.validation_end)
    write_cache(pd.DataFrame(1.0, index=index, columns=["AAA"]), settings.dates, tmp_path)
    write_fingerprint(settings, universe, tmp_path)

    class Source:
        def fetch_adjusted_close(self, tickers, start, end_exclusive):
            return pd.DataFrame(1.0, index=index, columns=["AAA"])

    def crash(*args, **kwargs):
        raise OSError("disk full while replacing the cache")

    monkeypatch.setattr(loader, "write_cache", crash)
    with pytest.raises(OSError, match="disk full"):
        fetch_and_cache(settings, universe, Source(), tmp_path)
    assert not (tmp_path / FINGERPRINT_FILE).exists()


def test_load_prices_does_not_fetch_when_the_cache_is_missing(settings, universe, tmp_path, monkeypatch):
    # R3-WHOLE-17: a missing cache is an error with a clear next step, not a silent download.
    def must_not_fetch(*args, **kwargs):
        raise AssertionError("load_prices fetched prices without being asked to refresh")

    monkeypatch.setattr(loader, "fetch_and_cache", must_not_fetch)
    with pytest.raises(CacheMismatch, match="lab data fetch"):
        load_prices(settings, universe, data_dir=tmp_path)


# ---- R5-F3: the fingerprint versions the cleaning rule (R4-SEC-05) -------------------------------


def _write_fingerprint_body(data_dir, fingerprint: dict) -> None:
    """Write a fingerprint file whose digest is correct for the body given (a valid, old-format file)."""
    body = {"sha256": loader._digest(fingerprint), "fingerprint": fingerprint}
    (data_dir / FINGERPRINT_FILE).write_text(json.dumps(body), encoding="utf-8")


def test_the_fingerprint_records_the_cleaning_rule(settings, universe):
    assert loader.cache_fingerprint(settings, universe)["cleaning_rule"] == loader.CLEANING_RULE


def test_a_cache_built_under_the_old_universe_rule_is_refused(settings, universe, tmp_path):
    # R4-SEC-05: HEAD fingerprinted the same tickers, dates and data settings with no cleaning rule.
    # Its digest and body match what the new code computes without the field, so the check must
    # refuse it on the missing rule, not accept it.
    index = pd.bdate_range(settings.dates.start, settings.dates.validation_end)
    write_cache(pd.DataFrame(1.0, index=index, columns=["AAA"]), settings.dates, tmp_path)
    old = loader.cache_fingerprint(settings, universe)
    old.pop("cleaning_rule", None)
    _write_fingerprint_body(tmp_path, old)

    with pytest.raises(CacheMismatch):
        load_prices(settings, universe, data_dir=tmp_path)


def test_a_cache_with_an_older_cleaning_rule_value_is_refused(settings, universe, tmp_path):
    index = pd.bdate_range(settings.dates.start, settings.dates.validation_end)
    write_cache(pd.DataFrame(1.0, index=index, columns=["AAA"]), settings.dates, tmp_path)
    older = loader.cache_fingerprint(settings, universe)
    older["cleaning_rule"] = loader.CLEANING_RULE - 1
    _write_fingerprint_body(tmp_path, older)

    with pytest.raises(CacheMismatch):
        load_prices(settings, universe, data_dir=tmp_path)
