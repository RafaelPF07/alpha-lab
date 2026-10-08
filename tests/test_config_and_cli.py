from datetime import date

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from alpha_lab.cli import app
from alpha_lab.config import Dates, load_settings, load_universe


def test_real_settings_load_and_pricing_is_verified():
    settings = load_settings()
    assert settings.pricing.verified_on == date(2026, 10, 8)
    assert settings.pricing.models["claude-haiku-5-5"].input_over_100k == 0.50


def test_dates_out_of_order_fail_loudly():
    with pytest.raises(ValidationError, match="start < train_end"):
        Dates(
            start=date(2012, 1, 1),
            train_end=date(2011, 1, 1),
            validation_end=date(2021, 12, 31),
            holdout_end=date(2025, 12, 31),
        )


def test_universe_is_the_approved_fifty_unique_tickers():
    universe = load_universe()
    assert len(universe.sector_etfs) == 9
    assert len(universe.stocks) == 41
    assert len(set(universe.tickers)) == 50


def test_cli_refuses_holdout_split():
    result = CliRunner().invoke(app, ["backtest", "momentum_12_1", "--split", "holdout"])
    assert result.exit_code == 2
    assert "holdout" in result.output


def test_cli_rejects_unknown_signal():
    result = CliRunner().invoke(app, ["backtest", "not_a_signal"])
    assert result.exit_code == 2
