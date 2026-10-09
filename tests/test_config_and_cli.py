import os
from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from alpha_lab import cli
from alpha_lab.cli import app
from alpha_lab.config import Dates, load_env_file, load_settings, load_universe
from alpha_lab.data.loader import write_cache, write_fingerprint
from alpha_lab.ledger.db import Ledger


def test_settings_file_loads_with_the_recorded_pricing_date_and_haiku_surcharge():
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


def test_universe_has_nine_etfs_41_stocks_and_no_duplicates():
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


def test_network_access_is_blocked_in_tests():
    import socket

    with pytest.raises(RuntimeError, match="network"):
        socket.create_connection(("example.com", 443), timeout=1)


# ---- .env parsing (SEC-17). Fake keys only; the real .env is never read here. -----------------

ENV_KEYS = ("ALPHA_TEST_BOM", "ALPHA_TEST_EXPORT", "ALPHA_TEST_QUOTED", "ALPHA_TEST_COMMENT")


def _clear_env(monkeypatch, keys):
    # setenv then delenv records the original state, so monkeypatch restores it afterwards.
    for key in keys:
        monkeypatch.setenv(key, "x")
        monkeypatch.delenv(key)


def test_env_file_accepts_bom_export_quotes_and_inline_comments(tmp_path, monkeypatch):
    _clear_env(monkeypatch, ENV_KEYS)
    env = tmp_path / ".env"
    env.write_bytes(
        (
            "﻿ALPHA_TEST_BOM=one\n"
            "export ALPHA_TEST_EXPORT=two\n"
            "ALPHA_TEST_QUOTED='three # not a comment'\n"
            "ALPHA_TEST_COMMENT=four  # trailing comment\n"
        ).encode()
    )
    load_env_file(env)
    assert os.environ["ALPHA_TEST_BOM"] == "one"
    assert os.environ["ALPHA_TEST_EXPORT"] == "two"
    assert os.environ["ALPHA_TEST_QUOTED"] == "three # not a comment"
    assert os.environ["ALPHA_TEST_COMMENT"] == "four"


def test_env_file_existing_variables_win(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPHA_TEST_WIN", "kept")
    env = tmp_path / ".env"
    env.write_text("ALPHA_TEST_WIN=other\n", encoding="utf-8")
    load_env_file(env)
    assert os.environ["ALPHA_TEST_WIN"] == "kept"


@pytest.mark.parametrize(
    "line",
    [
        "ALPHA_TEST_NO_EQUALS_SIGN_HERE",
        "ALPHA_TEST_Q='unterminated",
        "ALPHA_TEST_Q='closed' trailing",
        "ALPHA_TEST_Q=abc\"def",
        "1BAD_KEY=value",
    ],
)
def test_env_file_rejects_a_bad_line_and_never_echoes_it(tmp_path, line):
    env = tmp_path / ".env"
    env.write_text(f"# header\n{line}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2") as info:
        load_env_file(env)
    assert "ALPHA_TEST" not in str(info.value)


# ---- CLI: ledger show, report, holdout (QA-06, SEC-11, SEC-13, SEC-14, SEC-19) ----------------


@pytest.fixture
def temp_ledger(tmp_path, monkeypatch):
    """A ledger under tmp_path, with one run, one idea and one attempt. The CLI is pointed at it."""
    path = tmp_path / "ledger.db"
    monkeypatch.setattr(cli, "LEDGER_PATH", path)
    ledger = Ledger(path)
    ledger.create_run("run1", "{}", 1.0)
    ledger.add_idea(
        idea_id="idea1",
        run_id="run1",
        pipeline_id="p1",
        name="Momentum test",
        hypothesis="a hypothesis",
        description="a description",
        expected_holding_period_days=1,
    )
    ledger.add_attempt(
        attempt_id="att1",
        idea_id="idea1",
        run_id="run1",
        attempt_number=1,
        code="def signal(prices):\n    return prices\n\x1b[2Jhidden",
        code_hash="h1",
        status="validation_pass",
    )
    ledger.close()
    return path


def test_ledger_show_run_prints_the_run_and_strips_control_characters(temp_ledger):
    result = CliRunner().invoke(app, ["ledger", "show", "--run", "run1", "--code"])
    assert result.exit_code == 0, result.output
    assert "Run run1: running" in result.output
    assert "Idea: Momentum test" in result.output
    assert "attempt 1: validation_pass" in result.output
    assert "\x1b" not in result.output  # agent code cannot send terminal escape sequences


def test_ledger_show_unknown_run_exits_1(temp_ledger):
    result = CliRunner().invoke(app, ["ledger", "show", "--run", "missing"])
    assert result.exit_code == 1


def test_report_run_writes_into_the_redirected_runs_directory(temp_ledger, tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    monkeypatch.setattr(cli, "RUNS_DIR", runs_dir)
    result = CliRunner().invoke(app, ["report", "--run", "run1"])
    assert result.exit_code == 0, result.output
    assert (runs_dir / "run1" / "report.md").exists()


def test_holdout_refuses_when_stdin_is_not_a_terminal(temp_ledger):
    result = CliRunner().invoke(app, ["holdout", "--run", "run1", "--i-understand-this-is-final"])
    assert result.exit_code == 2
    assert "interactive terminal" in result.output


def _fake_report(attempt_ids):
    rows = [
        SimpleNamespace(
            attempt_id=attempt_id,
            name=attempt_id,
            passed=True,
            train_sharpe=0.1,
            validation_sharpe=0.2,
            deflated_sharpe=0.3,
        )
        for attempt_id in attempt_ids
    ]
    return SimpleNamespace(candidates=rows)


@pytest.fixture
def holdout_cli(temp_ledger, tmp_path, monkeypatch):
    """The holdout command with a terminal, a fake validated list and a temporary cache."""
    from alpha_lab.ledger import report

    monkeypatch.setattr(cli, "_stdin_is_tty", lambda: True)
    monkeypatch.setattr(report, "build_report_data", lambda ledger, run: _fake_report(["att1"]))
    data_dir = tmp_path / "data"
    settings, universe = load_settings(), load_universe()
    index = pd.bdate_range(settings.dates.start, settings.dates.validation_end)
    write_cache(pd.DataFrame(1.0, index=index, columns=["AAA"]), settings.dates, data_dir)
    write_fingerprint(settings, universe, data_dir)
    monkeypatch.setattr(cli, "DATA_DIR", data_dir)
    return temp_ledger


def test_holdout_with_a_wrong_typed_code_records_nothing(holdout_cli):
    result = CliRunner().invoke(
        app,
        ["holdout", "--run", "run1", "--i-understand-this-is-final"],
        input="WRONG\n",
    )
    assert result.exit_code == 1, result.output
    ledger = Ledger(holdout_cli)
    assert ledger.holdouts() == []
    ledger.close()


def test_holdout_attempt_selector_rejects_unknown_attempts(holdout_cli):
    result = CliRunner().invoke(
        app,
        ["holdout", "--run", "run1", "--i-understand-this-is-final", "--attempt", "nope"],
    )
    assert result.exit_code == 1
    assert "nope" in result.output
