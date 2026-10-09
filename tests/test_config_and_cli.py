import json
import os
import subprocess
import sys
from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from alpha_lab import cli
from alpha_lab.cli import app
from alpha_lab.config import (
    PROJECT_ROOT,
    BacktestConfig,
    Dates,
    GatesConfig,
    Settings,
    load_env_file,
    load_settings,
    load_universe,
)
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
        code="def signal(prices):\n    return prices  # \x1b[2Jhidden\n",
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


def _fake_report(attempt_ids, dsr_reason="", deflated=0.3, n_train_trials=10):
    rows = [
        SimpleNamespace(
            attempt_id=attempt_id,
            name=attempt_id,
            passed=True,
            train_sharpe=0.1,
            validation_sharpe=0.2,
            deflated_sharpe=deflated,
        )
        for attempt_id in attempt_ids
    ]
    return SimpleNamespace(
        candidates=rows,
        dsr_reason=dsr_reason,
        n_train_trials=n_train_trials,
        n_trials=n_train_trials,
    )


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
    # The attempt's code must pass the static filter. Otherwise the holdout refuses before the
    # confirmation, and these tests would no longer reach the prompt they are about (R5-F3).
    ledger = Ledger(temp_ledger)
    ledger.conn.execute(
        "UPDATE attempts SET code = ? WHERE attempt_id = 'att1'",
        ("import pandas as pd\n\ndef signal(prices):\n    return prices * 0.0 + 1.0 / 3\n",),
    )
    ledger.conn.commit()
    ledger.close()
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


# ---- R4: config bounds, windows, the .env parser and the CLI (R3-STAT-06, STAT-09, SEC-06, WHOLE-01/03)


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_gate_thresholds_reject_nan_and_infinity(value):
    with pytest.raises(ValidationError):
        GatesConfig(min_train_sharpe=value, min_validation_sharpe=0.3, max_turnover_daily=0.5)


def test_an_infinite_cost_is_refused():
    with pytest.raises(ValidationError):
        BacktestConfig(
            cost_bps_per_side=float("inf"),
            max_gross_leverage=1.0,
            annualization_days=252,
            min_history_days=0,
        )


def test_a_training_window_too_short_for_its_history_is_refused_at_startup():
    data = load_settings().model_dump()
    data["dates"] = {
        "start": "2010-01-01",
        "train_end": "2010-03-31",
        "validation_end": "2021-12-31",
        "holdout_end": "2025-12-31",
    }
    with pytest.raises(ValidationError, match="training window"):
        Settings.model_validate(data)


def test_env_file_that_is_not_utf8_raises_a_named_value_error(tmp_path):
    env = tmp_path / ".env"
    env.write_bytes("ALPHA_TEST_LATIN=caf\xe9\n".encode("latin-1"))
    with pytest.raises(ValueError, match="not UTF-8"):
        load_env_file(env)


def test_env_file_refuses_a_key_set_twice(tmp_path, monkeypatch):
    _clear_env(monkeypatch, ("ALPHA_TEST_DUP",))
    env = tmp_path / ".env"
    env.write_text("ALPHA_TEST_DUP=first\nALPHA_TEST_DUP=second\n", encoding="utf-8")
    with pytest.raises(ValueError, match="set twice"):
        load_env_file(env)
    assert "ALPHA_TEST_DUP" not in os.environ


def test_stdin_from_the_null_device_is_not_a_terminal():
    # R3-SEC-06. On Windows the null device reports isatty() True, so the check must look further.
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from alpha_lab import cli; print(cli._stdin_is_tty())"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code, str(PROJECT_ROOT / "src")],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert proc.stdout.strip() == "False", proc.stderr


def test_ledger_show_prints_n_a_for_a_null_metric_and_does_not_crash(tmp_path, monkeypatch):
    # R3-WHOLE-03: a flat scoring window stores a null Sharpe. The listing must still print.
    path = tmp_path / "ledger.db"
    monkeypatch.setattr(cli, "LEDGER_PATH", path)
    ledger = Ledger(path)
    ledger.create_run("run1", "{}", 1.0)
    ledger.add_idea(
        idea_id="idea1",
        run_id="run1",
        pipeline_id="p1",
        name="Flat",
        hypothesis="h",
        description="d",
        expected_holding_period_days=1,
    )
    flat = {"sharpe": None, "ann_return": 0.0, "max_drawdown": 0.0, "avg_daily_turnover": 0.0}
    ledger.add_attempt(
        attempt_id="att1",
        idea_id="idea1",
        run_id="run1",
        attempt_number=1,
        code="def signal(prices):\n    return prices\n",
        code_hash="h1",
        status="failed_gate",
        train_metrics_json=json.dumps(flat),
    )
    ledger.close()
    result = CliRunner().invoke(app, ["ledger", "show", "--run", "run1"])
    assert result.exit_code == 0, result.output
    assert "Sharpe n/a" in result.output
    assert "ann. return +0.00%" in result.output


def test_dashboard_passes_a_graceful_shutdown_timeout_to_uvicorn(tmp_path, monkeypatch):
    # R3-BE-02: an open event stream must not hold the server open after Ctrl+C.
    import uvicorn

    from alpha_lab.dashboard import server

    # Point at a ledger in tmp_path, not the owner's runs/ledger.db (the dashboard now checks its schema).
    monkeypatch.setattr(cli, "LEDGER_PATH", tmp_path / "ledger.db")
    seen = {}
    monkeypatch.setattr(server, "create_app", lambda **kwargs: object())
    monkeypatch.setattr(uvicorn, "run", lambda application, **kwargs: seen.update(kwargs))
    result = CliRunner().invoke(app, ["dashboard"])
    assert result.exit_code == 0, result.output
    assert seen["timeout_graceful_shutdown"] == cli.GRACEFUL_SHUTDOWN_SECONDS


@pytest.fixture
def run_cli(tmp_path, monkeypatch):
    """`lab run` with prices, the swarm and the ledger replaced. No API call and no .env read."""
    from alpha_lab.pipeline.run import RunSummary
    from alpha_lab.pipeline.single import PipelineResult

    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-tests")
    monkeypatch.setattr(cli, "load_env_file", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "LEDGER_PATH", tmp_path / "ledger.db")
    index = pd.bdate_range("2010-01-01", periods=5)
    monkeypatch.setattr(cli, "load_prices", lambda settings, universe: pd.DataFrame(1.0, index=index, columns=["A"]))

    async def fake_execute_run(**kwargs):
        results = [
            PipelineResult("p1", "validation_fail", None, "t", 1),
            PipelineResult("p2", "validation_pass", None, "t", 1),
        ]
        return RunSummary(
            run_id=kwargs["run_id"],
            status="completed",
            budget_usd=1.0,
            total_cost_usd=0.0,
            results=results,
        )

    subscriptions: list[int] = []
    unsubscriptions: list[int] = []

    class CountingBus(cli.EventBus):
        def subscribe(self):
            subscriptions.append(1)
            return super().subscribe()

        def unsubscribe(self, queue):
            unsubscriptions.append(1)
            return super().unsubscribe(queue)

    monkeypatch.setattr(cli, "execute_run", fake_execute_run)
    monkeypatch.setattr(cli, "EventBus", CountingBus)
    return SimpleNamespace(subscriptions=subscriptions, unsubscriptions=unsubscriptions)


def test_run_summary_prints_passed_critic_and_passed_validation_separately(run_cli):
    # R3-WHOLE-01 and DECISION-13: one validation_fail and one validation_pass.
    result = CliRunner().invoke(app, ["run", "--pipelines", "2"])
    assert result.exit_code == 0, result.output
    assert "passed critic: 2" in result.output
    assert "passed validation: 1" in result.output
    assert "critic passes" not in result.output


def test_run_subscribes_to_the_event_stream_only_with_verbose(run_cli):
    # R3-AGT-11: an undrained queue would grow for the whole run.
    CliRunner().invoke(app, ["run", "--pipelines", "1"])
    assert run_cli.subscriptions == []
    CliRunner().invoke(app, ["run", "--pipelines", "1", "--verbose"])
    assert run_cli.subscriptions == [1]


def test_a_bad_env_file_is_reported_by_run_with_exit_2(run_cli, monkeypatch):
    def bad_env(*args, **kwargs):
        raise ValueError(".env line 3: expected KEY=VALUE")

    monkeypatch.setattr(cli, "load_env_file", bad_env)
    result = CliRunner().invoke(app, ["run", "--pipelines", "1"])
    assert result.exit_code == 2
    assert "expected KEY=VALUE" in result.output


def test_holdout_top_refuses_when_the_deflated_sharpe_is_undefined(holdout_cli, monkeypatch):
    # R4-F1 request: --top ranks by the deflated Sharpe, so it refuses when that is not defined.
    from alpha_lab.ledger import report

    reason = "fewer than two trials have a finite train Sharpe (1 found)"
    monkeypatch.setattr(
        report, "build_report_data", lambda ledger, run: _fake_report(["att1"], dsr_reason=reason)
    )
    result = CliRunner().invoke(app, ["holdout", "--run", "run1", "--i-understand-this-is-final"])
    assert result.exit_code == 1
    assert "Refusing --top" in result.output
    ledger = Ledger(holdout_cli)
    assert ledger.holdouts() == []  # nothing was claimed
    ledger.close()


def test_holdout_attempt_selection_is_still_allowed_when_the_deflated_sharpe_is_undefined(
    holdout_cli, monkeypatch
):
    from alpha_lab.ledger import report

    reason = "fewer than two trials have a finite train Sharpe (1 found)"
    monkeypatch.setattr(
        report, "build_report_data", lambda ledger, run: _fake_report(["att1"], dsr_reason=reason)
    )
    result = CliRunner().invoke(
        app,
        ["holdout", "--run", "run1", "--i-understand-this-is-final", "--attempt", "att1"],
        input="WRONG\n",
    )
    assert "About to run the holdout" in result.output  # got past the refusal to the confirmation


def test_holdout_prints_n_a_for_a_nan_deflated_sharpe(holdout_cli, monkeypatch):
    # R4-F1 request: a NaN deflated Sharpe prints as n/a, never as "nan".
    from alpha_lab.ledger import report
    from alpha_lab.pipeline import holdout

    monkeypatch.setattr(
        report, "build_report_data", lambda ledger, run: _fake_report(["att1"], deflated=float("nan"))
    )
    monkeypatch.setattr(
        holdout,
        "run_holdout",
        lambda **kwargs: [holdout.HoldoutResult("att1", "att1", "skipped", None, "already run")],
    )
    result = CliRunner().invoke(
        app, ["holdout", "--run", "run1", "--i-understand-this-is-final", "--attempt", "att1"]
    )
    assert result.exit_code == 0, result.output
    assert "deflated SR n/a" in result.output
    assert "nan" not in result.output.lower()


# ---- R4-F6 items: pricing validators, a permanent guard on the shipped config, dashboard --run ---


def test_an_agent_model_without_a_price_is_refused_at_startup():
    data = load_settings().model_dump()
    data["agents"]["critic_model"] = "claude-unpriced-9"
    with pytest.raises(ValidationError, match="has no price"):
        Settings.model_validate(data)


def test_long_context_prices_must_come_in_pairs():
    data = load_settings().model_dump()
    data["pricing"]["models"]["claude-sonnet-5-5"]["input_over_100k"] = 3.0
    with pytest.raises(ValidationError, match="set together"):
        Settings.model_validate(data)


def test_the_shipped_settings_and_universe_pass_every_guard():
    # A permanent guard: whatever is committed in config/ must pass the rules the code relies on.
    settings = load_settings()
    universe = load_universe()
    agent_models = (
        settings.agents.ideas_model,
        settings.agents.coder_model,
        settings.agents.critic_model,
    )
    for model in agent_models:
        assert model in settings.pricing.models
    assert len(universe.tickers) == len(set(universe.tickers)) == 50
    assert settings.dates.start < settings.dates.train_end < settings.dates.validation_end


@pytest.fixture
def dashboard_ledger(tmp_path, monkeypatch):
    path = tmp_path / "ledger.db"
    monkeypatch.setattr(cli, "LEDGER_PATH", path)
    ledger = Ledger(path)
    ledger.create_run("run1", "{}", 1.0)
    ledger.close()
    return path


def test_dashboard_refuses_a_run_that_does_not_exist(dashboard_ledger, monkeypatch):
    import uvicorn

    from alpha_lab.dashboard import server

    monkeypatch.setattr(server, "create_app", lambda **kwargs: object())
    monkeypatch.setattr(uvicorn, "run", lambda application, **kwargs: None)
    result = CliRunner().invoke(app, ["dashboard", "--run", "missing"])
    assert result.exit_code == 1
    assert "no run with id" in result.output


def test_dashboard_accepts_a_run_that_exists(dashboard_ledger, monkeypatch):
    import uvicorn

    from alpha_lab.dashboard import server

    monkeypatch.setattr(server, "create_app", lambda **kwargs: object())
    monkeypatch.setattr(uvicorn, "run", lambda application, **kwargs: None)
    result = CliRunner().invoke(app, ["dashboard", "--run", "run1"])
    assert result.exit_code == 0, result.output


def test_run_unsubscribes_its_event_queue_when_it_ends(run_cli):
    # R3-AGT-11 (R4-F2 request): a finished run must not leave a queue subscribed to the bus.
    CliRunner().invoke(app, ["run", "--pipelines", "1", "--verbose"])
    assert run_cli.subscriptions == [1]
    assert run_cli.unsubscriptions == [1]


def test_cache_write_and_long_context_price_fields_are_optional_and_non_negative():
    from alpha_lab.config import ModelPricing

    plain = ModelPricing(input=1.0, output=2.0, cache_read=0.1)
    assert plain.cache_write_5m is None and plain.cache_read_over_100k is None
    priced = ModelPricing(input=1.0, output=2.0, cache_read=0.1, cache_write_5m=1.25, cache_write_1h=2.0)
    assert priced.cache_write_5m == 1.25
    with pytest.raises(ValidationError):
        ModelPricing(input=1.0, output=2.0, cache_read=0.1, cache_write_1h=-1.0)


# ---- R5-F3: read commands never create or migrate the ledger (R4-WHOLE-08, R4-AGT-13, R4-SEC-07) -----


def _drop_theme_column(path) -> None:
    """Make a ledger that an older version would have written: ideas has no theme column."""
    import sqlite3

    conn = sqlite3.connect(path)
    conn.execute("ALTER TABLE ideas DROP COLUMN theme")
    conn.commit()
    conn.close()


def _columns(path, table: str) -> set[str]:
    import sqlite3

    conn = sqlite3.connect(path)
    try:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    finally:
        conn.close()


def test_ledger_show_on_a_missing_ledger_creates_no_file(tmp_path, monkeypatch):
    path = tmp_path / "runs" / "ledger.db"
    monkeypatch.setattr(cli, "LEDGER_PATH", path)
    result = CliRunner().invoke(app, ["ledger", "show"])
    assert result.exit_code == 0, result.output
    assert "No runs yet" in result.output
    assert not path.exists()
    assert not path.parent.exists()


def test_report_on_a_missing_ledger_creates_no_file(tmp_path, monkeypatch):
    path = tmp_path / "runs" / "ledger.db"
    monkeypatch.setattr(cli, "LEDGER_PATH", path)
    result = CliRunner().invoke(app, ["report"])
    assert result.exit_code == 1
    assert not path.exists()
    assert not path.parent.exists()


def test_ledger_show_does_not_migrate_an_older_ledger(temp_ledger):
    # R4-WHOLE-08 and R4-AGT-13: a read refuses an old schema and leaves the file as it was.
    _drop_theme_column(temp_ledger)
    result = CliRunner().invoke(app, ["ledger", "show", "--run", "run1"])
    assert result.exit_code == 1, result.output
    assert "ledger migrate" in result.output
    assert "theme" not in _columns(temp_ledger, "ideas")


def test_dashboard_refuses_an_older_ledger_and_does_not_migrate_it(temp_ledger, monkeypatch):
    import uvicorn

    from alpha_lab.dashboard import server

    _drop_theme_column(temp_ledger)
    monkeypatch.setattr(server, "create_app", lambda **kwargs: object())
    monkeypatch.setattr(uvicorn, "run", lambda application, **kwargs: None)
    result = CliRunner().invoke(app, ["dashboard"])
    assert result.exit_code == 1, result.output
    assert "ledger migrate" in result.output
    assert "theme" not in _columns(temp_ledger, "ideas")


def test_ledger_migrate_brings_an_older_ledger_up_to_date(temp_ledger):
    _drop_theme_column(temp_ledger)
    result = CliRunner().invoke(app, ["ledger", "migrate"])
    assert result.exit_code == 0, result.output
    assert "theme" in _columns(temp_ledger, "ideas")
    # After the migration the read commands work again.
    assert CliRunner().invoke(app, ["ledger", "show", "--run", "run1"]).exit_code == 0


def test_the_holdout_invocation_does_not_record_a_console_flag(holdout_cli, monkeypatch):
    # R4-SEC-07: the TTY check refuses a non-terminal, and the recorded field proved nothing more.
    from alpha_lab.pipeline import holdout

    seen: dict = {}

    def fake_run_holdout(**kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr(holdout, "run_holdout", fake_run_holdout)
    result = CliRunner().invoke(
        app, ["holdout", "--run", "run1", "--i-understand-this-is-final"]
    )
    assert result.exit_code == 0, result.output
    assert "stdin_is_tty" not in seen["invocation"]
    assert seen["invocation"]["run_id"] == "run1"


def test_holdout_top_refuses_below_the_minimum_trial_count_and_prints_the_count(holdout_cli, monkeypatch):
    # R4-STAT-09 (R5-F1 request): --top ranks by a trial variance that needs MIN_TRIALS_FOR_VARIANCE trials.
    from alpha_lab.ledger import report

    monkeypatch.setattr(
        report, "build_report_data", lambda ledger, run: _fake_report(["att1"], n_train_trials=3)
    )
    result = CliRunner().invoke(app, ["holdout", "--run", "run1", "--i-understand-this-is-final"])
    assert result.exit_code == 1
    assert "3 trial(s)" in result.output
    assert f"at least {report.MIN_TRIALS_FOR_VARIANCE}" in result.output
    ledger = Ledger(holdout_cli)
    assert ledger.holdouts() == []  # refused before any claim
    ledger.close()


def test_holdout_attempt_selection_still_works_below_the_minimum_trial_count(holdout_cli, monkeypatch):
    from alpha_lab.ledger import report

    monkeypatch.setattr(
        report, "build_report_data", lambda ledger, run: _fake_report(["att1"], n_train_trials=3)
    )
    result = CliRunner().invoke(
        app,
        ["holdout", "--run", "run1", "--i-understand-this-is-final", "--attempt", "att1"],
        input="WRONG\n",
    )
    assert "About to run the holdout" in result.output  # past the refusal, to the confirmation
