import asyncio
import json

import numpy as np
import pandas as pd
import pytest
from fakes import RoutingAnthropic, tool_response, with_test_cache_prices

from alpha_lab.checks.sandbox import SandboxOutcome
from alpha_lab.config import GatesConfig, load_settings
from alpha_lab.events.bus import EventBus
from alpha_lab.ledger.db import Ledger
from alpha_lab.pipeline import validation as validation_module
from alpha_lab.pipeline.run import execute_run
from alpha_lab.pipeline.validation import validate_signal

GOOD = """
import pandas as pd


def signal(prices: pd.DataFrame) -> pd.DataFrame:
    score = prices.shift(21) / prices.shift(252) - 1.0
    ranks = score.rank(axis=1, pct=True)
    long = (ranks > 0.8).astype(float)
    short = (ranks <= 0.2).astype(float)
    n_long = long.sum(axis=1).replace(0.0, float("nan"))
    n_short = short.sum(axis=1).replace(0.0, float("nan"))
    return (long.div(n_long, axis=0) - short.div(n_short, axis=0)).fillna(0.0)
"""
LOW = {"concerns": [], "suspicion_level": "low", "suggested_change": "none"}
IDEA = {
    "name": "Twelve month winners",
    "hypothesis": "Stocks that rose over the past year keep rising for a while.",
    "description": "Rank by return from 252 to 21 days ago. Long the top fifth, short the bottom fifth.",
    "expected_holding_period_days": 21,
}


def relaxed_settings():
    settings = with_test_cache_prices(load_settings())
    return settings.model_copy(
        update={
            "gates": GatesConfig(
                min_train_sharpe=-100.0, min_validation_sharpe=0.3, max_turnover_daily=10.0
            )
        }
    )


def random_prices(end: str) -> pd.DataFrame:
    index = pd.bdate_range("2010-01-01", end)
    rng = np.random.default_rng(17)
    return pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, size=(len(index), 10)), axis=0)),
        index=index,
        columns=[f"T{i}" for i in range(10)],
    )


def test_validation_metrics_never_reach_any_agent(tmp_path):
    settings = with_test_cache_prices(load_settings())
    settings = settings.model_copy(
        update={
            "gates": GatesConfig(
                min_train_sharpe=-100.0, min_validation_sharpe=0.3, max_turnover_daily=10.0
            )
        }
    )
    index = pd.bdate_range("2010-01-01", "2021-12-31")
    rng = np.random.default_rng(17)
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, size=(len(index), 10)), axis=0)),
        index=index,
        columns=[f"T{i}" for i in range(10)],
    )
    ledger = Ledger(tmp_path / "l.db")
    client = RoutingAnthropic(
        {
            "submit_idea": [tool_response("submit_idea", IDEA)],
            "submit_code": [tool_response("submit_code", {"code": GOOD})],
            "submit_critique": [tool_response("submit_critique", LOW)],
        }
    )
    asyncio.run(
        execute_run(
            run_id="leak",
            settings=settings,
            prices=prices,
            n_pipelines=1,
            budget_usd=5.0,
            ledger=ledger,
            bus=EventBus(ledger),
            client=client,
        )
    )

    rows = ledger.validations()
    assert rows, "the signal should have reached validation"
    metrics = json.loads(rows[0]["metrics_json"])
    secrets = {
        repr(metrics["sharpe"]),
        f"{metrics['sharpe']:.3f}",
        repr(metrics["ann_return"]),
        repr(metrics["max_drawdown"]),
        repr(metrics["avg_daily_turnover"]),
    }

    sent = json.dumps(client.calls, default=str)
    for secret in secrets:
        assert secret not in sent, f"validation value {secret} leaked into an agent request"

    # The event stream only carries the boolean.
    for row in ledger.events_for_run("leak"):
        if row["type"] == "validation_result":
            assert json.loads(row["payload_json"]).keys() == {"passed"}


def test_a_window_that_cannot_be_scored_is_failed_backtest_not_validation_fail(tmp_path):
    # The prices stop in mid 2019, inside the validation window, so validation cannot be scored.
    ledger = Ledger(tmp_path / "l.db")
    client = RoutingAnthropic(
        {
            "submit_idea": [tool_response("submit_idea", IDEA)],
            "submit_code": [tool_response("submit_code", {"code": GOOD})],
            "submit_critique": [tool_response("submit_critique", LOW)],
        }
    )
    summary = asyncio.run(
        execute_run(
            run_id="gap",
            settings=relaxed_settings(),
            prices=random_prices("2019-06-28"),
            n_pipelines=1,
            budget_usd=5.0,
            ledger=ledger,
            bus=EventBus(ledger),
            client=client,
        )
    )

    assert summary.results[0].status == "failed_backtest"
    attempt = ledger.attempts_for_run("gap")[0]
    assert attempt["status"] == "failed_backtest"
    assert "validation could not be scored" in attempt["error"]
    assert ledger.validations() == []  # no pass or fail was recorded
    types = [row["type"] for row in ledger.events_for_run("gap")]
    assert "validation_result" not in types


def test_validate_signal_reports_an_unscorable_window_as_a_data_error():
    settings = relaxed_settings()
    outcome = validate_signal(GOOD, random_prices("2019-06-28"), settings, 21)
    assert outcome.passed is False
    assert outcome.no_verdict_status == "failed_backtest"
    assert "could not be scored" in outcome.error


def _sandbox_gives(outcome):
    def run(sandboxed, prices):
        return outcome

    return run


@pytest.mark.parametrize("status", ["timeout", "exception"])
def test_a_validation_sandbox_that_did_not_finish_gives_no_verdict(monkeypatch, status):
    # A timeout or a crashed child says nothing about the signal (R4-AGT-08, R4-WHOLE-03).
    outcome = SandboxOutcome(status=status, message="injected")
    monkeypatch.setattr(validation_module, "run_signal", _sandbox_gives(outcome))
    verdict = validate_signal(GOOD, random_prices("2021-12-31"), relaxed_settings(), 21)
    assert verdict.passed is False
    assert verdict.no_verdict_status == "failed_sandbox"
    assert verdict.metrics is None
    assert "injected" in verdict.error


def test_a_flat_validation_window_gives_no_verdict_as_training_does(monkeypatch):
    # Zero weights on every day: the Sharpe is not a number, so the gate cannot be applied. This is
    # recorded as failed_gate, the same as the training window, and not as validation_fail.
    prices = random_prices("2021-12-31")
    zeros = prices * 0.0
    monkeypatch.setattr(validation_module, "run_signal", _sandbox_gives(SandboxOutcome("ok", zeros)))
    verdict = validate_signal(GOOD, prices, relaxed_settings(), 1)
    assert verdict.passed is False
    assert verdict.no_verdict_status == "failed_gate"
    assert "not a finite number" in verdict.error


def test_a_harness_bug_is_not_recorded_as_a_validation_verdict(monkeypatch):
    def broken_backtest(*args, **kwargs):
        raise ValueError("injected harness bug")

    monkeypatch.setattr(validation_module, "run_backtest", broken_backtest)
    with pytest.raises(ValueError, match="injected harness bug"):
        validate_signal(GOOD, random_prices("2021-12-31"), relaxed_settings(), 21)


def test_a_harness_bug_in_validation_fails_the_pipeline_not_the_gate(tmp_path, monkeypatch):
    def broken_backtest(*args, **kwargs):
        raise ValueError("injected harness bug")

    monkeypatch.setattr(validation_module, "run_backtest", broken_backtest)
    ledger = Ledger(tmp_path / "l.db")
    client = RoutingAnthropic(
        {
            "submit_idea": [tool_response("submit_idea", IDEA)],
            "submit_code": [tool_response("submit_code", {"code": GOOD})],
            "submit_critique": [tool_response("submit_critique", LOW)],
        }
    )
    summary = asyncio.run(
        execute_run(
            run_id="bug",
            settings=relaxed_settings(),
            prices=random_prices("2021-12-31"),
            n_pipelines=1,
            budget_usd=5.0,
            ledger=ledger,
            bus=EventBus(ledger),
            client=client,
        )
    )

    assert summary.results[0].status == "failed_error", summary.results[0]
    assert ledger.validations() == []  # no pass or fail was recorded
    attempt = ledger.attempts_for_run("bug")[0]
    assert attempt["status"] == "aborted_error"
    assert "injected harness bug" in attempt["error"]


def test_an_error_escaping_the_sandbox_is_a_harness_failure_not_a_verdict(monkeypatch):
    from alpha_lab.checks.sandbox import SandboxedSignal, SandboxFailure

    def escaping_run(self, prices):
        raise PermissionError("a Windows cleanup error escaped run()")

    monkeypatch.setattr(SandboxedSignal, "run", escaping_run)
    with pytest.raises(SandboxFailure) as failure:
        validate_signal(GOOD, random_prices("2021-12-31"), relaxed_settings(), 21)
    assert failure.value.outcome.status == "exception"
    assert failure.value.outcome.error_type == "SandboxHarnessError"
    assert "PermissionError" in failure.value.outcome.message


ALTERNATING = """
import numpy as np
import pandas as pd


def signal(prices: pd.DataFrame) -> pd.DataFrame:
    sign = np.where(np.arange(len(prices.index)) % 2 == 0, 1.0, -1.0)
    weights = np.repeat(sign[:, None], prices.shape[1], axis=1) / prices.shape[1]
    return pd.DataFrame(weights, index=prices.index, columns=prices.columns)
"""


def turnover_settings(ceiling: float):
    settings = with_test_cache_prices(load_settings())
    return settings.model_copy(
        update={
            "gates": GatesConfig(
                min_train_sharpe=-100.0, min_validation_sharpe=-100.0, max_turnover_daily=ceiling
            )
        }
    )


def test_validation_turnover_above_the_ceiling_fails_the_gate():
    # Only the turnover ceiling can fail here, because the Sharpe floor is -100.
    prices = random_prices("2021-12-31")
    too_tight = validate_signal(ALTERNATING, prices, turnover_settings(0.5), 1)
    assert too_tight.metrics is not None
    assert too_tight.metrics["avg_daily_turnover"] > 0.5
    assert too_tight.passed is False

    loose = validate_signal(ALTERNATING, prices, turnover_settings(10.0), 1)
    assert loose.passed is True
