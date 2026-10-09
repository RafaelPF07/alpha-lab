import asyncio
import hashlib
import json
import re
import sqlite3
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import anthropic
import httpx2 as httpx
import numpy as np
import pandas as pd
import pytest
from fakes import (
    FakeAnthropic,
    RoutingAnthropic,
    text_response,
    tool_response,
    with_pricing,
    with_test_cache_prices,
)

from alpha_lab.agents.prompts import load_prompt
from alpha_lab.checks.sandbox import SandboxOutcome
from alpha_lab.config import GatesConfig, load_settings
from alpha_lab.events.bus import EventBus
from alpha_lab.ledger.db import Ledger
from alpha_lab.pipeline import run as run_module
from alpha_lab.pipeline import single as single_module
from alpha_lab.pipeline import validation as validation_module
from alpha_lab.pipeline.run import execute_run
from alpha_lab.statuses import (
    ATTEMPT_STATUSES,
    LLM_CALL_STATUSES,
    PIPELINE_STATUSES,
    RUN_FAILED_DATA,
    RUN_STATUSES,
)

IDEA = {
    "name": "Twelve month winners",
    "hypothesis": "Stocks that rose over the past year keep rising for a while.",
    "description": "Rank by return from 252 to 21 days ago. Long the top fifth, short the bottom fifth.",
    "expected_holding_period_days": 21,
}

GOOD_CODE = """
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

# Constant long-only weights. The only turnover is at entry, and the Sharpe follows the market.
EQUAL_WEIGHT_CODE = """
import pandas as pd


def signal(prices: pd.DataFrame) -> pd.DataFrame:
    return prices * 0.0 + 1.0 / prices.shape[1]
"""

# Weights flip sign every day. Held daily, that turns over about 2 per day, far above the limit.
ALTERNATING_CODE = """
import numpy as np
import pandas as pd


def signal(prices: pd.DataFrame) -> pd.DataFrame:
    sign = np.where(np.arange(len(prices.index)) % 2 == 0, 1.0, -1.0)
    weights = np.repeat(sign[:, None], prices.shape[1], axis=1) / prices.shape[1]
    return pd.DataFrame(weights, index=prices.index, columns=prices.columns)
"""

# Weights only in the warm-up rows, which the backtest does not score. The scored window then has
# zero returns and no variance, so the Sharpe ratio is not a number. (An all-zero output is
# rejected by the sandbox as all_zero, so it would never reach the gate.)
ZERO_VARIANCE_CODE = """
import pandas as pd


def signal(prices: pd.DataFrame) -> pd.DataFrame:
    weights = prices * 0.0
    weights.iloc[:100] = 1.0 / prices.shape[1]
    return weights
"""

LOW_CRITIC = {"concerns": [], "suspicion_level": "low", "suggested_change": "none"}

IMPORTS_OS = "import os\n" + GOOD_CODE
RAISES = "def signal(prices):\n    raise ValueError('nope')\n"
REQUEST = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def raising_code(i):
    """Same behaviour every time, but a different token, so it is not a duplicate."""
    return "def signal(prices):" + chr(10) + f"    raise ValueError('nope {i}')" + chr(10)


def distinct_idea(n: int) -> dict:
    """Ideas whose wording is unrelated to each other, so only true copies look alike."""
    digest = hashlib.sha256(f"idea-{n}".encode()).hexdigest()
    return {
        "name": f"signal {digest[:12]}",
        "hypothesis": f"Hypothesis built on {digest[12:40]} and an untested story.",
        "description": f"Rank stocks by {digest[40:64]} over a window of {n} days.",
        "expected_holding_period_days": 21,
    }


def relaxed_settings():
    """Same pipeline, with gates loose enough that the code can pass on random data."""
    settings = with_test_cache_prices(load_settings())
    return settings.model_copy(
        update={
            "gates": GatesConfig(
                min_train_sharpe=-100.0, min_validation_sharpe=-100.0, max_turnover_daily=10.0
            )
        }
    )


def default_gates_settings(max_revisions: int = 0):
    """The shipped settings, gates untouched. Only the revision cap is lowered, so one attempt runs."""
    settings = with_test_cache_prices(load_settings())
    return settings.model_copy(
        update={"agents": settings.agents.model_copy(update={"max_revisions": max_revisions})}
    )


def synthetic_prices() -> pd.DataFrame:
    """Ten tickers from 2010 through 2021, so the data runs past train_end into validation."""
    rng = np.random.default_rng(11)
    index = pd.bdate_range("2010-01-01", "2021-12-31")
    log_returns = rng.normal(0.0003, 0.012, size=(len(index), 10))
    prices = 100.0 * np.exp(np.cumsum(log_returns, axis=0))
    return pd.DataFrame(prices, index=index, columns=[f"T{i}" for i in range(10)])


def drifting_prices(drift_before: float, drift_after: float, split: str) -> pd.DataFrame:
    """Ten tickers, 2010 to 2021, with a daily drift that changes after `split`."""
    rng = np.random.default_rng(23)
    index = pd.bdate_range("2010-01-01", "2021-12-31")
    drift = np.where(index <= pd.Timestamp(split), drift_before, drift_after)
    log_returns = rng.normal(0.0, 0.01, size=(len(index), 10)) + drift[:, None]
    prices = 100.0 * np.exp(np.cumsum(log_returns, axis=0))
    return pd.DataFrame(prices, index=index, columns=[f"T{i}" for i in range(10)])


def strict_loads(text: str):
    """json.loads that refuses NaN and Infinity, as JSON.parse does in a browser."""

    def refuse(constant):
        raise ValueError(f"not strict JSON: {constant}")

    return json.loads(text, parse_constant=refuse)


@pytest.fixture
def ledger(tmp_path):
    db = Ledger(tmp_path / "ledger.db")
    yield db
    db.close()


def run_with(
    responses,
    ledger,
    budget_usd=1.0,
    settings=None,
    *,
    n_pipelines=1,
    prices=None,
    client=None,
):
    client = client if client is not None else FakeAnthropic(responses)
    bus = EventBus(ledger)
    summary = None

    async def scenario():
        nonlocal summary
        summary = await execute_run(
            run_id="test-run",
            settings=settings or relaxed_settings(),
            prices=prices if prices is not None else synthetic_prices(),
            n_pipelines=n_pipelines,
            budget_usd=budget_usd,
            ledger=ledger,
            bus=bus,
            client=client,
        )

    asyncio.run(scenario())
    return summary, client


def event_types(ledger):
    return [row["type"] for row in ledger.events_for_run("test-run")]


def assert_no_events_after_run_finished(ledger):
    types = event_types(ledger)
    assert types[-1] == "run_finished"
    assert types.count("run_finished") == 1


def attempt_detail(ledger):
    """What a failing assertion needs to say why: each attempt's status and error, and the sandbox
    status from the event stream. Used as the assertion message (R3-QA-01)."""
    sandbox = [
        json.loads(r["payload_json"])["status"]
        for r in ledger.events_for_run("test-run")
        if r["type"] == "sandbox_run"
    ]
    attempts = [(a["status"], a["error"]) for a in ledger.attempts_for_run("test-run")]
    return {"attempts": attempts, "sandbox_status": sandbox}


def test_passing_pipeline_runs_end_to_end_and_keeps_validation_out(ledger):
    responses = [
        tool_response("submit_idea", IDEA),
        tool_response("submit_code", {"code": GOOD_CODE}),
        tool_response("submit_critique", LOW_CRITIC),
    ]
    summary, client = run_with(responses, ledger)

    assert summary.status == "completed"
    assert summary.results[0].status == "validation_pass"
    assert summary.results[0].attempts == 1
    assert ledger.count_trials() == 1
    assert summary.total_cost_usd > 0

    types = event_types(ledger)
    for expected in [
        "run_started",
        "pipeline_started",
        "idea_created",
        "code_written",
        "static_check",
        "sandbox_run",
        "lookahead_check",
        "train_backtest",
        "pipeline_finished",
        "run_finished",
    ]:
        assert expected in types

    # Nothing after train_end (2018-12-31) may appear in an event payload.
    later_date = re.compile(r"20(19|2\d)-\d\d-\d\d")
    for row in ledger.events_for_run("test-run"):
        assert not later_date.search(row["payload_json"]), row["type"]

    # Agents see only training data, so nothing after train_end can be in their prompts either.
    for call in client.messages.calls:
        assert not later_date.search(repr(call["messages"]))


def test_static_failure_is_fed_back_to_the_coder(ledger):
    responses = [
        tool_response("submit_idea", IDEA),
        tool_response("submit_code", {"code": IMPORTS_OS}),
        tool_response("submit_code", {"code": GOOD_CODE}),
        tool_response("submit_critique", LOW_CRITIC),
    ]
    summary, client = run_with(responses, ledger)

    assert summary.results[0].status == "validation_pass"
    assert summary.results[0].attempts == 2
    revision_prompt = client.messages.calls[2]["messages"][0]["content"]
    assert (
        "It failed this check:\nStatic check failed:\n- import of 'os' is not allowed (line 1)"
        in revision_prompt
    )
    assert "Your previous code:" in revision_prompt


def test_revisions_are_capped_and_every_attempt_is_a_counted_trial(ledger):
    settings = relaxed_settings()
    max_attempts = settings.agents.max_revisions + 1
    responses = [tool_response("submit_idea", IDEA)] + [
        tool_response("submit_code", {"code": raising_code(i)}) for i in range(max_attempts)
    ]
    summary, client = run_with(responses, ledger, settings=settings)

    assert summary.results[0].status == "failed_sandbox"
    assert summary.results[0].attempts == max_attempts
    assert len(client.messages.calls) == 1 + max_attempts
    assert ledger.count_trials() == max_attempts


def test_budget_stops_the_run_cleanly(ledger):
    summary, client = run_with([], ledger, budget_usd=1e-9)

    assert summary.status == "budget_stopped"
    assert summary.results == []
    assert client.messages.calls == []
    assert ledger.get_run("test-run")["status"] == "budget_stopped"
    finished = [r for r in ledger.events_for_run("test-run") if r["type"] == "pipeline_finished"]
    assert len(finished) == 1
    assert json.loads(finished[0]["payload_json"])["status"] == "aborted_budget"


def test_budget_stop_at_the_critic_leaves_no_open_rows(ledger):
    # The ideas and coder calls fit in $0.01. The critic's worst-case reservation (about $0.04) does not.
    responses = [tool_response("submit_idea", IDEA), tool_response("submit_code", {"code": GOOD_CODE})]
    summary, _ = run_with(responses, ledger, budget_usd=0.01)

    assert summary.status == "budget_stopped"
    attempts = ledger.attempts_for_run("test-run")
    assert [a["status"] for a in attempts] == ["aborted_budget"]
    row = attempts[0]
    assert row["static_ok"] == 1 and row["sandbox_ok"] == 1 and row["lookahead_ok"] == 1
    assert row["train_metrics_json"] is not None  # what the attempt measured is kept
    assert "next call estimated" in row["error"]
    finished = [r for r in ledger.events_for_run("test-run") if r["type"] == "pipeline_finished"]
    assert json.loads(finished[0]["payload_json"])["status"] == "aborted_budget"
    assert ledger.count_trials() == 1


def test_critic_output_that_never_parses_ends_the_pipeline_and_the_run_completes(ledger):
    responses = [
        tool_response("submit_idea", IDEA),
        tool_response("submit_code", {"code": GOOD_CODE}),
        text_response(),  # the critic ignores the tool
        text_response(),  # and again after the repair message
    ]
    summary, _ = run_with(responses, ledger)

    assert summary.status == "completed"
    assert summary.results[0].status == "failed_critic_output", attempt_detail(ledger)
    statuses = [a["status"] for a in ledger.attempts_for_run("test-run")]
    assert statuses == ["failed_critic_output"]  # no row left 'running'
    assert_no_events_after_run_finished(ledger)
    # The ledger total is the sum of the llm_calls rows, which includes both critic calls.
    assert ledger.llm_cost_total("test-run") == pytest.approx(summary.total_cost_usd)


def test_an_unexpected_error_fails_one_pipeline_and_the_run_goes_on(ledger):
    client = RoutingAnthropic(
        {
            "submit_idea": [RuntimeError("boom"), tool_response("submit_idea", distinct_idea(2))],
            "submit_code": [tool_response("submit_code", {"code": GOOD_CODE})],
            "submit_critique": [tool_response("submit_critique", LOW_CRITIC)],
        }
    )
    summary, _ = run_with([], ledger, n_pipelines=2, client=client, settings=relaxed_settings())

    assert summary.status == "completed"
    statuses = sorted(r.status for r in summary.results)
    assert statuses[0] == "failed_error", (statuses, attempt_detail(ledger))
    assert statuses[1] in ("validation_pass", "validation_fail")
    failed = [r for r in ledger.events_for_run("test-run") if r["type"] == "pipeline_finished"]
    errors = [json.loads(r["payload_json"])["error"] for r in failed]
    assert any("RuntimeError: boom" in (e or "") for e in errors)
    assert_no_events_after_run_finished(ledger)


def test_a_bad_request_fails_its_pipeline_not_the_run(ledger):
    error = anthropic.BadRequestError(
        "credit balance too low", response=httpx.Response(400, request=REQUEST), body=None
    )
    summary, _ = run_with([error], ledger)

    # The run is not ended by an exception. It finishes, and as its only pipeline produced no
    # verdict, its status is failed (every pipeline ended in an API or unexpected error).
    assert summary.results[0].status == "failed_api"
    assert summary.status == "failed"
    assert ledger.get_run("test-run")["status"] == "failed"
    assert_no_events_after_run_finished(ledger)


def test_an_authentication_error_ends_the_run_and_finishes_the_run_row(ledger):
    error = anthropic.AuthenticationError(
        "invalid key", response=httpx.Response(401, request=REQUEST), body=None
    )
    # The second pipeline is waiting on the idea lock when the first one fails.
    client = RoutingAnthropic(
        {
            "submit_idea": [error, tool_response("submit_idea", distinct_idea(2))],
        },
        yield_once=True,
    )
    with pytest.raises(anthropic.AuthenticationError):
        run_with([], ledger, n_pipelines=2, client=client, settings=relaxed_settings())

    assert ledger.get_run("test-run")["status"] == "failed"
    assert_no_events_after_run_finished(ledger)
    assert all(a["status"] != "running" for a in ledger.attempts_for_run("test-run"))


def test_a_sibling_in_a_call_is_cancelled_and_recorded_before_the_run_ends(ledger, monkeypatch):
    budgets = []
    real_budget = run_module.Budget

    def recording_budget(limit_usd):
        budgets.append(real_budget(limit_usd))
        return budgets[-1]

    monkeypatch.setattr(run_module, "Budget", recording_budget)
    error = anthropic.AuthenticationError(
        "invalid key", response=httpx.Response(401, request=REQUEST), body=None
    )

    class Script:
        """Call 1 is pipeline 1's idea. Call 2 is pipeline 1's coder, which yields and then fails
        with an authentication error. Call 3 is pipeline 2's idea, which never returns."""

        def __init__(self):
            self.calls = []
            self.messages = self

        async def create(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return tool_response("submit_idea", distinct_idea(1))
            if len(self.calls) == 2:
                await asyncio.sleep(0)  # lets pipeline 2 start its own call
                raise error
            await asyncio.Event().wait()

    with pytest.raises(anthropic.AuthenticationError):
        run_with([], ledger, n_pipelines=2, client=Script(), settings=relaxed_settings())

    assert ledger.get_run("test-run")["status"] == "failed"
    statuses = [r["status"] for r in ledger.conn.execute("SELECT status FROM llm_calls ORDER BY rowid")]
    assert statuses == ["ok", "error", "cancelled"]  # the sibling's call was cancelled, not abandoned
    finished = [
        json.loads(r["payload_json"])["status"]
        for r in ledger.events_for_run("test-run")
        if r["type"] == "pipeline_finished"
    ]
    # The pipeline that made the fatal call failed with it. Its sibling was cancelled, which is its own status.
    assert finished == ["aborted_error", "aborted_cancelled"]
    assert_no_events_after_run_finished(ledger)
    # Nothing stays reserved, and the spend the budget holds is the cost in the ledger.
    [budget] = budgets
    assert budget.reserved_usd == pytest.approx(0.0)
    assert budget.spent_usd == pytest.approx(ledger.llm_cost_total("test-run"))


def test_a_failed_setup_still_finishes_the_run_row(ledger, monkeypatch):
    def missing_prompt(name):
        raise FileNotFoundError(f"no prompt named {name}")

    monkeypatch.setattr(run_module, "load_prompt", missing_prompt)
    with pytest.raises(FileNotFoundError):
        run_with([], ledger)

    assert ledger.get_run("test-run")["status"] == "failed"
    assert_no_events_after_run_finished(ledger)


def test_negative_training_sharpe_fails_the_default_gate(ledger):
    responses = [tool_response("submit_idea", IDEA), tool_response("submit_code", {"code": EQUAL_WEIGHT_CODE})]
    summary, client = run_with(
        responses,
        ledger,
        settings=default_gates_settings(),
        prices=drifting_prices(-0.001, -0.001, "2030-01-01"),
    )

    assert summary.results[0].status == "failed_gate", attempt_detail(ledger)
    error = ledger.attempts_for_run("test-run")[0]["error"]
    assert "Training gate failed" in error, error
    assert "is below the minimum 0.5" in error, error
    assert len(client.messages.calls) == 2  # no critic, no validation


def test_excess_turnover_fails_the_default_gate(ledger):
    daily_idea = dict(IDEA, expected_holding_period_days=1)
    responses = [tool_response("submit_idea", daily_idea), tool_response("submit_code", {"code": ALTERNATING_CODE})]
    summary, _ = run_with(
        responses,
        ledger,
        settings=default_gates_settings(),
        prices=drifting_prices(0.0003, 0.0003, "2030-01-01"),
    )

    assert summary.results[0].status == "failed_gate"
    error = ledger.attempts_for_run("test-run")[0]["error"]
    assert "average daily turnover" in error
    assert "is above the limit 0.5" in error


def test_validation_sharpe_below_the_default_minimum_is_validation_fail(ledger):
    responses = [
        tool_response("submit_idea", IDEA),
        tool_response("submit_code", {"code": EQUAL_WEIGHT_CODE}),
        tool_response("submit_critique", LOW_CRITIC),
    ]
    summary, _ = run_with(
        responses,
        ledger,
        settings=default_gates_settings(),
        prices=drifting_prices(0.0009, -0.0009, "2018-12-31"),
    )

    assert summary.results[0].status == "validation_fail"
    rows = ledger.validations()
    assert len(rows) == 1 and rows[0]["passed"] == 0
    assert json.loads(rows[0]["metrics_json"])["sharpe"] < 0.3


def test_zero_variance_signal_fails_the_gate_with_strict_json(ledger):
    responses = [tool_response("submit_idea", IDEA), tool_response("submit_code", {"code": ZERO_VARIANCE_CODE})]
    summary, _ = run_with(responses, ledger, settings=default_gates_settings())

    assert summary.results[0].status == "failed_gate", attempt_detail(ledger)
    attempt = ledger.attempts_for_run("test-run")[0]
    assert "not a finite number" in attempt["error"], attempt_detail(ledger)
    metrics = strict_loads(attempt["train_metrics_json"])  # raises on a bare NaN token
    assert metrics["sharpe"] is None
    train_events = [r for r in ledger.events_for_run("test-run") if r["type"] == "train_backtest"]
    assert strict_loads(train_events[0]["payload_json"])["sharpe"] is None


def test_validation_never_sees_prices_after_validation_end(ledger, monkeypatch):
    seen = []
    real_validate = single_module.validate_signal

    def spy(code, validation_prices, *args, **kwargs):
        seen.append(validation_prices.index.max())
        return real_validate(code, validation_prices, *args, **kwargs)

    monkeypatch.setattr(single_module, "validate_signal", spy)
    later = pd.bdate_range("2010-01-01", "2024-12-31")
    rng = np.random.default_rng(5)
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, size=(len(later), 10)), axis=0)),
        index=later,
        columns=[f"T{i}" for i in range(10)],
    )
    responses = [
        tool_response("submit_idea", IDEA),
        tool_response("submit_code", {"code": GOOD_CODE}),
        tool_response("submit_critique", LOW_CRITIC),
    ]
    run_with(responses, ledger, prices=prices)

    assert len(seen) == 1, attempt_detail(ledger)
    assert seen[0] <= pd.Timestamp("2021-12-31"), seen


def test_a_pipeline_cancelled_during_validation_is_aborted_cancelled(ledger, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def blocked_validation(code, validation_prices, settings, holding_days=1):
        entered.set()
        release.wait(timeout=30)  # the worker thread is not cancelled, so it must be released
        raise AssertionError("a cancelled pipeline must not finish validation")

    monkeypatch.setattr(single_module, "validate_signal", blocked_validation)
    responses = [
        tool_response("submit_idea", IDEA),
        tool_response("submit_code", {"code": GOOD_CODE}),
        tool_response("submit_critique", LOW_CRITIC),
    ]

    async def scenario():
        task = asyncio.ensure_future(
            execute_run(
                run_id="test-run",
                settings=relaxed_settings(),
                prices=synthetic_prices(),
                n_pipelines=1,
                budget_usd=1.0,
                ledger=ledger,
                bus=EventBus(ledger),
                client=FakeAnthropic(responses),
            )
        )
        try:
            # Bounded, so a run that never reaches validation fails here instead of hanging.
            await asyncio.to_thread(entered.wait, 300)
            assert entered.is_set(), attempt_detail(ledger)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release.set()

    asyncio.run(scenario())
    assert ledger.get_run("test-run")["status"] == "interrupted"
    finished = [
        json.loads(r["payload_json"])["status"]
        for r in ledger.events_for_run("test-run")
        if r["type"] == "pipeline_finished"
    ]
    assert finished == ["aborted_cancelled"], attempt_detail(ledger)
    attempts = ledger.attempts_for_run("test-run")
    assert [a["status"] for a in attempts] == ["aborted_cancelled"], attempt_detail(ledger)
    assert attempts[0]["error"] == "cancelled"
    assert ledger.validations() == []
    assert_no_events_after_run_finished(ledger)


def test_a_pipeline_cancelled_during_a_revision_is_aborted_cancelled(ledger):
    class RevisionScript:
        """Idea, then code that fails the static check, then a revision call that never returns."""

        def __init__(self, revision_started):
            self.calls = []
            self.messages = self
            self.revision_started = revision_started

        async def create(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return tool_response("submit_idea", IDEA)
            if len(self.calls) == 2:
                return tool_response("submit_code", {"code": IMPORTS_OS})
            self.revision_started.set()
            await asyncio.Event().wait()

    async def scenario():
        script = RevisionScript(asyncio.Event())
        task = asyncio.ensure_future(
            execute_run(
                run_id="test-run",
                settings=relaxed_settings(),
                prices=synthetic_prices(),
                n_pipelines=1,
                budget_usd=1.0,
                ledger=ledger,
                bus=EventBus(ledger),
                client=script,
            )
        )
        # Bounded, so a run that never reaches the revision fails here instead of hanging.
        await asyncio.wait_for(script.revision_started.wait(), timeout=300)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    types = event_types(ledger)
    assert "revision_requested" in types
    finished = [
        json.loads(r["payload_json"])["status"]
        for r in ledger.events_for_run("test-run")
        if r["type"] == "pipeline_finished"
    ]
    assert finished == ["aborted_cancelled"], attempt_detail(ledger)
    # The first attempt reached its own verdict before the revision, so it keeps that status.
    assert [a["status"] for a in ledger.attempts_for_run("test-run")] == ["failed_static"]
    assert_no_events_after_run_finished(ledger)


def test_a_coder_that_never_returns_code_gives_failed_coder(ledger):
    responses = [tool_response("submit_idea", IDEA), text_response(), text_response()]
    summary, _ = run_with(responses, ledger)

    # Nothing was learned, so a run whose only pipeline stopped at the coder is failed (R4-AGT-01).
    assert summary.status == "failed"
    assert summary.results[0].status == "failed_coder", attempt_detail(ledger)
    assert ledger.attempts_for_run("test-run") == []  # no attempt was made
    assert_no_events_after_run_finished(ledger)


def test_a_permission_error_ends_the_run_like_an_authentication_error(ledger):
    error = anthropic.PermissionDeniedError(
        "forbidden", response=httpx.Response(403, request=REQUEST), body=None
    )
    client = RoutingAnthropic(
        {"submit_idea": [error, tool_response("submit_idea", distinct_idea(2))]},
        yield_once=True,
    )
    with pytest.raises(anthropic.PermissionDeniedError):
        run_with([], ledger, n_pipelines=2, client=client, settings=relaxed_settings())

    assert ledger.get_run("test-run")["status"] == "failed"
    assert_no_events_after_run_finished(ledger)
    assert all(a["status"] != "running" for a in ledger.attempts_for_run("test-run"))


def test_a_model_without_pricing_stops_the_run_before_any_row_is_written(ledger):
    settings = relaxed_settings()
    agents = settings.agents.model_copy(update={"ideas_model": "no-such-model"})
    settings = settings.model_copy(update={"agents": agents})
    client = FakeAnthropic([])

    with pytest.raises(ValueError, match="no pricing configured for model 'no-such-model'"):
        run_with([], ledger, settings=settings, client=client)

    assert list(ledger.list_runs()) == []
    assert client.messages.calls == []


def test_a_run_whose_pipelines_all_fail_on_the_api_is_failed_not_completed(ledger):
    not_found = anthropic.NotFoundError(
        "model not found", response=httpx.Response(404, request=REQUEST), body=None
    )
    client = RoutingAnthropic({"submit_idea": [not_found, not_found]}, yield_once=True)
    summary, _ = run_with([], ledger, n_pipelines=2, client=client, settings=relaxed_settings())

    assert [r.status for r in summary.results] == ["failed_api", "failed_api"]
    assert summary.status == "failed"
    assert ledger.get_run("test-run")["status"] == "failed"


def test_a_run_whose_every_window_is_unscorable_is_failed_data(ledger):
    # The prices stop inside the validation window, so validation cannot be scored (failed_backtest).
    prices = synthetic_prices().loc[:"2019-06-28"]
    responses = [
        tool_response("submit_idea", IDEA),
        tool_response("submit_code", {"code": GOOD_CODE}),
        tool_response("submit_critique", LOW_CRITIC),
    ]
    summary, _ = run_with(responses, ledger, prices=prices)

    assert summary.results[0].status == "failed_backtest", attempt_detail(ledger)
    assert summary.status == RUN_FAILED_DATA
    assert ledger.get_run("test-run")["status"] == RUN_FAILED_DATA


def test_passed_critic_and_passed_validation_are_counted_separately(ledger):
    responses = [
        tool_response("submit_idea", IDEA),
        tool_response("submit_code", {"code": EQUAL_WEIGHT_CODE}),
        tool_response("submit_critique", LOW_CRITIC),
    ]
    summary, _ = run_with(
        responses,
        ledger,
        settings=default_gates_settings(),
        prices=drifting_prices(0.0009, -0.0009, "2018-12-31"),
    )

    # validation_fail passed the critic, so it counts there and not as a validation pass.
    assert summary.results[0].status == "validation_fail"
    assert summary.passed_critic == 1
    assert summary.passed_validation == 0


def test_every_status_the_pipeline_writes_is_declared_in_statuses():
    source = Path(single_module.__file__).read_text(encoding="utf-8")
    written = set(
        re.findall(
            r'"((?:failed|aborted)_[a-z_]+|critic_pass|validation_pass|validation_fail'
            r'|rejected_duplicate_code)"',
            source,
        )
    )
    assert written, "the scan found no status strings, so it checks nothing"
    assert written <= ATTEMPT_STATUSES | PIPELINE_STATUSES, written - (
        ATTEMPT_STATUSES | PIPELINE_STATUSES
    )


def test_every_run_status_the_run_writes_is_declared_in_statuses():
    source = Path(run_module.__file__).read_text(encoding="utf-8")
    written = set(re.findall(r'status = "([a-z_]+)"', source))
    assert written and written <= RUN_STATUSES, written


def test_the_stale_threshold_is_read_by_execute_run(ledger, monkeypatch):
    # Inside the reaper's window a run is left alone; past it, the run is abandoned at the next start.
    assert run_module.STALE_RUN_AFTER == timedelta(minutes=30)
    now = datetime.now(UTC)
    for run_id, age in (("fresh", timedelta(minutes=29)), ("stale", timedelta(minutes=31))):
        ledger.create_run(run_id, "{}", 1.0)
        ledger.add_event(
            f"e-{run_id}", run_id, None, None, None, None, "run", "run_started",
            (now - age).isoformat(timespec="milliseconds"), {},
        )

    def missing_prompt(name):
        raise FileNotFoundError(f"no prompt named {name}")

    # The reaper runs before the run row is created, so a failing setup still shows its effect.
    monkeypatch.setattr(run_module, "load_prompt", missing_prompt)
    prices = pd.DataFrame(
        {"T0": 1.0}, index=pd.bdate_range("2010-01-01", "2021-12-31")
    )
    with pytest.raises(FileNotFoundError):
        asyncio.run(
            execute_run(
                run_id="test-run",
                settings=relaxed_settings(),
                prices=prices,
                n_pipelines=1,
                budget_usd=1.0,
                ledger=ledger,
                bus=EventBus(ledger),
                client=FakeAnthropic([]),
            )
        )
    assert ledger.get_run("fresh")["status"] == "running"
    assert ledger.get_run("stale")["status"] == "abandoned"
    # The closing event is the same payload a live run writes, with the reaper's values.
    [closing] = [
        json.loads(r["payload_json"])
        for r in ledger.events_for_run("stale")
        if r["type"] == "run_finished"
    ]
    assert set(closing) == {
        "status", "total_cost_usd", "budget_usd", "reserved_usd", "data_failed_pipelines",
    }
    assert closing["status"] == "abandoned"
    assert closing["reserved_usd"] is None


def test_a_training_window_that_cannot_be_scored_is_failed_backtest(ledger):
    prices = synthetic_prices()
    prices.iloc[500, 0] = np.nan  # inside the training window, which ends 2018-12-31
    attempts = relaxed_settings().agents.max_revisions + 1
    # A failed_backtest attempt is revised like any failed attempt, so every attempt needs a reply.
    responses = [tool_response("submit_idea", IDEA)] + [
        tool_response("submit_code", {"code": GOOD_CODE.replace("shift(21)", f"shift({21 + i})")})
        for i in range(attempts)  # a different lookback each time, so no revision is a duplicate
    ]
    summary, client = run_with(responses, ledger, prices=prices)

    assert summary.results[0].status == "failed_backtest", attempt_detail(ledger)
    rows = ledger.attempts_for_run("test-run")
    assert [r["status"] for r in rows] == ["failed_backtest"] * attempts
    assert all("could not score the signal" in r["error"] for r in rows), attempt_detail(ledger)
    assert len(client.messages.calls) == 1 + attempts  # no critic and no validation were called


def test_the_ideas_call_row_names_the_idea_it_produced(ledger):
    responses = [
        tool_response("submit_idea", IDEA),
        tool_response("submit_code", {"code": GOOD_CODE}),
        tool_response("submit_critique", LOW_CRITIC),
    ]
    run_with(responses, ledger)

    [idea] = ledger.ideas_for_run("test-run")
    calls = ledger.conn.execute(
        "SELECT idea_id FROM llm_calls WHERE run_id = 'test-run' AND agent = 'ideas'"
    ).fetchall()
    assert calls, "the ideas call was not recorded"
    assert [c["idea_id"] for c in calls] == [idea["idea_id"]]


def test_the_ideas_prompt_shows_same_theme_ideas_even_when_they_are_older_than_thirty(ledger):
    ledger.create_run("r1", "{}", 1.0)
    ledger.add_idea(
        idea_id="old", run_id="r1", pipeline_id="p", name="Old momentum idea",
        hypothesis="h" * 12, description="d" * 12, expected_holding_period_days=1,
        theme="momentum", created_at="2026-01-01T00:00:00.000+00:00",
    )
    for i in range(40):
        ledger.add_idea(
            idea_id=f"n{i}", run_id="r1", pipeline_id="p", name=f"Value idea {i}",
            hypothesis="h" * 12, description="d" * 12, expected_holding_period_days=1,
            theme="value", created_at=f"2026-02-{i + 1:02d}T00:00:00.000+00:00",
        )
    shown = single_module.relevant_ideas(ledger, "momentum")
    names = [row["name"] for row in shown]
    assert "Old momentum idea" in names
    assert len(shown) == 30
    # The old rule (the 30 newest ideas) would have left it out.
    assert "Old momentum idea" not in [row["name"] for row in ledger.recent_ideas(30)]


def test_the_run_records_the_hash_of_each_prompt_it_used(ledger):
    responses = [
        tool_response("submit_idea", IDEA),
        tool_response("submit_code", {"code": GOOD_CODE}),
        tool_response("submit_critique", LOW_CRITIC),
    ]
    run_with(responses, ledger)

    config = json.loads(ledger.get_run("test-run")["config_json"])
    for key, name in (("ideas", "ideas_v1"), ("coder", "coder_v2"), ("critic", "critic_v1")):
        assert config["prompt_sha256"][key] == load_prompt(name).sha256


def test_a_run_whose_pipelines_all_stop_before_a_verdict_is_failed(ledger):
    # Every ideas reply lacks the tool call, so each pipeline stops at failed_ideas (R4-AGT-01).
    responses = [text_response() for _ in range(12)]
    summary, _ = run_with(responses, ledger, n_pipelines=2)

    assert [r.status for r in summary.results] == ["failed_ideas", "failed_ideas"]
    assert summary.status == "failed"
    assert ledger.get_run("test-run")["status"] == "failed"


def test_a_failed_abort_write_leaves_no_attempt_running(ledger, monkeypatch):
    # The attempt's own abort write fails, as on a locked database. The attempt must still end
    # closed, and the original harness error must still be what the pipeline reports (R4-AGT-02).
    real_update = ledger.update_attempt

    def flaky_update(attempt_id, **fields):
        if fields.get("status") == "aborted_error":
            raise sqlite3.OperationalError("database is locked")
        return real_update(attempt_id, **fields)

    monkeypatch.setattr(ledger, "update_attempt", flaky_update)

    def harness_boom(*args, **kwargs):
        raise RuntimeError("harness boom")

    monkeypatch.setattr(single_module, "run_signal", harness_boom)
    responses = [
        tool_response("submit_idea", IDEA),
        tool_response("submit_code", {"code": GOOD_CODE}),
        tool_response("submit_critique", LOW_CRITIC),
    ]
    summary, _ = run_with(responses, ledger)

    statuses = [a["status"] for a in ledger.attempts_for_run("test-run")]
    assert "running" not in statuses, attempt_detail(ledger)
    assert summary.results[0].status == "failed_error"
    assert summary.status == "failed"
    assert ledger.get_run("test-run")["status"] == "failed"
    assert_no_events_after_run_finished(ledger)


def test_a_validation_sandbox_timeout_is_failed_sandbox_not_a_verdict(ledger, monkeypatch):
    # The signal ran on the training window and passed the critic. On the validation window the
    # sandbox timed out, so there is no verdict, and the pipeline must not count as passing the critic.
    def timed_out(sandboxed, prices):
        return SandboxOutcome(status="timeout", message="the signal took too long")

    monkeypatch.setattr(validation_module, "run_signal", timed_out)
    responses = [
        tool_response("submit_idea", IDEA),
        tool_response("submit_code", {"code": EQUAL_WEIGHT_CODE}),
        tool_response("submit_critique", LOW_CRITIC),
    ]
    summary, _ = run_with(responses, ledger)

    assert summary.results[0].status == "failed_sandbox", attempt_detail(ledger)
    assert summary.passed_critic == 0
    assert ledger.validations() == []
    assert "validation_result" not in event_types(ledger)


def test_a_model_without_a_cache_write_price_stops_the_run_before_any_row_is_written(ledger):
    settings = with_pricing(relaxed_settings(), "claude-haiku-5-5", cache_write_5m=None)
    client = FakeAnthropic([])

    with pytest.raises(ValueError, match="cache_write_5m"):
        run_with([], ledger, settings=settings, client=client)

    assert list(ledger.list_runs()) == []
    assert client.messages.calls == []


def _stored_statuses(db):
    """Every status value the ledger holds, read back from the rows and the event payloads."""

    def column(sql):
        return {row[0] for row in db.conn.execute(sql).fetchall()}

    def payload_statuses(event_type):
        return {
            json.loads(row[0])["status"]
            for row in db.conn.execute(
                "SELECT payload_json FROM events WHERE type = ?", (event_type,)
            ).fetchall()
        }

    return {
        "attempts": column("SELECT status FROM attempts"),
        "pipelines": payload_statuses("pipeline_finished"),
        "runs": column("SELECT status FROM runs"),
        "run_events": payload_statuses("run_finished"),
        "llm_calls": column("SELECT status FROM llm_calls"),
    }


def test_every_status_a_run_stores_is_declared_in_statuses(ledger, tmp_path, monkeypatch):
    """Reads back what was stored, instead of scanning source text, so a typo in a keyword
    argument or a comment cannot pass (R4-AGT-07). Three runs reach the writers: a run that
    validates a signal, one that fails its static check, and one that the reaper abandons."""
    run_with(
        [
            tool_response("submit_idea", IDEA),
            tool_response("submit_code", {"code": EQUAL_WEIGHT_CODE}),
            tool_response("submit_critique", LOW_CRITIC),
        ],
        ledger,
        settings=default_gates_settings(),
        prices=drifting_prices(0.0009, -0.0009, "2018-12-31"),
    )
    second = Ledger(tmp_path / "second.db")
    run_with(
        [
            tool_response("submit_idea", IDEA),
            tool_response("submit_code", {"code": IMPORTS_OS}),
        ],
        second,
        settings=default_gates_settings(),
    )
    # A run that a dead process left open: one pipeline started, one attempt running.
    stale_at = datetime.now(UTC) - timedelta(minutes=31)
    second.create_run("stale", "{}", 1.0)
    second.add_event(
        "e-stale-run", "stale", None, None, None, None, "run", "run_started",
        stale_at.isoformat(timespec="milliseconds"), {},
    )
    second.add_event(
        "e-stale-pipe", "stale", "p-stale", None, None, None, "pipeline", "pipeline_started",
        stale_at.isoformat(timespec="milliseconds"), {"theme": "momentum"},
    )
    second.add_idea(
        idea_id="i-stale", run_id="stale", pipeline_id="p-stale", name="Stale idea",
        hypothesis="h", description="d", expected_holding_period_days=5, theme="momentum",
    )
    second.add_attempt(
        attempt_id="a-stale", idea_id="i-stale", run_id="stale", attempt_number=1,
        code="x", code_hash="h", status="running",
    )

    def no_prompts(name):
        raise FileNotFoundError(f"no prompt named {name}")

    monkeypatch.setattr(run_module, "load_prompt", no_prompts)
    with pytest.raises(FileNotFoundError):
        asyncio.run(
            execute_run(
                run_id="next",
                settings=relaxed_settings(),
                prices=synthetic_prices(),
                n_pipelines=1,
                budget_usd=1.0,
                ledger=second,
                bus=EventBus(second),
                client=FakeAnthropic([]),
            )
        )
    assert second.get_run("stale")["status"] == "abandoned"

    stored = {}
    for db in (ledger, second):
        for key, values in _stored_statuses(db).items():
            stored.setdefault(key, set()).update(values)
    assert {"validation_fail", "failed_static", "abandoned"} <= (
        stored["attempts"] | stored["runs"] | stored["run_events"]
    )
    assert stored["attempts"] <= ATTEMPT_STATUSES
    assert stored["pipelines"] <= PIPELINE_STATUSES
    assert stored["runs"] <= RUN_STATUSES
    assert stored["run_events"] <= RUN_STATUSES
    assert stored["llm_calls"] <= LLM_CALL_STATUSES
    second.close()


def test_a_mixed_run_counts_its_data_failures_in_run_finished(ledger, monkeypatch):
    # R4-STAT-08: 49 windows that could not be scored among 50 pipelines must show in the record,
    # even though the run is not failed_data (one pipeline passed the critic).
    from alpha_lab.pipeline.single import PipelineResult

    results = [PipelineResult(f"p{i}", "failed_backtest", None, "momentum", 1) for i in range(49)]
    results.append(PipelineResult("pass", "critic_pass", None, "momentum", 1))

    async def fake_swarm(ctx, budget, n, concurrency, seed, progress=None, on_progress=None):
        return results

    monkeypatch.setattr(run_module, "run_swarm", fake_swarm)
    summary, _ = run_with([], ledger, n_pipelines=50)

    assert summary.status == "completed"
    [finished] = [
        json.loads(r["payload_json"])
        for r in ledger.events_for_run("test-run")
        if r["type"] == "run_finished"
    ]
    assert finished["data_failed_pipelines"] == 49
    assert finished["status"] == "completed"
