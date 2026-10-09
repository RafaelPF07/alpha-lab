import asyncio
import hashlib
import json
import re

import anthropic
import httpx2 as httpx
import numpy as np
import pandas as pd
import pytest
from fakes import FakeAnthropic, RoutingAnthropic, text_response, tool_response

from alpha_lab.config import GatesConfig, load_settings
from alpha_lab.events.bus import EventBus
from alpha_lab.ledger.db import Ledger
from alpha_lab.pipeline import run as run_module
from alpha_lab.pipeline import single as single_module
from alpha_lab.pipeline.run import execute_run

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
    settings = load_settings()
    return settings.model_copy(
        update={
            "gates": GatesConfig(
                min_train_sharpe=-100.0, min_validation_sharpe=-100.0, max_turnover_daily=10.0
            )
        }
    )


def default_gates_settings(max_revisions: int = 0):
    """The shipped settings, gates untouched. Only the revision cap is lowered, so one attempt runs."""
    settings = load_settings()
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
    assert summary.results[0].status == "failed_critic_output"
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
    assert statuses[0] == "failed_error"
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

    assert summary.status == "completed"
    assert summary.results[0].status == "failed_api"
    assert ledger.get_run("test-run")["status"] == "completed"
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


def test_a_sibling_in_a_call_is_cancelled_and_recorded_before_the_run_ends(ledger):
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
    assert finished == ["aborted_error", "aborted_error"]
    assert_no_events_after_run_finished(ledger)


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

    assert summary.results[0].status == "failed_gate"
    error = ledger.attempts_for_run("test-run")[0]["error"]
    assert "Training gate failed" in error
    assert "is below the minimum 0.5" in error
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

    assert summary.results[0].status == "failed_gate"
    attempt = ledger.attempts_for_run("test-run")[0]
    assert "not a finite number" in attempt["error"]
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

    assert len(seen) == 1
    assert seen[0] <= pd.Timestamp("2021-12-31")
