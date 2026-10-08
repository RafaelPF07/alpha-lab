import re

import numpy as np
import pandas as pd
import pytest
from fakes import FakeAnthropic, tool_response

from alpha_lab.config import GatesConfig, load_settings
from alpha_lab.events.bus import EventBus
from alpha_lab.ledger.db import Ledger
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

LOW_CRITIC = {"concerns": [], "suspicion_level": "low", "suggested_change": "none"}

IMPORTS_OS = "import os\n" + GOOD_CODE
RAISES = "def signal(prices):\n    raise ValueError('nope')\n"


def raising_code(i):
    """Same behaviour every time, but a different token, so it is not a duplicate."""
    return "def signal(prices):" + chr(10) + f"    raise ValueError('nope {i}')" + chr(10)


def relaxed_settings():
    """Same pipeline, with gates loose enough that the code can pass on random data."""
    settings = load_settings()
    return settings.model_copy(
        update={
            "gates": GatesConfig(
                min_train_sharpe=-100.0, min_validation_sharpe=0.3, max_turnover_daily=10.0
            )
        }
    )


def synthetic_prices() -> pd.DataFrame:
    """Ten tickers from 2010 through 2021, so the data runs past train_end into validation."""
    rng = np.random.default_rng(11)
    index = pd.bdate_range("2010-01-01", "2021-12-31")
    log_returns = rng.normal(0.0003, 0.012, size=(len(index), 10))
    prices = 100.0 * np.exp(np.cumsum(log_returns, axis=0))
    return pd.DataFrame(prices, index=index, columns=[f"T{i}" for i in range(10)])


@pytest.fixture
def ledger(tmp_path):
    db = Ledger(tmp_path / "ledger.db")
    yield db
    db.close()


def run_with(responses, ledger, budget_usd=1.0, settings=None):
    client = FakeAnthropic(responses)
    bus = EventBus(ledger)
    summary = None

    import asyncio

    async def scenario():
        nonlocal summary
        summary = await execute_run(
            run_id="test-run",
            settings=settings or relaxed_settings(),
            prices=synthetic_prices(),
            n_pipelines=1,
            budget_usd=budget_usd,
            ledger=ledger,
            bus=bus,
            client=client,
        )

    asyncio.run(scenario())
    return summary, client


def test_passing_pipeline_runs_end_to_end_and_keeps_validation_out(ledger):
    responses = [
        tool_response("submit_idea", IDEA),
        tool_response("submit_code", {"code": GOOD_CODE}),
        tool_response("submit_critique", LOW_CRITIC),
    ]
    summary, client = run_with(responses, ledger)

    assert summary.status == "completed"
    assert summary.results[0].status == "critic_pass"
    assert summary.results[0].attempts == 1
    assert ledger.count_trials() == 1
    assert summary.total_cost_usd > 0

    types = [row["type"] for row in ledger.events_for_run("test-run")]
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

    assert summary.results[0].status == "critic_pass"
    assert summary.results[0].attempts == 2
    revision_prompt = client.messages.calls[2]["messages"][0]["content"]
    assert "Static check failed" in revision_prompt
    assert "os" in revision_prompt


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


def test_unexpected_api_error_marks_the_run_failed(ledger):
    import anthropic
    import httpx2 as httpx

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    error = anthropic.BadRequestError(
        "credit balance too low", response=httpx.Response(400, request=request), body=None
    )
    with pytest.raises(anthropic.BadRequestError):
        run_with([error], ledger)
    assert ledger.get_run("test-run")["status"] == "failed"
