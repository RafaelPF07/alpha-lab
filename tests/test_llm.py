import asyncio

import anthropic
import httpx2 as httpx
import pytest
from fakes import FakeAnthropic, tool_response
from pydantic import BaseModel, Field

from alpha_lab.agents.budget import Budget, BudgetExceeded
from alpha_lab.agents.llm import (
    BACKOFF_BASE_SECONDS,
    ContextLimitExceeded,
    LLMClient,
    PromptSpec,
    StepFailed,
    Usage,
    compute_cost,
)
from alpha_lab.config import load_settings
from alpha_lab.events.bus import EventBus
from alpha_lab.ledger.db import Ledger


class Out(BaseModel):
    name: str = Field(min_length=3)
    count: int = Field(ge=1)


PROMPT = PromptSpec(name="test_v1", text="You are a test agent.")
REQUEST = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def rate_limit():
    return anthropic.RateLimitError(
        "slow down", response=httpx.Response(429, request=REQUEST), body=None
    )


def bad_request():
    return anthropic.BadRequestError(
        "bad", response=httpx.Response(400, request=REQUEST), body=None
    )


@pytest.fixture
def settings():
    return load_settings()


@pytest.fixture
def ledger(tmp_path):
    db = Ledger(tmp_path / "ledger.db")
    db.create_run("run1", "{}", 5.0)
    yield db
    db.close()


def make_llm(settings, ledger, responses, budget_usd=5.0, sleeps=None):
    client = FakeAnthropic(responses)
    recorded = [] if sleeps is None else sleeps

    async def fake_sleep(seconds):
        recorded.append(seconds)

    llm = LLMClient(
        client=client,
        settings=settings,
        budget=Budget(budget_usd),
        ledger=ledger,
        bus=EventBus(ledger),
        run_id="run1",
        sleep=fake_sleep,
    )
    return llm, client


async def call(llm, text="make a thing"):
    return await llm.structured_call(
        agent="test",
        prompt=PROMPT,
        model="claude-haiku-5-5",
        user_text=text,
        output_model=Out,
        tool_name="submit",
        tool_description="submit it",
    )


def test_cost_formula_under_threshold(settings):
    pricing = settings.pricing.models["claude-haiku-5-5"]
    # 1000 fresh input at $0.10/M, 500 output at $0.50/M
    assert compute_cost(pricing, Usage(1000, 500, 0, 0), 100_000) == pytest.approx(0.00035)
    # 1000 fresh input + 50,000 cache reads at $0.01/M, 500 output. The prompt is 51K tokens.
    assert compute_cost(pricing, Usage(1000, 500, 50_000, 0), 100_000) == pytest.approx(0.00085)


def test_cost_formula_over_threshold_uses_long_context_prices(settings):
    pricing = settings.pricing.models["claude-haiku-5-5"]
    # 150,000 input at $0.50/M, 500 output at $2.50/M
    assert compute_cost(pricing, Usage(150_000, 500, 0, 0), 100_000) == pytest.approx(0.07625)


def test_budget_is_checked_before_anything_is_sent(settings, ledger):
    async def scenario():
        llm, client = make_llm(
            settings,
            ledger,
            [tool_response("submit", {"name": "abc", "count": 1})],
            budget_usd=1e-9,
        )
        with pytest.raises(BudgetExceeded):
            await call(llm)
        assert client.messages.calls == []

    asyncio.run(scenario())


def test_oversized_request_is_refused_not_truncated(settings, ledger):
    async def scenario():
        llm, client = make_llm(settings, ledger, [])
        with pytest.raises(ContextLimitExceeded):
            await call(llm, text="x" * (settings.agents.context_limit_tokens * 4))
        assert client.messages.calls == []

    asyncio.run(scenario())


def test_rate_limits_are_retried_with_backoff(settings, ledger):
    async def scenario():
        sleeps = []
        responses = [
            rate_limit(),
            rate_limit(),
            tool_response("submit", {"name": "abc", "count": 2}),
        ]
        llm, client = make_llm(settings, ledger, responses, sleeps=sleeps)
        result = await call(llm)
        assert result.count == 2
        assert len(client.messages.calls) == 3
        assert sleeps == [BACKOFF_BASE_SECONDS, BACKOFF_BASE_SECONDS * 2]

    asyncio.run(scenario())


def test_gives_up_after_four_attempts(settings, ledger):
    async def scenario():
        sleeps = []
        llm, client = make_llm(settings, ledger, [rate_limit() for _ in range(4)], sleeps=sleeps)
        with pytest.raises(anthropic.RateLimitError):
            await call(llm)
        assert len(client.messages.calls) == 4
        assert len(sleeps) == 3

    asyncio.run(scenario())


def test_bad_requests_are_never_retried(settings, ledger):
    async def scenario():
        llm, client = make_llm(settings, ledger, [bad_request()])
        with pytest.raises(anthropic.BadRequestError):
            await call(llm)
        assert len(client.messages.calls) == 1

    asyncio.run(scenario())


def test_invalid_output_gets_one_repair_attempt(settings, ledger):
    async def scenario():
        responses = [
            tool_response("submit", {"name": "ab", "count": 0}),  # fails validation
            tool_response("submit", {"name": "abc", "count": 3}),  # repaired
        ]
        llm, client = make_llm(settings, ledger, responses)
        result = await call(llm)
        assert result.name == "abc"
        repair_messages = client.messages.calls[1]["messages"]
        tool_result = repair_messages[-1]["content"][0]
        assert tool_result["type"] == "tool_result"
        assert tool_result["is_error"] is True

    asyncio.run(scenario())


def test_second_invalid_output_fails_the_step(settings, ledger):
    async def scenario():
        responses = [
            tool_response("submit", {"name": "ab", "count": 0}),
            tool_response("submit", {"name": "ab", "count": 0}),
        ]
        llm, client = make_llm(settings, ledger, responses)
        with pytest.raises(StepFailed):
            await call(llm)
        assert len(client.messages.calls) == 2

    asyncio.run(scenario())


def test_every_call_is_logged_with_cost(settings, ledger):
    async def scenario():
        responses = [
            tool_response(
                "submit", {"name": "abc", "count": 1}, input_tokens=1000, output_tokens=500
            )
        ]
        llm, _ = make_llm(settings, ledger, responses)
        await call(llm)
        rows = ledger.conn.execute("SELECT * FROM llm_calls").fetchall()
        assert len(rows) == 1
        assert rows[0]["agent"] == "test"
        assert rows[0]["prompt_version"] == "test_v1"
        assert rows[0]["cost_usd"] == pytest.approx(0.00035)
        assert llm.budget.spent_usd == pytest.approx(0.00035)
        assert llm.budget.reserved_usd == pytest.approx(0.0)

    asyncio.run(scenario())
