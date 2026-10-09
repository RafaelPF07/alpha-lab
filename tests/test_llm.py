import asyncio
import re

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
    OutputTruncated,
    PromptSpec,
    StepFailed,
    Usage,
    compute_cost,
)
from alpha_lab.agents.models import CODER_CODE_MAX_CHARS
from alpha_lab.agents.prompts import load_prompt
from alpha_lab.checks.static import FORBIDDEN_BUILTINS, FORBIDDEN_IO_METHODS, FORBIDDEN_MODULE_NAMES
from alpha_lab.config import load_settings
from alpha_lab.events.bus import EventBus
from alpha_lab.ledger.db import Ledger


class Out(BaseModel):
    name: str = Field(min_length=3)
    count: int = Field(ge=1)


PROMPT = PromptSpec(name="test_v1", text="You are a test agent.")
REQUEST = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def rate_limit(headers=None):
    return anthropic.RateLimitError(
        "slow down", response=httpx.Response(429, headers=headers, request=REQUEST), body=None
    )


def status_error(cls, status, headers=None):
    return cls("boom", response=httpx.Response(status, headers=headers, request=REQUEST), body=None)


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


async def call(llm, text="make a thing", model="claude-haiku-5-5"):
    return await llm.structured_call(
        agent="test",
        prompt=PROMPT,
        model=model,
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


def test_budget_refusal_leaves_a_refused_row_and_an_event(settings, ledger):
    async def scenario():
        llm, _ = make_llm(settings, ledger, [], budget_usd=1e-9)
        with pytest.raises(BudgetExceeded):
            await call(llm)

    asyncio.run(scenario())
    rows = ledger.conn.execute("SELECT status, cost_usd FROM llm_calls").fetchall()
    assert [(r["status"], r["cost_usd"]) for r in rows] == [("refused", 0.0)]
    types = [row["type"] for row in ledger.events_for_run("run1")]
    assert "llm_call_refused" in types
    assert "llm_call_started" not in types


def test_oversized_request_is_refused_not_truncated(settings, ledger):
    async def scenario():
        llm, client = make_llm(settings, ledger, [])
        with pytest.raises(ContextLimitExceeded):
            await call(llm, text="x" * (settings.agents.context_limit_tokens * 4))
        assert client.messages.calls == []

    asyncio.run(scenario())
    statuses = [r["status"] for r in ledger.conn.execute("SELECT status FROM llm_calls")]
    assert statuses == ["refused"]


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


def test_retry_after_from_the_server_sets_the_wait(settings, ledger):
    async def scenario():
        sleeps = []
        responses = [
            rate_limit({"retry-after": "7"}),
            tool_response("submit", {"name": "abc", "count": 2}),
        ]
        llm, _ = make_llm(settings, ledger, responses, sleeps=sleeps)
        await call(llm)
        assert sleeps == [7.0]

    asyncio.run(scenario())


def test_retry_after_is_capped(settings, ledger):
    async def scenario():
        sleeps = []
        responses = [
            rate_limit({"retry-after-ms": "900000"}),  # 900 seconds
            tool_response("submit", {"name": "abc", "count": 2}),
        ]
        llm, _ = make_llm(settings, ledger, responses, sleeps=sleeps)
        await call(llm)
        assert sleeps == [60.0]

    asyncio.run(scenario())


def test_timeouts_and_conflicts_are_retried_like_the_sdk_does(settings, ledger):
    async def scenario():
        sleeps = []
        responses = [
            status_error(anthropic.APIStatusError, 408),
            status_error(anthropic.APIStatusError, 409),
            tool_response("submit", {"name": "abc", "count": 2}),
        ]
        llm, client = make_llm(settings, ledger, responses, sleeps=sleeps)
        result = await call(llm)
        assert result.count == 2
        assert len(client.messages.calls) == 3

    asyncio.run(scenario())


def test_x_should_retry_false_stops_the_retries(settings, ledger):
    async def scenario():
        error = status_error(
            anthropic.InternalServerError, 500, headers={"x-should-retry": "false"}
        )
        llm, client = make_llm(settings, ledger, [error])
        with pytest.raises(anthropic.InternalServerError):
            await call(llm)
        assert len(client.messages.calls) == 1

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


def test_a_failed_call_is_charged_its_reservation_and_releases_it(settings, ledger):
    async def scenario():
        llm, _ = make_llm(settings, ledger, [bad_request()])
        with pytest.raises(anthropic.BadRequestError):
            await call(llm)
        assert llm.budget.reserved_usd == pytest.approx(0.0)
        assert llm.budget.spent_usd > 0
        # The ledger total and the budget agree, so the run cost is the sum of the llm_calls rows.
        assert ledger.llm_cost_total("run1") == pytest.approx(llm.budget.spent_usd)

    asyncio.run(scenario())


def test_a_cancelled_call_is_recorded_as_cancelled(settings, ledger):
    class HangingMessages:
        async def create(self, **kwargs):
            await asyncio.Event().wait()

    class HangingClient:
        messages = HangingMessages()

    async def scenario():
        llm = LLMClient(
            client=HangingClient(),
            settings=settings,
            budget=Budget(5.0),
            ledger=ledger,
            bus=EventBus(ledger),
            run_id="run1",
        )
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(call(llm), timeout=0.05)
        assert llm.budget.reserved_usd == pytest.approx(0.0)
        assert llm.budget.spent_usd > 0

    asyncio.run(scenario())
    statuses = [r["status"] for r in ledger.conn.execute("SELECT status FROM llm_calls")]
    assert statuses == ["cancelled"]


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


def test_repair_message_names_the_field_and_the_rule_without_echoing_the_input(settings, ledger):
    async def scenario():
        secret = "SENTINEL_TEXT_THAT_MUST_NOT_BE_ECHOED"
        responses = [
            tool_response("submit", {"name": "ab", "count": 1, "note": secret}),
            tool_response("submit", {"name": "ab", "count": 1, "note": secret}),
        ]
        llm, client = make_llm(settings, ledger, responses)
        with pytest.raises(StepFailed) as failure:
            await call(llm)
        repair_result = client.messages.calls[1]["messages"][-1]["content"][0]
        assert repair_result["content"] == (
            "the tool input failed validation: name: String should have at least 3 characters"
        )
        assert secret not in repair_result["content"]
        assert secret not in str(failure.value)

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


def test_a_reply_cut_off_at_max_tokens_is_not_repaired(settings, ledger):
    async def scenario():
        truncated = tool_response("submit", {"name": "abc"})  # the input stops before count
        truncated.stop_reason = "max_tokens"
        llm, client = make_llm(settings, ledger, [truncated])
        with pytest.raises(OutputTruncated) as failure:
            await call(llm)
        assert len(client.messages.calls) == 1
        assert "max_tokens=4000" in str(failure.value)
        assert "not repaired" in str(failure.value)

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


def test_haiku_is_sent_a_forced_tool_choice_without_parallel_tool_use(settings, ledger):
    async def scenario():
        llm, client = make_llm(settings, ledger, [tool_response("submit", {"name": "abc", "count": 1})])
        await call(llm, model="claude-haiku-5-5")
        assert client.messages.calls[0]["tool_choice"] == {
            "type": "tool",
            "name": "submit",
            "disable_parallel_tool_use": True,
        }

    asyncio.run(scenario())


def test_sonnet_is_sent_auto_tool_choice_without_parallel_tool_use(settings, ledger):
    async def scenario():
        llm, client = make_llm(settings, ledger, [tool_response("submit", {"name": "abc", "count": 1})])
        await call(llm, model="claude-sonnet-5-5")
        assert client.messages.calls[0]["tool_choice"] == {
            "type": "auto",
            "disable_parallel_tool_use": True,
        }

    asyncio.run(scenario())


def test_coder_output_cap_fits_the_token_budget(settings):
    # Python is about 3 characters per token. The longest allowed program should fit in 90% of the
    # reply budget, which leaves room for the JSON wrapper.
    assert CODER_CODE_MAX_CHARS / 3 <= 0.9 * settings.agents.max_tokens_per_call


def test_coder_prompt_states_every_rule_the_checker_enforces():
    text = load_prompt("coder_v1").text
    for name in sorted(FORBIDDEN_BUILTINS | FORBIDDEN_MODULE_NAMES | FORBIDDEN_IO_METHODS):
        assert re.search(rf"(?<![\w.]){re.escape(name)}(?!\w)", text), f"prompt lacks {name}"
    assert "starts with `read_`" in text
    assert "starts with two underscores" in text
    assert "starts with one underscore" in text
    assert "Star imports are forbidden" in text
    assert "to_frame, are allowed" in text
