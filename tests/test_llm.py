import asyncio
import json
import re
import sqlite3

import anthropic
import httpx2 as httpx
import pytest
from fakes import (
    FakeAnthropic,
    text_response,
    tool_response,
    with_pricing,
    with_test_cache_prices,
)
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
    missing_prices,
)
from alpha_lab.agents.models import CODER_CODE_MAX_CHARS
from alpha_lab.agents.prompts import load_prompt
from alpha_lab.checks.static import FORBIDDEN_BUILTINS, FORBIDDEN_IO_METHODS, FORBIDDEN_MODULE_NAMES
from alpha_lab.config import ModelPricing, load_settings
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
    return with_test_cache_prices(load_settings())


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


# Test prices only. They are chosen so each term is easy to check by hand, not Anthropic's prices.
def test_a_configured_cache_write_price_is_charged(settings):
    pricing = ModelPricing(input=1.0, output=1.0, cache_read=0.1, cache_write_5m=1.25)
    # 1,000,000 cache-write tokens at 1.25 per million
    assert compute_cost(pricing, Usage(0, 0, 0, 1_000_000), 200_000) == pytest.approx(1.25)


def test_long_context_cache_reads_use_their_configured_price(settings):
    pricing = ModelPricing(
        input=1.0,
        output=1.0,
        input_over_100k=2.0,
        output_over_100k=2.0,
        cache_read=0.1,
        cache_read_over_100k=0.5,
        cache_write_5m=1.25,
    )
    # The prompt is 150,000 tokens, over the threshold, so the over-100K cache-read price applies
    assert compute_cost(pricing, Usage(0, 0, 150_000, 0), 100_000) == pytest.approx(0.075)
    # Under the threshold the ordinary cache-read price applies
    assert compute_cost(pricing, Usage(0, 0, 50_000, 0), 100_000) == pytest.approx(0.005)


def test_long_context_cache_writes_use_their_own_price(settings):
    pricing = settings.pricing.models["claude-haiku-5-5"]
    # The shipped Haiku prices: 0.125 per million for 5-minute cache writes, 0.625 above 100K.
    # A 150,000-token prompt that writes its whole cache is billed at the over-100K write price.
    assert compute_cost(pricing, Usage(0, 0, 0, 150_000), 100_000) == pytest.approx(0.09375)
    assert compute_cost(pricing, Usage(0, 0, 0, 50_000), 100_000) == pytest.approx(0.00625)


def test_a_tier_without_its_long_context_write_price_is_reported_missing():
    pricing = ModelPricing(
        input=1.0,
        output=1.0,
        input_over_100k=2.0,
        output_over_100k=2.0,
        cache_read=0.1,
        cache_read_over_100k=0.5,
        cache_write_5m=1.25,
    )
    assert missing_prices(pricing) == ["cache_write_5m_over_100k"]


def test_a_cache_write_without_its_price_is_refused_not_costed_at_the_input_rate():
    pricing = ModelPricing(input=1.0, output=1.0, cache_read=0.1)
    with pytest.raises(ValueError, match="cache_write_5m"):
        compute_cost(pricing, Usage(0, 0, 0, 1000), 200_000)


def test_a_model_without_a_cache_write_price_is_refused_before_anything_is_sent(settings, ledger):
    priced_out = with_pricing(settings, "claude-haiku-5-5", cache_write_5m=None)

    async def scenario():
        llm, client = make_llm(
            priced_out, ledger, [tool_response("submit", {"name": "abc", "count": 1})]
        )
        with pytest.raises(ValueError, match="cache_write_5m"):
            await call(llm)
        assert client.messages.calls == []
        assert llm.budget.spent_usd == 0.0
        assert llm.budget.reserved_usd == 0.0

    asyncio.run(scenario())


def test_the_reservation_assumes_cache_writes_at_their_price(settings, ledger):
    # The input is about a few hundred tokens. At 10,000 per million for cache writes, the
    # reservation is several dollars, which a 0.50 budget must refuse before anything is sent.
    expensive = with_pricing(settings, "claude-haiku-5-5", cache_write_5m=10_000.0)

    async def scenario():
        llm, client = make_llm(expensive, ledger, [], budget_usd=0.5)
        with pytest.raises(BudgetExceeded):
            await call(llm)
        assert client.messages.calls == []

    asyncio.run(scenario())


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
    # A transport failure may have reached the server, so its reservation is charged (R4-AGT-03).
    # A 4xx answer is charged nothing, as the tests below check.
    async def scenario():
        no_answer = anthropic.APIConnectionError(request=REQUEST)
        llm, _ = make_llm(settings, ledger, [no_answer] * 4)
        with pytest.raises(anthropic.APIConnectionError):
            await call(llm)
        assert llm.budget.reserved_usd == pytest.approx(0.0)
        assert llm.budget.spent_usd > 0
        # The ledger total and the budget agree, so the run cost is the sum of the llm_calls rows.
        assert ledger.llm_cost_total("run1") == pytest.approx(llm.budget.spent_usd)

    asyncio.run(scenario())


def test_a_cancelled_call_is_recorded_as_cancelled(settings, ledger):
    async def scenario():
        in_flight = asyncio.Event()

        class HangingMessages:
            async def create(self, **kwargs):
                in_flight.set()  # the request is on the wire; nothing else is awaited
                await asyncio.Event().wait()

        class HangingClient:
            messages = HangingMessages()

        llm = LLMClient(
            client=HangingClient(),
            settings=settings,
            budget=Budget(5.0),
            ledger=ledger,
            bus=EventBus(ledger),
            run_id="run1",
        )
        task = asyncio.ensure_future(call(llm))
        await in_flight.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
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
        llm, client = make_llm(
            settings, ledger, [tool_response("submit", {"name": "abc", "count": 1})]
        )
        await call(llm, model="claude-haiku-5-5")
        assert client.messages.calls[0]["tool_choice"] == {
            "type": "tool",
            "name": "submit",
            "disable_parallel_tool_use": True,
        }

    asyncio.run(scenario())


def test_sonnet_is_sent_auto_tool_choice_without_parallel_tool_use(settings, ledger):
    async def scenario():
        llm, client = make_llm(
            settings, ledger, [tool_response("submit", {"name": "abc", "count": 1})]
        )
        await call(llm, model="claude-sonnet-5-5")
        assert client.messages.calls[0]["tool_choice"] == {
            "type": "auto",
            "disable_parallel_tool_use": True,
        }

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("cls", "status"),
    [(anthropic.AuthenticationError, 401), (anthropic.PermissionDeniedError, 403)],
)
def test_a_rejected_key_is_run_fatal_even_when_the_server_asks_for_a_retry(
    settings, ledger, cls, status
):
    # DECISION-10: no later call can succeed with a rejected key, so the header is not honoured.
    async def scenario():
        sleeps = []
        error = status_error(cls, status, headers={"x-should-retry": "true"})
        llm, client = make_llm(settings, ledger, [error], sleeps=sleeps)
        with pytest.raises(cls):
            await call(llm)
        assert len(client.messages.calls) == 1
        assert sleeps == []

    asyncio.run(scenario())


def test_each_llm_call_row_records_the_hash_of_the_prompt_text(settings, ledger):
    async def scenario():
        llm, _ = make_llm(settings, ledger, [tool_response("submit", {"name": "abc", "count": 1})])
        await call(llm)

    asyncio.run(scenario())
    [row] = ledger.conn.execute("SELECT prompt_version, prompt_sha256 FROM llm_calls").fetchall()
    assert row["prompt_version"] == "test_v1"
    assert row["prompt_sha256"] == PROMPT.sha256
    edited = PromptSpec(name="test_v1", text=PROMPT.text + " One more sentence.")
    assert edited.sha256 != PROMPT.sha256  # an edit without a rename shows


def test_each_retry_emits_a_heartbeat_event_before_it_waits(settings, ledger):
    def retry_events_written():
        return [
            json.loads(r["payload_json"])
            for r in ledger.events_for_run("run1")
            if r["type"] == "llm_call_retrying"
        ]

    # At each wait, record how many heartbeats are already in the ledger (R4-AGT-06).
    waits = []

    async def recording_sleep(seconds):
        waits.append((seconds, len(retry_events_written())))

    responses = [
        rate_limit({"retry-after": "7"}),
        rate_limit(),
        tool_response("submit", {"name": "abc", "count": 2}),
    ]
    llm = LLMClient(
        client=FakeAnthropic(responses),
        settings=settings,
        budget=Budget(5.0),
        ledger=ledger,
        bus=EventBus(ledger),
        run_id="run1",
        sleep=recording_sleep,
    )

    asyncio.run(call(llm))

    # Each wait comes after its heartbeat: the first wait sees one event, the second sees two.
    assert waits == [(7.0, 1), (BACKOFF_BASE_SECONDS * 2, 2)]
    retries = retry_events_written()
    assert [p["attempt"] for p in retries] == [1, 2]
    assert [p["delay_seconds"] for p in retries] == [7.0, BACKOFF_BASE_SECONDS * 2]


def test_a_call_cancelled_during_backoff_is_charged_nothing(settings, ledger):
    # No request is in flight during the wait, so the server cannot have billed one.
    async def cancel_during_the_wait(seconds):
        raise asyncio.CancelledError()

    async def scenario():
        llm = LLMClient(
            client=FakeAnthropic([rate_limit()]),
            settings=settings,
            budget=Budget(5.0),
            ledger=ledger,
            bus=EventBus(ledger),
            run_id="run1",
            sleep=cancel_during_the_wait,
        )
        with pytest.raises(asyncio.CancelledError):
            await call(llm)
        assert llm.budget.reserved_usd == pytest.approx(0.0)
        assert llm.budget.spent_usd == pytest.approx(0.0)

    asyncio.run(scenario())
    rows = ledger.conn.execute("SELECT status, cost_usd FROM llm_calls ORDER BY rowid").fetchall()
    assert [(r["status"], r["cost_usd"]) for r in rows] == [("cancelled_before_send", 0.0)]


def test_a_rejected_key_is_charged_nothing(settings, ledger):
    # The server answered 401, so it did not process the request (R4-AGT-03).
    async def scenario():
        llm, _ = make_llm(settings, ledger, [status_error(anthropic.AuthenticationError, 401)])
        with pytest.raises(anthropic.AuthenticationError):
            await call(llm)
        assert llm.budget.spent_usd == 0.0
        assert llm.budget.reserved_usd == 0.0

    asyncio.run(scenario())
    rows = ledger.conn.execute("SELECT status, cost_usd FROM llm_calls").fetchall()
    assert [(r["status"], r["cost_usd"]) for r in rows] == [("error", 0.0)]


def test_a_final_rate_limit_answer_is_charged_nothing(settings, ledger):
    async def scenario():
        llm, _ = make_llm(settings, ledger, [rate_limit()] * 4)
        with pytest.raises(anthropic.RateLimitError):
            await call(llm)
        assert llm.budget.spent_usd == 0.0

    asyncio.run(scenario())


def test_a_server_error_answer_keeps_its_reservation(settings, ledger):
    # A 5xx may come after the server processed the request, so the reservation stays charged.
    async def scenario():
        llm, _ = make_llm(settings, ledger, [status_error(anthropic.InternalServerError, 500)] * 4)
        with pytest.raises(anthropic.InternalServerError):
            await call(llm)
        assert llm.budget.spent_usd > 0.0
        assert llm.budget.reserved_usd == pytest.approx(0.0)

    asyncio.run(scenario())


def test_a_refusal_that_cannot_be_recorded_still_raises_the_budget_refusal(
    settings, ledger, monkeypatch, capsys
):
    def broken_write(**fields):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(ledger, "add_llm_call", broken_write)

    async def scenario():
        llm, client = make_llm(settings, ledger, [], budget_usd=0.0001)
        with pytest.raises(BudgetExceeded):
            await call(llm)
        assert client.messages.calls == []

    asyncio.run(scenario())
    assert "could not record" in capsys.readouterr().err


def test_a_truncated_reply_is_recorded_as_discarded_and_is_not_a_call(settings, ledger):
    async def scenario():
        reply = tool_response("submit", {"name": "abc", "count": 1}, stop_reason="max_tokens")
        llm, _ = make_llm(settings, ledger, [reply])
        with pytest.raises(OutputTruncated):
            await call(llm)

    asyncio.run(scenario())
    rows = ledger.conn.execute("SELECT status, cost_usd FROM llm_calls").fetchall()
    assert [r["status"] for r in rows] == ["discarded"]
    assert rows[0]["cost_usd"] > 0  # the reply was billed, so it still costs
    [agent] = ledger.llm_cost_by_agent("run1")
    assert agent["calls"] == 0


def test_a_reply_without_the_tool_is_discarded_and_the_repair_is_the_only_call(settings, ledger):
    async def scenario():
        llm, _ = make_llm(
            settings,
            ledger,
            [text_response(), tool_response("submit", {"name": "abc", "count": 1})],
        )
        assert (await call(llm)).count == 1

    asyncio.run(scenario())
    rows = ledger.conn.execute("SELECT status FROM llm_calls ORDER BY rowid").fetchall()
    assert [r["status"] for r in rows] == ["discarded", "ok"]
    [agent] = ledger.llm_cost_by_agent("run1")
    assert agent["calls"] == 1


def test_a_failed_ledger_write_keeps_the_call_result_and_releases_the_reservation(
    settings, ledger, capsys, monkeypatch
):
    def broken_write(**fields):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(ledger, "add_llm_call", broken_write)

    async def scenario():
        llm, _ = make_llm(settings, ledger, [tool_response("submit", {"name": "abc", "count": 1})])
        result = await call(llm)
        assert result.count == 1
        assert llm.budget.reserved_usd == pytest.approx(0.0)
        assert llm.budget.spent_usd > 0

    asyncio.run(scenario())
    assert "could not record" in capsys.readouterr().err
    finished = [
        json.loads(r["payload_json"])
        for r in ledger.events_for_run("run1")
        if r["type"] == "llm_call_finished"
    ]
    assert finished[0]["recorded"] is False


def test_a_call_that_overshoots_the_budget_is_reported_and_the_next_is_refused(settings, ledger):
    async def scenario():
        llm, _ = make_llm(
            settings,
            ledger,
            [tool_response("submit", {"name": "abc", "count": 1}, input_tokens=100_000)],
            budget_usd=0.003,
        )
        await call(llm)  # the reservation fits; the billed usage does not
        assert llm.budget.spent_usd > llm.budget.limit_usd
        with pytest.raises(BudgetExceeded):
            await call(llm)
        assert len(llm.client.messages.calls) == 1

    asyncio.run(scenario())
    types = [row["type"] for row in ledger.events_for_run("run1")]
    assert "budget_overrun" in types


def test_coder_output_cap_fits_the_token_budget(settings):
    # Python is about 3 characters per token. The longest allowed program should fit in 90% of the
    # reply budget, which leaves room for the JSON wrapper.
    assert CODER_CODE_MAX_CHARS / 3 <= 0.9 * settings.agents.max_tokens_per_call


def test_coder_prompt_states_every_rule_the_checker_enforces():
    text = load_prompt("coder_v2").text
    for name in sorted(FORBIDDEN_BUILTINS | FORBIDDEN_MODULE_NAMES | FORBIDDEN_IO_METHODS):
        assert re.search(rf"(?<![\w.]){re.escape(name)}(?!\w)", text), f"prompt lacks {name}"
    assert "starts with `read_`" in text
    assert "starts with two underscores" in text
    assert "starts with one underscore" in text
    assert "Star imports are forbidden" in text
    assert "to_frame, are allowed" in text
