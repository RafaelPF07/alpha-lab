"""The one client every agent uses. Cost tracking, budget guard, context guard, retries, logging.

Agents never call the SDK directly. This wrapper is the only place that sends
requests, which keeps the budget and the ledger honest.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

import anthropic
from pydantic import BaseModel, ValidationError

from alpha_lab.agents.budget import Budget, BudgetExceeded
from alpha_lab.config import ModelPricing, Settings
from alpha_lab.events.bus import EventBus, PipelineEvent
from alpha_lab.ledger.db import Ledger

MAX_ATTEMPTS = 4

# claude-sonnet-5-5 rejected a forced tool choice with a 400 error in a live critic call (commit
# 23cd375). Those models are asked to use the tool, and the one-repair logic handles a miss.
MODELS_WITHOUT_FORCED_TOOL_CHOICE = frozenset({"claude-sonnet-5-5"})


def _tool_choice(model: str, tool_name: str) -> dict[str, Any]:
    """The tool_choice sent with every request. Parallel tool use is off, so a reply holds at most
    one tool call, and the repair message can answer it. disable_parallel_tool_use is a field of
    the SDK's ToolChoiceAutoParam and ToolChoiceToolParam."""
    if model in MODELS_WITHOUT_FORCED_TOOL_CHOICE:
        return {"type": "auto", "disable_parallel_tool_use": True}
    return {"type": "tool", "name": tool_name, "disable_parallel_tool_use": True}


BACKOFF_BASE_SECONDS = 1.0
# A retry-after from the server is honoured, but never for longer than this.
MAX_RETRY_AFTER_SECONDS = 60.0
# The SDK's own list. 5xx and 529 (overloaded) are retried as well.
RETRYABLE_STATUS_CODES = frozenset({408, 409, 429})
# DECISION-10: a rejected key or a forbidden request cannot succeed on a later call, so it is never
# retried, even when the server's x-should-retry header says so. The pipeline layer ends the run.
RUN_FATAL_STATUS_CODES = frozenset({401, 403})
# A rough character-to-token ratio for budget reservation, not a tokenizer count (R3-AGT-09).
CHARS_PER_TOKEN = 3.5
# stop_reason values that mean the reply was cut off before it was complete.
TRUNCATED_STOP_REASONS = frozenset({"max_tokens", "model_context_window_exceeded"})


class ContextLimitExceeded(ValueError):
    """The request is too large for context_limit_tokens. Raised instead of truncating."""


class StepFailed(RuntimeError):
    """The model did not return valid output, even after one repair attempt."""


class OutputTruncated(StepFailed):
    """The reply hit max_tokens before the tool call was complete. Not repaired, because the same
    request would be cut off again."""


@dataclass(frozen=True)
class PromptSpec:
    name: str  # e.g. "ideas_v1"
    text: str  # the static system prompt, cached across calls

    @property
    def sha256(self) -> str:
        """The hash of the prompt text. Each llm_calls row stores it, so an edit without a rename shows (WHOLE-15)."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int


class MissingPrice(ValueError):
    """The model's price table lacks a price that a call needs. Nothing is sent, and no price is guessed."""


def missing_prices(pricing: ModelPricing) -> list[str]:
    """The configured prices that a call to this model can need and that are not set.

    Every request carries a cache breakpoint (system block with cache_control), so a cache write can
    happen on any call, and cache_write_5m is always needed. A model with a long-context tier
    (input_over_100k set) also needs cache_write_5m_over_100k and cache_read_over_100k, so the tier
    is priced in full. A model with no tier reads its cache at the ordinary cache_read price, as it
    always has.
    """
    missing = []
    if pricing.cache_write_5m is None:
        missing.append("cache_write_5m")
    if pricing.input_over_100k is not None:
        if pricing.cache_write_5m_over_100k is None:
            missing.append("cache_write_5m_over_100k")
        if pricing.cache_read_over_100k is None:
            missing.append("cache_read_over_100k")
    return missing


def compute_cost(pricing: ModelPricing, usage: Usage, long_context_threshold: int) -> float:
    """USD for one call. Prompts above the threshold use the *_over_100k prices when given.

    Cache writes are billed at the 5-minute write price, because no request asks for a longer cache
    lifetime. Above the threshold they use cache_write_5m_over_100k when the model has a long-context
    tier, and a cache read uses cache_read_over_100k. A token class that the call used but the table
    does not price raises MissingPrice, instead of being costed at another rate.
    """
    prompt_size = usage.input_tokens + usage.cache_read_tokens + usage.cache_write_tokens
    over = prompt_size > long_context_threshold
    input_price = (
        pricing.input_over_100k if over and pricing.input_over_100k is not None else pricing.input
    )
    output_price = (
        pricing.output_over_100k
        if over and pricing.output_over_100k is not None
        else pricing.output
    )
    cache_write_cost = 0.0
    if usage.cache_write_tokens:
        write_field = "cache_write_5m"
        write_price = pricing.cache_write_5m
        if over and pricing.input_over_100k is not None:
            write_field = "cache_write_5m_over_100k"
            write_price = pricing.cache_write_5m_over_100k
        if write_price is None:
            raise MissingPrice(f"this call wrote the cache, but {write_field} is not configured")
        cache_write_cost = usage.cache_write_tokens * write_price
    cache_read_cost = 0.0
    if usage.cache_read_tokens:
        read_price = pricing.cache_read
        if over and pricing.input_over_100k is not None:
            read_price = pricing.cache_read_over_100k
        if read_price is None:
            raise MissingPrice(
                "this call read a long-context cache, but cache_read_over_100k is not configured"
            )
        cache_read_cost = usage.cache_read_tokens * read_price
    cost = (
        usage.input_tokens * input_price
        + cache_write_cost
        + cache_read_cost
        + usage.output_tokens * output_price
    )
    return cost / 1_000_000


def estimate_tokens(texts: list[str]) -> int:
    return int(sum(len(t) for t in texts) / CHARS_PER_TOKEN) + 1


def _is_retryable(exc: Exception) -> bool:
    """Retry what the SDK retries: timeouts, conflicts, rate limits, 5xx and connection errors.

    As in the SDK, an x-should-retry header overrides the status code, except for the run-fatal
    codes (DECISION-10), which are never retried.
    """
    if isinstance(exc, anthropic.APIStatusError):
        if exc.status_code in RUN_FATAL_STATUS_CODES:
            return False
        hint = exc.response.headers.get("x-should-retry")
        if hint == "false":
            return False
        if hint == "true":
            return True
        return exc.status_code in RETRYABLE_STATUS_CODES or exc.status_code >= 500
    return isinstance(exc, anthropic.APIConnectionError)  # includes timeouts


def _retry_delay(exc: Exception, attempt: int) -> float:
    """Seconds before the next attempt: the server's retry-after when it sent one, else backoff."""
    if isinstance(exc, anthropic.APIStatusError):
        requested = _retry_after(exc.response.headers)
        if requested is not None and requested > 0:
            return min(requested, MAX_RETRY_AFTER_SECONDS)
    return BACKOFF_BASE_SECONDS * 2 ** (attempt - 1)


def _retry_after(headers: Mapping[str, str]) -> float | None:
    """Seconds from retry-after-ms or retry-after. The HTTP-date form is not parsed, so backoff applies."""
    for name, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
        raw = headers.get(name)
        if raw is None:
            continue
        try:
            return float(raw) * scale
        except ValueError:
            continue
    return None


def _describe_errors(exc: ValidationError) -> str:
    """One line per problem, naming the field and the rule. The rejected input is left out, so the
    repair message never repeats the model's output."""
    parts = []
    for error in exc.errors(include_url=False, include_input=False, include_context=False):
        where = ".".join(str(part) for part in error["loc"]) or "input"
        parts.append(f"{where}: {error['msg']}")
    return "; ".join(parts)


@dataclass
class _Flight:
    """Whether a request is on the wire, and the status of the last answered error. A call cancelled
    while waiting to retry has none in flight. An answered error says whether the server replied."""

    in_flight: bool = False
    answered_status: int | None = None


class LLMClient:
    def __init__(
        self,
        *,
        client: Any,
        settings: Settings,
        budget: Budget,
        ledger: Ledger,
        bus: EventBus,
        run_id: str,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    ) -> None:
        self.client = client
        self.settings = settings
        self.budget = budget
        self.ledger = ledger
        self.bus = bus
        self.run_id = run_id
        self._sleep = sleep

    async def structured_call(
        self,
        *,
        agent: str,
        prompt: PromptSpec,
        model: str,
        user_text: str,
        output_model: type[BaseModel],
        tool_name: str,
        tool_description: str,
        pipeline_id: str | None = None,
        idea_id: str | None = None,
    ) -> BaseModel:
        """Ask the model to call one tool whose input matches `output_model`. One repair attempt."""
        tool = {
            "name": tool_name,
            "description": tool_description,
            "input_schema": output_model.model_json_schema(),
        }
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_text}]
        last_problem = ""
        for repair in (False, True):
            if repair and messages[-1]["role"] == "assistant":
                # No tool call was made, so the conversation ends on the assistant turn.
                messages.append(
                    {
                        "role": "user",
                        "content": f"Call the {tool_name} tool. Problem: {last_problem}",
                    }
                )
            response, call_id = await self._send(
                agent=agent,
                prompt=prompt,
                model=model,
                messages=messages,
                tool=tool,
                pipeline_id=pipeline_id,
                idea_id=idea_id,
            )
            if response.stop_reason in TRUNCATED_STOP_REASONS:
                self._mark_discarded(call_id)
                raise OutputTruncated(
                    f"{agent} reply was cut off at max_tokens="
                    f"{self.settings.agents.max_tokens_per_call} before the tool call was complete. "
                    "It was not repaired, because the same request would be cut off again."
                )
            messages.append({"role": "assistant", "content": _blocks_to_dicts(response.content)})
            tool_block = next(
                (b for b in response.content if b.type == "tool_use" and b.name == tool_name),
                None,
            )
            if tool_block is None:
                self._mark_discarded(call_id)
                last_problem = f"you did not call the {tool_name} tool"
                continue
            try:
                return output_model.model_validate(tool_block.input)
            except ValidationError as exc:
                self._mark_discarded(call_id)
                last_problem = f"the tool input failed validation: {_describe_errors(exc)}"
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_block.id,
                                "content": last_problem,
                                "is_error": True,
                            }
                        ],
                    }
                )
        raise StepFailed(f"{agent} output invalid after one repair: {last_problem}")

    async def _send(
        self,
        *,
        agent: str,
        prompt: PromptSpec,
        model: str,
        messages: list[dict[str, Any]],
        tool: dict[str, Any],
        pipeline_id: str | None,
        idea_id: str | None,
    ) -> tuple[Any, str]:
        """Send one request and record it. Returns the reply and the call_id of its llm_calls row."""
        max_tokens = self.settings.agents.max_tokens_per_call
        system = [{"type": "text", "text": prompt.text, "cache_control": {"type": "ephemeral"}}]
        pricing = self.settings.pricing.models.get(model)
        if pricing is None:
            raise KeyError(f"no pricing configured for model {model!r}")
        absent = missing_prices(pricing)
        if absent:
            # Checked before anything is reserved or sent, so no money is spent on an unpriced call.
            raise MissingPrice(
                f"no price for {', '.join(absent)} on model {model!r} in [pricing.models]; "
                "add the value from the pricing page before this model is run"
            )

        call_id = uuid.uuid4().hex
        input_text = [prompt.text, json.dumps(messages, default=str), json.dumps(tool)]
        est_input = estimate_tokens(input_text)
        if est_input > self.settings.agents.context_limit_tokens:
            reason = (
                f"{agent} request is about {est_input} tokens, over the limit of "
                f"{self.settings.agents.context_limit_tokens}; refusing to truncate"
            )
            self._refuse(call_id, agent, model, prompt, pipeline_id, idea_id, None, reason)
            raise ContextLimitExceeded(reason)

        # The reservation is a worst case. Every input token might be a cache write, which costs more
        # than a plain input token, and a prompt above the threshold uses the long-context prices.
        threshold = self.settings.pricing.long_context_threshold_tokens
        long_context = est_input > threshold and pricing.input_over_100k is not None
        write_rate = pricing.cache_write_5m or 0.0
        if long_context and pricing.cache_write_5m_over_100k is not None:
            write_rate = pricing.cache_write_5m_over_100k
        input_rate = max(pricing.input_over_100k if long_context else pricing.input, write_rate)
        output_rate = pricing.output
        if long_context and pricing.output_over_100k is not None:
            output_rate = pricing.output_over_100k
        est_cost = (est_input * input_rate + max_tokens * output_rate) / 1_000_000
        try:
            self.budget.reserve(est_cost)
        except BudgetExceeded as exc:
            self._refuse(call_id, agent, model, prompt, pipeline_id, idea_id, est_cost, str(exc))
            raise
        started = time.perf_counter()
        flight = _Flight()
        response = None
        usage: Usage | None = None
        status = "error"
        try:
            # Inside the try, so the reservation is released whatever happens before the request.
            self._emit(
                "llm_call_started",
                "llm",
                agent,
                pipeline_id,
                idea_id,
                {
                    "call_id": call_id,
                    "model": model,
                    "prompt_version": prompt.name,
                    "estimated_input_tokens": est_input,
                },
            )
            response = await self._call_with_retries(
                flight,
                call_id=call_id,
                agent=agent,
                pipeline_id=pipeline_id,
                idea_id=idea_id,
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
                tool=tool,
            )
            usage = Usage(
                input_tokens=response.usage.input_tokens or 0,
                output_tokens=response.usage.output_tokens or 0,
                cache_read_tokens=response.usage.cache_read_input_tokens or 0,
                cache_write_tokens=response.usage.cache_creation_input_tokens or 0,
            )
            status = "ok"
        except asyncio.CancelledError:
            status = "cancelled"
            raise
        finally:
            latency_ms = int((time.perf_counter() - started) * 1000)
            if usage is not None:
                actual_cost = compute_cost(
                    pricing, usage, self.settings.pricing.long_context_threshold_tokens
                )
            elif status == "cancelled" and not flight.in_flight:
                # Cancelled while it waited to retry. No request was in flight, so nothing was billed.
                status = "cancelled_before_send"
                actual_cost = 0.0
            elif flight.answered_status is not None and flight.answered_status < 500:
                # A 4xx answer means the server rejected the request, so nothing was processed or
                # billed (R4-AGT-03). This assumes the API bills no 4xx response; check the pricing page.
                actual_cost = 0.0
            else:
                # A transport error, a 5xx answer or a cancellation with a request in flight. The
                # server may have processed the request, so the reservation is charged.
                actual_cost = est_cost
            recorded = self._record_call(
                call_id=call_id,
                agent=agent,
                model=model,
                prompt_version=prompt.name,
                prompt_sha256=prompt.sha256,
                idea_id=idea_id,
                usage=usage,
                cost_usd=actual_cost,
                latency_ms=latency_ms,
                status=status,
            )
            self.budget.settle(est_cost, actual_cost)
            self._emit(
                "llm_call_finished",
                "llm",
                agent,
                pipeline_id,
                idea_id,
                {
                    "call_id": call_id,
                    "status": status,
                    "cost_usd": actual_cost,
                    "latency_ms": latency_ms,
                    "input_tokens": usage.input_tokens if usage else None,
                    "output_tokens": usage.output_tokens if usage else None,
                    # False when the ledger write failed. The spend is still counted in the budget.
                    "recorded": recorded,
                },
            )
            self._emit(
                "budget_update",
                "budget",
                agent,
                pipeline_id,
                idea_id,
                {"spent_usd": self.budget.spent_usd, "limit_usd": self.budget.limit_usd},
            )
            if self.budget.spent_usd > self.budget.limit_usd:
                # The reservation fitted but the billed usage did not. Later calls are refused by
                # Budget.reserve, so the overshoot is bounded by this one call.
                self._emit(
                    "budget_overrun",
                    "budget",
                    agent,
                    pipeline_id,
                    idea_id,
                    {"spent_usd": self.budget.spent_usd, "limit_usd": self.budget.limit_usd},
                )
        return response, call_id

    def _record_call(
        self,
        *,
        call_id: str,
        agent: str,
        model: str,
        prompt_version: str,
        prompt_sha256: str,
        idea_id: str | None,
        usage: Usage | None,
        cost_usd: float,
        latency_ms: int,
        status: str,
    ) -> bool:
        """Write the llm_calls row. A failed write is reported and not raised, so it cannot replace
        the call's result. Returns False when the row was not written."""
        try:
            self.ledger.add_llm_call(
                call_id=call_id,
                run_id=self.run_id,
                idea_id=idea_id,
                agent=agent,
                model=model,
                prompt_version=prompt_version,
                prompt_sha256=prompt_sha256,
                input_tokens=usage.input_tokens if usage else None,
                output_tokens=usage.output_tokens if usage else None,
                cache_read_tokens=usage.cache_read_tokens if usage else None,
                cache_write_tokens=usage.cache_write_tokens if usage else None,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
                status=status,
            )
        except Exception as exc:  # noqa: BLE001 - the call's result matters more than its row
            print(
                f"[llm] could not record call {call_id} ({agent}, ${cost_usd:.6f}): {exc}",
                file=sys.stderr,
            )
            return False
        return True

    def _mark_discarded(self, call_id: str) -> None:
        """The reply was billed but the harness cannot use it. The row keeps its cost, and it is not
        counted as a successful call (R4-AGT-12). A failed write is reported and not raised."""
        try:
            self.ledger.set_llm_call_status(call_id, "discarded")
        except Exception as exc:  # noqa: BLE001 - the repair or the failure matters more than the label
            print(f"[llm] could not mark call {call_id} discarded: {exc}", file=sys.stderr)

    def _refuse(
        self,
        call_id: str,
        agent: str,
        model: str,
        prompt: PromptSpec,
        pipeline_id: str | None,
        idea_id: str | None,
        estimated_cost: float | None,
        reason: str,
    ) -> None:
        """Record a call that was never sent, so the refusal leaves a trace. It costs nothing. A failed
        write is reported and not raised, so the refusal itself is what the caller sees (R4-AGT-02)."""
        try:
            self.ledger.add_llm_call(
                call_id=call_id,
                run_id=self.run_id,
                idea_id=idea_id,
                agent=agent,
                model=model,
                prompt_version=prompt.name,
                prompt_sha256=prompt.sha256,
                input_tokens=None,
                output_tokens=None,
                cache_read_tokens=None,
                cache_write_tokens=None,
                cost_usd=0.0,
                latency_ms=None,
                status="refused",
            )
        except Exception as exc:  # noqa: BLE001 - the refusal matters more than its row
            print(f"[llm] could not record refusal {call_id} ({agent}): {exc}", file=sys.stderr)
        self._emit(
            "llm_call_refused",
            "llm",
            agent,
            pipeline_id,
            idea_id,
            {
                "call_id": call_id,
                "model": model,
                "reason": reason,
                "estimated_cost_usd": estimated_cost,
            },
        )

    async def _call_with_retries(
        self,
        flight: _Flight,
        *,
        call_id: str,
        agent: str,
        pipeline_id: str | None,
        idea_id: str | None,
        **kwargs: Any,
    ) -> Any:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            flight.in_flight = True
            flight.answered_status = (
                None  # a cancelled request has no answer from an earlier attempt
            )
            try:
                response = await self.client.messages.create(
                    tool_choice=_tool_choice(kwargs["model"], kwargs["tool"]["name"]),
                    tools=[kwargs["tool"]],
                    model=kwargs["model"],
                    max_tokens=kwargs["max_tokens"],
                    system=kwargs["system"],
                    messages=list(kwargs["messages"]),  # snapshot: the caller keeps appending
                )
            except Exception as exc:
                # The request was answered with an error, so nothing is in flight during the wait.
                flight.in_flight = False
                flight.answered_status = (
                    exc.status_code if isinstance(exc, anthropic.APIStatusError) else None
                )
                if not _is_retryable(exc) or attempt == MAX_ATTEMPTS:
                    raise
                delay = _retry_delay(exc, attempt)
                # A heartbeat before each wait, so a live run never looks silent while it backs off.
                self._emit(
                    "llm_call_retrying",
                    "llm",
                    agent,
                    pipeline_id,
                    idea_id,
                    {
                        "call_id": call_id,
                        "attempt": attempt,
                        "delay_seconds": delay,
                        "error_type": type(exc).__name__,
                    },
                )
                await self._sleep(delay)
                continue
            flight.in_flight = False
            return response
        raise AssertionError("unreachable")

    def _emit(
        self,
        event_type: str,
        stage: str,
        agent: str | None,
        pipeline_id: str | None,
        idea_id: str | None,
        payload: dict[str, Any],
    ) -> None:
        self.bus.emit(
            PipelineEvent(
                run_id=self.run_id,
                pipeline_id=pipeline_id,
                idea_id=idea_id,
                agent=agent,
                stage=stage,
                type=event_type,  # type: ignore[arg-type]
                payload=payload,
            )
        )


def _blocks_to_dicts(blocks: list[Any]) -> list[dict[str, Any]]:
    return [b.model_dump(exclude_none=True) for b in blocks]
