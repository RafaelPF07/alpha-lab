"""The one client every agent uses. Cost tracking, budget guard, context guard, retries, logging.

Agents never call the SDK directly. This wrapper is the only place that sends
requests, which keeps the budget and the ledger honest.
"""

from __future__ import annotations

import asyncio
import json
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
CHARS_PER_TOKEN = 3.5  # conservative estimate, used instead of a count_tokens round trip
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


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int


def compute_cost(pricing: ModelPricing, usage: Usage, long_context_threshold: int) -> float:
    """USD for one call. Prompts above the threshold use the *_over_100k prices when given.

    Cache writes are charged at the base input rate, because settings.toml has no cache-write
    price. The documented 5-minute write price is higher (1.25 times the input price), so writes
    are under-counted until that price is configured.
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
    cost = (
        (usage.input_tokens + usage.cache_write_tokens) * input_price
        + usage.cache_read_tokens * pricing.cache_read
        + usage.output_tokens * output_price
    )
    return cost / 1_000_000


def estimate_tokens(texts: list[str]) -> int:
    return int(sum(len(t) for t in texts) / CHARS_PER_TOKEN) + 1


def _is_retryable(exc: Exception) -> bool:
    """Retry what the SDK retries: timeouts, conflicts, rate limits, 5xx and connection errors.

    As in the SDK, an x-should-retry header overrides the status code.
    """
    if isinstance(exc, anthropic.APIStatusError):
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
            response = await self._send(
                agent=agent,
                prompt=prompt,
                model=model,
                messages=messages,
                tool=tool,
                pipeline_id=pipeline_id,
                idea_id=idea_id,
            )
            if response.stop_reason in TRUNCATED_STOP_REASONS:
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
                last_problem = f"you did not call the {tool_name} tool"
                continue
            try:
                return output_model.model_validate(tool_block.input)
            except ValidationError as exc:
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
    ) -> Any:
        max_tokens = self.settings.agents.max_tokens_per_call
        system = [{"type": "text", "text": prompt.text, "cache_control": {"type": "ephemeral"}}]
        pricing = self.settings.pricing.models.get(model)
        if pricing is None:
            raise KeyError(f"no pricing configured for model {model!r}")

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

        est_cost = (est_input * pricing.input + max_tokens * pricing.output) / 1_000_000
        try:
            self.budget.reserve(est_cost)
        except BudgetExceeded as exc:
            self._refuse(call_id, agent, model, prompt, pipeline_id, idea_id, est_cost, str(exc))
            raise
        started = time.perf_counter()
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

        response = None
        usage: Usage | None = None
        status = "error"
        try:
            response = await self._call_with_retries(
                model=model, max_tokens=max_tokens, system=system, messages=messages, tool=tool
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
            if usage is None:
                # No usage report (an error or a cancellation). The request may still have been
                # processed, so the reservation is charged. This is the conservative choice.
                actual_cost = est_cost
            else:
                actual_cost = compute_cost(
                    pricing, usage, self.settings.pricing.long_context_threshold_tokens
                )
            self.budget.settle(est_cost, actual_cost)
            self.ledger.add_llm_call(
                call_id=call_id,
                run_id=self.run_id,
                idea_id=idea_id,
                agent=agent,
                model=model,
                prompt_version=prompt.name,
                input_tokens=usage.input_tokens if usage else None,
                output_tokens=usage.output_tokens if usage else None,
                cache_read_tokens=usage.cache_read_tokens if usage else None,
                cache_write_tokens=usage.cache_write_tokens if usage else None,
                cost_usd=actual_cost,
                latency_ms=latency_ms,
                status=status,
            )
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
        return response

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
        """Record a call that was never sent, so the refusal leaves a trace. It costs nothing."""
        self.ledger.add_llm_call(
            call_id=call_id,
            run_id=self.run_id,
            idea_id=idea_id,
            agent=agent,
            model=model,
            prompt_version=prompt.name,
            input_tokens=None,
            output_tokens=None,
            cache_read_tokens=None,
            cache_write_tokens=None,
            cost_usd=0.0,
            latency_ms=None,
            status="refused",
        )
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

    async def _call_with_retries(self, **kwargs: Any) -> Any:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                return await self.client.messages.create(
                    tool_choice=_tool_choice(kwargs["model"], kwargs["tool"]["name"]),
                    tools=[kwargs["tool"]],
                    model=kwargs["model"],
                    max_tokens=kwargs["max_tokens"],
                    system=kwargs["system"],
                    messages=list(kwargs["messages"]),  # snapshot: the caller keeps appending
                )
            except Exception as exc:
                if not _is_retryable(exc) or attempt == MAX_ATTEMPTS:
                    raise
                await self._sleep(_retry_delay(exc, attempt))
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
