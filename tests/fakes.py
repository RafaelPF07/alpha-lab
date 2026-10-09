"""A scripted stand-in for the Anthropic async client. No network, no API key."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from alpha_lab.config import Settings


def with_pricing(settings: Settings, model: str, **updates: Any) -> Settings:
    """A copy of `settings` with some prices of one model changed. For tests only."""
    models = dict(settings.pricing.models)
    models[model] = models[model].model_copy(update=updates)
    pricing = settings.pricing.model_copy(update={"models": models})
    return settings.model_copy(update={"pricing": pricing})


def with_test_cache_prices(settings: Settings) -> Settings:
    """The shipped settings with the cache prices filled in as TEST VALUES, so the test models are
    runnable. These are not Anthropic's prices. A test that checks a price sets its own value."""
    for model, price in settings.pricing.models.items():
        updates: dict[str, float] = {}
        if price.cache_write_5m is None:
            updates["cache_write_5m"] = 2.0 * price.input
        if price.input_over_100k is not None and price.cache_write_5m_over_100k is None:
            updates["cache_write_5m_over_100k"] = 2.0 * (price.cache_write_5m or price.input)
        if price.input_over_100k is not None and price.cache_read_over_100k is None:
            updates["cache_read_over_100k"] = 2.0 * price.cache_read
        if updates:
            settings = with_pricing(settings, model, **updates)
    return settings


def tool_response(
    name: str,
    tool_input: dict[str, Any],
    *,
    input_tokens: int = 1000,
    output_tokens: int = 200,
    cache_read: int = 0,
    cache_write: int = 0,
    tool_id: str = "toolu_test",
    stop_reason: str = "tool_use",
) -> SimpleNamespace:
    block = SimpleNamespace(type="tool_use", id=tool_id, name=name, input=tool_input)
    block.model_dump = lambda exclude_none=False: {  # mirrors the SDK block model
        "type": "tool_use",
        "id": tool_id,
        "name": name,
        "input": tool_input,
    }
    usage = SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_write,
    )
    return SimpleNamespace(content=[block], usage=usage, stop_reason=stop_reason)


def text_response(text: str = "Here is my answer in prose.") -> SimpleNamespace:
    """A reply with no tool call, which is what a model returns when it ignores the tool."""
    block = SimpleNamespace(type="text", text=text)
    block.model_dump = lambda exclude_none=False: {"type": "text", "text": text}
    usage = SimpleNamespace(
        input_tokens=1000,
        output_tokens=50,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )
    return SimpleNamespace(content=[block], usage=usage, stop_reason="end_turn")


class _Messages:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("fake client ran out of scripted responses")
        item = self._responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class FakeAnthropic:
    def __init__(self, responses: list[Any]) -> None:
        self.messages = _Messages(responses)


class RoutingAnthropic:
    """Replies by tool name, so concurrent pipelines get the right answer regardless of call order.

    Each tool has a queue of scripted replies. For submit_idea, an empty queue yields a
    new, unique idea, so swarms never accidentally produce duplicates unless a test asks.
    """

    def __init__(
        self, scripts: dict[str, list[Any]], idea_factory: Any = None, yield_once: bool = False
    ) -> None:
        self._scripts = {k: list(v) for k, v in scripts.items()}
        self._idea_factory = idea_factory
        self._counter = 0
        # yield_once makes each call suspend once before it answers, as a real network call does.
        # Without it, a pipeline runs to its next real suspension point without any interleaving.
        self._yield_once = yield_once
        self.calls: list[dict[str, Any]] = []
        self.messages = self

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._yield_once:
            await asyncio.sleep(0)
        tool = kwargs["tools"][0]["name"]
        queue = self._scripts.get(tool, [])
        if queue:
            item = queue.pop(0)
        elif tool == "submit_idea" and self._idea_factory is not None:
            self._counter += 1
            item = tool_response("submit_idea", self._idea_factory(self._counter))
        else:
            raise AssertionError(f"no scripted reply left for {tool}")
        if isinstance(item, BaseException):
            raise item
        return item
