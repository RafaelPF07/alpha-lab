"""A scripted stand-in for the Anthropic async client. No network, no API key."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def tool_response(
    name: str,
    tool_input: dict[str, Any],
    *,
    input_tokens: int = 1000,
    output_tokens: int = 200,
    cache_read: int = 0,
    cache_write: int = 0,
    tool_id: str = "toolu_test",
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
    return SimpleNamespace(content=[block], usage=usage, stop_reason="tool_use")


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
