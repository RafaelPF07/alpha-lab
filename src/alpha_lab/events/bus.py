"""Typed pipeline events: persisted to the ledger and broadcast to live subscribers.

Emitting must never crash or noticeably slow the pipeline. Failures are logged
to stderr and dropped. Holdout data must never appear in a payload.
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from alpha_lab.ledger.db import Ledger

EventType = Literal[
    "run_started",
    "pipeline_started",
    "idea_created",
    "idea_rejected_duplicate",
    "llm_call_started",
    "llm_call_finished",
    "code_written",
    "static_check",
    "sandbox_run",
    "lookahead_check",
    "train_backtest",
    "critic_review",
    "revision_requested",
    "validation_result",
    "pipeline_finished",
    "budget_update",
    "run_finished",
]


def _now_ms() -> datetime:
    now = datetime.now(UTC)
    return now.replace(microsecond=now.microsecond // 1000 * 1000)


class PipelineEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    run_id: str
    pipeline_id: str | None = None
    idea_id: str | None = None
    attempt_number: int | None = None
    agent: str | None = None
    stage: str
    type: EventType
    timestamp: datetime = Field(default_factory=_now_ms)
    payload: dict[str, Any] = Field(default_factory=dict)


class EventBus:
    def __init__(self, ledger: Ledger, queue_size: int = 10_000) -> None:
        self._ledger = ledger
        self._queue_size = queue_size
        self._subscribers: list[asyncio.Queue[PipelineEvent]] = []

    def subscribe(self) -> asyncio.Queue[PipelineEvent]:
        queue: asyncio.Queue[PipelineEvent] = asyncio.Queue(maxsize=self._queue_size)
        self._subscribers.append(queue)
        return queue

    def emit(self, event: PipelineEvent) -> None:
        try:
            self._ledger.add_event(
                event.event_id,
                event.run_id,
                event.pipeline_id,
                event.idea_id,
                event.attempt_number,
                event.agent,
                event.stage,
                event.type,
                event.timestamp.isoformat(timespec="milliseconds"),
                event.payload,
            )
        except Exception as exc:  # noqa: BLE001 - emitting must never crash the pipeline
            print(f"[events] failed to persist {event.type}: {exc}", file=sys.stderr)
        for queue in self._subscribers:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                print(f"[events] subscriber queue full, dropped {event.type}", file=sys.stderr)
