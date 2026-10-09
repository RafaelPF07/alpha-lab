"""Typed pipeline events: persisted to the ledger and broadcast to live subscribers.

Emitting must never crash or noticeably slow the pipeline. Failures are logged
to stderr and dropped. Holdout data must never appear in a payload. A payload is
strict JSON: NaN and infinity become null when the event is built.
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from alpha_lab.ledger.db import Ledger, json_ready

EventType = Literal[
    "run_started",
    "pipeline_started",
    "idea_created",
    "idea_rejected_duplicate",
    "llm_call_started",
    "llm_call_refused",
    "llm_call_finished",
    "code_written",
    "duplicate_code",
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

# The first dropped event is reported, then one report per this many, so a full queue cannot flood stderr.
DROP_REPORT_EVERY = 1000


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

    @field_validator("payload", mode="after")
    @classmethod
    def _strict_payload(cls, payload: dict[str, Any]) -> dict[str, Any]:
        """NaN and infinity become null here, so no subscriber and no ledger row ever sees them."""
        return json_ready(payload)


class EventBus:
    def __init__(self, ledger: Ledger, queue_size: int = 10_000) -> None:
        self._ledger = ledger
        self._queue_size = queue_size
        self._subscribers: list[asyncio.Queue[PipelineEvent]] = []
        self._dropped = 0

    def subscribe(self) -> asyncio.Queue[PipelineEvent]:
        queue: asyncio.Queue[PipelineEvent] = asyncio.Queue(maxsize=self._queue_size)
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[PipelineEvent]) -> None:
        """Stop sending events to this queue. Events emitted from now on are not queued for it."""
        if queue in self._subscribers:
            self._subscribers.remove(queue)

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
                self._dropped += 1
                if self._dropped % DROP_REPORT_EVERY == 1:
                    print(
                        f"[events] subscriber queue full: {self._dropped} events dropped so far, "
                        f"latest {event.type}",
                        file=sys.stderr,
                    )
