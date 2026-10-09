"""Run orchestration: one run, one shared budget, N pipelines.

The run row always ends with a status. If the process dies first, a later run marks the row
'abandoned' (see Ledger.abandon_stale_runs).
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import pandas as pd

from alpha_lab.agents.budget import Budget
from alpha_lab.agents.llm import LLMClient
from alpha_lab.agents.prompts import load_prompt
from alpha_lab.config import Settings
from alpha_lab.events.bus import EventBus, PipelineEvent
from alpha_lab.ledger.db import Ledger
from alpha_lab.pipeline.single import PipelineContext, PipelineResult
from alpha_lab.pipeline.swarm import SwarmProgress, run_swarm

IDEAS_PROMPT = "ideas_v1"
CODER_PROMPT = "coder_v1"
CRITIC_PROMPT = "critic_v1"
# A live run writes events all the time. A run silent for this long has lost its process.
STALE_RUN_AFTER = timedelta(minutes=30)


@dataclass
class RunSummary:
    run_id: str
    status: str
    budget_usd: float
    total_cost_usd: float
    results: list[PipelineResult] = field(default_factory=list)

    @property
    def pipeline_statuses(self) -> Counter:
        return Counter(r.status for r in self.results)

    @property
    def train_passes(self) -> int:
        return sum(1 for r in self.results if r.status in ("critic_pass", "validation_pass"))


def _config_snapshot(settings: Settings) -> str:
    return json.dumps(
        {
            "settings": settings.model_dump(mode="json"),
            "prompts": {"ideas": IDEAS_PROMPT, "coder": CODER_PROMPT, "critic": CRITIC_PROMPT},
        },
        sort_keys=True,
    )


async def execute_run(
    *,
    run_id: str,
    settings: Settings,
    prices: pd.DataFrame,
    n_pipelines: int,
    budget_usd: float,
    ledger: Ledger,
    bus: EventBus,
    client: Any,
    on_progress: Callable[[SwarmProgress], None] | None = None,
    progress_sink: dict | None = None,
) -> RunSummary:
    """Run `n_pipelines` pipelines as a swarm under one budget."""
    # Agents get training data only. The validation gate gets history up to validation_end, and no
    # later. This function is the one place that cuts the frame, so holdout prices stay out of it.
    train_prices = prices.loc[: pd.Timestamp(settings.dates.train_end)]
    validation_prices = prices.loc[: pd.Timestamp(settings.dates.validation_end)]
    budget = Budget(budget_usd)
    ledger.abandon_stale_runs(STALE_RUN_AFTER)

    status = "failed"  # an error that nothing below handles ends the run as failed
    created = False
    total_cost = 0.0
    results: list[PipelineResult] = []
    try:
        ledger.create_run(run_id, _config_snapshot(settings), budget_usd)
        created = True
        bus.emit(
            PipelineEvent(
                run_id=run_id,
                stage="run",
                type="run_started",
                payload={
                    "pipelines": n_pipelines,
                    "budget_usd": budget_usd,
                    "concurrency": settings.run.max_concurrency,
                },
            )
        )

        llm = LLMClient(
            client=client, settings=settings, budget=budget, ledger=ledger, bus=bus, run_id=run_id
        )
        ctx = PipelineContext(
            run_id=run_id,
            settings=settings,
            train_prices=train_prices,
            validation_prices=validation_prices,
            llm=llm,
            ledger=ledger,
            bus=bus,
            idea_prompt=load_prompt(IDEAS_PROMPT),
            coder_prompt=load_prompt(CODER_PROMPT),
            critic_prompt=load_prompt(CRITIC_PROMPT),
            idea_lock=asyncio.Lock(),
        )

        progress = SwarmProgress(total=n_pipelines, budget=budget)
        if progress_sink is not None:
            progress_sink["progress"] = progress
        results = await run_swarm(
            ctx,
            budget,
            n_pipelines,
            settings.run.max_concurrency,
            settings.run.seed,
            progress=progress,
            on_progress=on_progress,
        )
        # A pipeline that the budget stopped returns no result, so fewer results means the budget ran out.
        status = "budget_stopped" if len(results) < n_pipelines else "completed"
    except Exception:
        status = "failed"
        raise
    except BaseException:  # KeyboardInterrupt, cancellation
        status = "interrupted"
        raise
    finally:
        if created:
            # Cost is the sum of the llm_calls rows: every call that was charged, failed or not.
            total_cost = ledger.llm_cost_total(run_id)
            ledger.finish_run(run_id, status, total_cost)
            bus.emit(
                PipelineEvent(
                    run_id=run_id,
                    stage="run",
                    type="run_finished",
                    payload={
                        "status": status,
                        "total_cost_usd": total_cost,
                        "budget_usd": budget_usd,
                        "reserved_usd": budget.reserved_usd,
                    },
                )
            )

    return RunSummary(
        run_id=run_id,
        status=status,
        budget_usd=budget_usd,
        total_cost_usd=total_cost,
        results=results,
    )
