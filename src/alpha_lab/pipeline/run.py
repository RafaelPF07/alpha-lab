"""Run orchestration: one run, a shared budget, N pipelines, a ledger row that always ends."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from alpha_lab.agents.budget import Budget, BudgetExceeded
from alpha_lab.agents.llm import LLMClient
from alpha_lab.agents.prompts import load_prompt
from alpha_lab.config import Settings
from alpha_lab.events.bus import EventBus, PipelineEvent
from alpha_lab.ledger.db import Ledger
from alpha_lab.pipeline.single import PipelineContext, PipelineResult, run_single_pipeline

IDEAS_PROMPT = "ideas_v1"
CODER_PROMPT = "coder_v1"


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
        return sum(1 for r in self.results if r.status == "train_pass")


def _config_snapshot(settings: Settings) -> str:
    return json.dumps(
        {
            "settings": settings.model_dump(mode="json"),
            "prompts": {"ideas": IDEAS_PROMPT, "coder": CODER_PROMPT},
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
) -> RunSummary:
    """Run `n_pipelines` pipelines one after another. Phase 2 replaces this with a swarm."""
    # Agents see training data only. Validation and holdout prices never leave this function.
    train_prices = prices.loc[: pd.Timestamp(settings.dates.train_end)]
    budget = Budget(budget_usd)
    ledger.create_run(run_id, _config_snapshot(settings), budget_usd)
    bus.emit(
        PipelineEvent(
            run_id=run_id,
            stage="run",
            type="run_started",
            payload={"pipelines": n_pipelines, "budget_usd": budget_usd},
        )
    )

    llm = LLMClient(
        client=client, settings=settings, budget=budget, ledger=ledger, bus=bus, run_id=run_id
    )
    ctx = PipelineContext(
        run_id=run_id,
        settings=settings,
        train_prices=train_prices,
        llm=llm,
        ledger=ledger,
        bus=bus,
        idea_prompt=load_prompt(IDEAS_PROMPT),
        coder_prompt=load_prompt(CODER_PROMPT),
    )

    results: list[PipelineResult] = []
    status = "completed"
    try:
        for _ in range(n_pipelines):
            results.append(await run_single_pipeline(ctx))
    except BudgetExceeded:
        status = "budget_stopped"
    except asyncio.CancelledError:
        status = "interrupted"
        raise
    finally:
        ledger.finish_run(run_id, status, budget.spent_usd)
        bus.emit(
            PipelineEvent(
                run_id=run_id,
                stage="run",
                type="run_finished",
                payload={"status": status, "total_cost_usd": budget.spent_usd},
            )
        )

    return RunSummary(
        run_id=run_id,
        status=status,
        budget_usd=budget_usd,
        total_cost_usd=budget.spent_usd,
        results=results,
    )
