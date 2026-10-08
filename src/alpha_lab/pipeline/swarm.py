"""Run many pipelines concurrently under one shared budget."""

from __future__ import annotations

import asyncio
import random
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

from alpha_lab.agents.budget import Budget, BudgetExceeded
from alpha_lab.pipeline.single import PipelineContext, PipelineResult, run_single_pipeline


@dataclass
class SwarmProgress:
    total: int
    budget: Budget | None = None
    started_at: float = field(default_factory=time.monotonic)
    running: int = 0
    done: int = 0
    passed: int = 0
    statuses: Counter = field(default_factory=Counter)

    @property
    def spent_usd(self) -> float:
        return self.budget.spent_usd if self.budget else 0.0

    @property
    def spend_rate_per_min(self) -> float:
        minutes = max(time.monotonic() - self.started_at, 1e-6) / 60
        return self.spent_usd / minutes


def pick_themes(themes: list[str], n: int, seed: int) -> list[str]:
    """One theme per pipeline, reproducible from the run seed."""
    rng = random.Random(seed)
    return [rng.choice(themes) for _ in range(n)]


async def run_swarm(
    ctx: PipelineContext,
    budget: Budget,
    n_pipelines: int,
    max_concurrency: int,
    seed: int,
    progress: SwarmProgress | None = None,
    on_progress: Callable[[SwarmProgress], None] | None = None,
) -> list[PipelineResult]:
    """Start N pipelines, at most `max_concurrency` at a time. Stops starting new ones when the budget is gone."""
    themes = pick_themes(ctx.settings.diversity.themes, n_pipelines, seed)
    progress = progress or SwarmProgress(total=n_pipelines, budget=budget)
    semaphore = asyncio.Semaphore(max_concurrency)
    stop = asyncio.Event()
    results: list[PipelineResult] = []

    async def one(theme: str) -> None:
        if stop.is_set():
            return
        async with semaphore:
            if stop.is_set():
                return
            progress.running += 1
            try:
                result = await run_single_pipeline(ctx, theme)
            except BudgetExceeded:
                stop.set()  # no new pipelines start; in-flight ones stop at their next LLM call
                return
            finally:
                progress.running -= 1
            results.append(result)
            progress.done += 1
            progress.statuses[result.status] += 1
            if result.status in ("critic_pass", "validation_pass"):
                progress.passed += 1
            if on_progress:
                on_progress(progress)

    await asyncio.gather(*(one(theme) for theme in themes))
    return results
