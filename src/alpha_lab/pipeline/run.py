"""Run orchestration: one run, one shared budget, N pipelines.

The run row always ends with a status. If the process dies first, a later run marks the row
'abandoned' and closes its open pipelines (see Ledger.abandon_stale_runs). The status names are
in alpha_lab.statuses (DECISION-13).
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import pandas as pd

from alpha_lab.agents.budget import Budget
from alpha_lab.agents.llm import LLMClient, missing_prices
from alpha_lab.agents.prompts import load_prompt
from alpha_lab.config import Settings
from alpha_lab.events.bus import EventBus, PipelineEvent
from alpha_lab.ledger.db import Ledger
from alpha_lab.pipeline.single import PipelineContext, PipelineResult
from alpha_lab.pipeline.swarm import SwarmProgress, run_swarm
from alpha_lab.statuses import (
    PASSED_CRITIC,
    PASSED_VALIDATION,
    RUN_FAILED_DATA,
    RUN_FAILING_PIPELINE_STATUSES,
)

IDEAS_PROMPT = "ideas_v1"
CODER_PROMPT = "coder_v2"
CRITIC_PROMPT = "critic_v1"
# A live run writes events all the time, so a run silent for this long has lost its process. The
# longest silence a live run can have is one API attempt, plus at most MAX_RETRY_AFTER_SECONDS of
# waiting. One attempt can last the SDK's timeout of 600 s, but only if the SDK does not retry on
# its own. cli.py builds the client with max_retries=0, and LLMClient does the retrying, so there
# is one attempt of at most 600 s. Each retry writes llm_call_retrying before it waits, so the
# silence stays near 11 minutes. If the SDK retried too, one call could be silent for about 30
# minutes, and this threshold would be too short.
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
    def passed_critic(self) -> int:
        """Pipelines whose signal passed the critic, including those that then failed validation."""
        return sum(1 for r in self.results if r.status in PASSED_CRITIC)

    @property
    def passed_validation(self) -> int:
        return sum(1 for r in self.results if r.status in PASSED_VALIDATION)


def run_finished_payload(
    *,
    status: str,
    total_cost_usd: float,
    budget_usd: float,
    reserved_usd: float | None,
    data_failed_pipelines: int | None = None,
) -> dict[str, Any]:
    """The one definition of the run_finished payload. The live run, the reaper and the demo all use it,
    so the keys cannot drift apart (R3-BE-06). data_failed_pipelines counts the pipelines whose window
    could not be scored (failed_backtest), so a mixed run shows them (R4-STAT-08). None when unknown."""
    return {
        "status": status,
        "total_cost_usd": total_cost_usd,
        "budget_usd": budget_usd,
        "reserved_usd": reserved_usd,
        "data_failed_pipelines": data_failed_pipelines,
    }


def _check_pricing(settings: Settings) -> None:
    """Every model the agents use needs every price a call can need. Without one, each pipeline would
    fail at its first call, and the run would still read 'completed'. Nothing is guessed: a model
    with a missing price is refused here, before the run row is written."""
    agents = settings.agents
    for model in (agents.ideas_model, agents.coder_model, agents.critic_model):
        if model not in settings.pricing.models:
            raise ValueError(
                f"no pricing configured for model {model!r}; add it under [pricing.models] "
                "in settings.toml"
            )
        absent = missing_prices(settings.pricing.models[model])
        if absent:
            raise ValueError(
                f"model {model!r} has no {', '.join(absent)} in [pricing.models]; add the value "
                "from the pricing page to settings.toml before running this model"
            )


def _prompt_hashes() -> dict[str, str]:
    """SHA-256 of each prompt's text. A prompt that cannot be read is recorded as unavailable, so the
    run row is still written and the setup error that follows fails the run (R3-WHOLE-15)."""
    hashes: dict[str, str] = {}
    for key, name in (("ideas", IDEAS_PROMPT), ("coder", CODER_PROMPT), ("critic", CRITIC_PROMPT)):
        try:
            hashes[key] = load_prompt(name).sha256
        except OSError as exc:
            hashes[key] = f"unavailable: {type(exc).__name__}"
    return hashes


def _config_snapshot(settings: Settings) -> str:
    """The settings and the prompt versions and hashes this run used. The hashes show an edit made
    without a rename (WHOLE-15)."""
    return json.dumps(
        {
            "settings": settings.model_dump(mode="json"),
            "prompts": {"ideas": IDEAS_PROMPT, "coder": CODER_PROMPT, "critic": CRITIC_PROMPT},
            "prompt_sha256": _prompt_hashes(),
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
    _check_pricing(settings)
    # Agents get training data only. The validation gate gets history up to validation_end, and no
    # later. This function is the one place that cuts the frame, so holdout prices stay out of it.
    train_prices = prices.loc[: pd.Timestamp(settings.dates.train_end)]
    validation_prices = prices.loc[: pd.Timestamp(settings.dates.validation_end)]
    budget = Budget(budget_usd)
    for abandoned_id in ledger.abandon_stale_runs(STALE_RUN_AFTER):
        # The reaper closed the pipelines and the run's cost in the ledger. The closing event is written
        # here, with the same payload builder as a live run (R3-BE-06).
        bus.emit(
            PipelineEvent(
                run_id=abandoned_id,
                stage="run",
                type="run_finished",
                payload=run_finished_payload(
                    status="abandoned",
                    total_cost_usd=ledger.llm_cost_total(abandoned_id),
                    budget_usd=ledger.get_run(abandoned_id)["budget_usd"],
                    reserved_usd=None,  # the process that held the reservation is gone
                    data_failed_pipelines=None,  # the pipelines' outcomes were not kept
                ),
            )
        )

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
        # When every pipeline stopped on an API or unexpected error, nothing was learned: the run failed.
        if len(results) < n_pipelines:
            status = "budget_stopped"
        elif results and all(r.status in RUN_FAILING_PIPELINE_STATUSES for r in results):
            status = "failed"
        elif results and all(r.status == "failed_backtest" for r in results):
            status = RUN_FAILED_DATA
        else:
            status = "completed"
    except Exception:
        status = "failed"
        raise
    except BaseException:  # KeyboardInterrupt, cancellation
        status = "interrupted"
        raise
    finally:
        if created:
            try:
                # An attempt whose own status write failed would stay 'running' in a finished run.
                # Every pipeline has ended by now, so any attempt still open is closed here (R4-AGT-02).
                ledger.close_running_attempts(
                    run_id, f"the run ended ({status}) before this attempt did"
                )
            except Exception as exc:  # noqa: BLE001 - closing must not replace the run's own outcome
                print(f"[run] could not close open attempts of {run_id}: {exc}", file=sys.stderr)
            # Cost is the sum of the llm_calls rows: every call that was charged, failed or not.
            total_cost = ledger.llm_cost_total(run_id)
            ledger.finish_run(run_id, status, total_cost)
            bus.emit(
                PipelineEvent(
                    run_id=run_id,
                    stage="run",
                    type="run_finished",
                    payload=run_finished_payload(
                        status=status,
                        total_cost_usd=total_cost,
                        budget_usd=budget_usd,
                        reserved_usd=budget.reserved_usd,
                        data_failed_pipelines=sum(
                            1 for r in results if r.status == "failed_backtest"
                        ),
                    ),
                )
            )

    return RunSummary(
        run_id=run_id,
        status=status,
        budget_usd=budget_usd,
        total_cost_usd=total_cost,
        results=results,
    )
