"""One idea -> code -> checks -> train backtest -> revise loop.

This stage only ever sees prices up to train_end. Validation and holdout
data never reach an agent, a sandbox or an event payload.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from alpha_lab.agents.coder import Revision, write_signal_code
from alpha_lab.agents.ideas import generate_idea
from alpha_lab.agents.llm import ContextLimitExceeded, LLMClient, PromptSpec, StepFailed
from alpha_lab.agents.models import CoderOutput, IdeaOutput
from alpha_lab.backtest.engine import run_backtest
from alpha_lab.checks.lookahead import LookaheadResult, run_lookahead_tests
from alpha_lab.checks.sandbox import SandboxedSignal, SandboxFailure, SandboxOutcome
from alpha_lab.checks.static import static_check
from alpha_lab.config import Settings
from alpha_lab.events.bus import EventBus, PipelineEvent
from alpha_lab.ledger.db import Ledger


@dataclass(frozen=True)
class PipelineContext:
    run_id: str
    settings: Settings
    train_prices: pd.DataFrame  # prices through train_end only
    llm: LLMClient
    ledger: Ledger
    bus: EventBus
    idea_prompt: PromptSpec
    coder_prompt: PromptSpec


@dataclass(frozen=True)
class PipelineResult:
    pipeline_id: str
    status: str
    idea_name: str | None
    attempts: int


@dataclass(frozen=True)
class AttemptOutcome:
    status: str  # train_pass | failed_static | failed_sandbox | failed_lookahead | failed_gate
    failure: str | None = None
    metrics: dict[str, Any] | None = None


def code_hash(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def _emit(
    ctx: PipelineContext,
    event_type: str,
    stage: str,
    pipeline_id: str,
    idea_id: str | None,
    attempt: int | None,
    agent: str | None,
    payload: dict,
) -> None:
    ctx.bus.emit(
        PipelineEvent(
            run_id=ctx.run_id,
            pipeline_id=pipeline_id,
            idea_id=idea_id,
            attempt_number=attempt,
            agent=agent,
            stage=stage,
            type=event_type,  # type: ignore[arg-type]
            payload=payload,
        )
    )


async def run_single_pipeline(ctx: PipelineContext) -> PipelineResult:
    settings = ctx.settings
    pipeline_id = uuid.uuid4().hex
    _emit(ctx, "pipeline_started", "pipeline", pipeline_id, None, None, None, {})

    try:
        idea = await generate_idea(
            ctx.llm,
            prompt=ctx.idea_prompt,
            model=settings.agents.ideas_model,
            recent_ideas=list(ctx.ledger.recent_ideas(30)),
            pipeline_id=pipeline_id,
        )
    except (StepFailed, ContextLimitExceeded) as exc:
        return _finish(ctx, pipeline_id, "failed_ideas", None, 0, str(exc))

    idea_id = uuid.uuid4().hex
    ctx.ledger.add_idea(
        idea_id=idea_id,
        run_id=ctx.run_id,
        pipeline_id=pipeline_id,
        name=idea.name,
        hypothesis=idea.hypothesis,
        description=idea.description,
        expected_holding_period_days=idea.expected_holding_period_days,
    )
    _emit(
        ctx,
        "idea_created",
        "ideas",
        pipeline_id,
        idea_id,
        None,
        "ideas",
        {
            "name": idea.name,
            "hypothesis": idea.hypothesis,
            "description": idea.description,
            "expected_holding_period_days": idea.expected_holding_period_days,
        },
    )

    try:
        code = await write_signal_code(
            ctx.llm,
            prompt=ctx.coder_prompt,
            model=settings.agents.coder_model,
            idea=idea,
            revision=None,
            pipeline_id=pipeline_id,
            idea_id=idea_id,
        )
    except (StepFailed, ContextLimitExceeded) as exc:
        return _finish(ctx, pipeline_id, "failed_coder", idea, 0, str(exc))

    status = "failed"
    attempts = 0
    max_attempts = settings.agents.max_revisions + 1
    for attempt_number in range(1, max_attempts + 1):
        attempts = attempt_number
        outcome = await _check_attempt(ctx, pipeline_id, idea_id, idea, code, attempt_number)
        status = outcome.status
        if status == "train_pass":
            break
        if attempt_number == max_attempts:
            break

        _emit(
            ctx,
            "revision_requested",
            "coder",
            pipeline_id,
            idea_id,
            attempt_number + 1,
            "coder",
            {"reason": status},
        )
        try:
            code = await write_signal_code(
                ctx.llm,
                prompt=ctx.coder_prompt,
                model=settings.agents.coder_model,
                idea=idea,
                revision=Revision(
                    previous_code=code.code,
                    failure=outcome.failure or status,
                    train_metrics=outcome.metrics,
                ),
                pipeline_id=pipeline_id,
                idea_id=idea_id,
            )
        except (StepFailed, ContextLimitExceeded) as exc:
            status = "failed_coder"
            return _finish(ctx, pipeline_id, status, idea, attempts, str(exc))

    return _finish(ctx, pipeline_id, status, idea, attempts, None)


def _finish(
    ctx: PipelineContext,
    pipeline_id: str,
    status: str,
    idea: IdeaOutput | None,
    attempts: int,
    error: str | None,
) -> PipelineResult:
    _emit(
        ctx,
        "pipeline_finished",
        "pipeline",
        pipeline_id,
        None,
        None,
        None,
        {"status": status, "attempts": attempts, "error": error},
    )
    return PipelineResult(
        pipeline_id=pipeline_id,
        status=status,
        idea_name=idea.name if idea else None,
        attempts=attempts,
    )


async def _check_attempt(
    ctx: PipelineContext,
    pipeline_id: str,
    idea_id: str,
    idea: IdeaOutput,
    code: CoderOutput,
    attempt_number: int,
) -> AttemptOutcome:
    settings = ctx.settings
    allowed = settings.sandbox.allowed_imports
    digest = code_hash(code.code)
    attempt_id = ctx.ledger.add_attempt(
        attempt_id=uuid.uuid4().hex,
        idea_id=idea_id,
        run_id=ctx.run_id,
        attempt_number=attempt_number,
        code=code.code,
        code_hash=digest,
        status="running",
    )

    def emit(event_type: str, stage: str, payload: dict, agent: str | None = None) -> None:
        _emit(ctx, event_type, stage, pipeline_id, idea_id, attempt_number, agent, payload)

    emit(
        "code_written", "coder", {"code_hash": digest, "lines": code.code.count("\n") + 1}, "coder"
    )

    report = static_check(code.code, allowed)
    emit(
        "static_check",
        "static",
        {"ok": report.ok, "errors": report.errors, "warnings": report.warnings},
    )
    if not report.ok:
        failure = "Static check failed:\n- " + "\n- ".join(report.errors)
        return _record(ctx, attempt_id, "failed_static", failure, None, static_ok=False)

    sandboxed = SandboxedSignal(code.code, settings.sandbox.timeout_seconds, allowed)
    ctx.ledger.add_trial(uuid.uuid4().hex, ctx.run_id, attempt_id, digest)
    outcome: SandboxOutcome = await asyncio.to_thread(sandboxed.run, ctx.train_prices)
    emit(
        "sandbox_run",
        "sandbox",
        {"status": outcome.status, "error_type": outcome.error_type, "message": outcome.message},
    )
    if not outcome.ok:
        failure = f"Sandbox run failed with status {outcome.status}: {outcome.message}"
        if outcome.traceback:
            failure += "\nTraceback (last lines):\n" + outcome.traceback[-1500:]
        return _record(
            ctx, attempt_id, "failed_sandbox", failure, None, static_ok=True, sandbox_ok=False
        )

    try:
        results: list[LookaheadResult] = await asyncio.to_thread(
            run_lookahead_tests, sandboxed, ctx.train_prices
        )
    except SandboxFailure as exc:
        failure = f"Lookahead tests could not run: {exc}"
        return _record(
            ctx, attempt_id, "failed_sandbox", failure, None, static_ok=True, sandbox_ok=True
        )

    for result in results:
        emit(
            "lookahead_check",
            "lookahead",
            {
                "test": result.test,
                "passed": result.passed,
                "cutoff": _iso(result.cutoff),
                "first_offending_date": _iso(result.first_offending_date),
                "ticker": result.ticker,
            },
        )
    failed = [r for r in results if not r.passed]
    if failed:
        r = failed[0]
        failure = (
            f"Lookahead {r.test} test failed. Output on {_iso(r.first_offending_date)} "
            f"for ticker {r.ticker} changed when data after {_iso(r.cutoff)} was altered. "
            "The signal uses future information."
        )
        return _record(
            ctx,
            attempt_id,
            "failed_lookahead",
            failure,
            None,
            static_ok=True,
            sandbox_ok=True,
            lookahead_ok=False,
        )

    backtest = settings.backtest
    result = run_backtest(
        outcome.weights,
        ctx.train_prices,
        settings.dates.start,
        settings.dates.train_end,
        backtest,
    )
    metrics = {k: float(v) for k, v in asdict(result.metrics).items()}
    emit("train_backtest", "train", metrics)

    gates = settings.gates
    problems = []
    if not metrics["sharpe"] >= gates.min_train_sharpe:
        problems.append(
            f"Sharpe {metrics['sharpe']:.3f} is below the minimum {gates.min_train_sharpe}"
        )
    if not metrics["avg_daily_turnover"] <= gates.max_turnover_daily:
        problems.append(
            f"average daily turnover {metrics['avg_daily_turnover']:.3f} is above the limit "
            f"{gates.max_turnover_daily}"
        )
    if problems:
        failure = "Training gate failed: " + "; ".join(problems) + "."
        return _record(
            ctx,
            attempt_id,
            "failed_gate",
            failure,
            metrics,
            static_ok=True,
            sandbox_ok=True,
            lookahead_ok=True,
        )

    return _record(
        ctx,
        attempt_id,
        "train_pass",
        None,
        metrics,
        static_ok=True,
        sandbox_ok=True,
        lookahead_ok=True,
    )


def _record(
    ctx: PipelineContext,
    attempt_id: str,
    status: str,
    failure: str | None,
    metrics: dict[str, Any] | None,
    *,
    static_ok: bool,
    sandbox_ok: bool | None = None,
    lookahead_ok: bool | None = None,
) -> AttemptOutcome:
    ctx.ledger.update_attempt(
        attempt_id,
        status=status,
        static_ok=int(static_ok),
        sandbox_ok=None if sandbox_ok is None else int(sandbox_ok),
        lookahead_ok=None if lookahead_ok is None else int(lookahead_ok),
        error=failure,
        train_metrics_json=None if metrics is None else json.dumps(metrics),
    )
    return AttemptOutcome(status=status, failure=failure, metrics=metrics)


def _iso(value: pd.Timestamp | None) -> str | None:
    return None if value is None else pd.Timestamp(value).date().isoformat()
