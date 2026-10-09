"""One idea -> dedup -> code -> checks -> train backtest -> critic -> revise loop.

This stage only ever sees prices up to train_end. Validation and holdout
data never reach an agent, a sandbox or an event payload.

Attempt statuses, in funnel order:
  failed_static, rejected_duplicate_code, failed_sandbox, failed_lookahead,
  failed_gate, failed_critic, critic_pass
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
from alpha_lab.agents.critic import critique_signal
from alpha_lab.agents.ideas import generate_idea
from alpha_lab.agents.llm import ContextLimitExceeded, LLMClient, PromptSpec, StepFailed
from alpha_lab.agents.models import CoderOutput, CriticOutput, IdeaOutput
from alpha_lab.backtest.engine import run_backtest
from alpha_lab.backtest.holding import hold_weights
from alpha_lab.checks.dedup import idea_similarity, normalised_code_hash
from alpha_lab.checks.lookahead import LookaheadResult, run_lookahead_tests
from alpha_lab.checks.sandbox import SandboxedSignal, SandboxFailure, SandboxOutcome
from alpha_lab.checks.static import static_check
from alpha_lab.config import Settings
from alpha_lab.events.bus import EventBus, PipelineEvent
from alpha_lab.ledger.db import Ledger
from alpha_lab.pipeline.validation import validate_signal

IDEA_RETRIES_ON_DUPLICATE = 2


@dataclass(frozen=True)
class PipelineContext:
    run_id: str
    settings: Settings
    train_prices: pd.DataFrame  # prices through train_end only
    validation_prices: pd.DataFrame  # prices through validation_end, used after the loop only
    llm: LLMClient
    ledger: Ledger
    bus: EventBus
    idea_prompt: PromptSpec
    coder_prompt: PromptSpec
    critic_prompt: PromptSpec
    idea_lock: asyncio.Lock


@dataclass(frozen=True)
class PipelineResult:
    pipeline_id: str
    status: str
    idea_name: str | None
    theme: str
    attempts: int


@dataclass(frozen=True)
class AttemptOutcome:
    status: str
    failure: str | None = None
    metrics: dict[str, Any] | None = None
    critic: CriticOutput | None = None
    attempt_id: str | None = None


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


async def _draft_unique_idea(
    ctx: PipelineContext, pipeline_id: str, theme: str
) -> tuple[IdeaOutput, str]:
    """Draft an idea that is not a near-copy of the ledger, and record it.

    Holds ctx.idea_lock across the draft, the duplicate check and the insert.
    Without the lock, two pipelines could both pass the check with the same
    idea before either one was written down.
    """
    settings = ctx.settings
    threshold = settings.dedup.similarity_threshold
    async with ctx.idea_lock:
        for _ in range(IDEA_RETRIES_ON_DUPLICATE + 1):
            idea = await generate_idea(
                ctx.llm,
                prompt=ctx.idea_prompt,
                model=settings.agents.ideas_model,
                recent_ideas=list(ctx.ledger.recent_ideas(30)),
                theme=theme,
                pipeline_id=pipeline_id,
            )
            best_score, best_id = 0.0, None
            for old in ctx.ledger.all_ideas():
                score = idea_similarity(
                    idea.name, idea.description, old["name"], old["description"]
                )
                if score > best_score:
                    best_score, best_id = score, old["idea_id"]
            if best_score < threshold:
                idea_id = uuid.uuid4().hex
                ctx.ledger.add_idea(
                    idea_id=idea_id,
                    run_id=ctx.run_id,
                    pipeline_id=pipeline_id,
                    name=idea.name,
                    hypothesis=idea.hypothesis,
                    description=idea.description,
                    expected_holding_period_days=idea.expected_holding_period_days,
                    theme=theme,
                )
                return idea, idea_id
            ctx.ledger.add_rejected_idea(
                rejection_id=uuid.uuid4().hex,
                run_id=ctx.run_id,
                pipeline_id=pipeline_id,
                name=idea.name,
                similar_to_idea_id=best_id,
                similarity=best_score,
            )
            _emit(
                ctx,
                "idea_rejected_duplicate",
                "ideas",
                pipeline_id,
                None,
                None,
                "ideas",
                {"name": idea.name, "similarity": round(best_score, 3), "threshold": threshold},
            )
    raise StepFailed("ideas kept duplicating earlier ideas")


async def run_single_pipeline(ctx: PipelineContext, theme: str) -> PipelineResult:
    settings = ctx.settings
    pipeline_id = uuid.uuid4().hex
    _emit(ctx, "pipeline_started", "pipeline", pipeline_id, None, None, None, {"theme": theme})

    try:
        idea, idea_id = await _draft_unique_idea(ctx, pipeline_id, theme)
    except (StepFailed, ContextLimitExceeded) as exc:
        return _finish(ctx, pipeline_id, "failed_ideas", None, theme, 0, str(exc))

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
            "theme": theme,
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
        return _finish(ctx, pipeline_id, "failed_coder", idea, theme, 0, str(exc))

    status = "failed"
    attempts = 0
    last: AttemptOutcome | None = None
    max_attempts = settings.agents.max_revisions + 1
    for attempt_number in range(1, max_attempts + 1):
        attempts = attempt_number
        outcome = await _check_attempt(ctx, pipeline_id, idea_id, idea, code, attempt_number)
        status = outcome.status
        last = outcome
        if status == "critic_pass":
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
            return _finish(ctx, pipeline_id, "failed_coder", idea, theme, attempts, str(exc))

    if status == "critic_pass" and last is not None and last.attempt_id is not None:
        verdict = await asyncio.to_thread(
            validate_signal,
            code.code,
            ctx.validation_prices,
            settings,
            idea.expected_holding_period_days,
        )
        ctx.ledger.add_validation(
            last.attempt_id,
            ctx.run_id,
            verdict.metrics,
            verdict.passed,
            verdict.error,
            verdict.returns,
        )
        status = "validation_pass" if verdict.passed else "validation_fail"
        ctx.ledger.update_attempt(last.attempt_id, status=status)
        _emit(
            ctx,
            "validation_result",
            "validation",
            pipeline_id,
            idea_id,
            attempts,
            "validation",
            {"passed": verdict.passed},
        )
    return _finish(ctx, pipeline_id, status, idea, theme, attempts, None)


def _finish(
    ctx: PipelineContext,
    pipeline_id: str,
    status: str,
    idea: IdeaOutput | None,
    theme: str,
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
        {"status": status, "attempts": attempts, "error": error, "theme": theme},
    )
    return PipelineResult(
        pipeline_id=pipeline_id,
        status=status,
        idea_name=idea.name if idea else None,
        theme=theme,
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
    normalised = normalised_code_hash(code.code)
    earlier = ctx.ledger.find_normalised_code(normalised)  # checked before this attempt is stored
    attempt_id = ctx.ledger.add_attempt(
        attempt_id=uuid.uuid4().hex,
        idea_id=idea_id,
        run_id=ctx.run_id,
        attempt_number=attempt_number,
        code=code.code,
        code_hash=digest,
        normalised_hash=normalised,
        status="running",
    )

    def emit(event_type: str, stage: str, payload: dict, agent: str | None = None) -> None:
        _emit(ctx, event_type, stage, pipeline_id, idea_id, attempt_number, agent, payload)

    emit(
        "code_written",
        "coder",
        {"code_hash": digest, "lines": len(code.code.splitlines())},
        "coder",
    )

    if earlier is not None:
        failure = (
            "Duplicate code: this is the same program as an earlier attempt, apart from comments "
            "and formatting. Write a genuinely different implementation."
        )
        emit("static_check", "static", {"ok": False, "errors": [failure], "warnings": []})
        return _record(ctx, attempt_id, "rejected_duplicate_code", failure, None, static_ok=False)

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

    result = run_backtest(
        hold_weights(outcome.weights, idea.expected_holding_period_days),
        ctx.train_prices,
        settings.dates.start,
        settings.dates.train_end,
        settings.backtest,
    )
    metrics = {k: float(v) for k, v in asdict(result.metrics).items()}
    ctx.ledger.update_attempt(
        attempt_id,
        train_returns_json=json.dumps(
            {str(day.date()): float(v) for day, v in result.net_returns.items()}
        ),
    )
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

    critic = await critique_signal(
        ctx.llm,
        prompt=ctx.critic_prompt,
        model=settings.agents.critic_model,
        idea_name=idea.name,
        hypothesis=idea.hypothesis,
        code=code.code,
        metrics=metrics,
        pipeline_id=pipeline_id,
        idea_id=idea_id,
    )
    emit(
        "critic_review",
        "critic",
        {
            "suspicion_level": critic.suspicion_level,
            "concerns": critic.concerns,
            "suggested_change": critic.suggested_change,
        },
        "critic",
    )
    ctx.ledger.update_attempt(
        attempt_id,
        critic_json=critic.model_dump_json(),
        suspicion_level=critic.suspicion_level,
    )
    if critic.suspicion_level == "high":
        failure = (
            "The critic rated this signal as high suspicion, probably an artefact. Concerns: "
            + "; ".join(critic.concerns or ["none given"])
            + f". Suggested change: {critic.suggested_change}"
        )
        return _record(
            ctx,
            attempt_id,
            "failed_critic",
            failure,
            metrics,
            static_ok=True,
            sandbox_ok=True,
            lookahead_ok=True,
            critic=critic,
        )

    return _record(
        ctx,
        attempt_id,
        "critic_pass",
        None,
        metrics,
        static_ok=True,
        sandbox_ok=True,
        lookahead_ok=True,
        critic=critic,
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
    critic: CriticOutput | None = None,
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
    return AttemptOutcome(
        status=status, failure=failure, metrics=metrics, critic=critic, attempt_id=attempt_id
    )


def _iso(value: pd.Timestamp | None) -> str | None:
    return None if value is None else pd.Timestamp(value).date().isoformat()
