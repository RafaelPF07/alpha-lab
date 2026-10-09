"""One idea -> dedup -> code -> checks -> train backtest -> critic -> revise loop -> validation.

Agents see training data only (prices through train_end). The validation gate runs once, after
the revision loop, on history through validation_end. Holdout data never reaches an agent, a
sandbox or an event payload.

Every pipeline that starts gets exactly one pipeline_finished event, written in a finally. A
process that is killed leaves its open pipelines without one, and Ledger.abandon_stale_runs writes
them on a later run.

The status sets are defined in alpha_lab.statuses (DECISION-13). Attempt statuses (attempts.status).
A failed attempt is followed by a revision, until the revision cap is reached, except
failed_critic_output, which ends the pipeline:
  failed_static, rejected_duplicate_code, failed_sandbox, failed_lookahead, failed_backtest,
  failed_gate, failed_critic, failed_critic_output, critic_pass,
  validation_pass, validation_fail (set on the last attempt only),
  aborted_budget, aborted_error (an exception stopped the attempt),
  aborted_cancelled (the attempt was cancelled).
The ledger reaper writes abandoned, not this module.

Pipeline statuses (pipeline_finished and PipelineResult.status): the attempt statuses above, plus
  failed_ideas, failed_coder (the model gave no usable output),
  failed_api (the API failed after retries), failed_error (an unexpected error),
  aborted_budget, aborted_error, aborted_cancelled.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import sys
import uuid
from dataclasses import asdict, dataclass
from typing import Any

import anthropic
import pandas as pd

from alpha_lab.agents.budget import BudgetExceeded
from alpha_lab.agents.coder import Revision, write_signal_code
from alpha_lab.agents.critic import critique_signal
from alpha_lab.agents.ideas import AVOID_LIST_SIZE, generate_idea
from alpha_lab.agents.llm import ContextLimitExceeded, LLMClient, PromptSpec, StepFailed
from alpha_lab.agents.models import CoderOutput, CriticOutput, IdeaOutput
from alpha_lab.backtest.engine import BacktestDataError, run_backtest
from alpha_lab.backtest.holding import hold_weights
from alpha_lab.checks.dedup import idea_similarity, normalised_code_hash
from alpha_lab.checks.lookahead import LookaheadResult, run_lookahead_tests
from alpha_lab.checks.sandbox import SandboxedSignal, SandboxFailure, SandboxOutcome
from alpha_lab.checks.static import static_check
from alpha_lab.config import Settings
from alpha_lab.events.bus import EventBus, PipelineEvent
from alpha_lab.ledger.db import Ledger, strict_json
from alpha_lab.pipeline.validation import run_signal, validate_signal

IDEA_RETRIES_ON_DUPLICATE = 2
# No revision after these: the critic gave no usable review, so there is no feedback to send the coder.
NO_REVISION_STATUSES = frozenset({"failed_critic_output"})
# API errors that end the whole run. Every later call would fail the same way.
FATAL_API_ERRORS = (anthropic.AuthenticationError, anthropic.PermissionDeniedError)


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


@dataclass
class _PipelineProgress:
    """How far a pipeline got. Read when it ends, whatever the reason."""

    status: str = "aborted_error"
    error: str | None = "the pipeline ended before it finished"
    idea: IdeaOutput | None = None
    attempts: int = 0

    def end(self, status: str, error: str | None) -> None:
        """Set the status for an early stop. Every exit path in run_single_pipeline goes through here,
        so the status cannot be left as the verdict of an earlier attempt."""
        self.status, self.error = status, error


@dataclass
class _AttemptChecks:
    """The checks an attempt has passed so far. Written to the ledger if an exception stops it."""

    static_ok: bool | None = None
    sandbox_ok: bool | None = None
    lookahead_ok: bool | None = None
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


def relevant_ideas(ledger: Ledger, theme: str, limit: int = AVOID_LIST_SIZE) -> list:
    """The earlier ideas the ideas prompt shows, newest first. Ideas from the same theme come first, so
    the model sees the family it is asked to stay in, even when those ideas are old. The rest fill up
    with the newest ideas. Dedup still compares every idea, so this only changes what the model is
    told (R3-WHOLE-09)."""
    rows = list(reversed(ledger.all_ideas()))  # newest first
    same_theme = [row["idea_id"] for row in rows if row["theme"] == theme][: limit // 2]
    chosen = set(same_theme)
    for row in rows:
        if len(chosen) >= limit:
            break
        chosen.add(row["idea_id"])
    return [row for row in rows if row["idea_id"] in chosen][:limit]


async def _draft_unique_idea(
    ctx: PipelineContext, pipeline_id: str, theme: str
) -> tuple[IdeaOutput, str]:
    """Draft an idea that is not a near-copy of the ledger, and record it.

    Drafts are serialised on ctx.idea_lock, so each draft is written after the ideas accepted
    before it, and its prompt lists them. That avoids drafting duplicates. Safety does not depend
    on the lock: the duplicate check and the insert run with no await between them, so two
    pipelines cannot both accept the same idea.
    """
    settings = ctx.settings
    threshold = settings.dedup.similarity_threshold
    async with ctx.idea_lock:
        for _ in range(IDEA_RETRIES_ON_DUPLICATE + 1):
            # The id is chosen before the call, so the ideas call's llm_calls row names the idea it
            # produced (WHOLE-07). A rejected draft's id is never stored as an idea.
            candidate_id = uuid.uuid4().hex
            idea = await generate_idea(
                ctx.llm,
                prompt=ctx.idea_prompt,
                model=settings.agents.ideas_model,
                recent_ideas=relevant_ideas(ctx.ledger, theme),
                theme=theme,
                pipeline_id=pipeline_id,
                idea_id=candidate_id,
            )
            best_score, best_id = 0.0, None
            for old in ctx.ledger.all_ideas():
                score = idea_similarity(
                    idea.name, idea.description, old["name"], old["description"], floor=threshold
                )
                if score > best_score:
                    best_score, best_id = score, old["idea_id"]
            if best_score < threshold:
                idea_id = candidate_id
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
    """Run one pipeline to a terminal status and emit its pipeline_finished event in a finally.

    BudgetExceeded and the fatal API errors are raised, because they end the whole run. So are
    cancellation and the other BaseException exits, after their status is recorded. Every other
    failure becomes this pipeline's status, and the run goes on. A cancellation is recorded as
    aborted_cancelled, whichever stage it reached, and then re-raised.
    """
    pipeline_id = uuid.uuid4().hex
    _emit(ctx, "pipeline_started", "pipeline", pipeline_id, None, None, None, {"theme": theme})
    progress = _PipelineProgress()
    try:
        await _run_stages(ctx, pipeline_id, theme, progress)
    except BudgetExceeded as exc:
        progress.end("aborted_budget", str(exc))
        raise
    except FATAL_API_ERRORS as exc:
        progress.end("aborted_error", f"{type(exc).__name__}: {exc}")
        raise
    except asyncio.CancelledError:
        # A sibling's fatal error, Ctrl+C or shutdown. The stage it was in does not matter.
        progress.end("aborted_cancelled", "cancelled")
        raise
    except anthropic.APIError as exc:
        progress.end("failed_api", f"{type(exc).__name__}: {exc}")
    except Exception as exc:  # noqa: BLE001 - one pipeline's failure must not end the run
        progress.end("failed_error", f"{type(exc).__name__}: {exc}")
    except BaseException as exc:  # KeyboardInterrupt and other exits: record, then let them go on
        progress.end("aborted_error", f"{type(exc).__name__}: {exc}")
        raise
    finally:
        _emit(
            ctx,
            "pipeline_finished",
            "pipeline",
            pipeline_id,
            None,
            None,
            None,
            {
                "status": progress.status,
                "attempts": progress.attempts,
                "error": progress.error,
                "theme": theme,
            },
        )
    return PipelineResult(
        pipeline_id=pipeline_id,
        status=progress.status,
        idea_name=progress.idea.name if progress.idea else None,
        theme=theme,
        attempts=progress.attempts,
    )


async def _run_stages(
    ctx: PipelineContext, pipeline_id: str, theme: str, progress: _PipelineProgress
) -> None:
    settings = ctx.settings
    try:
        idea, idea_id = await _draft_unique_idea(ctx, pipeline_id, theme)
    except (StepFailed, ContextLimitExceeded) as exc:
        progress.status, progress.error = "failed_ideas", str(exc)
        return
    progress.idea = idea

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
        progress.status, progress.error = "failed_coder", str(exc)
        return

    max_attempts = settings.agents.max_revisions + 1
    last: AttemptOutcome | None = None
    for attempt_number in range(1, max_attempts + 1):
        progress.attempts = attempt_number
        outcome = await _check_attempt(ctx, pipeline_id, idea_id, idea, code, attempt_number)
        last = outcome
        progress.status, progress.error = outcome.status, outcome.failure
        if (
            outcome.status == "critic_pass"
            or outcome.status in NO_REVISION_STATUSES
            or attempt_number == max_attempts
        ):
            break

        _emit(
            ctx,
            "revision_requested",
            "coder",
            pipeline_id,
            idea_id,
            attempt_number + 1,
            "coder",
            {"reason": outcome.status},
        )
        try:
            code = await write_signal_code(
                ctx.llm,
                prompt=ctx.coder_prompt,
                model=settings.agents.coder_model,
                idea=idea,
                revision=Revision(
                    previous_code=code.code,
                    failure=outcome.failure or outcome.status,
                    train_metrics=outcome.metrics,
                ),
                pipeline_id=pipeline_id,
                idea_id=idea_id,
            )
        except (StepFailed, ContextLimitExceeded) as exc:
            progress.status, progress.error = "failed_coder", str(exc)
            return

    if progress.status == "critic_pass" and last is not None and last.attempt_id is not None:
        await _validate(ctx, pipeline_id, idea_id, idea, code, last.attempt_id, progress)


async def _validate(
    ctx: PipelineContext,
    pipeline_id: str,
    idea_id: str,
    idea: IdeaOutput,
    code: CoderOutput,
    attempt_id: str,
    progress: _PipelineProgress,
) -> None:
    try:
        verdict = await asyncio.to_thread(
            validate_signal,
            code.code,
            ctx.validation_prices,
            ctx.settings,
            idea.expected_holding_period_days,
        )
    except BaseException as exc:
        # The attempt passed the critic, but validation did not finish. It must not stay critic_pass.
        # A failed write here must not replace the exception, so it is reported and the original is raised.
        try:
            if isinstance(exc, asyncio.CancelledError):
                ctx.ledger.update_attempt(attempt_id, status="aborted_cancelled", error="cancelled")
            else:
                ctx.ledger.update_attempt(
                    attempt_id, status="aborted_error", error=f"{type(exc).__name__}: {exc}"
                )
        except Exception as write_exc:  # noqa: BLE001 - the original exception matters more
            print(f"[single] could not record attempt {attempt_id}: {write_exc}", file=sys.stderr)
        raise
    if verdict.no_verdict_status is not None:
        # The validation window gave no verdict: it could not be scored, the sandbox did not finish, or
        # the Sharpe is not a number. Nothing is recorded as a pass or a fail, so the attempt does not
        # count as having passed the critic's gate (R4-WHOLE-03, R4-AGT-08).
        ctx.ledger.update_attempt(attempt_id, status=verdict.no_verdict_status, error=verdict.error)
        progress.status, progress.error = verdict.no_verdict_status, verdict.error
        return
    status = "validation_pass" if verdict.passed else "validation_fail"
    with ctx.ledger.transaction():
        ctx.ledger.add_validation(
            attempt_id,
            ctx.run_id,
            verdict.metrics,
            verdict.passed,
            verdict.error,
            verdict.returns,
        )
        ctx.ledger.update_attempt(attempt_id, status=status)
    progress.status, progress.error = status, None
    _emit(
        ctx,
        "validation_result",
        "validation",
        pipeline_id,
        idea_id,
        progress.attempts,
        "validation",
        {"passed": verdict.passed},
    )


async def _check_attempt(
    ctx: PipelineContext,
    pipeline_id: str,
    idea_id: str,
    idea: IdeaOutput,
    code: CoderOutput,
    attempt_number: int,
) -> AttemptOutcome:
    digest = code_hash(code.code)
    normalised = normalised_code_hash(code.code)
    earlier = ctx.ledger.find_normalised_code(normalised)  # looked up before this attempt is stored
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
    checks = _AttemptChecks()
    try:
        return await _run_checks(
            ctx,
            pipeline_id,
            idea_id,
            idea,
            code,
            attempt_number,
            attempt_id,
            digest,
            earlier,
            checks,
        )
    except BudgetExceeded as exc:
        _abort(ctx, attempt_id, "aborted_budget", str(exc), checks)
        raise
    except asyncio.CancelledError:
        _abort(ctx, attempt_id, "aborted_cancelled", "cancelled", checks)
        raise
    except BaseException as exc:  # no attempt stays 'running'
        _abort(ctx, attempt_id, "aborted_error", f"{type(exc).__name__}: {exc}", checks)
        raise


async def _run_checks(
    ctx: PipelineContext,
    pipeline_id: str,
    idea_id: str,
    idea: IdeaOutput,
    code: CoderOutput,
    attempt_number: int,
    attempt_id: str,
    digest: str,
    earlier: Any,
    checks: _AttemptChecks,
) -> AttemptOutcome:
    settings = ctx.settings
    allowed = settings.sandbox.allowed_imports

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
            "Duplicate code: this is the same program as an earlier attempt, apart from comments, "
            "docstrings and formatting. Write a genuinely different implementation."
        )
        emit(
            "duplicate_code",
            "static",
            {"earlier_attempt_id": earlier["attempt_id"], "message": failure},
        )
        return _record(ctx, attempt_id, "rejected_duplicate_code", failure, None, static_ok=None)

    report = static_check(code.code, allowed)
    emit(
        "static_check",
        "static",
        {"ok": report.ok, "errors": report.errors, "warnings": report.warnings},
    )
    if not report.ok:
        failure = "Static check failed:\n- " + "\n- ".join(report.errors)
        return _record(ctx, attempt_id, "failed_static", failure, None, static_ok=False)
    checks.static_ok = True

    sandboxed = SandboxedSignal(code.code, settings.sandbox.timeout_seconds, allowed)
    ctx.ledger.add_trial(uuid.uuid4().hex, ctx.run_id, attempt_id, digest)
    outcome: SandboxOutcome = await asyncio.to_thread(run_signal, sandboxed, ctx.train_prices)
    # Which containment mechanism ran (job-object, process-group or taskkill), kept for the audit.
    ctx.ledger.update_attempt(attempt_id, containment=outcome.containment)
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
    checks.sandbox_ok = True

    try:
        results: list[LookaheadResult] = await asyncio.to_thread(
            run_lookahead_tests, sandboxed, ctx.train_prices
        )
    except SandboxFailure as exc:
        failure = f"Lookahead tests could not run: {exc}"
        return _record(
            ctx, attempt_id, "failed_sandbox", failure, None, static_ok=True, sandbox_ok=False
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
    checks.lookahead_ok = True

    held = hold_weights(outcome.weights, idea.expected_holding_period_days)
    try:
        result = run_backtest(
            held,
            ctx.train_prices,
            settings.dates.start,
            settings.dates.train_end,
            settings.backtest,
        )
    except BacktestDataError as exc:
        failure = f"The training backtest could not score the signal: {exc}"
        return _record(
            ctx,
            attempt_id,
            "failed_backtest",
            failure,
            None,
            static_ok=True,
            sandbox_ok=True,
            lookahead_ok=True,
        )
    metrics = {k: float(v) for k, v in asdict(result.metrics).items()}
    checks.metrics = metrics
    ctx.ledger.update_attempt(
        attempt_id,
        train_returns_json=strict_json(
            {str(day.date()): float(v) for day, v in result.net_returns.items()}
        ),
    )
    emit("train_backtest", "train", metrics)

    gates = settings.gates
    problems: list[str] = []
    sharpe = metrics["sharpe"]
    if not math.isfinite(sharpe):
        problems.append("Sharpe is not a finite number, because the scoring window has no variance")
    elif sharpe < gates.min_train_sharpe:
        problems.append(f"Sharpe {sharpe:.3f} is below the minimum {gates.min_train_sharpe}")
    turnover = metrics["avg_daily_turnover"]
    if not math.isfinite(turnover):
        problems.append("average daily turnover is not a finite number")
    elif turnover > gates.max_turnover_daily:
        problems.append(
            f"average daily turnover {turnover:.3f} is above the limit {gates.max_turnover_daily}"
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

    try:
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
    except (StepFailed, ContextLimitExceeded) as exc:
        failure = f"The critic gave no usable review: {exc}"
        return _record(
            ctx,
            attempt_id,
            "failed_critic_output",
            failure,
            metrics,
            static_ok=True,
            sandbox_ok=True,
            lookahead_ok=True,
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
    static_ok: bool | None,
    sandbox_ok: bool | None = None,
    lookahead_ok: bool | None = None,
    critic: CriticOutput | None = None,
) -> AttemptOutcome:
    """Write an attempt's final status. None means the check never ran, so the column stays NULL."""
    ctx.ledger.update_attempt(
        attempt_id,
        status=status,
        static_ok=_flag(static_ok),
        sandbox_ok=_flag(sandbox_ok),
        lookahead_ok=_flag(lookahead_ok),
        error=failure,
        train_metrics_json=None if metrics is None else strict_json(metrics),
    )
    return AttemptOutcome(
        status=status, failure=failure, metrics=metrics, critic=critic, attempt_id=attempt_id
    )


def _abort(
    ctx: PipelineContext,
    attempt_id: str,
    status: str,
    error: str,
    checks: _AttemptChecks,
) -> None:
    """Record what an attempt reached before an exception stopped it, so no row stays 'running'.

    A failed write here is reported and not raised. The exception that stopped the attempt is the one
    the caller must see. execute_run closes any attempt still left open when the run ends (R4-AGT-02).
    """
    fields: dict[str, Any] = {"status": status, "error": error}
    if checks.static_ok is not None:
        fields["static_ok"] = int(checks.static_ok)
    if checks.sandbox_ok is not None:
        fields["sandbox_ok"] = int(checks.sandbox_ok)
    if checks.lookahead_ok is not None:
        fields["lookahead_ok"] = int(checks.lookahead_ok)
    if checks.metrics is not None:
        fields["train_metrics_json"] = strict_json(checks.metrics)
    try:
        ctx.ledger.update_attempt(attempt_id, **fields)
    except Exception as exc:  # noqa: BLE001 - the original exception matters more than this write
        print(f"[single] could not record attempt {attempt_id} as {status}: {exc}", file=sys.stderr)


def _flag(value: bool | None) -> int | None:
    return None if value is None else int(value)


def _iso(value: pd.Timestamp | None) -> str | None:
    return None if value is None else pd.Timestamp(value).date().isoformat()
