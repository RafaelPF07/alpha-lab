"""Synthetic but realistic run for building and demoing the dashboard without spending API money.

It writes the same ledger rows and the same event payload keys as a real run. The keys are the
ones the real emitters write (pipeline/single.py, pipeline/run.py, agents/llm.py and
checks/lookahead.py), so the dashboard code path is identical. The numbers are invented and are
labelled as demo in the UI.

The demo never writes holdout rows. The holdout runs only through `lab holdout`, on real data.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import random
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_lab.backtest.metrics import Metrics
from alpha_lab.config import Settings, load_settings
from alpha_lab.events.bus import EventBus, PipelineEvent
from alpha_lab.ledger.db import LEDGER_PATH, Ledger

DEMO_RUN_ID = "demo"
BUDGET_USD = 2.0
THEMES = [
    "momentum",
    "mean reversion",
    "volatility",
    "cross-sectional ranking",
    "sector rotation",
    "seasonality",
    "trend filters",
    "pairs and relative value",
]
MODELS = {"ideas": "claude-haiku-5-5", "coder": "claude-haiku-5-5", "critic": "claude-sonnet-5-5"}
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
}  # invented, USD per million tokens
TOKENS = {  # (input range, output range) per agent
    "ideas": ((2500, 4800), (250, 700)),
    "coder": ((3000, 6000), (400, 1400)),
    "critic": ((3000, 5000), (200, 600)),
}
TICKERS = ("AAPL", "MSFT", "JPM", "XOM")
DUPLICATE_DRAFT_RATE = 0.125  # share of first drafts rejected as near copies of an earlier idea
DUPLICATE_CODE_RATE = 0.2  # share of revisions that come back as the same program
REFUSED_PIPELINE = 3  # its first coder call is refused, so the refusal path is always visible
TRAIN_DAYS = pd.bdate_range("2010-01-01", "2018-12-31")[:400]
VALIDATION_DAYS = pd.bdate_range("2019-01-01", "2021-12-31")
LOOP_ENDS = ("validation_pass", "validation_fail")  # after these the revision loop stops
ERRORS = {
    "failed_static": "Static check failed:\n- import of 'os' is not allowed",
    "rejected_duplicate_code": (
        "Duplicate code: this is the same program as an earlier attempt, apart from comments, "
        "docstrings and formatting. Write a genuinely different implementation."
    ),
    "failed_sandbox": "Sandbox run failed with status exception: KeyError: 'demo_column'",
    "failed_lookahead": "Lookahead truncation test failed. The signal uses future information.",
    "failed_gate": "Training gate failed: demo metrics are below the configured gates.",
    "failed_critic": "The critic rated this signal as high suspicion, probably an artefact.",
}


def _cost(agent: str, tokens_in: int, tokens_out: int) -> float:
    p_in, p_out = PRICES[MODELS[agent]]
    return (tokens_in * p_in + tokens_out * p_out) / 1_000_000


def _returns(rng: np.random.Generator, index: pd.DatetimeIndex, drift: float) -> dict[str, float]:
    values = rng.normal(drift, 0.008, size=len(index))
    return {str(d.date()): float(v) for d, v in zip(index, values)}


def _demo_idea(rng: random.Random, n: int) -> dict:
    theme = THEMES[n % len(THEMES)]
    return {
        "name": f"{theme} signal {n:02d}",
        "hypothesis": f"Demo hypothesis {n}: {theme} predicts next-period returns in large caps.",
        "description": f"Demo description {n}: rank by a {rng.randint(5, 120)}-day {theme} score.",
        "expected_holding_period_days": rng.choice([5, 10, 21, 63]),
        "theme": theme,
    }


def _train_metrics(rng: random.Random, drift_shift: float = 0.0) -> dict[str, float]:
    """Metric keys come from the real Metrics dataclass, as pipeline/single.py writes them."""
    sharpe = rng.gauss(0.2 + drift_shift, 0.6)
    metrics = Metrics(
        n_days=1800,
        ann_return=sharpe * 0.1,
        ann_volatility=0.1,
        sharpe=sharpe,
        max_drawdown=-abs(rng.gauss(0.12, 0.05)),
        avg_daily_turnover=abs(rng.gauss(0.12, 0.08)),
        hit_rate=0.5 + rng.gauss(0.0, 0.02),
    )
    return {k: float(v) for k, v in dataclasses.asdict(metrics).items()}


def _attempt_status(rng: random.Random, outcome: float, attempt: int) -> str:
    if outcome < 0.2 and attempt == 1:
        return "failed_static"
    if outcome < 0.3:
        return "failed_sandbox"
    if outcome < 0.38:
        return "failed_lookahead"
    if outcome < 0.9:
        return "failed_gate" if rng.random() < 0.8 or attempt < 3 else "failed_critic"
    if rng.random() < 0.2:
        return "failed_critic"
    return "validation_pass" if rng.random() < 0.45 else "validation_fail"


@dataclass
class _Run:
    """One demo run's shared state: the ledger and bus, the random streams, and the running spend."""

    ledger: Ledger
    bus: EventBus
    run_id: str
    rng: random.Random
    np_rng: np.random.Generator
    pace: float
    spent: float = 0.0

    def emit(
        self,
        type_: str,
        stage: str,
        payload: dict,
        *,
        pipeline_id: str | None = None,
        idea_id: str | None = None,
        attempt: int | None = None,
        agent: str | None = None,
    ) -> None:
        self.bus.emit(
            PipelineEvent(
                run_id=self.run_id,
                pipeline_id=pipeline_id,
                idea_id=idea_id,
                attempt_number=attempt,
                agent=agent,
                stage=stage,
                type=type_,  # type: ignore[arg-type]
                payload=payload,
            )
        )

    def refuse(self, agent: str, pipeline_id: str, idea_id: str | None, reason: str) -> None:
        """A call refused before it was sent. It costs nothing, as agents/llm.py _refuse records it."""
        call_id, model = uuid.uuid4().hex, MODELS[agent]
        self.ledger.add_llm_call(
            call_id=call_id,
            run_id=self.run_id,
            idea_id=idea_id,
            agent=agent,
            model=model,
            prompt_version=f"{agent}_v1",
            input_tokens=None,
            output_tokens=None,
            cache_read_tokens=None,
            cache_write_tokens=None,
            cost_usd=0.0,
            latency_ms=None,
            status="refused",
        )
        self.emit(
            "llm_call_refused",
            "llm",
            {"call_id": call_id, "model": model, "reason": reason, "estimated_cost_usd": None},
            pipeline_id=pipeline_id,
            idea_id=idea_id,
            agent=agent,
        )

    def llm(self, agent: str, pipeline_id: str, idea_id: str | None) -> None:
        """One simulated model call, written and announced the way agents/llm.py does it."""
        (lo_in, hi_in), (lo_out, hi_out) = TOKENS[agent]
        tokens_in, tokens_out = self.rng.randint(lo_in, hi_in), self.rng.randint(lo_out, hi_out)
        cost = _cost(agent, tokens_in, tokens_out)
        latency = self.rng.randint(900, 4200)
        call_id, model, prompt = uuid.uuid4().hex, MODELS[agent], f"{agent}_v1"
        self.emit(
            "llm_call_started",
            "llm",
            {
                "call_id": call_id,
                "model": model,
                "prompt_version": prompt,
                "estimated_input_tokens": tokens_in,
            },
            pipeline_id=pipeline_id,
            idea_id=idea_id,
            agent=agent,
        )
        self.ledger.add_llm_call(
            call_id=call_id,
            run_id=self.run_id,
            idea_id=idea_id,
            agent=agent,
            model=model,
            prompt_version=prompt,
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            cache_read_tokens=0,
            cache_write_tokens=0,
            cost_usd=cost,
            latency_ms=latency,
            status="ok",
        )
        self.spent += cost
        self.emit(
            "llm_call_finished",
            "llm",
            {
                "call_id": call_id,
                "status": "ok",
                "cost_usd": cost,
                "latency_ms": latency,
                "input_tokens": tokens_in,
                "output_tokens": tokens_out,
            },
            pipeline_id=pipeline_id,
            idea_id=idea_id,
            agent=agent,
        )
        self.emit(
            "budget_update",
            "budget",
            {"spent_usd": self.spent, "limit_usd": BUDGET_USD},
            pipeline_id=pipeline_id,
            idea_id=idea_id,
            agent=agent,
        )


async def _attempt(
    run: _Run, pid: str, idea_id: str, number: int, status: str, earlier_id: str | None
) -> str:
    """One coder attempt through the checks, in the order pipeline/single.py runs them.

    Returns the new attempt's id. A duplicate is compared with `earlier_id`, the previous attempt.
    """
    rng, ledger = run.rng, run.ledger
    attempt_id = uuid.uuid4().hex
    ledger.add_attempt(
        attempt_id=attempt_id,
        idea_id=idea_id,
        run_id=run.run_id,
        attempt_number=number,
        code=f"# demo code, attempt {number}",
        code_hash=attempt_id,
        normalised_hash=attempt_id,
        status="running",
    )

    def finish(
        final: str,
        *,
        static_ok: bool | None,
        sandbox_ok: bool | None = None,
        lookahead_ok: bool | None = None,
        metrics: dict | None = None,
    ) -> None:
        """Record the attempt's final status and check flags, as pipeline/single.py _record does."""
        ledger.update_attempt(
            attempt_id,
            status=final,
            error=ERRORS.get(final),
            static_ok=None if static_ok is None else int(static_ok),
            sandbox_ok=None if sandbox_ok is None else int(sandbox_ok),
            lookahead_ok=None if lookahead_ok is None else int(lookahead_ok),
            train_metrics_json=None if metrics is None else json.dumps(metrics),
        )

    where = {"pipeline_id": pid, "idea_id": idea_id, "attempt": number}
    digest = uuid.uuid4().hex
    run.emit(
        "code_written",
        "coder",
        {"code_hash": digest, "lines": rng.randint(18, 60)},
        agent="coder",
        **where,
    )
    if status == "rejected_duplicate_code":
        run.emit(
            "duplicate_code",
            "static",
            {"earlier_attempt_id": earlier_id, "message": ERRORS[status]},
            **where,
        )
        finish(status, static_ok=None)
        return attempt_id
    static_ok = status != "failed_static"
    run.emit(
        "static_check",
        "static",
        {
            "ok": static_ok,
            "errors": [] if static_ok else ["import of 'os' is not allowed"],
            "warnings": [],
        },
        **where,
    )
    if not static_ok:
        finish("failed_static", static_ok=False)
        return attempt_id

    # The real pipeline records the trial just before the sandbox runs, so trials match attempts
    # that reached the sandbox.
    ledger.add_trial(uuid.uuid4().hex, run.run_id, attempt_id, digest)
    sandbox_ok = status != "failed_sandbox"
    run.emit(
        "sandbox_run",
        "sandbox",
        {
            "status": "ok" if sandbox_ok else "exception",
            "error_type": None if sandbox_ok else "KeyError",
            "message": None if sandbox_ok else "demo_column not found",
        },
        **where,
    )
    if not sandbox_ok:
        finish("failed_sandbox", static_ok=True, sandbox_ok=False)
        return attempt_id

    # Both lookahead tests always run, as checks/lookahead.run_lookahead_tests does.
    broken = rng.choice(("truncation", "perturbation")) if status == "failed_lookahead" else None
    for test in ("truncation", "perturbation"):
        await asyncio.sleep(run.pace * 0.2)
        if test == broken:
            payload = {
                "test": test,
                "passed": False,
                "cutoff": "2016-09-30",
                "first_offending_date": "2016-10-03",
                "ticker": rng.choice(TICKERS),
            }
        else:
            payload = {
                "test": test,
                "passed": True,
                "cutoff": None,
                "first_offending_date": None,
                "ticker": None,
            }
        run.emit("lookahead_check", "lookahead", payload, **where)
    if broken is not None:
        finish("failed_lookahead", static_ok=True, sandbox_ok=True, lookahead_ok=False)
        return attempt_id

    metrics = _train_metrics(rng)
    ledger.update_attempt(
        attempt_id,
        train_returns_json=json.dumps(_returns(run.np_rng, TRAIN_DAYS, 0.0003)),
    )
    run.emit("train_backtest", "train", metrics, **where)
    if status == "failed_gate":
        finish("failed_gate", static_ok=True, sandbox_ok=True, lookahead_ok=True, metrics=metrics)
        return attempt_id

    run.llm("critic", pid, idea_id)
    high = status == "failed_critic"
    critic = {
        "concerns": ["returns look too smooth for a daily signal"] if high else [],
        "suspicion_level": "high" if high else "low",
        "suggested_change": "use only prices up to the previous close" if high else "none",
    }
    run.emit("critic_review", "critic", critic, agent="critic", **where)
    ledger.update_attempt(
        attempt_id,
        critic_json=json.dumps(critic),
        suspicion_level=critic["suspicion_level"],
    )
    if high:
        finish("failed_critic", static_ok=True, sandbox_ok=True, lookahead_ok=True, metrics=metrics)
        return attempt_id

    passed = status == "validation_pass"
    ledger.add_validation(
        attempt_id,
        run.run_id,
        _train_metrics(rng, drift_shift=-0.6),
        passed,
        None,
        _returns(run.np_rng, VALIDATION_DAYS, 0.0004 if passed else -0.0002),
    )
    finish(status, static_ok=True, sandbox_ok=True, lookahead_ok=True, metrics=metrics)
    run.emit("validation_result", "validation", {"passed": passed}, agent="validation", **where)
    return attempt_id


async def _pipeline(run: _Run, settings: Settings, n: int) -> None:
    """One pipeline: draft an idea, then revise and check it until it stops, as pipeline/single.py does."""
    rng = run.rng
    pid = uuid.uuid4().hex
    theme = THEMES[n % len(THEMES)]
    run.emit("pipeline_started", "pipeline", {"theme": theme}, pipeline_id=pid)
    await asyncio.sleep(run.pace * rng.uniform(0.2, 1.5))

    idea = _demo_idea(rng, n)
    threshold = settings.dedup.similarity_threshold
    if rng.random() < DUPLICATE_DRAFT_RATE:  # a first draft that is too close to an earlier idea
        run.llm("ideas", pid, None)
        similarity = round(threshold + rng.uniform(0.02, 0.1), 3)
        run.ledger.add_rejected_idea(
            rejection_id=uuid.uuid4().hex,
            run_id=run.run_id,
            pipeline_id=pid,
            name=f"{idea['name']} (variant)",
            similar_to_idea_id=None,
            similarity=similarity,
        )
        run.emit(
            "idea_rejected_duplicate",
            "ideas",
            {"name": f"{idea['name']} (variant)", "similarity": similarity, "threshold": threshold},
            pipeline_id=pid,
            agent="ideas",
        )
    run.llm("ideas", pid, None)
    idea_id = uuid.uuid4().hex
    run.ledger.add_idea(
        idea_id=idea_id,
        run_id=run.run_id,
        pipeline_id=pid,
        name=idea["name"],
        hypothesis=idea["hypothesis"],
        description=idea["description"],
        expected_holding_period_days=idea["expected_holding_period_days"],
        theme=theme,
    )
    run.emit(
        "idea_created",
        "ideas",
        {
            "name": idea["name"],
            "hypothesis": idea["hypothesis"],
            "description": idea["description"],
            "expected_holding_period_days": idea["expected_holding_period_days"],
            "theme": theme,
        },
        pipeline_id=pid,
        idea_id=idea_id,
        agent="ideas",
    )

    outcome = rng.random()
    max_attempts = settings.agents.max_revisions + 1
    final, error, made, previous = "failed_coder", None, 0, None
    for number in range(1, max_attempts + 1):
        await asyncio.sleep(run.pace * rng.uniform(0.3, 1.2))
        if n == REFUSED_PIPELINE and number == 1:
            run.refuse("coder", pid, idea_id, "context limit exceeded")
            error = "coder request is over the context limit"
            break
        run.llm("coder", pid, idea_id)
        if number > 1 and rng.random() < DUPLICATE_CODE_RATE:
            status = "rejected_duplicate_code"
        else:
            status = _attempt_status(rng, outcome, number)
        previous = await _attempt(run, pid, idea_id, number, status, previous)
        made, final = number, status
        if status in LOOP_ENDS or number == max_attempts:
            break
        run.emit(
            "revision_requested",
            "coder",
            {"reason": status},
            pipeline_id=pid,
            idea_id=idea_id,
            attempt=number + 1,
            agent="coder",
        )
    run.emit(
        "pipeline_finished",
        "pipeline",
        {"status": final, "attempts": made, "error": error, "theme": theme},
        pipeline_id=pid,
    )


async def run_demo(
    ledger: Ledger,
    bus: EventBus,
    n_pipelines: int = 40,
    seed: int = 7,
    pace: float = 0.35,
    run_id: str = DEMO_RUN_ID,
) -> None:
    """Stream a demo run into the ledger. `pace` is seconds per simulated step.

    The run row is written first, and it is finished as `completed`, `interrupted` (cancelled)
    or `failed`, so a stopped demo never sits at `running`. A demo that already has a run row is
    not generated again.
    """
    if Path(ledger.path).resolve() == LEDGER_PATH.resolve():
        raise ValueError("the demo must not write to the real ledger; give it its own path")
    if ledger.get_run(run_id) is not None:
        return  # already generated; reruns would duplicate rows
    settings = load_settings()
    run = _Run(
        ledger=ledger,
        bus=bus,
        run_id=run_id,
        rng=random.Random(seed),
        np_rng=np.random.default_rng(seed),
        pace=pace,
    )
    ledger.create_run(run_id, json.dumps({"demo": True, "pipelines": n_pipelines}), BUDGET_USD)
    run.emit(
        "run_started",
        "run",
        {
            "pipelines": n_pipelines,
            "budget_usd": BUDGET_USD,
            "concurrency": settings.run.max_concurrency,
        },
    )

    async def staggered(n: int) -> None:
        await asyncio.sleep(pace * 0.5 * (n % 10))
        await _pipeline(run, settings, n)

    status = "completed"
    try:
        async with asyncio.TaskGroup() as group:
            for n in range(n_pipelines):
                group.create_task(staggered(n))
    except asyncio.CancelledError:
        status = "interrupted"
        raise
    except Exception:
        status = "failed"
        raise
    finally:
        ledger.finish_run(run_id, status, run.spent)
        run.emit("run_finished", "run", {"status": status, "total_cost_usd": run.spent})
