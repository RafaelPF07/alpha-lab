"""Synthetic but realistic run for building and demoing the dashboard without spending API money.

It writes the same ledger tables and events as a real run, so the dashboard code
path is identical. The numbers are invented and are labelled as demo in the UI.
"""

from __future__ import annotations

import asyncio
import json
import random
import uuid

import numpy as np
import pandas as pd

from alpha_lab.config import load_settings
from alpha_lab.events.bus import EventBus, PipelineEvent
from alpha_lab.ledger.db import Ledger

DEMO_RUN_ID = "demo"
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


def _cost(agent: str, tokens_in: int, tokens_out: int) -> float:
    prices = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.00, 10.00)}
    p_in, p_out = prices[MODELS[agent]]
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


async def run_demo(
    ledger: Ledger,
    bus: EventBus,
    n_pipelines: int = 40,
    seed: int = 7,
    pace: float = 0.35,
    run_id: str = DEMO_RUN_ID,
) -> None:
    """Stream a demo run into the ledger. `pace` is seconds per simulated step."""
    if ledger.get_run(run_id) is not None:
        return  # already generated; reruns would duplicate rows
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    settings = load_settings()
    budget = 2.0
    ledger.create_run(run_id, json.dumps({"demo": True, "pipelines": n_pipelines}), budget)
    bus.emit(
        PipelineEvent(
            run_id=run_id,
            stage="run",
            type="run_started",
            payload={"pipelines": n_pipelines, "budget_usd": budget, "demo": True},
        )
    )
    index = pd.bdate_range("2019-01-01", "2021-12-31")

    async def pipeline(n: int) -> None:
        pid = uuid.uuid4().hex
        theme = THEMES[n % len(THEMES)]
        idea = _demo_idea(rng, n)
        idea_id = uuid.uuid4().hex
        bus.emit(
            PipelineEvent(
                run_id=run_id,
                pipeline_id=pid,
                stage="pipeline",
                type="pipeline_started",
                payload={"theme": theme},
            )
        )
        await asyncio.sleep(pace * rng.uniform(0.2, 1.5))

        tin, tout = rng.randint(2500, 4800), rng.randint(250, 700)
        cost = _cost("ideas", tin, tout)
        _llm(bus, ledger, run_id, pid, idea_id, "ideas", tin, tout, cost, rng, pace)
        ledger.add_idea(
            idea_id=idea_id,
            run_id=run_id,
            pipeline_id=pid,
            name=idea["name"],
            hypothesis=idea["hypothesis"],
            description=idea["description"],
            expected_holding_period_days=idea["expected_holding_period_days"],
            theme=theme,
        )
        bus.emit(
            PipelineEvent(
                run_id=run_id,
                pipeline_id=pid,
                idea_id=idea_id,
                stage="ideas",
                agent="ideas",
                type="idea_created",
                payload={"name": idea["name"], "theme": theme},
            )
        )

        outcome = rng.random()
        for attempt in range(1, 4):
            await asyncio.sleep(pace * rng.uniform(0.3, 1.2))
            tin, tout = rng.randint(3000, 6000), rng.randint(400, 1400)
            _llm(
                bus,
                ledger,
                run_id,
                pid,
                idea_id,
                "coder",
                tin,
                tout,
                _cost("coder", tin, tout),
                rng,
                pace,
            )
            attempt_id = uuid.uuid4().hex
            status = _attempt_status(rng, outcome, attempt)
            metrics = None
            if status not in ("failed_static",):
                metrics = _train_metrics(rng)
            ledger.add_attempt(
                attempt_id=attempt_id,
                idea_id=idea_id,
                run_id=run_id,
                attempt_number=attempt,
                code=f"# demo code {attempt}",
                code_hash=attempt_id,
                normalised_hash=attempt_id,
                status=status,
                static_ok=int(status != "failed_static"),
                sandbox_ok=int(status not in ("failed_static", "failed_sandbox")),
                lookahead_ok=int(
                    status not in ("failed_static", "failed_sandbox", "failed_lookahead")
                ),
                train_metrics_json=None if metrics is None else json.dumps(metrics),
                error=None if status not in ("failed_gate", "failed_critic") else "demo failure",
            )
            ledger.add_trial(uuid.uuid4().hex, run_id, attempt_id, attempt_id)
            for stage, etype in [
                ("static", "static_check"),
                ("sandbox", "sandbox_run"),
                ("lookahead", "lookahead_check"),
            ]:
                bus.emit(
                    PipelineEvent(
                        run_id=run_id,
                        pipeline_id=pid,
                        idea_id=idea_id,
                        attempt_number=attempt,
                        stage=stage,
                        type=etype,
                        payload={"ok": status != f"failed_{stage}"},
                    )
                )
                await asyncio.sleep(pace * 0.2)
            if metrics is not None:
                bus.emit(
                    PipelineEvent(
                        run_id=run_id,
                        pipeline_id=pid,
                        idea_id=idea_id,
                        attempt_number=attempt,
                        stage="train",
                        type="train_backtest",
                        payload=metrics,
                    )
                )
            if (
                status == "critic_pass"
                or status == "validation_pass"
                or status == "validation_fail"
            ):
                tin, tout = rng.randint(3000, 5000), rng.randint(200, 600)
                _llm(
                    bus,
                    ledger,
                    run_id,
                    pid,
                    idea_id,
                    "critic",
                    tin,
                    tout,
                    _cost("critic", tin, tout),
                    rng,
                    pace,
                )
                suspicion = "high" if status == "failed_critic" else "low"
                bus.emit(
                    PipelineEvent(
                        run_id=run_id,
                        pipeline_id=pid,
                        idea_id=idea_id,
                        attempt_number=attempt,
                        agent="critic",
                        stage="critic",
                        type="critic_review",
                        payload={"suspicion_level": suspicion, "concerns": []},
                    )
                )
                if status in ("validation_pass", "validation_fail"):
                    val_metrics = _train_metrics(rng, drift_shift=-0.6)
                    passed = status == "validation_pass"
                    ledger.add_validation(
                        attempt_id,
                        run_id,
                        val_metrics,
                        passed,
                        None,
                        _returns(np_rng, index, 0.0004 if passed else -0.0002),
                    )
                    ledger.update_attempt(
                        attempt_id,
                        train_returns_json=json.dumps(
                            _returns(
                                np_rng, pd.bdate_range("2010-01-01", "2018-12-31")[:400], 0.0003
                            )
                        ),
                    )
                    bus.emit(
                        PipelineEvent(
                            run_id=run_id,
                            pipeline_id=pid,
                            idea_id=idea_id,
                            attempt_number=attempt,
                            stage="validation",
                            type="validation_result",
                            payload={"passed": passed},
                        )
                    )
                break
            if status == "failed_static" or attempt == 3:
                break
            bus.emit(
                PipelineEvent(
                    run_id=run_id,
                    pipeline_id=pid,
                    idea_id=idea_id,
                    attempt_number=attempt + 1,
                    agent="coder",
                    stage="coder",
                    type="revision_requested",
                    payload={"reason": status},
                )
            )
        final = ledger.attempts_for_run(run_id)
        last = [a for a in final if a["idea_id"] == idea_id]
        final_status = last[-1]["status"] if last else "failed_coder"
        bus.emit(
            PipelineEvent(
                run_id=run_id,
                pipeline_id=pid,
                idea_id=idea_id,
                stage="pipeline",
                type="pipeline_finished",
                payload={"status": final_status, "attempts": len(last)},
            )
        )

    async def staggered(n: int) -> None:
        await asyncio.sleep(pace * 0.5 * (n % 10))
        await pipeline(n)

    await asyncio.gather(*(staggered(n) for n in range(n_pipelines)))
    total = sum(
        r["cost_usd"] or 0.0
        for r in ledger.conn.execute(
            "SELECT cost_usd FROM llm_calls WHERE run_id = ?", (run_id,)
        ).fetchall()
    )
    ledger.finish_run(run_id, "completed", total)
    bus.emit(
        PipelineEvent(
            run_id=run_id,
            stage="run",
            type="run_finished",
            payload={"status": "completed", "total_cost_usd": total, "demo": True},
        )
    )
    _ = settings  # kept so the demo validates against the real settings file


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


def _train_metrics(rng: random.Random, drift_shift: float = 0.0) -> dict:
    sharpe = rng.gauss(0.2 + drift_shift, 0.6)
    return {
        "n_days": 1800,
        "ann_return": sharpe * 0.1,
        "ann_volatility": 0.1,
        "sharpe": sharpe,
        "max_drawdown": -abs(rng.gauss(0.12, 0.05)),
        "avg_daily_turnover": abs(rng.gauss(0.12, 0.08)),
        "hit_rate": 0.5 + rng.gauss(0.0, 0.02),
    }


def _llm(bus, ledger, run_id, pid, idea_id, agent, tin, tout, cost, rng, pace) -> None:
    call_id = uuid.uuid4().hex
    latency = rng.randint(900, 4200)
    bus.emit(
        PipelineEvent(
            run_id=run_id,
            pipeline_id=pid,
            idea_id=idea_id,
            agent=agent,
            stage="llm",
            type="llm_call_started",
            payload={"call_id": call_id, "model": MODELS[agent]},
        )
    )
    ledger.add_llm_call(
        call_id=call_id,
        run_id=run_id,
        idea_id=idea_id,
        agent=agent,
        model=MODELS[agent],
        prompt_version=f"{agent}_v1",
        input_tokens=tin,
        output_tokens=tout,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=cost,
        latency_ms=latency,
        status="ok",
    )
    bus.emit(
        PipelineEvent(
            run_id=run_id,
            pipeline_id=pid,
            idea_id=idea_id,
            agent=agent,
            stage="llm",
            type="llm_call_finished",
            payload={
                "call_id": call_id,
                "status": "ok",
                "cost_usd": cost,
                "latency_ms": latency,
                "input_tokens": tin,
                "output_tokens": tout,
            },
        )
    )
    bus.emit(
        PipelineEvent(
            run_id=run_id, stage="budget", type="budget_update", payload={"spent_usd": cost}
        )
    )
