"""Read-only queries behind the dashboard. Plain functions over the ledger, so they can be tested without a server.

Holdout rule: a signal's holdout metrics are returned only when its holdout run is
recorded as complete. Otherwise the holdout is reported as "locked" and no number is sent.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime

import pandas as pd

from alpha_lab.data.loader import read_pre_holdout
from alpha_lab.ledger.db import Ledger
from alpha_lab.ledger.report import build_report_data, equity_curves

TRAIN_PASS = ("critic_pass", "failed_critic", "validation_pass", "validation_fail")


def _finite(x):
    return x if isinstance(x, (int, float)) and math.isfinite(x) else None


def list_runs(ledger: Ledger) -> list[dict]:
    return [
        {
            "run_id": r["run_id"],
            "status": r["status"],
            "started_at": r["started_at"],
            "finished_at": r["finished_at"],
            "budget_usd": r["budget_usd"],
            "total_cost_usd": r["total_cost_usd"],
        }
        for r in ledger.list_runs(100)
    ]


def run_summary(ledger: Ledger, run_id: str) -> dict | None:
    run = ledger.get_run(run_id)
    if run is None:
        return None
    ideas = ledger.ideas_for_run(run_id)
    attempts = ledger.attempts_for_run(run_id)
    pipelines = {
        e["pipeline_id"]
        for e in ledger.events_for_run(run_id)
        if e["type"] == "pipeline_started" and e["pipeline_id"]
    }
    return {
        "run_id": run_id,
        "status": run["status"],
        "budget_usd": run["budget_usd"],
        "spent_usd": run["total_cost_usd"],
        "ideas": len(ideas),
        "pipelines": len(pipelines),
        "train_passes": len({a["idea_id"] for a in attempts if a["status"] in TRAIN_PASS}),
        "validation_passes": len(
            {a["idea_id"] for a in attempts if a["status"] == "validation_pass"}
        ),
        "rejected_ideas": ledger.count_rejected_ideas(),
    }


def funnel(ledger: Ledger, run_id: str) -> list[dict]:
    data = build_report_data(ledger, run_id)
    return [{"stage": stage, "count": count} for stage, count in data.funnel]


def llm_series(ledger: Ledger, run_id: str) -> list[dict]:
    rows = ledger.conn.execute(
        """SELECT created_at, agent, model, input_tokens, output_tokens, cost_usd, latency_ms, status
           FROM llm_calls WHERE run_id = ? ORDER BY created_at""",
        (run_id,),
    ).fetchall()
    return [
        {
            "t": r["created_at"],
            "agent": r["agent"],
            "model": r["model"],
            "input_tokens": r["input_tokens"],
            "output_tokens": r["output_tokens"],
            "cost_usd": r["cost_usd"],
            "latency_ms": r["latency_ms"],
            "status": r["status"],
        }
        for r in rows
    ]


def spend_by_agent(ledger: Ledger, run_id: str) -> list[dict]:
    return [
        {
            "agent": r["agent"],
            "model": r["model"],
            "calls": r["calls"],
            "cost_usd": r["cost_usd"] or 0.0,
        }
        for r in ledger.llm_cost_by_agent(run_id)
    ]


def galaxy(ledger: Ledger, run_id: str) -> dict:
    data = build_report_data(ledger, run_id)
    return {
        "points": [
            {
                "attempt_id": c.attempt_id,
                "name": c.name,
                "theme": c.theme,
                "train_sharpe": _finite(c.train_sharpe),
                "validation_sharpe": _finite(c.validation_sharpe),
                "deflated_sharpe": _finite(c.deflated_sharpe),
                "turnover": _finite(c.turnover),
                "revisions": c.revisions,
                "passed": c.passed,
            }
            for c in data.candidates
        ],
        "luck": {
            "train_sharpes": [_finite(s) for s in data.all_train_sharpes if _finite(s) is not None],
            "expected_max": _finite(data.expected_max_annual),
            "n_trials": data.n_trials,
        },
    }


def themes(ledger: Ledger, run_id: str) -> list[dict]:
    # NaN is not valid JSON, so empty medians are sent as null.
    return [
        {k: (_finite(v) if isinstance(v, float) else v) for k, v in row.items()}
        for row in build_report_data(ledger, run_id).themes
    ]


def failures(ledger: Ledger, run_id: str) -> list[dict]:
    return [{"status": s, "count": n} for s, n in build_report_data(ledger, run_id).failures]


def economics(ledger: Ledger) -> list[dict]:
    """Cost and outcome per agent and model, across all runs, so future model comparisons line up."""
    rows = ledger.conn.execute(
        """SELECT agent, model, COUNT(*) AS calls, SUM(cost_usd) AS cost,
                  SUM(CASE WHEN status = 'ok' THEN 1 ELSE 0 END) AS ok_calls
           FROM llm_calls GROUP BY agent, model ORDER BY agent, model"""
    ).fetchall()
    return [
        {
            "agent": r["agent"],
            "model": r["model"],
            "calls": r["calls"],
            "cost_usd": r["cost"] or 0.0,
            "success_rate": (r["ok_calls"] / r["calls"]) if r["calls"] else None,
        }
        for r in rows
    ]


def timeline(ledger: Ledger, run_id: str) -> list[dict]:
    """One lane per pipeline: LLM calls as segments, pipeline and attempt stages as markers."""
    lanes: dict[str, dict] = defaultdict(
        lambda: {"pipeline_id": None, "segments": [], "markers": []}
    )
    for e in ledger.events_for_run(run_id):
        pid = e["pipeline_id"]
        if not pid:
            continue
        lane = lanes[pid]
        lane["pipeline_id"] = pid
        payload = json.loads(e["payload_json"])
        if e["type"] == "llm_call_finished":
            lane["segments"].append(
                {
                    "t": e["timestamp"],
                    "agent": e["agent"],
                    "latency_ms": payload.get("latency_ms"),
                    "status": payload.get("status"),
                }
            )
        elif e["type"] in (
            "static_check",
            "sandbox_run",
            "lookahead_check",
            "train_backtest",
            "critic_review",
            "validation_result",
            "revision_requested",
            "pipeline_finished",
            "idea_rejected_duplicate",
        ):
            lane["markers"].append(
                {
                    "t": e["timestamp"],
                    "type": e["type"],
                    "stage": e["stage"],
                    "attempt": e["attempt_number"],
                }
            )
    return list(lanes.values())


def events_after(ledger: Ledger, run_id: str, after_seq: int, limit: int = 500) -> list[dict]:
    rows = ledger.conn.execute(
        """SELECT rowid AS seq, * FROM events WHERE run_id = ? AND rowid > ?
           ORDER BY rowid LIMIT ?""",
        (run_id, after_seq, limit),
    ).fetchall()
    return [
        {
            "seq": r["seq"],
            "type": r["type"],
            "stage": r["stage"],
            "agent": r["agent"],
            "pipeline_id": r["pipeline_id"],
            "idea_id": r["idea_id"],
            "attempt_number": r["attempt_number"],
            "timestamp": r["timestamp"],
            "payload": json.loads(r["payload_json"]),
        }
        for r in rows
    ]


def signal_detail(ledger: Ledger, attempt_id: str) -> dict | None:
    row = ledger.conn.execute(
        "SELECT * FROM attempts WHERE attempt_id = ?", (attempt_id,)
    ).fetchone()
    if row is None:
        # Accept an idea id too, and resolve it to that idea's latest attempt.
        row = ledger.conn.execute(
            "SELECT * FROM attempts WHERE idea_id = ? ORDER BY attempt_number DESC LIMIT 1",
            (attempt_id,),
        ).fetchone()
    if row is None:
        return None
    attempt_id = row["attempt_id"]
    idea = ledger.conn.execute(
        "SELECT * FROM ideas WHERE idea_id = ?", (row["idea_id"],)
    ).fetchone()
    siblings = ledger.conn.execute(
        "SELECT attempt_id, attempt_number, status, code, error FROM attempts WHERE idea_id = ? "
        "ORDER BY attempt_number",
        (row["idea_id"],),
    ).fetchall()
    calls = ledger.conn.execute(
        "SELECT agent, model, input_tokens, output_tokens, cost_usd, status FROM llm_calls "
        "WHERE idea_id = ? ORDER BY created_at",
        (row["idea_id"],),
    ).fetchall()
    validation = ledger.conn.execute(
        "SELECT passed, metrics_json FROM validations WHERE attempt_id = ?", (attempt_id,)
    ).fetchone()
    holdout = ledger.get_holdout(attempt_id)
    # Holdout numbers appear only once the holdout run is complete.
    holdout_view: dict = {"status": "locked"}
    if holdout is not None:
        holdout_view = {"status": holdout["status"]}
        if holdout["status"] == "complete" and holdout["metrics_json"]:
            holdout_view["metrics"] = json.loads(holdout["metrics_json"])
    return {
        "attempt_id": attempt_id,
        "idea": {
            "name": idea["name"],
            "hypothesis": idea["hypothesis"],
            "description": idea["description"],
            "theme": idea["theme"],
        }
        if idea
        else None,
        "attempts": [
            {
                "attempt_id": a["attempt_id"],
                "attempt_number": a["attempt_number"],
                "status": a["status"],
                "error": a["error"],
                "code": a["code"],
            }
            for a in siblings
        ],
        "critic": json.loads(row["critic_json"]) if row["critic_json"] else None,
        "train_metrics": json.loads(row["train_metrics_json"])
        if row["train_metrics_json"]
        else None,
        "validation": (
            {
                "passed": bool(validation["passed"]),
                "metrics": json.loads(validation["metrics_json"])
                if validation["metrics_json"]
                else None,
            }
            if validation is not None
            else None
        ),
        "holdout": holdout_view,
        "llm_calls": [dict(c) for c in calls],
    }


def equity(ledger: Ledger, run_id: str, top: int = 5) -> list[dict]:
    data = build_report_data(ledger, run_id)
    out = []
    for c in data.candidates[:top]:
        curve = equity_curves(c)
        running_max = curve.cummax()
        drawdown = curve / running_max - 1.0
        holdout = ledger.get_holdout(c.attempt_id)
        out.append(
            {
                "attempt_id": c.attempt_id,
                "name": c.name,
                "train_end": _iso(c.train_returns.index.max()) if len(c.train_returns) else None,
                "validation_end": _iso(c.validation_returns.index.max())
                if len(c.validation_returns)
                else None,
                "dates": [_iso(d) for d in curve.index],
                "equity": [float(v) for v in curve.values],
                "drawdown": [float(v) for v in drawdown.values],
                "holdout": "complete"
                if holdout is not None and holdout["status"] == "complete"
                else "locked",
            }
        )
    return out


def _iso(value) -> str:
    return datetime.fromisoformat(str(value)).date().isoformat() if value is not None else None


def correlation(ledger: Ledger, run_id: str) -> dict:
    data = build_report_data(ledger, run_id)
    if data.correlation is None:
        return {"names": [], "matrix": [], "clusters": []}
    names = list(data.correlation.columns)
    return {
        "names": names,
        "matrix": [[float(v) for v in row] for row in data.correlation.values],
        "clusters": data.clusters,
    }


def equal_weight_benchmark(curves_dates: list[str]) -> dict | None:
    """Equal weight across the universe, using pre-holdout prices only, over the given dates."""
    if not curves_dates:
        return None
    prices = read_pre_holdout()
    start, end = pd.Timestamp(curves_dates[0]), pd.Timestamp(curves_dates[-1])
    window = prices.loc[start:end]
    returns = window.pct_change(fill_method=None).mean(axis=1).fillna(0.0)
    equity_curve = (1.0 + returns).cumprod()
    lookup = {str(d.date()): float(v) for d, v in equity_curve.items()}
    dates = [d for d in curves_dates if d in lookup]
    return {"dates": dates, "equity": [lookup[d] for d in dates]}
