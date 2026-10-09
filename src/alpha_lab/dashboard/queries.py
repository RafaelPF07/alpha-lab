"""Read-only queries behind the dashboard. Plain functions over the ledger, so they can be tested without a server.

Holdout rule (DECISION-1): a signal's holdout metrics are returned only when its holdout run is
recorded as complete. Otherwise the holdout is reported as "locked" and no holdout number is sent.
Training and validation numbers are shown on the research pages and in the signal inspector.
"""

from __future__ import annotations

import json
import math
import threading
from collections import OrderedDict, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_lab.config import DATA_DIR
from alpha_lab.data.loader import PRE_HOLDOUT_FILE, read_pre_holdout
from alpha_lab.ledger.db import Ledger
from alpha_lab.ledger.report import ReportData, build_report_data, equity_curves

TRAIN_PASS = ("critic_pass", "failed_critic", "validation_pass", "validation_fail")
TIMELINE_MARKERS = (
    "static_check",
    "sandbox_run",
    "lookahead_check",
    "train_backtest",
    "critic_review",
    "validation_result",
    "revision_requested",
    "pipeline_finished",
    "idea_rejected_duplicate",
    "duplicate_code",
)
EQUITY_MAX_POINTS = 500
REPORT_CACHE_SIZE = 16


def _finite(x):
    return x if isinstance(x, (int, float)) and math.isfinite(x) else None


# The report is built from ledger-wide data (the trial count and the spread of trial Sharpes), and it
# reads status and metric columns that are updated in place. So a cached copy is keyed by all of that.
_report_cache: OrderedDict[tuple, ReportData] = OrderedDict()
_report_lock = threading.Lock()


def _report_key(ledger: Ledger, run_id: str) -> tuple | None:
    """Cache key for a finished run. None for a run that is still running, or that does not exist."""
    row = ledger.conn.execute("SELECT status FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if row is None or row["status"] == "running":
        return None
    counts = ledger.conn.execute(
        """SELECT (SELECT COUNT(*) FROM attempts) AS attempts,
                  (SELECT COUNT(*) FROM attempts WHERE train_metrics_json IS NOT NULL) AS scored,
                  (SELECT COUNT(*) FROM validations) AS validations,
                  (SELECT COUNT(*) FROM trials) AS trials"""
    ).fetchone()
    return (
        str(ledger.path),
        run_id,
        counts["attempts"],
        counts["scored"],
        counts["validations"],
        counts["trials"],
    )


def report_for(ledger: Ledger, run_id: str) -> ReportData:
    """The run's report. A finished run is served from memory until the counts above change.
    A running run is rebuilt on every call, because its rows are still being written."""
    key = _report_key(ledger, run_id)
    if key is not None:
        with _report_lock:
            cached = _report_cache.get(key)
            if cached is not None:
                _report_cache.move_to_end(key)
                return cached
    data = build_report_data(ledger, run_id)
    if key is not None:
        with _report_lock:
            _report_cache[key] = data
            _report_cache.move_to_end(key)
            while len(_report_cache) > REPORT_CACHE_SIZE:
                _report_cache.popitem(last=False)
    return data


def _report(ledger: Ledger, run_id: str, data: ReportData | None) -> ReportData:
    return report_for(ledger, run_id) if data is None else data


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
    conn = ledger.conn
    attempts = conn.execute(
        "SELECT idea_id, status FROM attempts WHERE run_id = ?", (run_id,)
    ).fetchall()
    ideas = conn.execute("SELECT COUNT(*) FROM ideas WHERE run_id = ?", (run_id,)).fetchone()[0]
    pipelines = conn.execute(
        """SELECT COUNT(DISTINCT pipeline_id) FROM events
           WHERE run_id = ? AND type = 'pipeline_started' AND pipeline_id IS NOT NULL
             AND pipeline_id != ''""",
        (run_id,),
    ).fetchone()[0]
    rejected = conn.execute(
        "SELECT COUNT(*) FROM rejected_ideas WHERE run_id = ?", (run_id,)
    ).fetchone()[0]
    return {
        "run_id": run_id,
        "status": run["status"],
        "budget_usd": run["budget_usd"],
        "spent_usd": run["total_cost_usd"],
        "ideas": ideas,
        "pipelines": pipelines,
        "train_passes": len({a["idea_id"] for a in attempts if a["status"] in TRAIN_PASS}),
        "validation_passes": len(
            {a["idea_id"] for a in attempts if a["status"] == "validation_pass"}
        ),
        "rejected_ideas": rejected,
    }


def funnel(ledger: Ledger, run_id: str) -> list[dict]:
    data = report_for(ledger, run_id)
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


def galaxy(ledger: Ledger, run_id: str, data: ReportData | None = None) -> dict:
    data = _report(ledger, run_id, data)
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


def themes(ledger: Ledger, run_id: str, data: ReportData | None = None) -> list[dict]:
    # NaN is not valid JSON, so empty medians are sent as null.
    return [
        {k: (_finite(v) if isinstance(v, float) else v) for k, v in row.items()}
        for row in _report(ledger, run_id, data).themes
    ]


def failures(ledger: Ledger, run_id: str, data: ReportData | None = None) -> list[dict]:
    return [{"status": s, "count": n} for s, n in _report(ledger, run_id, data).failures]


def research(ledger: Ledger, run_id: str) -> dict:
    """Everything the research page needs. The report is built once and shared by the three parts."""
    data = report_for(ledger, run_id)
    return {
        "galaxy": galaxy(ledger, run_id, data),
        "themes": themes(ledger, run_id, data),
        "failures": failures(ledger, run_id, data),
        "economics": economics(ledger),
    }


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
        if e["type"] in ("llm_call_finished", "llm_call_refused"):
            refused = e["type"] == "llm_call_refused"  # never sent, so there is no latency
            payload = json.loads(e["payload_json"])
            lane["segments"].append(
                {
                    "t": e["timestamp"],
                    "agent": e["agent"],
                    "latency_ms": payload.get("latency_ms"),
                    "status": "refused" if refused else payload.get("status"),
                }
            )
        elif e["type"] in TIMELINE_MARKERS:
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
        "run_id": row["run_id"],
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


def _thin(n: int, deepest: int, max_points: int) -> list[int]:
    """Positions to keep: evenly spaced, plus both ends and the deepest drawdown, so the worst loss survives."""
    if n <= max_points:
        return list(range(n))
    keep = {int(i) for i in np.linspace(0, n - 1, max_points).round()}
    keep.update((0, n - 1, deepest))
    return sorted(keep)


def equity(
    ledger: Ledger,
    run_id: str,
    top: int = 5,
    max_points: int = EQUITY_MAX_POINTS,
    data: ReportData | None = None,
) -> list[dict]:
    """Equity and drawdown for the top candidates by deflated Sharpe, thinned to at most max_points.

    A candidate whose validation row has no returns (legacy rows) is left out. The drawdown is measured
    from max(running peak, 1.0), the rule in backtest/metrics.py, so it agrees with the reported max drawdown.
    """
    data = _report(ledger, run_id, data)
    out = []
    usable = [c for c in data.candidates if len(c.validation_returns) > 0]
    for c in usable[:top]:
        curve = equity_curves(c)
        values = curve.to_numpy(dtype=float)
        peak = np.maximum(np.maximum.accumulate(values), 1.0)
        drawdown = values / peak - 1.0
        keep = _thin(len(values), int(np.argmin(drawdown)), max_points)
        holdout = ledger.get_holdout(c.attempt_id)
        out.append(
            {
                "attempt_id": c.attempt_id,
                "name": c.name,
                "train_end": _iso(c.train_returns.index.max()) if len(c.train_returns) else None,
                "validation_end": _iso(c.validation_returns.index.max()),
                "dates": [_iso(curve.index[i]) for i in keep],
                "equity": [float(values[i]) for i in keep],
                "drawdown": [float(drawdown[i]) for i in keep],
                "holdout": "complete"
                if holdout is not None and holdout["status"] == "complete"
                else "locked",
            }
        )
    return out


def _iso(value) -> str:
    return datetime.fromisoformat(str(value)).date().isoformat() if value is not None else None


def correlation(ledger: Ledger, run_id: str, data: ReportData | None = None) -> dict:
    data = _report(ledger, run_id, data)
    if data.correlation is None:
        return {"names": [], "matrix": [], "clusters": []}
    names = list(data.correlation.columns)
    return {
        "names": names,
        "matrix": [[float(v) for v in row] for row in data.correlation.values],
        "clusters": data.clusters,
    }


_price_cache: dict[Path, tuple[int, pd.DataFrame]] = {}


def _pre_holdout_prices() -> pd.DataFrame:
    """Pre-holdout prices, re-read only when the cache file changes. Raises FileNotFoundError if it is missing."""
    path = DATA_DIR / PRE_HOLDOUT_FILE
    stamp = path.stat().st_mtime_ns
    cached = _price_cache.get(path)
    if cached is None or cached[0] != stamp:
        cached = (stamp, read_pre_holdout(DATA_DIR))
        _price_cache[path] = cached
    return cached[1]


def equal_weight_benchmark(curves_dates: list[str]) -> dict | None:
    """Equal weight across the universe, using pre-holdout prices only, over the given dates."""
    if not curves_dates:
        return None
    try:
        prices = _pre_holdout_prices()
    except FileNotFoundError:
        return {
            "dates": [],
            "equity": [],
            "reason": "pre-holdout prices are not cached yet; run `lab data fetch`",
        }
    start, end = pd.Timestamp(curves_dates[0]), pd.Timestamp(curves_dates[-1])
    window = prices.loc[start:end]
    returns = window.pct_change(fill_method=None).mean(axis=1).fillna(0.0)
    equity_curve = (1.0 + returns).cumprod()
    lookup = {str(d.date()): float(v) for d, v in equity_curve.items()}
    dates = [d for d in curves_dates if d in lookup]
    return {"dates": dates, "equity": [lookup[d] for d in dates]}
