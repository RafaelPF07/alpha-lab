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
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_lab import statuses
from alpha_lab.config import DATA_DIR, load_settings, load_universe
from alpha_lab.data.loader import (
    PRE_HOLDOUT_FILE,
    CacheMismatch,
    check_fingerprint,
    read_checked_pre_holdout,
)
from alpha_lab.ledger.db import Ledger
from alpha_lab.ledger.report import ReportData, build_report_data, equity_curves
from alpha_lab.pipeline.run import STALE_RUN_AFTER

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
# A holdout row still 'running' after this long was probably left by a killed process (R3-SEC-11).
HOLDOUT_STALE_AFTER = timedelta(hours=6)
REPORT_CACHE_SIZE = 16  # at most this many finished-run reports are kept ...
REPORT_CACHE_CANDIDATES = (
    400  # ... and at most this many candidates in total (about 18 MB at 100 each)
)


def _finite(x):
    return x if isinstance(x, (int, float)) and math.isfinite(x) else None


# The report is built from ledger-wide data (the trial count and the spread of trial Sharpes), and it
# reads status and metric columns that are updated in place. So a cached copy is keyed by all of that.
# Each entry keeps its candidate count, which is what the memory budget is counted in.
_report_cache: OrderedDict[tuple, tuple[ReportData, int]] = OrderedDict()
_report_lock = threading.Lock()


def _report_key(ledger: Ledger, run_id: str) -> tuple | None:
    """Cache key for a finished run. None for a run that is still running, or that does not exist.
    The newest rowid of attempts and validations changes when a row is replaced, even if the counts
    stay the same. The key runs on every request, so it reads only counts and maxima: a scan of
    attempts.train_metrics_json would read every attempt's code too (R4-BE-05)."""
    row = ledger.conn.execute("SELECT status FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if row is None or row["status"] == "running":
        return None
    counts = ledger.conn.execute(
        """SELECT (SELECT COUNT(*) FROM attempts) AS attempts,
                  (SELECT COUNT(*) FROM validations) AS validations,
                  (SELECT COUNT(*) FROM trials) AS trials,
                  (SELECT MAX(rowid) FROM attempts) AS last_attempt,
                  (SELECT MAX(rowid) FROM validations) AS last_validation"""
    ).fetchone()
    return (
        str(ledger.path),
        run_id,
        counts["attempts"],
        counts["validations"],
        counts["trials"],
        counts["last_attempt"],
        counts["last_validation"],
    )


def _trim_report_cache() -> None:
    """Drop the least recently used reports until both budgets hold. The newest entry always stays."""
    total = sum(weight for _, weight in _report_cache.values())
    while len(_report_cache) > 1 and (
        len(_report_cache) > REPORT_CACHE_SIZE or total > REPORT_CACHE_CANDIDATES
    ):
        _, (_, weight) = _report_cache.popitem(last=False)
        total -= weight


def report_for(ledger: Ledger, run_id: str) -> ReportData:
    """The run's report. A finished run is served from memory until the key above changes.
    A running run is rebuilt on every call, because its rows are still being written."""
    key = _report_key(ledger, run_id)
    if key is not None:
        with _report_lock:
            cached = _report_cache.get(key)
            if cached is not None:
                _report_cache.move_to_end(key)
                return cached[0]
    data = build_report_data(ledger, run_id)
    if key is not None:
        with _report_lock:
            _report_cache[key] = (data, max(1, len(data.candidates)))
            _report_cache.move_to_end(key)
            _trim_report_cache()
    return data


def _report(ledger: Ledger, run_id: str, data: ReportData | None) -> ReportData:
    return report_for(ledger, run_id) if data is None else data


def stale_flag(stored_status: str, shown_status: str | None) -> bool:
    """`stale` is only ever set for a status that is not final, and no run-level status is both
    non-final and silent. A silent run is shown as 'abandoned' (final), so this is False for every run.
    The ledger's own status is still sent as `stored_status`, which stays 'running' for such a run.
    Holdout rows keep their own stale flag (signal_detail), which applies to a 'running' status."""
    return False


def list_runs(ledger: Ledger) -> list[dict]:
    """Runs with the status a reader should show. A dead run reads 'abandoned' before the reaper runs
    (Ledger.effective_run_status). See stale_flag for when 'stale' is set."""
    rows = []
    for r in ledger.list_runs(100):
        shown = ledger.effective_run_status(r["run_id"], STALE_RUN_AFTER)
        rows.append(
            {
                "run_id": r["run_id"],
                "status": shown,
                "stored_status": r["status"],
                "stale": stale_flag(r["status"], shown),
                "started_at": r["started_at"],
                "finished_at": r["finished_at"],
                "budget_usd": r["budget_usd"],
                "total_cost_usd": r["total_cost_usd"],
            }
        )
    return rows


def run_state(ledger: Ledger, run_id: str) -> dict:
    """The status fields a reader shows for one run. The stream sends them as its closing frame, so a
    page knows why the stream ended. An unknown run has no status (R4-BE-11)."""
    run = ledger.get_run(run_id)
    if run is None:
        return {"status": None, "stored_status": None, "stale": False}
    shown = ledger.effective_run_status(run_id, STALE_RUN_AFTER)
    return {
        "status": shown,
        "stored_status": run["status"],
        "stale": stale_flag(run["status"], shown),
    }


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
    shown = ledger.effective_run_status(run_id, STALE_RUN_AFTER)
    return {
        "run_id": run_id,
        "status": shown,
        "stored_status": run["status"],
        "stale": stale_flag(run["status"], shown),
        "budget_usd": run["budget_usd"],
        "spent_usd": run["total_cost_usd"],
        "ideas": ideas,
        "pipelines": pipelines,
        # Counts of distinct ideas, named after the statuses.py set each one counts (R4-WHOLE-10).
        "passed_train_gate": len(
            {a["idea_id"] for a in attempts if a["status"] in statuses.PASSED_TRAIN_GATE}
        ),
        "passed_critic": len(
            {a["idea_id"] for a in attempts if a["status"] in statuses.PASSED_CRITIC}
        ),
        "passed_validation": len(
            {a["idea_id"] for a in attempts if a["status"] in statuses.PASSED_VALIDATION}
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
    attempt_status = {
        r["attempt_id"]: r["status"]
        for r in ledger.conn.execute(
            "SELECT attempt_id, status FROM attempts WHERE run_id = ?", (run_id,)
        )
    }
    return {
        "points": [
            {
                "attempt_id": c.attempt_id,
                "name": c.name,
                "theme": c.theme,
                "status": attempt_status.get(c.attempt_id),
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
            payload = _parsed_payload(e["payload_json"])["payload"] or {}
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


def has_current_schema(ledger: Ledger) -> bool:
    """True when the events table has the seq key. Ledgers written before it need `lab ledger migrate`,
    which a read-only connection cannot run (the server answers 503 with that instruction)."""
    return any(row["name"] == "seq" for row in ledger.conn.execute("PRAGMA table_info(events)"))


def _parsed_payload(text: str) -> dict:
    """The payload as an object. A stored text that is not JSON gives None and a payload_error, so one
    bad row cannot fail the page or the stream (R4-BE-03). The ledger's own writers always store JSON."""
    try:
        return {"payload": json.loads(text)}
    except (TypeError, ValueError):
        return {"payload": None, "payload_error": "payload_json is not valid JSON"}


def events_after(ledger: Ledger, run_id: str, after_seq: int, limit: int = 500) -> list[dict]:
    rows = ledger.conn.execute(
        """SELECT seq, * FROM events WHERE run_id = ? AND seq > ?
           ORDER BY seq LIMIT ?""",
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
            **_parsed_payload(r["payload_json"]),
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
    holdout_view: dict = {"status": statuses.HOLDOUT_LOCKED_VIEW}
    if holdout is not None:
        holdout_view = {"status": holdout["status"]}
        if holdout["status"] == "running":
            # Still "running" with no finish: the status stays locked, and the page is told it is stale.
            started = datetime.fromisoformat(holdout["created_at"])
            if started.tzinfo is None:
                started = started.replace(tzinfo=UTC)
            holdout_view["started_at"] = holdout["created_at"]
            holdout_view["stale"] = datetime.now(UTC) - started > HOLDOUT_STALE_AFTER
        if holdout["status"] == "complete" and holdout["metrics_json"]:
            holdout_view["metrics"] = json.loads(holdout["metrics_json"])
    return {
        "attempt_id": attempt_id,
        "run_id": row["run_id"],
        # Run-level, as on the equity curves: empty when the deflated Sharpe is defined.
        "dsr_reason": report_for(ledger, row["run_id"]).dsr_reason,
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
    """Positions to keep: evenly spaced, plus both ends and the deepest drawdown, so the worst loss survives.
    The result has at most max_points positions (max_points of 3 or more). When the deepest point is off
    the grid, it replaces the grid point nearest to it, so the count does not grow (R4-BE-09)."""
    if n <= max_points:
        return list(range(n))
    grid = {int(i) for i in np.linspace(0, n - 1, max_points).round()}
    if deepest not in grid and len(grid) > 2:
        grid.discard(min(grid - {0, n - 1}, key=lambda i: abs(i - deepest)))
        grid.add(deepest)
    return sorted(grid)


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
                # The validation verdict, as the report counts it (validations.passed).
                "passed": c.passed,
                "holdout": "complete"
                if holdout is not None and holdout["status"] == "complete"
                else statuses.HOLDOUT_LOCKED_VIEW,
                # Run-level: empty when the deflated Sharpe is defined, else why it is not (DECISION-12).
                "dsr_reason": data.dsr_reason,
            }
        )
    return out


def limits() -> dict:
    """Thresholds the context and cost panels draw as lines. They come from settings.toml, not from the pages."""
    settings = load_settings()
    return {
        "pricing_tier_tokens": settings.pricing.long_context_threshold_tokens,
        "context_limit_tokens": settings.agents.context_limit_tokens,
    }


def status_sets() -> dict[str, list[str]]:
    """The status sets from statuses.py, as sorted lists. The pages get them here and do not hard-code them."""
    sets = {
        "attempt": statuses.ATTEMPT_STATUSES,
        "pipeline": statuses.PIPELINE_STATUSES,
        "run": statuses.RUN_STATUSES,
        "aborted": statuses.ABORTED_STATUSES,
        "error_pipeline": statuses.ERROR_PIPELINE_STATUSES,
        "passed_train_gate": statuses.PASSED_TRAIN_GATE,
        "passed_critic": statuses.PASSED_CRITIC,
        "passed_validation": statuses.PASSED_VALIDATION,
        "validation_verdicts": statuses.VALIDATION_VERDICTS,
        "llm_call": statuses.LLM_CALL_STATUSES,
        # The holdout values the API sends: the stored statuses and the "locked" view (R4-BE-07).
        "holdout": statuses.HOLDOUT_VIEWS,
    }
    return {name: sorted(values) for name, values in sets.items()}


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
    """Pre-holdout prices through the checked loader: a cache built for other settings raises CacheMismatch.
    The parquet is re-read only when it changes. Raises FileNotFoundError if it is missing."""
    path = DATA_DIR / PRE_HOLDOUT_FILE
    stamp = path.stat().st_mtime_ns
    settings, universe = load_settings(), load_universe()
    cached = _price_cache.get(path)
    if cached is not None and cached[0] == stamp:
        check_fingerprint(settings, universe, DATA_DIR)  # settings may have changed since the read
        return cached[1]
    prices = read_checked_pre_holdout(settings, universe, DATA_DIR)
    _price_cache[path] = (stamp, prices)
    return prices


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
    except CacheMismatch as error:
        return {"dates": [], "equity": [], "reason": f"cache stale: {error}"}
    start, end = pd.Timestamp(curves_dates[0]), pd.Timestamp(curves_dates[-1])
    window = prices.loc[start:end]
    returns = window.pct_change(fill_method=None).mean(axis=1).fillna(0.0)
    equity_curve = (1.0 + returns).cumprod()
    lookup = {str(d.date()): float(v) for d, v in equity_curve.items()}
    dates = [d for d in curves_dates if d in lookup]
    return {"dates": dates, "equity": [lookup[d] for d in dates]}
