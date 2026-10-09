"""Run report: funnel, spend, deflated Sharpe, correlation clusters, research figures.

Everything here is read from the ledger. Holdout results are included only when
an official holdout run exists for the signal (Phase 4), and are never inferred.
"""

from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd

from alpha_lab.backtest.metrics import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    moments,
    per_period_sharpe,
)
from alpha_lab.config import PROJECT_ROOT
from alpha_lab.ledger.db import Ledger

TRAIN_GATE_STATUSES = ("critic_pass", "failed_critic", "validation_pass", "validation_fail")
CRITIC_STATUSES = ("critic_pass", "validation_pass", "validation_fail")
CLUSTER_CORRELATION = 0.7
TOP_N = 10
ANNUALISATION = 252


@dataclass
class CandidateRow:
    name: str
    theme: str
    idea_id: str
    attempt_id: str
    train_sharpe: float
    validation_sharpe: float
    deflated_sharpe: float
    turnover: float
    revisions: int
    validation_returns: pd.Series
    train_returns: pd.Series
    passed: bool


@dataclass
class ReportData:
    run_id: str
    funnel: list[tuple[str, int]]
    spend: list[tuple[str, str, int, float]]
    candidates: list[CandidateRow] = field(default_factory=list)
    clusters: list[list[str]] = field(default_factory=list)
    correlation: pd.DataFrame | None = None
    all_train_sharpes: list[float] = field(default_factory=list)
    expected_max_annual: float = 0.0
    n_trials: int = 0
    trial_sharpe_variance: float = 0.0
    failures: list[tuple[str, int]] = field(default_factory=list)
    themes: list[dict] = field(default_factory=list)


def _series(data: str | None) -> pd.Series:
    if not data:
        return pd.Series(dtype="float64")
    obj = json.loads(data)
    s = pd.Series(obj, dtype="float64")
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def _clusters(corr: pd.DataFrame, threshold: float) -> list[list[str]]:
    parent = {c: c for c in corr.columns}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            if corr.loc[a, b] > threshold:
                parent[find(a)] = find(b)
    groups: dict[str, list[str]] = {}
    for c in cols:
        groups.setdefault(find(c), []).append(c)
    return [sorted(g) for g in groups.values() if len(g) > 1]


def build_report_data(
    ledger: Ledger, run_id: str, annualization: int = ANNUALISATION
) -> ReportData:
    conn = ledger.conn
    ideas = conn.execute("SELECT * FROM ideas WHERE run_id = ?", (run_id,)).fetchall()
    attempts = conn.execute("SELECT * FROM attempts WHERE run_id = ?", (run_id,)).fetchall()

    def distinct(rows, predicate) -> int:
        return len({r["idea_id"] for r in rows if predicate(r)})

    funnel = [
        ("ideas generated", len(ideas)),
        ("coded (at least one attempt)", distinct(attempts, lambda r: True)),
        ("passed static checks", distinct(attempts, lambda r: r["static_ok"] == 1)),
        ("ran in sandbox", distinct(attempts, lambda r: r["sandbox_ok"] == 1)),
        ("passed lookahead tests", distinct(attempts, lambda r: r["lookahead_ok"] == 1)),
        ("passed train gate", distinct(attempts, lambda r: r["status"] in TRAIN_GATE_STATUSES)),
        ("passed critic", distinct(attempts, lambda r: r["status"] in CRITIC_STATUSES)),
        ("passed validation", distinct(attempts, lambda r: r["status"] == "validation_pass")),
    ]

    spend = [
        (r["agent"], r["model"], int(r["calls"]), float(r["cost_usd"] or 0.0))
        for r in ledger.llm_cost_by_agent(run_id)
    ]

    # Trials and the spread of Sharpe ratios across them, from the whole ledger (the honest denominator).
    trial_sharpes = []
    for row in conn.execute(
        "SELECT train_metrics_json FROM attempts WHERE train_metrics_json IS NOT NULL"
    ):
        sharpe = json.loads(row["train_metrics_json"])["sharpe"]
        if math.isfinite(sharpe):
            trial_sharpes.append(sharpe / math.sqrt(annualization))  # per-period units
    n_trials = ledger.count_trials()
    variance = float(np.var(trial_sharpes, ddof=1)) if len(trial_sharpes) > 1 else 0.0
    expected_max = expected_max_sharpe(n_trials, variance) * math.sqrt(annualization)

    data = ReportData(
        run_id=run_id,
        funnel=funnel,
        spend=spend,
        all_train_sharpes=[s * math.sqrt(annualization) for s in trial_sharpes],
        expected_max_annual=expected_max,
        n_trials=n_trials,
        trial_sharpe_variance=variance,
    )

    failures = conn.execute(
        "SELECT status, COUNT(*) AS n FROM attempts WHERE run_id = ? GROUP BY status ORDER BY n DESC",
        (run_id,),
    ).fetchall()
    data.failures = [(r["status"], int(r["n"])) for r in failures]

    # Validation survivors (and every validated candidate) with their deflated Sharpe.
    candidates: list[CandidateRow] = []
    for v in ledger.validations():
        if v["run_id"] != run_id or v["metrics_json"] is None:
            continue
        metrics = json.loads(v["metrics_json"])
        val_returns = _series(v["returns_json"])
        attempt = conn.execute(
            "SELECT * FROM attempts WHERE attempt_id = ?", (v["attempt_id"],)
        ).fetchone()
        train_returns = _series(attempt["train_returns_json"])
        train_metrics = (
            json.loads(attempt["train_metrics_json"]) if attempt["train_metrics_json"] else {}
        )
        skew, kurt = moments(val_returns) if len(val_returns) > 2 else (0.0, 3.0)
        dsr = deflated_sharpe_ratio(
            sharpe=per_period_sharpe(val_returns) if len(val_returns) > 2 else 0.0,
            n_obs=len(val_returns),
            skew=skew,
            kurtosis=kurt,
            n_trials=max(n_trials, 2),
            sharpe_variance=variance,
        )
        candidates.append(
            CandidateRow(
                name=v["name"],
                theme=v["theme"] or "unknown",
                idea_id=attempt["idea_id"],
                attempt_id=attempt["attempt_id"],
                train_sharpe=float(train_metrics.get("sharpe", float("nan"))),
                validation_sharpe=float(metrics["sharpe"]),
                deflated_sharpe=float(dsr),
                turnover=float(metrics["avg_daily_turnover"]),
                revisions=int(attempt["attempt_number"]) - 1,
                validation_returns=val_returns,
                train_returns=train_returns,
                passed=bool(v["passed"]),
            )
        )
    candidates.sort(key=lambda c: -(c.deflated_sharpe if math.isfinite(c.deflated_sharpe) else -1))
    data.candidates = candidates

    survivors = [c for c in candidates if c.passed and c.validation_returns.size > 2]
    if len(survivors) >= 2:
        frame = pd.concat({c.name: c.validation_returns for c in survivors}, axis=1)
        data.correlation = frame.corr()
        data.clusters = _clusters(data.correlation, CLUSTER_CORRELATION)

    themes: dict[str, dict] = {}
    for idea in ideas:
        theme = idea["theme"] or "unknown"
        t = themes.setdefault(
            theme,
            {
                "theme": theme,
                "ideas": 0,
                "train": [],
                "val": [],
                "stage_pass": {"critic": 0, "validation": 0},
            },
        )
        t["ideas"] += 1
    for c in candidates:
        theme = c.theme
        t = themes.setdefault(
            theme,
            {
                "theme": theme,
                "ideas": 0,
                "train": [],
                "val": [],
                "stage_pass": {"critic": 0, "validation": 0},
            },
        )
        t["val"].append(c.validation_sharpe)
        if math.isfinite(c.train_sharpe):
            t["train"].append(c.train_sharpe)
    for row in attempts:
        if row["status"] in CRITIC_STATUSES:
            idea = next((i for i in ideas if i["idea_id"] == row["idea_id"]), None)
            if idea is not None:
                themes[idea["theme"] or "unknown"]["stage_pass"]["critic"] += 1
        if row["status"] == "validation_pass":
            idea = next((i for i in ideas if i["idea_id"] == row["idea_id"]), None)
            if idea is not None:
                themes[idea["theme"] or "unknown"]["stage_pass"]["validation"] += 1
    data.themes = [
        {
            "theme": t["theme"],
            "ideas": t["ideas"],
            "median_train_sharpe": median(t["train"]) if t["train"] else float("nan"),
            "median_validation_sharpe": median(t["val"]) if t["val"] else float("nan"),
            "critic_passes": t["stage_pass"]["critic"],
            "validation_passes": t["stage_pass"]["validation"],
        }
        for t in sorted(themes.values(), key=lambda x: x["theme"])
    ]
    return data


def top_by_deflated_sharpe(data: ReportData, n: int = TOP_N) -> list[CandidateRow]:
    return [c for c in data.candidates[:n]]


def equity_curves(candidate: CandidateRow) -> pd.Series:
    combined = pd.concat([candidate.train_returns, candidate.validation_returns]).sort_index()
    return (1.0 + combined).cumprod()


def render_markdown(data: ReportData) -> str:
    lines = [f"# Alpha Lab run report: {data.run_id}", ""]
    lines += ["## Funnel", "", "| stage | signals |", "|---|---:|"]
    lines += [f"| {stage} | {count} |" for stage, count in data.funnel]
    lines += [
        "",
        "## Spend by agent and model",
        "",
        "| agent | model | calls | USD |",
        "|---|---|---:|---:|",
    ]
    lines += [f"| {a} | {m} | {n} | {c:.4f} |" for a, m, n, c in data.spend]
    total = sum(c for *_, c in data.spend)
    lines += [f"| **total** | | | **{total:.4f}** |", ""]

    lines += [
        "## Multiple testing",
        "",
        f"- trials in the ledger (N): {data.n_trials}",
        f"- variance of train Sharpe ratios across trials (per period): {data.trial_sharpe_variance:.6g}",
        f"- expected maximum annualised Sharpe from pure noise at N: {data.expected_max_annual:.3f}",
        "",
    ]

    lines += [
        f"## Top {TOP_N} by deflated Sharpe",
        "",
        "| signal | theme | train SR | validation SR | DSR | turnover | revisions |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for c in top_by_deflated_sharpe(data):
        lines.append(
            f"| {c.name} | {c.theme} | {c.train_sharpe:.3f} | {c.validation_sharpe:.3f} | "
            f"{c.deflated_sharpe:.3f} | {c.turnover:.4f} | {c.revisions} |"
        )
    if not data.candidates:
        lines.append("| (no signal reached validation) | | | | | | |")

    lines += ["", "## Correlation clusters (daily returns above 0.7)", ""]
    if data.clusters:
        lines += [f"- {', '.join(cluster)}" for cluster in data.clusters]
    else:
        lines.append("None found among validation survivors.")

    lines += [
        "",
        "## Theme leaderboard",
        "",
        "| theme | ideas | median train SR | median validation SR | critic passes | validation passes |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for t in data.themes:
        lines.append(
            f"| {t['theme']} | {t['ideas']} | {_fmt(t['median_train_sharpe'])} | "
            f"{_fmt(t['median_validation_sharpe'])} | {t['critic_passes']} | {t['validation_passes']} |"
        )

    lines += [
        "",
        "## Failure anatomy (attempts by outcome)",
        "",
        "| outcome | attempts |",
        "|---|---:|",
    ]
    lines += [f"| {s} | {n} |" for s, n in data.failures]

    lines += [
        "",
        "## Known limitations",
        "",
        "- Survivorship bias: the universe is today's large caps, so results are optimistic.",
        "- Daily close-to-close execution, fixed basis-point costs, no borrowing costs, no slippage beyond that.",
        "- Turnover ignores drift, so constant-weight signals show zero turnover.",
        "- The LLM may have learned about the post-2010 market during training (subtle lookahead).",
        "- The deflated Sharpe assumes the trial Sharpe ratios are the search space that was actually explored.",
        "- Small universe of 50 names. Results are not evidence of a tradable edge.",
        "",
    ]
    return "\n".join(lines)


def _fmt(x: float) -> str:
    return "n/a" if x is None or not math.isfinite(x) else f"{x:.3f}"


def write_report(ledger: Ledger, run_id: str, out_dir: Path | None = None) -> Path:
    import plotly.graph_objects as go

    data = build_report_data(ledger, run_id)
    out_dir = out_dir or (PROJECT_ROOT / "runs" / run_id)
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    markdown = render_markdown(data)
    (out_dir / "report.md").write_text(markdown, encoding="utf-8")

    sections: list[str] = []

    # Signal galaxy: train SR (x) against validation SR (y), with the y = x line.
    if data.candidates:
        xs = [c.train_sharpe for c in data.candidates]
        ys = [c.validation_sharpe for c in data.candidates]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                text=[c.name for c in data.candidates],
                marker={"size": [max(4, 40 * c.turnover) for c in data.candidates]},
            )
        )
        lo, hi = min(xs + ys), max(xs + ys)
        fig.add_trace(
            go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="y = x", line={"dash": "dash"})
        )
        fig.update_layout(
            title="Signal galaxy: train vs validation Sharpe",
            xaxis_title="train Sharpe (annualised)",
            yaxis_title="validation Sharpe (annualised)",
        )
        sections.append(_save(fig, figures / "galaxy.html", "Signal galaxy", embed_js=not sections))

    # Luck distribution: all trial Sharpes, with the expected maximum from noise.
    if data.all_train_sharpes:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=data.all_train_sharpes, name="train Sharpe, all trials"))
        fig.add_vline(
            x=data.expected_max_annual, line_dash="dash", annotation_text="expected max from noise"
        )
        fig.update_layout(
            title="Luck distribution across trials",
            xaxis_title="train Sharpe (annualised)",
            yaxis_title="trials",
        )
        sections.append(
            _save(fig, figures / "luck.html", "Luck distribution", embed_js=not sections)
        )

    # Equity curves for the top five by deflated Sharpe.
    top5 = top_by_deflated_sharpe(data, 5)
    if top5:
        fig = go.Figure()
        for c in top5:
            fig.add_trace(
                go.Scatter(x=equity_curves(c).index, y=equity_curves(c).values, name=c.name)
            )
        fig.update_layout(
            title="Equity curves, top 5 by deflated Sharpe (train then validation)",
            yaxis_title="growth of 1.0",
            xaxis_title="date",
        )
        sections.append(_save(fig, figures / "equity.html", "Equity curves", embed_js=not sections))

    # Correlation heatmap for survivors.
    if data.correlation is not None:
        fig = go.Figure(
            go.Heatmap(
                z=data.correlation.values,
                x=list(data.correlation.columns),
                y=list(data.correlation.index),
                zmin=-1,
                zmax=1,
            )
        )
        fig.update_layout(title="Daily-return correlation between validation survivors")
        sections.append(
            _save(fig, figures / "correlation.html", "Correlation map", embed_js=not sections)
        )

    body = "\n".join(sections)
    page = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>Alpha Lab report {html.escape(run_id)}</title>"
        "<style>body{font-family:system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem}"
        "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px 8px}</style>"
        "</head><body>"
        f"<pre style='white-space:pre-wrap'>{html.escape(markdown)}</pre>"
        f"{body}</body></html>"
    )
    (out_dir / "report.html").write_text(page, encoding="utf-8")
    return out_dir / "report.md"


def _save(fig, path: Path, title: str, embed_js: bool) -> str:
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)
    fragment = fig.to_html(full_html=False, include_plotlyjs=embed_js)
    return f"<h2>{html.escape(title)}</h2>" + fragment
