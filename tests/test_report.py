import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fakes import RoutingAnthropic, tool_response

from alpha_lab.backtest.metrics import deflated_sharpe_ratio, expected_max_sharpe, moments
from alpha_lab.config import GatesConfig, load_settings
from alpha_lab.events.bus import EventBus
from alpha_lab.ledger.db import Ledger
from alpha_lab.ledger.report import build_report_data, render_markdown, write_report
from alpha_lab.pipeline.run import execute_run

GOOD = """
import pandas as pd


def signal(prices: pd.DataFrame) -> pd.DataFrame:
    score = prices.shift(21) / prices.shift(252) - 1.0
    ranks = score.rank(axis=1, pct=True)
    long = (ranks > 0.8).astype(float)
    short = (ranks <= 0.2).astype(float)
    n_long = long.sum(axis=1).replace(0.0, float("nan"))
    n_short = short.sum(axis=1).replace(0.0, float("nan"))
    return (long.div(n_long, axis=0) - short.div(n_short, axis=0)).fillna(0.0)
"""
LOW = {"concerns": [], "suspicion_level": "low", "suggested_change": "none"}
IDEA = {
    "name": "Twelve month winners",
    "hypothesis": "Stocks that rose over the past year keep rising for a while.",
    "description": "Rank by return from 252 to 21 days ago. Long the top fifth, short the bottom fifth.",
    "expected_holding_period_days": 21,
}


def test_report_is_written_with_funnel_dsr_and_figures(tmp_path):
    settings = load_settings()
    settings = settings.model_copy(
        update={
            "gates": GatesConfig(
                min_train_sharpe=-100.0, min_validation_sharpe=-100.0, max_turnover_daily=10.0
            )
        }
    )
    index = pd.bdate_range("2010-01-01", "2021-12-31")
    rng = np.random.default_rng(23)
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, size=(len(index), 10)), axis=0)),
        index=index,
        columns=[f"T{i}" for i in range(10)],
    )
    ledger = Ledger(tmp_path / "report.db")
    client = RoutingAnthropic(
        {
            "submit_idea": [tool_response("submit_idea", IDEA)],
            "submit_code": [tool_response("submit_code", {"code": GOOD})],
            "submit_critique": [tool_response("submit_critique", LOW)],
        }
    )
    asyncio.run(
        execute_run(
            run_id="rep",
            settings=settings,
            prices=prices,
            n_pipelines=1,
            budget_usd=5.0,
            ledger=ledger,
            bus=EventBus(ledger),
            client=client,
        )
    )

    data = build_report_data(ledger, "rep")
    assert data.funnel[0] == ("ideas generated", 1)
    assert data.funnel[-1][0] == "passed validation"
    assert len(data.candidates) == 1
    assert np.isfinite(data.candidates[0].deflated_sharpe)

    markdown = render_markdown(data)
    for heading in [
        "## Funnel",
        "## Spend by agent and model",
        "## Multiple testing",
        "## Known limitations",
    ]:
        assert heading in markdown

    path = write_report(ledger, "rep", out_dir=tmp_path / "out")
    assert path.exists()
    assert (tmp_path / "out" / "report.html").exists()
    for name in ["galaxy.html", "equity.html"]:
        assert (tmp_path / "out" / "figures" / name).exists()
    assert json.loads(json.dumps(data.spend))  # spend rows are plain data


VAL_DAYS = pd.bdate_range("2019-01-01", periods=400)


def _add_validated(ledger, run_id, n, name, returns, passed=True):
    """One idea, one attempt with a train backtest, and one validation with daily returns."""
    idea_id, attempt_id = f"idea{n}", f"attempt{n}"
    ledger.add_idea(
        idea_id=idea_id,
        run_id=run_id,
        pipeline_id=f"p{n}",
        name=name,
        hypothesis="Test hypothesis text here.",
        description="Test description here.",
        expected_holding_period_days=21,
        theme="momentum",
    )
    ledger.add_attempt(
        attempt_id=attempt_id,
        idea_id=idea_id,
        run_id=run_id,
        attempt_number=1,
        status="validation_pass",
        static_ok=1,
        sandbox_ok=1,
        lookahead_ok=1,
        train_metrics_json=json.dumps({"sharpe": 0.5, "n_days": 1000.0}),
    )
    ledger.add_trial(f"trial{n}", run_id, attempt_id, attempt_id)
    series = None if returns is None else {str(d.date()): float(r) for d, r in returns.items()}
    ledger.add_validation(
        attempt_id, run_id, {"sharpe": 0.9, "avg_daily_turnover": 0.1}, passed, None, series
    )


def test_validation_rows_without_enough_returns_are_skipped_and_counted(tmp_path):
    ledger = Ledger(tmp_path / "short.db")
    ledger.create_run("short", "{}", 1.0)
    rng = np.random.default_rng(1)
    _add_validated(ledger, "short", 0, "no returns", None)
    _add_validated(
        ledger, "short", 1, "two returns", pd.Series(rng.normal(0, 0.01, 2), VAL_DAYS[:2])
    )
    _add_validated(ledger, "short", 2, "usable", pd.Series(rng.normal(0, 0.01, 60), VAL_DAYS[:60]))

    data = build_report_data(ledger, "short")

    assert [c.name for c in data.candidates] == ["usable"]
    assert data.skipped_validations == 2
    markdown = render_markdown(data)
    assert "validation rows skipped (no returns, or fewer than three daily returns): 2" in markdown
    # BE-02: the whole report, not only the data, must be written for this run.
    path = write_report(ledger, "short", out_dir=tmp_path / "out")
    assert path.exists()


def test_same_named_survivors_both_reach_the_correlation_matrix(tmp_path):
    rng = np.random.default_rng(2)
    days = VAL_DAYS[:300]
    base = rng.normal(0.0004, 0.01, 300)
    twin = base + rng.normal(0.0, 0.002, 300)  # a near-copy under the same name
    other = rng.normal(0.0003, 0.01, 300)
    ledger = Ledger(tmp_path / "dup.db")
    ledger.create_run("dup", "{}", 1.0)
    _add_validated(ledger, "dup", 0, "Twelve month winners", pd.Series(base, days))
    _add_validated(ledger, "dup", 1, "Twelve month winners", pd.Series(twin, days))
    _add_validated(ledger, "dup", 2, "Sector rotation", pd.Series(other, days))

    data = build_report_data(ledger, "dup")

    assert data.correlation is not None
    assert data.correlation.shape == (3, 3)
    assert data.correlation.columns.is_unique
    assert any(len(cluster) == 2 for cluster in data.clusters)


def test_trial_sharpes_use_each_runs_own_annualisation(tmp_path):
    ledger = Ledger(tmp_path / "ann.db")
    ledger.create_run(
        "old", json.dumps({"settings": {"backtest": {"annualization_days": 252}}}), 1.0
    )
    ledger.create_run(
        "new", json.dumps({"settings": {"backtest": {"annualization_days": 365}}}), 1.0
    )
    per_period = {"old": [0.02, -0.01, 0.03], "new": [0.01, 0.04, -0.02]}
    annual = {"old": 252, "new": 365}
    for run_id, values in per_period.items():
        for i, sr in enumerate(values):
            ledger.add_attempt(
                attempt_id=f"{run_id}{i}",
                idea_id="idea",
                run_id=run_id,
                attempt_number=1,
                status="failed_gate",
                train_metrics_json=json.dumps(
                    {"sharpe": sr * np.sqrt(annual[run_id]), "n_days": 1000.0}
                ),
            )
            ledger.add_trial(f"trial_{run_id}{i}", run_id, f"{run_id}{i}", f"{run_id}{i}")

    data = build_report_data(ledger, "new")

    all_per_period = [sr for values in per_period.values() for sr in values]
    variance = float(np.var(all_per_period, ddof=1))
    assert data.trial_sharpe_variance == pytest.approx(variance)
    assert data.expected_max_annual == pytest.approx(
        expected_max_sharpe(6, variance) * np.sqrt(365)
    )


def _best_of_noise_trials(rng, n_trials=200, t_train=1000, t_val=400):
    """Pure noise: n_trials strategies with zero true edge. Returns the report's DSR for the
    trial with the best validation Sharpe, and the same DSR computed by hand."""
    train = rng.normal(0.0, 0.01, (n_trials, t_train))
    val = rng.normal(0.0, 0.01, (n_trials, t_val))
    sr_train = train.mean(axis=1) / train.std(axis=1, ddof=1)  # per period
    sr_val = val.mean(axis=1) / val.std(axis=1, ddof=1)
    best = int(np.argmax(sr_val))

    # In-memory ledger: file-backed commits are about 60 times slower, and nothing needs the disk.
    ledger = Ledger(Path(":memory:"))
    ledger.create_run("noise", "{}", 1.0)
    ledger.add_idea(
        idea_id="idea",
        run_id="noise",
        pipeline_id="p",
        name="best",
        hypothesis="Test hypothesis text here.",
        description="Test description here.",
        expected_holding_period_days=21,
        theme="noise",
    )
    # Bulk inserts: Ledger.add_* commits on every row, which is too slow for a simulation.
    ledger.conn.executemany(
        "INSERT INTO attempts (attempt_id, idea_id, run_id, attempt_number, status, created_at,"
        " train_metrics_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (
                f"a{i}",
                "idea",
                "noise",
                1,
                "failed_gate",
                "2020-01-01",
                json.dumps({"sharpe": float(s) * np.sqrt(252), "n_days": float(t_train)}),
            )
            for i, s in enumerate(sr_train)
        ],
    )
    ledger.conn.executemany(
        "INSERT INTO trials (trial_id, run_id, attempt_id, code_hash, created_at)"
        " VALUES (?, ?, ?, ?, ?)",
        [(f"t{i}", "noise", f"a{i}", f"h{i}", "2020-01-01") for i in range(n_trials)],
    )
    ledger.conn.commit()
    days = pd.bdate_range("2019-01-01", periods=t_val)
    ledger.add_validation(
        f"a{best}",
        "noise",
        {"sharpe": float(sr_val[best]) * np.sqrt(252), "avg_daily_turnover": 0.1},
        False,
        None,
        {str(d.date()): float(r) for d, r in zip(days, val[best])},
    )
    data = build_report_data(ledger, "noise")
    ledger.close()

    trial_variance = float(np.var(sr_train, ddof=1))
    matched = (
        trial_variance * (t_train - 1) / (t_val - 1)
    )  # DECISION-6: rescale to validation length
    skew, kurt = moments(pd.Series(val[best]))
    by_hand = deflated_sharpe_ratio(float(sr_val[best]), t_val, skew, kurt, n_trials, matched)
    return data.candidates[0].deflated_sharpe, by_hand


def test_best_of_noise_trials_gets_a_dsr_near_one_half(tmp_path):
    # STAT-03: with the trial variance left at train length, the best noise trial
    # scored a median DSR of about 0.85. Length-matched, it is about 0.5.
    rng = np.random.default_rng(20260109)
    results = [_best_of_noise_trials(rng) for _ in range(120)]
    reported = np.array([r[0] for r in results])
    by_hand = np.array([r[1] for r in results])

    assert np.allclose(reported, by_hand, atol=1e-9)
    assert 0.35 < float(np.median(reported)) < 0.65
