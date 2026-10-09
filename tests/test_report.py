import asyncio
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fakes import RoutingAnthropic, tool_response, with_test_cache_prices

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
    settings = with_test_cache_prices(load_settings())
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
    # One train-backtested trial cannot estimate the trial variance, so the DSR is NaN (DECISION-12).
    assert math.isnan(data.candidates[0].deflated_sharpe)
    assert "fewer than two" in data.dsr_reason

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


def _add_validated(
    ledger, run_id, n, name, returns, passed=True, train_sharpe=0.5, n_days=1000.0, lag1_sharpe=None
):
    """One idea, one attempt with a train backtest, and one validation with daily returns.

    n_days=None leaves n_days out of the train metrics (a ledger without train lengths). lag1_sharpe
    adds the lag-one Sharpe to the validation metrics, as the engine writes it (STAT-07).
    """
    idea_id, attempt_id = f"idea{n}", f"attempt{n}"
    train_metrics = (
        {"sharpe": train_sharpe}
        if n_days is None
        else {
            "sharpe": train_sharpe,
            "n_days": n_days,
        }
    )
    validation_metrics = {"sharpe": 0.9, "avg_daily_turnover": 0.1}
    if lag1_sharpe is not None:
        validation_metrics["sharpe_lag1"] = lag1_sharpe
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
        train_metrics_json=json.dumps(train_metrics),
    )
    ledger.add_trial(f"trial{n}", run_id, attempt_id, attempt_id)
    series = None if returns is None else {str(d.date()): float(r) for d, r in returns.items()}
    ledger.add_validation(attempt_id, run_id, validation_metrics, passed, None, series)


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
        {str(d.date()): float(r) for d, r in zip(days, val[best], strict=True)},
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


def _add_sandboxed_trials(ledger, run_id, n, idea_id="idea0", status="failed_sandbox"):
    """Trials that reached the sandbox but have no train Sharpe (no train backtest)."""
    for i in range(n):
        ledger.add_attempt(
            attempt_id=f"{status}{i}",
            idea_id=idea_id,
            run_id=run_id,
            attempt_number=2,
            status=status,
        )
        ledger.add_trial(f"{status}_t{i}", run_id, f"{status}{i}", f"{status}{i}")


def test_one_train_trial_among_fifty_has_no_deflated_sharpe(tmp_path):
    # R3-STAT-01 / R3-WHOLE-04. Before the fix, V was set to 0, SR0 to 0, and the DSR printed
    # 0.289 (the plain PSR) for N = 50. DECISION-12: it must be NaN with a reason.
    ledger = Ledger(tmp_path / "one.db")
    ledger.create_run("one", "{}", 1.0)
    rng = np.random.default_rng(5)
    returns = pd.Series(rng.normal(0.001, 0.01, 300), VAL_DAYS[:300])
    _add_validated(ledger, "one", 0, "lone survivor", returns)
    _add_sandboxed_trials(ledger, "one", 49)

    data = build_report_data(ledger, "one")

    assert data.n_trials == 50
    assert data.n_train_trials == 1
    assert math.isnan(data.candidates[0].deflated_sharpe)
    assert math.isnan(data.expected_max_annual)
    assert "fewer than two" in data.dsr_reason
    markdown = render_markdown(data)
    assert "trials in the ledger (N): 50" in markdown
    assert "with a finite train Sharpe (V): 1" in markdown
    assert "DSR: n/a" in markdown
    assert "fewer than two" in markdown


def test_trials_with_no_spread_have_no_deflated_sharpe(tmp_path):
    # Two trials with identical train Sharpe ratios: V = 0, so SR0 = 0 and the DSR would equal the
    # PSR. It must be NaN (DECISION-12).
    ledger = Ledger(tmp_path / "flat.db")
    ledger.create_run("flat", "{}", 1.0)
    rng = np.random.default_rng(6)
    for n, name in enumerate(["first", "second"]):
        returns = pd.Series(rng.normal(0.001, 0.01, 300), VAL_DAYS[:300])
        _add_validated(ledger, "flat", n, name, returns)

    data = build_report_data(ledger, "flat")

    assert all(math.isnan(c.deflated_sharpe) for c in data.candidates)
    assert "no spread" in data.dsr_reason


def test_few_train_trials_get_a_warning_in_the_report(tmp_path):
    ledger = Ledger(tmp_path / "few.db")
    ledger.create_run("few", "{}", 1.0)
    rng = np.random.default_rng(7)
    for n, sharpe in enumerate([0.1, 0.4, 0.7]):
        _add_validated(
            ledger,
            "few",
            n,
            f"signal {n}",
            pd.Series(rng.normal(0.001, 0.01, 300), VAL_DAYS[:300]),
            train_sharpe=sharpe,
        )

    data = build_report_data(ledger, "few")
    markdown = render_markdown(data)

    assert data.dsr_reason == ""
    assert all(math.isfinite(c.deflated_sharpe) for c in data.candidates)
    assert "warning" in markdown and "fewer than ten" in markdown


def test_failed_backtests_are_counted_in_the_multiple_testing_section(tmp_path):
    # STAT-02, report side: attempts that failed the train backtest are in N and not in V.
    ledger = Ledger(tmp_path / "fb.db")
    ledger.create_run("fb", "{}", 1.0)
    survivor = pd.Series(np.random.default_rng(8).normal(0.001, 0.01, 60), VAL_DAYS[:60])
    _add_validated(ledger, "fb", 0, "survivor", survivor)
    _add_sandboxed_trials(ledger, "fb", 2, status="failed_backtest")

    markdown = render_markdown(build_report_data(ledger, "fb"))

    assert "attempts that ended in failed_backtest (a data error" in markdown
    assert "failed_backtest" in markdown and "| 2 |" in markdown


def test_an_idea_whose_critic_output_failed_still_passed_the_train_gate(tmp_path):
    # R3-WHOLE-20 and DECISION-13. failed_critic_output comes after the train gate (single.py),
    # so the funnel must count it as past the gate and not as past the critic.
    ledger = Ledger(tmp_path / "critic.db")
    ledger.create_run("critic", "{}", 1.0)
    ledger.add_idea(
        idea_id="idea",
        run_id="critic",
        pipeline_id="p",
        name="no usable review",
        hypothesis="Test hypothesis text here.",
        description="Test description here.",
        expected_holding_period_days=21,
        theme="momentum",
    )
    ledger.add_attempt(
        attempt_id="attempt",
        idea_id="idea",
        run_id="critic",
        attempt_number=1,
        status="failed_critic_output",
        static_ok=1,
        sandbox_ok=1,
        lookahead_ok=1,
        train_metrics_json=json.dumps({"sharpe": 0.5, "n_days": 1000.0}),
    )

    funnel = dict(build_report_data(ledger, "critic").funnel)

    assert funnel["passed train gate"] == 1
    assert funnel["passed critic"] == 0


def test_no_train_lengths_means_no_validation_dsr(tmp_path):
    # R4-STAT-07. Without train lengths the trial variance cannot be length-matched (DECISION-6). The
    # DSR must fail closed with a reason, not fall back to the unmatched variance (DECISION-12).
    ledger = Ledger(tmp_path / "nolen.db")
    ledger.create_run("nolen", "{}", 1.0)
    rng = np.random.default_rng(9)
    for n, sharpe in enumerate([0.1, 0.9]):
        _add_validated(
            ledger,
            "nolen",
            n,
            f"signal {n}",
            pd.Series(rng.normal(0.001, 0.01, 300), VAL_DAYS[:300]),
            train_sharpe=sharpe,
            n_days=None,
        )

    data = build_report_data(ledger, "nolen")

    assert all(math.isnan(c.deflated_sharpe) for c in data.candidates)
    assert "length" in data.dsr_reason
    assert "DSR: n/a" in render_markdown(data)


def test_trials_differing_only_by_rounding_have_no_deflated_sharpe(tmp_path):
    # R4-STAT-01. Fifty train Sharpes that differ by 1e-10 relative are not a spread: the DSR must be
    # NaN with the reason, not the PSR.
    ledger = Ledger(tmp_path / "rounding.db")
    ledger.create_run("rounding", "{}", 1.0)
    rng = np.random.default_rng(10)
    for n in range(50):
        _add_validated(
            ledger,
            "rounding",
            n,
            f"signal {n}",
            pd.Series(rng.normal(0.001, 0.01, 300), VAL_DAYS[:300]),
            train_sharpe=0.5 * (1.0 + 1e-10 * n),
        )

    data = build_report_data(ledger, "rounding")

    assert data.n_train_trials == 50
    assert all(math.isnan(c.deflated_sharpe) for c in data.candidates)
    assert "no spread" in data.dsr_reason


def test_the_lag_one_sharpe_sits_next_to_the_default_in_the_report(tmp_path):
    # STAT-07. The report states the validation Sharpe at lag 1 next to the default. A row from a run
    # with no lag-one figure shows n/a rather than a number.
    ledger = Ledger(tmp_path / "lag.db")
    ledger.create_run("lag", "{}", 1.0)
    rng = np.random.default_rng(12)
    _add_validated(
        ledger,
        "lag",
        0,
        "with lag",
        pd.Series(rng.normal(0.001, 0.01, 300), VAL_DAYS[:300]),
        lag1_sharpe=0.4,
    )
    _add_validated(
        ledger,
        "lag",
        1,
        "older run",
        pd.Series(rng.normal(0.001, 0.01, 300), VAL_DAYS[:300]),
    )

    data = build_report_data(ledger, "lag")
    by_name = {c.name: c for c in data.candidates}
    assert by_name["with lag"].validation_sharpe_lag1 == pytest.approx(0.4)
    assert math.isnan(by_name["older run"].validation_sharpe_lag1)

    markdown = render_markdown(data)
    section = markdown.split("## Execution lag", 1)[1].split("## Theme leaderboard", 1)[0]
    assert "validation SR, lag 0" in section and "validation SR, lag 1" in section
    assert "| with lag | 0.900 | 0.400 |" in section
    assert "| older run | 0.900 | n/a |" in section
