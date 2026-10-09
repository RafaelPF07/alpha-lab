"""Dashboard data layer and demo stream. No web server and no network: these call the query functions directly.

The HTTP routes, strict JSON on the wire and the live stream are tested in test_dashboard_http.py.
"""

import asyncio
import dataclasses
import json
import uuid
from datetime import UTC, datetime
from typing import get_args

import numpy as np
import pandas as pd
import pytest

from alpha_lab.backtest.metrics import Metrics, compute_metrics
from alpha_lab.config import load_settings
from alpha_lab.dashboard import queries
from alpha_lab.data.loader import PRE_HOLDOUT_FILE
from alpha_lab.events import demo
from alpha_lab.events.bus import EventBus, EventType, PipelineEvent
from alpha_lab.events.demo import DEMO_RUN_ID, run_demo
from alpha_lab.ledger.db import LEDGER_PATH, Ledger

# Payload keys each event type carries, as the real emitters write them: pipeline/single.py,
# pipeline/run.py, agents/llm.py and checks/lookahead.py. train_backtest carries the Metrics fields.
REAL_PAYLOAD_KEYS = {
    "run_started": {"pipelines", "budget_usd", "concurrency"},
    "run_finished": {
        "status",
        "total_cost_usd",
        "budget_usd",
        "reserved_usd",
        "data_failed_pipelines",
    },
    "pipeline_started": {"theme"},
    "pipeline_finished": {"status", "attempts", "error", "theme"},
    "idea_created": {"name", "hypothesis", "description", "expected_holding_period_days", "theme"},
    "idea_rejected_duplicate": {"name", "similarity", "threshold"},
    "llm_call_started": {"call_id", "model", "prompt_version", "estimated_input_tokens"},
    "llm_call_finished": {
        "call_id",
        "status",
        "cost_usd",
        "latency_ms",
        "input_tokens",
        "output_tokens",
    },
    "budget_update": {"spent_usd", "limit_usd"},
    "code_written": {"code_hash", "lines"},
    "duplicate_code": {"earlier_attempt_id", "message"},
    "llm_call_refused": {"call_id", "model", "reason", "estimated_cost_usd"},
    "llm_call_retrying": {"call_id", "attempt", "delay_seconds", "error_type"},
    "budget_overrun": {"spent_usd", "limit_usd"},
    "static_check": {"ok", "errors", "warnings"},
    "sandbox_run": {"status", "error_type", "message"},
    "lookahead_check": {"test", "passed", "cutoff", "first_offending_date", "ticker"},
    "train_backtest": {field.name for field in dataclasses.fields(Metrics)},
    "critic_review": {"suspicion_level", "concerns", "suggested_change"},
    "revision_requested": {"reason"},
    "validation_result": {"passed"},
}


def demo_ledger(tmp_path, pipelines=8, seed=3):
    ledger = Ledger(tmp_path / "demo.db")
    bus = EventBus(ledger)
    asyncio.run(run_demo(ledger, bus, n_pipelines=pipelines, pace=0.0, seed=seed))
    return ledger


def new_ledger(tmp_path, run_id="r1"):
    ledger = Ledger(tmp_path / "edge.db")
    ledger.create_run(run_id, "{}", 1.0)
    return ledger


def add_candidate(ledger, name, *, train, validation, run_id="r1", status="validation_pass"):
    """One idea with one attempt, one trial and one validation row, from explicit return series."""
    idea_id = uuid.uuid4().hex
    ledger.add_idea(
        idea_id=idea_id,
        run_id=run_id,
        pipeline_id=uuid.uuid4().hex,
        name=name,
        hypothesis="h",
        description="d",
        expected_holding_period_days=21,
        theme="momentum",
    )
    attempt_id = uuid.uuid4().hex
    ledger.add_attempt(
        attempt_id=attempt_id,
        idea_id=idea_id,
        run_id=run_id,
        attempt_number=1,
        code="def signal(prices): return prices * 0.0",
        code_hash=attempt_id,
        normalised_hash=attempt_id,
        status=status,
        static_ok=1,
        sandbox_ok=1,
        lookahead_ok=1,
        train_metrics_json=json.dumps({"sharpe": 0.5}),
        train_returns_json=None if train is None else json.dumps(train),
    )
    ledger.add_trial(uuid.uuid4().hex, run_id, attempt_id, attempt_id)
    ledger.add_validation(
        attempt_id, run_id, {"sharpe": 0.6, "avg_daily_turnover": 0.1}, True, None, validation
    )
    return attempt_id


@pytest.fixture(scope="module")
def big_demo(tmp_path_factory):
    """40 pipelines with seed 7: every event type appears. Read-only for the tests that use it."""
    ledger = demo_ledger(tmp_path_factory.mktemp("big"), pipelines=40, seed=7)
    yield ledger
    ledger.close()


@pytest.fixture(scope="module")
def small_demo(tmp_path_factory):
    """8 pipelines with seed 3. Read-only for the tests that use it."""
    ledger = demo_ledger(tmp_path_factory.mktemp("small"))
    yield ledger
    ledger.close()


def test_demo_payloads_have_the_real_event_keys(big_demo):
    seen = set()
    for row in big_demo.events_for_run(DEMO_RUN_ID):
        payload = json.loads(row["payload_json"])
        # Every row must parse back into the typed event model.
        PipelineEvent(
            run_id=row["run_id"],
            stage=row["stage"] or "run",
            type=row["type"],
            pipeline_id=row["pipeline_id"],
            payload=payload,
        )
        assert set(payload) == REAL_PAYLOAD_KEYS[row["type"]], row["type"]
        seen.add(row["type"])
    # budget_overrun needs the run to spend past its budget. The default demo spends about 0.24 of its
    # 2 USD, so it does not emit that type. Its keys are still checked against the table above.
    assert seen == set(get_args(EventType)) - {"budget_overrun"}, (
        "the demo should exercise every event type it can"
    )


def test_demo_has_one_started_and_one_finished_pipeline_per_idea(small_demo):
    rows = small_demo.events_for_run(DEMO_RUN_ID)
    started = [r for r in rows if r["type"] == "pipeline_started"]
    finished = [r for r in rows if r["type"] == "pipeline_finished"]
    assert len(started) == 8
    assert len(finished) == 8


def test_demo_run_is_not_regenerated(tmp_path):
    ledger = demo_ledger(tmp_path)
    before = len(ledger.events_for_run(DEMO_RUN_ID))
    asyncio.run(run_demo(ledger, EventBus(ledger), n_pipelines=8, pace=0.0, seed=3))
    assert len(ledger.events_for_run(DEMO_RUN_ID)) == before
    ledger.close()


def test_demo_writes_no_holdout_rows(big_demo):
    assert big_demo.holdouts() == []


def test_demo_records_a_trial_only_for_attempts_that_reached_the_sandbox(big_demo):
    # Static failures and duplicate code stop before the sandbox, so they have no trial.
    stopped_early = {"failed_static", "rejected_duplicate_code"}
    reached = [
        a for a in big_demo.attempts_for_run(DEMO_RUN_ID) if a["status"] not in stopped_early
    ]
    assert big_demo.count_trials() == len(reached)


def test_lookahead_events_name_the_test_they_ran(big_demo):
    tests = [
        json.loads(r["payload_json"])["test"]
        for r in big_demo.events_for_run(DEMO_RUN_ID)
        if r["type"] == "lookahead_check"
    ]
    assert set(tests) == {"truncation", "perturbation"}
    assert tests.count("truncation") == tests.count("perturbation")


def test_demo_refuses_to_write_to_the_real_ledger(tmp_path):
    ledger = Ledger(tmp_path / "demo.db")
    ledger.path = LEDGER_PATH  # the attribute only: the guard must raise before any write
    with pytest.raises(ValueError, match="real ledger"):
        asyncio.run(run_demo(ledger, EventBus(ledger), n_pipelines=2, pace=0.0))
    assert ledger.get_run(DEMO_RUN_ID) is None
    ledger.close()


def test_an_interrupted_demo_is_marked_interrupted(tmp_path):
    ledger = Ledger(tmp_path / "demo.db")

    async def cancel_midway():
        task = asyncio.create_task(
            run_demo(ledger, EventBus(ledger), n_pipelines=8, pace=0.2, seed=3)
        )
        await asyncio.sleep(0.5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_midway())
    run = ledger.get_run(DEMO_RUN_ID)
    assert run["status"] == "interrupted"
    assert run["finished_at"] is not None
    ledger.close()


def test_summary_and_funnel_come_from_the_ledger(small_demo):
    summary = queries.run_summary(small_demo, DEMO_RUN_ID)
    assert summary["pipelines"] == 8
    assert summary["ideas"] == 8
    funnel = queries.funnel(small_demo, DEMO_RUN_ID)
    assert funnel[0] == {"stage": "ideas generated", "count": 8}


def test_summary_counts_rejected_ideas_for_this_run_only(tmp_path):
    ledger = demo_ledger(tmp_path)
    ledger.create_run("other", "{}", 1.0)
    ledger.add_rejected_idea(
        rejection_id=uuid.uuid4().hex,
        run_id="other",
        pipeline_id=uuid.uuid4().hex,
        name="elsewhere",
        similar_to_idea_id=None,
        similarity=0.99,
    )
    own = ledger.conn.execute(
        "SELECT COUNT(*) FROM rejected_ideas WHERE run_id = ?", (DEMO_RUN_ID,)
    ).fetchone()[0]
    assert queries.run_summary(ledger, DEMO_RUN_ID)["rejected_ideas"] == own
    ledger.close()


def test_holdout_numbers_are_hidden_until_the_holdout_run_is_complete(small_demo):
    candidates = queries.galaxy(small_demo, DEMO_RUN_ID)["points"]
    for point in candidates:
        detail = queries.signal_detail(small_demo, point["attempt_id"])
        assert detail["holdout"] == {"status": "locked"}
        assert "metrics" not in detail["holdout"]
    for curve in queries.equity(small_demo, DEMO_RUN_ID):
        assert curve["holdout"] == "locked"


def test_recorded_holdout_metrics_appear_only_after_completion(tmp_path):
    ledger = demo_ledger(tmp_path, pipelines=20)
    points = queries.galaxy(ledger, DEMO_RUN_ID)["points"]
    assert points, "the demo should contain at least one validated signal"
    point = points[0]
    attempt_id = point["attempt_id"]
    ledger.add_holdout("h1", attempt_id, DEMO_RUN_ID, "running", None)
    assert queries.signal_detail(ledger, attempt_id)["holdout"]["status"] == "running"
    assert "metrics" not in queries.signal_detail(ledger, attempt_id)["holdout"]
    ledger.update_holdout(attempt_id, "complete", {"sharpe": 0.4})
    assert queries.signal_detail(ledger, attempt_id)["holdout"]["metrics"] == {"sharpe": 0.4}
    ledger.close()


def test_signal_detail_carries_the_run_id(small_demo):
    attempt_id = small_demo.attempts_for_run(DEMO_RUN_ID)[0]["attempt_id"]
    assert queries.signal_detail(small_demo, attempt_id)["run_id"] == DEMO_RUN_ID


def test_signal_endpoint_accepts_an_idea_id(small_demo):
    idea_id = small_demo.ideas_for_run(DEMO_RUN_ID)[0]["idea_id"]
    detail = queries.signal_detail(small_demo, idea_id)
    assert detail is not None
    assert detail["idea"]["name"]


def test_events_after_pages_through_the_stream(small_demo):
    everything = queries.events_after(small_demo, DEMO_RUN_ID, 0, limit=100000)
    first_page = queries.events_after(small_demo, DEMO_RUN_ID, 0, limit=5)
    rest = queries.events_after(small_demo, DEMO_RUN_ID, first_page[-1]["seq"], limit=100000)
    assert [e["seq"] for e in first_page + rest] == [e["seq"] for e in everything]


def test_timeline_has_one_lane_per_pipeline(small_demo):
    lanes = queries.timeline(small_demo, DEMO_RUN_ID)
    assert len(lanes) == 8
    assert all(lane["pipeline_id"] for lane in lanes)


def test_economics_has_one_row_per_agent_and_model(small_demo):
    rows = queries.economics(small_demo)
    assert {r["agent"] for r in rows} >= {"ideas", "coder", "critic"}
    assert len({(r["agent"], r["model"]) for r in rows}) == len(rows)
    assert all(0.0 <= r["success_rate"] <= 1.0 for r in rows)


def test_research_builds_the_report_once_and_a_finished_run_is_cached(tmp_path, monkeypatch):
    ledger = demo_ledger(tmp_path)
    builds = []
    original = queries.build_report_data

    def counting(led, run_id, *args, **kwargs):
        builds.append(run_id)
        return original(led, run_id, *args, **kwargs)

    monkeypatch.setattr(queries, "build_report_data", counting)
    queries.research(ledger, DEMO_RUN_ID)
    assert builds == [DEMO_RUN_ID], "galaxy, themes and failures share one build"
    queries.research(ledger, DEMO_RUN_ID)
    assert len(builds) == 1, "a finished run is served from the cache"
    # A trial row changes the ledger-wide count the report depends on, so the cache must not be used.
    ledger.add_trial(uuid.uuid4().hex, DEMO_RUN_ID, "unknown-attempt", "hash")
    queries.research(ledger, DEMO_RUN_ID)
    assert len(builds) == 2
    ledger.close()


def test_a_running_run_is_rebuilt_on_every_request(tmp_path, monkeypatch):
    ledger = Ledger(tmp_path / "live.db")
    ledger.create_run("live", "{}", 1.0)  # status stays "running"
    builds = []
    original = queries.build_report_data

    def counting(led, run_id, *args, **kwargs):
        builds.append(run_id)
        return original(led, run_id, *args, **kwargs)

    monkeypatch.setattr(queries, "build_report_data", counting)
    queries.research(ledger, "live")
    queries.research(ledger, "live")
    assert builds == ["live", "live"]
    ledger.close()


def test_drawdown_starts_from_one_so_it_matches_the_reported_max_drawdown(tmp_path):
    train = {"2018-01-02": -0.10, "2018-01-03": 0.05, "2018-01-04": 0.0}
    validation = {"2019-01-02": 0.0, "2019-01-03": 0.01, "2019-01-04": -0.02, "2019-01-07": 0.0}
    ledger = new_ledger(tmp_path)
    add_candidate(ledger, "Candidate", train=train, validation=validation)
    curve = queries.equity(ledger, "r1", top=1)[0]
    combined = pd.Series({**train, **validation}, dtype=float)
    combined.index = pd.to_datetime(combined.index)
    expected = compute_metrics(combined, pd.Series(0.0, index=combined.index), 252).max_drawdown
    assert min(curve["drawdown"]) == pytest.approx(expected)
    assert min(curve["drawdown"]) == pytest.approx(-0.10)
    ledger.close()


def test_equity_is_thinned_but_keeps_the_deepest_drawdown(tmp_path):
    rng = np.random.default_rng(5)
    days = pd.bdate_range("2019-01-01", periods=1500)
    validation = {
        str(d.date()): float(v)
        for d, v in zip(days, rng.normal(0.0002, 0.01, len(days)), strict=True)
    }
    ledger = new_ledger(tmp_path)
    add_candidate(ledger, "Candidate", train=None, validation=validation)
    full = queries.equity(ledger, "r1", top=1, max_points=10**6)[0]
    thin = queries.equity(ledger, "r1", top=1, max_points=50)[0]
    assert len(full["dates"]) == 1500
    assert len(thin["dates"]) <= 53
    assert min(thin["drawdown"]) == pytest.approx(min(full["drawdown"]))
    assert thin["dates"][0] == full["dates"][0]
    assert thin["dates"][-1] == full["dates"][-1]
    ledger.close()


def test_equity_leaves_out_a_validation_row_without_returns(tmp_path):
    ledger = new_ledger(tmp_path)
    add_candidate(
        ledger,
        "Has returns",
        train={"2018-01-02": 0.01, "2018-01-03": -0.01},
        validation={"2019-01-02": 0.0, "2019-01-03": 0.01, "2019-01-04": 0.02},
    )
    add_candidate(ledger, "Legacy row", train={"2018-01-02": 0.01}, validation=None)
    names = [c["name"] for c in queries.equity(ledger, "r1", top=5)]
    assert names == ["Has returns"]
    ledger.close()


def test_correlation_leaves_out_a_validation_row_without_returns(tmp_path):
    ledger = new_ledger(tmp_path)
    days = {"2019-01-02": 0.01, "2019-01-03": -0.02, "2019-01-04": 0.005, "2019-01-07": 0.0}
    add_candidate(ledger, "First", train=None, validation=days)
    add_candidate(ledger, "Second", train=None, validation={k: -v for k, v in days.items()})
    add_candidate(ledger, "Legacy survivor", train=None, validation=None)
    corr = queries.correlation(ledger, "r1")
    # Labels are "name (attempt prefix)" so same-named survivors stay apart; the name part decides.
    assert sorted(label.split(" (")[0] for label in corr["names"]) == ["First", "Second"]
    ledger.close()


def test_benchmark_says_why_it_has_no_data_when_the_price_cache_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(queries, "DATA_DIR", tmp_path)  # an empty folder: no parquet file
    result = queries.equal_weight_benchmark(["2019-01-02", "2019-01-03"])
    assert result["dates"] == []
    assert result["equity"] == []
    assert "lab data fetch" in result["reason"]


def test_timeline_segments_are_the_llm_calls_that_were_sent_or_refused(small_demo):
    lanes = queries.timeline(small_demo, DEMO_RUN_ID)
    segments = [s for lane in lanes for s in lane["segments"]]
    calls = [
        r
        for r in small_demo.events_for_run(DEMO_RUN_ID)
        if r["type"] in ("llm_call_finished", "llm_call_refused")
    ]
    assert segments, "the demo makes LLM calls, so there should be segments"
    assert len(segments) == len(calls)
    assert {s["status"] for s in segments} <= {"ok", "refused"}


def test_demo_prices_and_models_come_from_the_settings(tmp_path, monkeypatch):
    base = load_settings()
    haiku = base.pricing.models["claude-haiku-5-5"].model_copy(update={"input": 3.0, "output": 4.0})
    models = {**base.pricing.models, "claude-haiku-5-5": haiku}
    priced = base.model_copy(update={"pricing": base.pricing.model_copy(update={"models": models})})
    monkeypatch.setattr(demo, "load_settings", lambda: priced)
    ledger = Ledger(tmp_path / "demo.db")
    asyncio.run(run_demo(ledger, EventBus(ledger), n_pipelines=8, pace=0.0, seed=3))
    rows = ledger.conn.execute(
        "SELECT model, input_tokens, output_tokens, cost_usd FROM llm_calls "
        "WHERE run_id = ? AND status = 'ok'",
        (DEMO_RUN_ID,),
    ).fetchall()
    ledger.close()
    assert {r["model"] for r in rows} == {
        base.agents.ideas_model,
        base.agents.critic_model,
        base.agents.coder_model,
    }
    haiku_rows = [r for r in rows if r["model"] == "claude-haiku-5-5"]
    assert haiku_rows, "the demo should price some Haiku calls"
    for r in haiku_rows:
        expected = (r["input_tokens"] * 3.0 + r["output_tokens"] * 4.0) / 1_000_000
        assert r["cost_usd"] == pytest.approx(expected)


def test_a_reached_critic_with_no_usable_review_counts_as_a_train_pass(tmp_path):
    # failed_critic_output is written after the training gate, so the summary counts it as a train pass.
    ledger = new_ledger(tmp_path)
    add_candidate(ledger, "No review", train=None, validation=None, status="failed_critic_output")
    add_candidate(ledger, "Gate failed", train=None, validation=None, status="failed_gate")
    add_candidate(ledger, "Validated", train=None, validation=None, status="validation_pass")
    summary = queries.run_summary(ledger, "r1")
    # One vocabulary (statuses.py): counts of distinct ideas, named after the set they count (R4-WHOLE-10).
    assert summary["passed_train_gate"] == 2, "the critic-output case and the validated one"
    assert summary["passed_critic"] == 1, "only the validated idea passed the critic"
    assert summary["passed_validation"] == 1
    assert "train_passes" not in summary and "validation_passes" not in summary
    ledger.close()


def test_equity_and_signal_payloads_say_why_the_dsr_is_missing(tmp_path):
    ledger = new_ledger(tmp_path)
    attempt_id = add_candidate(
        ledger,
        "Lone candidate",
        train={"2018-01-02": 0.01, "2018-01-03": -0.01},
        validation={"2019-01-02": 0.0, "2019-01-03": 0.01, "2019-01-04": -0.01},
    )
    curve = queries.equity(ledger, "r1", top=1)[0]
    assert "two trials" in curve["dsr_reason"], "one trial cannot estimate the trial variance"
    assert queries.signal_detail(ledger, attempt_id)["dsr_reason"] == curve["dsr_reason"]
    ledger.close()


def test_the_limits_come_from_the_settings():
    settings = load_settings()
    assert queries.limits() == {
        "pricing_tier_tokens": settings.pricing.long_context_threshold_tokens,
        "context_limit_tokens": settings.agents.context_limit_tokens,
    }
    assert queries.limits() == {"pricing_tier_tokens": 100000, "context_limit_tokens": 90000}


def test_benchmark_refuses_a_price_cache_built_for_other_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(queries, "DATA_DIR", tmp_path)
    prices = pd.DataFrame(
        {"AAPL": [100.0, 101.0]}, index=pd.to_datetime(["2019-01-02", "2019-01-03"])
    )
    prices.to_parquet(tmp_path / PRE_HOLDOUT_FILE)  # a cache with no fingerprint
    result = queries.equal_weight_benchmark(["2019-01-02", "2019-01-03"])
    assert result["dates"] == []
    assert result["equity"] == []
    assert result["reason"].startswith("cache stale")
    assert "different settings" in result["reason"]


def test_a_finished_run_is_rebuilt_when_a_row_changes_but_the_counts_do_not(tmp_path, monkeypatch):
    ledger = demo_ledger(tmp_path)
    builds = []
    original = queries.build_report_data

    def counting(led, run_id, *args, **kwargs):
        builds.append(run_id)
        return original(led, run_id, *args, **kwargs)

    monkeypatch.setattr(queries, "build_report_data", counting)
    queries.research(ledger, DEMO_RUN_ID)
    assert len(builds) == 1
    # Add one validation and remove the oldest: the row count is unchanged, but the newest rowid moves on.
    old_validation = ledger.conn.execute(
        "SELECT validation_id FROM validations ORDER BY rowid LIMIT 1"
    ).fetchone()[0]
    unvalidated = ledger.conn.execute(
        "SELECT attempt_id FROM attempts WHERE run_id = ? AND attempt_id NOT IN "
        "(SELECT attempt_id FROM validations) LIMIT 1",
        (DEMO_RUN_ID,),
    ).fetchone()
    assert unvalidated is not None, "the demo should have an attempt with no validation row"
    ledger.add_validation(unvalidated[0], DEMO_RUN_ID, {"sharpe": 0.9}, True, None, None)
    ledger.conn.execute("DELETE FROM validations WHERE validation_id = ?", (old_validation,))
    ledger.conn.commit()
    queries.research(ledger, DEMO_RUN_ID)
    assert len(builds) == 2, "a replaced row must not be served from the old cache entry"
    ledger.close()


def test_the_report_cache_is_bounded_by_candidates_as_well_as_entries(tmp_path, monkeypatch):
    queries._report_cache.clear()  # other tests share the module-level cache
    ledger = demo_ledger(tmp_path)
    weight = len(queries.report_for(ledger, DEMO_RUN_ID).candidates)
    assert weight >= 1
    ledger.create_run("second", "{}", 1.0)
    ledger.finish_run("second", "completed", 0.0)
    # The demo run is cached with a weight equal to the budget. Caching the second run pushes the
    # total over the budget, so the demo run is evicted and must be built again when it is asked for.
    monkeypatch.setattr(queries, "REPORT_CACHE_CANDIDATES", weight)
    queries.report_for(ledger, "second")
    assert len(queries._report_cache) == 1, "the older entry was evicted by the candidate budget"
    builds = []
    original = queries.build_report_data

    def counting(led, run_id, *args, **kwargs):
        builds.append(run_id)
        return original(led, run_id, *args, **kwargs)

    monkeypatch.setattr(queries, "build_report_data", counting)
    queries.report_for(ledger, DEMO_RUN_ID)
    queries.report_for(ledger, DEMO_RUN_ID)  # cached again, because the budget holds one entry
    assert builds == [DEMO_RUN_ID], "the second request is served from the cache"
    ledger.close()


def test_a_holdout_left_running_for_hours_is_marked_stale(tmp_path):
    ledger = demo_ledger(tmp_path, pipelines=20)
    attempt_id = queries.galaxy(ledger, DEMO_RUN_ID)["points"][0]["attempt_id"]
    ledger.add_holdout("h-old", attempt_id, DEMO_RUN_ID, "running", None)
    ledger.conn.execute(
        "UPDATE holdout_runs SET created_at = ? WHERE attempt_id = ?",
        ("2020-01-01T00:00:00.000+00:00", attempt_id),
    )
    ledger.conn.commit()
    holdout = queries.signal_detail(ledger, attempt_id)["holdout"]
    assert holdout["status"] == "running"
    assert holdout["stale"] is True, "a running row years old was probably left by a killed process"
    assert holdout["started_at"] == "2020-01-01T00:00:00.000+00:00"
    ledger.conn.execute(
        "UPDATE holdout_runs SET created_at = ? WHERE attempt_id = ?",
        (datetime.now(UTC).isoformat(), attempt_id),
    )
    ledger.conn.commit()
    assert queries.signal_detail(ledger, attempt_id)["holdout"]["stale"] is False
    ledger.close()


def test_a_running_run_silent_for_longer_than_the_stale_limit_is_flagged(tmp_path):
    ledger = Ledger(tmp_path / "stale.db")
    ledger.create_run("dead", "{}", 1.0)
    ledger.create_run("live", "{}", 1.0)
    for run_id, when in (
        ("dead", "2020-01-01T00:00:00.000+00:00"),
        ("live", datetime.now(UTC).isoformat()),
    ):
        ledger.conn.execute(
            "INSERT INTO events (event_id, run_id, stage, type, timestamp, payload_json) "
            "VALUES (?, ?, 'run', 'run_started', ?, '{}')",
            (uuid.uuid4().hex, run_id, when),
        )
    ledger.conn.commit()
    rows = {r["run_id"]: r for r in queries.list_runs(ledger)}
    assert {k: r["status"] for k, r in rows.items()} == {"dead": "abandoned", "live": "running"}
    assert rows["dead"]["stored_status"] == "running", (
        "the ledger still stores the dead run as running"
    )
    # 'abandoned' is a final status, so it never carries stale: true (R4-BE-06 and the R5 stale rule).
    assert rows["dead"]["stale"] is False
    assert rows["live"]["stale"] is False
    assert queries.run_summary(ledger, "dead")["status"] == "abandoned"
    assert queries.run_summary(ledger, "dead")["stale"] is False
    assert queries.run_summary(ledger, "live")["stale"] is False
    ledger.close()


def test_the_demo_writes_run_finished_through_the_shared_payload(tmp_path, monkeypatch):
    calls = []
    shared = demo.run_finished_payload

    def spy(**kwargs):
        calls.append(kwargs)
        return shared(**kwargs)

    monkeypatch.setattr(demo, "run_finished_payload", spy)
    ledger = demo_ledger(tmp_path)
    ledger.close()
    assert len(calls) == 1
    assert calls[0]["status"] == "completed"


def test_the_demo_and_the_stream_use_the_status_sets_from_statuses_py():
    from alpha_lab import statuses
    from alpha_lab.dashboard import server

    assert demo.LOOP_ENDS == statuses.VALIDATION_VERDICTS
    assert server.ENDED_RUN_STATUSES == statuses.RUN_STATUSES - {"running"}


def test_a_silent_run_keeps_its_stored_running_status_and_is_not_stale(tmp_path):
    ledger = Ledger(tmp_path / "flags.db")
    old = "2020-01-01T00:00:00.000+00:00"
    for run_id, stored in (("final", "abandoned"), ("dead", "running")):
        ledger.create_run(run_id, "{}", 1.0)
        ledger.conn.execute(
            "INSERT INTO events (event_id, run_id, stage, type, timestamp, payload_json) "
            "VALUES (?, ?, 'run', 'run_started', ?, '{}')",
            (uuid.uuid4().hex, run_id, old),
        )
        if stored != "running":
            ledger.finish_run(run_id, stored, 0.0)
    ledger.conn.commit()
    rows = {r["run_id"]: r for r in queries.list_runs(ledger)}
    assert rows["final"]["status"] == "abandoned"
    assert rows["final"]["stored_status"] == "abandoned"
    assert rows["final"]["stale"] is False, "a final stored status is never stale"
    assert rows["dead"]["status"] == "abandoned"
    assert rows["dead"]["stored_status"] == "running"
    assert rows["dead"]["stale"] is False, "abandoned is final, so it never carries stale"
    assert queries.run_summary(ledger, "final")["stale"] is False
    ledger.close()


def test_no_run_status_that_is_final_carries_stale(tmp_path):
    """Every status a reader can show for a run is checked: only a non-final status may carry stale."""
    ledger = Ledger(tmp_path / "every.db")
    old = "2020-01-01T00:00:00.000+00:00"
    for run_id in ("silent", "done"):
        ledger.create_run(run_id, "{}", 1.0)
        ledger.conn.execute(
            "INSERT INTO events (event_id, run_id, stage, type, timestamp, payload_json) "
            "VALUES (?, ?, 'run', 'run_started', ?, '{}')",
            (uuid.uuid4().hex, run_id, old),
        )
    ledger.finish_run("done", "completed", 0.0)
    ledger.conn.commit()
    for row in queries.list_runs(ledger):
        final = row["status"] != "running"
        assert not (final and row["stale"]), row
    ledger.close()


def test_equity_thinning_never_exceeds_max_points():
    # 1000 points, the deepest drawdown at index 1 (off the even grid): it must replace a grid point.
    kept = queries._thin(1000, 1, 500)
    assert len(kept) <= 500
    assert {0, 999, 1} <= set(kept)
    # When the deepest point is already on the grid, the output is exactly max_points.
    assert len(queries._thin(1000, 0, 500)) == 500


def test_the_report_key_does_not_read_the_metrics_column(tmp_path):
    """The cache key runs on every request, so it may use only indexed counts and maxima (R4-BE-05)."""
    ledger = demo_ledger(tmp_path, pipelines=20)
    statements: list[str] = []
    ledger.conn.set_trace_callback(statements.append)
    queries._report_key(ledger, DEMO_RUN_ID)
    ledger.conn.set_trace_callback(None)
    assert statements, "the key should run at least one statement"
    assert not any("train_metrics_json" in s for s in statements), statements
    ledger.close()


def test_the_holdout_statuses_come_from_statuses_py(tmp_path):
    from alpha_lab import statuses

    sets = queries.status_sets()
    assert sets["holdout"] == sorted(statuses.HOLDOUT_VIEWS)
    assert set(statuses.HOLDOUT_VIEWS) == set(statuses.HOLDOUT_STATUSES) | {
        statuses.HOLDOUT_LOCKED_VIEW
    }
    ledger = demo_ledger(tmp_path, pipelines=20)
    attempt_id = queries.galaxy(ledger, DEMO_RUN_ID)["points"][0]["attempt_id"]
    ledger.add_holdout("h-state", attempt_id, DEMO_RUN_ID, "running", None)
    assert queries.signal_detail(ledger, attempt_id)["holdout"]["status"] in sets["holdout"]
    ledger.finish_holdout(attempt_id, "error", None, None)
    assert queries.signal_detail(ledger, attempt_id)["holdout"]["status"] == "error"
    assert queries.signal_detail(ledger, attempt_id)["holdout"]["status"] in sets["holdout"]
    ledger.close()


def test_galaxy_points_carry_their_status_and_equity_curves_their_passed_flag(big_demo):
    from alpha_lab import statuses

    points = {p["attempt_id"]: p for p in queries.galaxy(big_demo, DEMO_RUN_ID)["points"]}
    assert points, "the demo should contain validated candidates"
    for point in points.values():
        assert point["status"] in statuses.ATTEMPT_STATUSES
    curves = queries.equity(big_demo, DEMO_RUN_ID, top=5)
    assert curves
    for curve in curves:
        assert isinstance(curve["passed"], bool)
        assert curve["passed"] == points[curve["attempt_id"]]["passed"]
