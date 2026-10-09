"""Dashboard data layer and demo stream. No web server and no network: these call the query functions directly.

The HTTP routes, strict JSON on the wire and the live stream are tested in test_dashboard_http.py.
"""

import asyncio
import dataclasses
import json
import uuid
from typing import get_args

import numpy as np
import pandas as pd
import pytest

from alpha_lab.backtest.metrics import Metrics, compute_metrics
from alpha_lab.dashboard import queries
from alpha_lab.events.bus import EventBus, EventType, PipelineEvent
from alpha_lab.events.demo import DEMO_RUN_ID, run_demo
from alpha_lab.ledger.db import LEDGER_PATH, Ledger

# Payload keys each event type carries, as the real emitters write them: pipeline/single.py,
# pipeline/run.py, agents/llm.py and checks/lookahead.py. train_backtest carries the Metrics fields.
REAL_PAYLOAD_KEYS = {
    "run_started": {"pipelines", "budget_usd", "concurrency"},
    "run_finished": {"status", "total_cost_usd"},
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
    assert seen == set(get_args(EventType)), "the demo should exercise every event type"


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
    assert queries.signal_detail(ledger, attempt_id)["holdout"] == {"status": "running"}
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
        str(d.date()): float(v) for d, v in zip(days, rng.normal(0.0002, 0.01, len(days)))
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
