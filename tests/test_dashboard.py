"""Dashboard data layer and demo stream. No web server and no network: these call the query functions directly."""

import asyncio
import json

from alpha_lab.dashboard import queries
from alpha_lab.events.bus import EventBus, PipelineEvent
from alpha_lab.events.demo import DEMO_RUN_ID, run_demo
from alpha_lab.ledger.db import Ledger


def demo_ledger(tmp_path, pipelines=8):
    ledger = Ledger(tmp_path / "demo.db")
    bus = EventBus(ledger)
    asyncio.run(run_demo(ledger, bus, n_pipelines=pipelines, pace=0.0, seed=3))
    return ledger


def test_demo_event_stream_is_schema_valid(tmp_path):
    ledger = demo_ledger(tmp_path)
    rows = ledger.events_for_run(DEMO_RUN_ID)
    assert rows, "the demo should write events"
    for row in rows:
        payload = json.loads(row["payload_json"])
        # Every row must parse back into the typed event model.
        PipelineEvent(
            run_id=row["run_id"],
            stage=row["stage"] or "run",
            type=row["type"],
            pipeline_id=row["pipeline_id"],
            payload=payload,
        )
    started = [r for r in rows if r["type"] == "pipeline_started"]
    finished = [r for r in rows if r["type"] == "pipeline_finished"]
    assert len(started) == 8
    assert len(finished) == 8
    ledger.close()


def test_demo_run_is_not_regenerated(tmp_path):
    ledger = demo_ledger(tmp_path)
    before = len(ledger.events_for_run(DEMO_RUN_ID))
    asyncio.run(run_demo(ledger, EventBus(ledger), n_pipelines=8, pace=0.0, seed=3))
    assert len(ledger.events_for_run(DEMO_RUN_ID)) == before
    ledger.close()


def test_summary_and_funnel_come_from_the_ledger(tmp_path):
    ledger = demo_ledger(tmp_path)
    summary = queries.run_summary(ledger, DEMO_RUN_ID)
    assert summary["pipelines"] == 8
    assert summary["ideas"] == 8
    funnel = queries.funnel(ledger, DEMO_RUN_ID)
    assert funnel[0] == {"stage": "ideas generated", "count": 8}
    ledger.close()


def test_holdout_numbers_are_hidden_until_the_holdout_run_is_complete(tmp_path):
    ledger = demo_ledger(tmp_path)
    candidates = queries.galaxy(ledger, DEMO_RUN_ID)["points"]
    for point in candidates:
        detail = queries.signal_detail(ledger, point["attempt_id"])
        assert detail["holdout"] == {"status": "locked"}
        assert "metrics" not in detail["holdout"]
    for curve in queries.equity(ledger, DEMO_RUN_ID):
        assert curve["holdout"] == "locked"
    ledger.close()


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


def test_signal_endpoint_accepts_an_idea_id(tmp_path):
    ledger = demo_ledger(tmp_path)
    idea_id = ledger.ideas_for_run(DEMO_RUN_ID)[0]["idea_id"]
    detail = queries.signal_detail(ledger, idea_id)
    assert detail is not None
    assert detail["idea"]["name"]
    ledger.close()


def test_events_after_pages_through_the_stream(tmp_path):
    ledger = demo_ledger(tmp_path)
    everything = queries.events_after(ledger, DEMO_RUN_ID, 0, limit=100000)
    first_page = queries.events_after(ledger, DEMO_RUN_ID, 0, limit=5)
    rest = queries.events_after(ledger, DEMO_RUN_ID, first_page[-1]["seq"], limit=100000)
    assert [e["seq"] for e in first_page + rest] == [e["seq"] for e in everything]
    ledger.close()


def test_timeline_has_one_lane_per_pipeline(tmp_path):
    ledger = demo_ledger(tmp_path)
    lanes = queries.timeline(ledger, DEMO_RUN_ID)
    assert len(lanes) == 8
    assert all(lane["pipeline_id"] for lane in lanes)
    ledger.close()


def test_economics_groups_by_agent_and_model(tmp_path):
    ledger = demo_ledger(tmp_path)
    rows = queries.economics(ledger)
    assert {r["agent"] for r in rows} >= {"ideas", "coder", "critic"}
    assert all(0.0 <= r["success_rate"] <= 1.0 for r in rows)
    ledger.close()
