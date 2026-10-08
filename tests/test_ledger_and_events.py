import asyncio
import json

import pytest

from alpha_lab.events.bus import EventBus, PipelineEvent
from alpha_lab.ledger.db import Ledger


@pytest.fixture
def ledger(tmp_path):
    db = Ledger(tmp_path / "ledger.db")
    yield db
    db.close()


def test_run_idea_attempt_round_trip(ledger):
    ledger.create_run("r1", '{"seed": 42}', 2.0)
    ledger.add_idea(
        idea_id="i1",
        run_id="r1",
        pipeline_id="p1",
        name="Slow reversal",
        hypothesis="Overreaction fades",
        description="Buy losers",
        expected_holding_period_days=5,
    )
    ledger.add_attempt(
        attempt_id="a1",
        idea_id="i1",
        run_id="r1",
        attempt_number=1,
        code="x",
        code_hash="h",
        status="running",
    )
    ledger.update_attempt(
        "a1", status="train_pass", static_ok=1, train_metrics_json='{"sharpe": 1.0}'
    )
    ledger.finish_run("r1", "completed", 0.12)

    run = ledger.get_run("r1")
    assert run["status"] == "completed"
    assert run["total_cost_usd"] == pytest.approx(0.12)
    attempt = ledger.attempts_for_run("r1")[0]
    assert attempt["status"] == "train_pass"
    assert json.loads(attempt["train_metrics_json"])["sharpe"] == 1.0


def test_trials_are_counted_and_never_deleted(ledger):
    ledger.create_run("r1", "{}", 1.0)
    ledger.add_attempt(
        attempt_id="a1",
        idea_id="i",
        run_id="r1",
        attempt_number=1,
        code="c",
        code_hash="h1",
        status="failed_gate",
    )
    ledger.add_trial("t1", "r1", "a1", "h1")
    ledger.add_trial("t2", "r1", "a1", "h2")
    assert ledger.count_trials() == 2


def test_recent_ideas_are_newest_first(ledger):
    ledger.create_run("r1", "{}", 1.0)
    for i in range(3):
        ledger.add_idea(
            idea_id=f"i{i}",
            run_id="r1",
            pipeline_id="p",
            name=f"idea {i}",
            hypothesis="h" * 12,
            description="d" * 12,
            expected_holding_period_days=1,
            created_at=f"2026-01-0{i + 1}T00:00:00.000+00:00",
        )
    names = [row["name"] for row in ledger.recent_ideas(2)]
    assert names == ["idea 2", "idea 1"]


def test_bus_persists_and_broadcasts(ledger):
    ledger.create_run("r1", "{}", 1.0)
    bus = EventBus(ledger)

    async def scenario():
        queue = bus.subscribe()
        bus.emit(PipelineEvent(run_id="r1", stage="run", type="run_started", payload={"n": 1}))
        event = await asyncio.wait_for(queue.get(), timeout=1)
        assert event.type == "run_started"

    asyncio.run(scenario())
    rows = ledger.events_for_run("r1")
    assert len(rows) == 1
    assert json.loads(rows[0]["payload_json"]) == {"n": 1}


def test_emit_never_crashes_the_pipeline(ledger, capsys):
    ledger.create_run("r1", "{}", 1.0)
    bus = EventBus(ledger)
    ledger.conn.close()  # simulate a broken ledger

    bus.emit(PipelineEvent(run_id="r1", stage="run", type="run_started"))

    assert "failed to persist run_started" in capsys.readouterr().err
