import asyncio
import json
import sqlite3
from datetime import timedelta

import pytest

from alpha_lab.events.bus import EventBus, PipelineEvent
from alpha_lab.ledger.db import Ledger, now_iso


@pytest.fixture
def ledger(tmp_path):
    db = Ledger(tmp_path / "ledger.db")
    yield db
    db.close()


def strict_loads(text):
    def refuse(constant):
        raise ValueError(f"not strict JSON: {constant}")

    return json.loads(text, parse_constant=refuse)


def add_idea_and_attempt(ledger, run_id="r1", attempt_id="a1", status="running"):
    ledger.add_idea(
        idea_id="i1",
        run_id=run_id,
        pipeline_id="p1",
        name="Slow reversal",
        hypothesis="Overreaction fades",
        description="Buy losers",
        expected_holding_period_days=5,
    )
    ledger.add_attempt(
        attempt_id=attempt_id,
        idea_id="i1",
        run_id=run_id,
        attempt_number=1,
        code="x",
        code_hash="h",
        status=status,
    )


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


def test_trials_are_counted(ledger):
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


def test_trials_cannot_be_deleted_or_changed(ledger):
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
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        ledger.conn.execute("DELETE FROM trials")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        ledger.conn.execute("UPDATE trials SET code_hash = 'changed'")
    assert ledger.count_trials() == 1


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


def test_nan_never_reaches_a_subscriber_or_the_ledger(ledger):
    ledger.create_run("r1", "{}", 1.0)
    bus = EventBus(ledger)
    queue = bus.subscribe()
    event = PipelineEvent(
        run_id="r1",
        stage="train",
        type="train_backtest",
        payload={"sharpe": float("nan"), "ret": 0.5, "nested": [float("inf")]},
    )
    assert event.payload == {"sharpe": None, "ret": 0.5, "nested": [None]}
    bus.emit(event)
    assert queue.get_nowait().payload["sharpe"] is None
    stored = strict_loads(ledger.events_for_run("r1")[0]["payload_json"])
    assert stored == {"sharpe": None, "ret": 0.5, "nested": [None]}


def test_a_non_finite_value_is_stored_as_null_in_a_validation(ledger):
    ledger.create_run("r1", "{}", 1.0)
    add_idea_and_attempt(ledger, status="validation_fail")
    ledger.add_validation(
        "a1", "r1", {"sharpe": float("nan"), "avg": 0.5}, False, None, {"2020-01-02": float("inf")}
    )
    row = ledger.validations()[0]
    assert strict_loads(row["metrics_json"]) == {"sharpe": None, "avg": 0.5}
    assert strict_loads(row["returns_json"]) == {"2020-01-02": None}


def test_writes_with_unknown_column_names_are_refused(ledger):
    ledger.create_run("r1", "{}", 1.0)
    add_idea_and_attempt(ledger)
    with pytest.raises(ValueError, match="no column named"):
        ledger.update_attempt("a1", **{"status = 'pwned', static_ok": 0})
    assert ledger.attempts_for_run("r1")[0]["status"] == "running"
    with pytest.raises(ValueError, match="no column named"):
        ledger.add_idea(
            idea_id="x",
            run_id="r1",
            pipeline_id="p",
            name="n",
            hypothesis="h",
            description="d",
            expected_holding_period_days=1,
            drop_table="ideas",
        )


def test_a_transaction_rolls_back_every_write_in_it(ledger):
    ledger.create_run("r1", "{}", 1.0)
    add_idea_and_attempt(ledger)
    with pytest.raises(RuntimeError), ledger.transaction():
        ledger.add_validation("a1", "r1", {"sharpe": 1.0}, True, None, None)
        ledger.update_attempt("a1", status="validation_pass")
        raise RuntimeError("crash between the two writes")
    assert ledger.validations() == []
    assert ledger.attempts_for_run("r1")[0]["status"] == "running"


def test_a_transaction_commits_all_of_its_writes(ledger):
    ledger.create_run("r1", "{}", 1.0)
    add_idea_and_attempt(ledger)
    with ledger.transaction():
        ledger.add_validation("a1", "r1", {"sharpe": 1.0}, True, None, None)
        ledger.update_attempt("a1", status="validation_pass")
    assert len(ledger.validations()) == 1
    assert ledger.attempts_for_run("r1")[0]["status"] == "validation_pass"


def test_run_cost_is_the_sum_of_the_llm_calls(ledger):
    ledger.create_run("r1", "{}", 1.0)
    for i, (cost, status) in enumerate([(0.01, "ok"), (0.0, "refused"), (0.25, "error")]):
        ledger.add_llm_call(
            call_id=f"c{i}",
            run_id="r1",
            idea_id=None,
            agent="coder",
            model="m",
            prompt_version="p",
            input_tokens=None,
            output_tokens=None,
            cache_read_tokens=None,
            cache_write_tokens=None,
            cost_usd=cost,
            latency_ms=1,
            status=status,
        )
    assert ledger.llm_cost_total("r1") == pytest.approx(0.26)
    assert ledger.llm_cost_total("missing") == 0.0
    [agent_row] = ledger.llm_cost_by_agent("r1")
    assert agent_row["calls"] == 1  # only the call that succeeded
    assert agent_row["cost_usd"] == pytest.approx(0.26)  # the charged failure is included


def test_stale_runs_and_their_open_attempts_are_abandoned(ledger):
    ledger.create_run("old", "{}", 1.0)
    ledger.add_event(
        "e1", "old", None, None, None, None, "run", "run_started", "2020-01-01T00:00:00.000+00:00", {}
    )
    add_idea_and_attempt(ledger, run_id="old", attempt_id="a1")
    assert ledger.abandon_stale_runs(timedelta(minutes=30)) == ["old"]
    assert ledger.get_run("old")["status"] == "abandoned"
    assert ledger.attempts_for_run("old")[0]["status"] == "abandoned"


def test_a_run_with_recent_events_is_left_alone(ledger):
    ledger.create_run("fresh", "{}", 1.0)
    ledger.add_event("e1", "fresh", None, None, None, None, "run", "run_started", now_iso(), {})
    assert ledger.abandon_stale_runs(timedelta(minutes=30)) == []
    assert ledger.get_run("fresh")["status"] == "running"


def test_run_lookups_have_indexes(ledger):
    names = {row["name"] for row in ledger.conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")}
    for index in [
        "ideas_run",
        "attempts_run",
        "attempts_normalised_hash",
        "llm_calls_run",
        "trials_run",
        "events_run",
        "rejected_ideas_run",
        "validations_run",
        "holdout_runs_run",
    ]:
        assert index in names, index


def test_dropped_events_are_reported_rarely_and_unsubscribe_stops_delivery(ledger, capsys):
    ledger.create_run("r1", "{}", 1.0)
    bus = EventBus(ledger, queue_size=1)
    bus.subscribe()  # never drained, so every event after the first is dropped
    for _ in range(1002):
        bus.emit(PipelineEvent(run_id="r1", stage="run", type="run_started"))
    reports = [line for line in capsys.readouterr().err.splitlines() if "queue full" in line]
    assert len(reports) == 2  # reported on the first drop and on the 1001st
    assert len(ledger.events_for_run("r1")) == 1002  # persistence is not affected

    queue = bus.subscribe()
    bus.unsubscribe(queue)
    bus.emit(PipelineEvent(run_id="r1", stage="run", type="run_started"))
    assert queue.empty()
