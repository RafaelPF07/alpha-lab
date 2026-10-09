import asyncio
import json
import shutil
import sqlite3
from datetime import timedelta

import pytest

from alpha_lab.events.bus import EventBus, PipelineEvent
from alpha_lab.ledger.db import HoldoutConflict, Ledger, now_iso


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
        "attempts_idea",
        "llm_calls_idea",
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


def test_an_abandoned_run_gets_its_cost_and_closing_events(ledger):
    ledger.create_run("old", "{}", 2.0)
    ledger.add_event("e1", "old", None, None, None, None, "run", "run_started", "2020-01-01T00:00:00.000+00:00", {})
    ledger.add_event("e2", "old", "p1", None, None, None, "pipeline", "pipeline_started", "2020-01-01T00:00:01.000+00:00", {"theme": "momentum"})
    ledger.add_event("e3", "old", "p2", None, None, None, "pipeline", "pipeline_started", "2020-01-01T00:00:02.000+00:00", {"theme": "value"})
    ledger.add_event("e4", "old", "p2", None, 1, None, "pipeline", "pipeline_finished", "2020-01-01T00:00:03.000+00:00", {"status": "critic_pass"})
    add_idea_and_attempt(ledger, run_id="old", attempt_id="a1")
    for i, cost in enumerate([0.50, 0.25]):
        ledger.add_llm_call(
            call_id=f"c{i}", run_id="old", idea_id="i1", agent="coder", model="m",
            prompt_version="p", input_tokens=None, output_tokens=None, cache_read_tokens=None,
            cache_write_tokens=None, cost_usd=cost, latency_ms=1, status="ok",
        )

    assert ledger.abandon_stale_runs(timedelta(minutes=30)) == ["old"]

    run = ledger.get_run("old")
    assert run["status"] == "abandoned"
    assert run["total_cost_usd"] == pytest.approx(0.75)
    assert run["total_cost_usd"] == pytest.approx(ledger.llm_cost_total("old"))
    rows = ledger.events_for_run("old")
    finished = {
        r["pipeline_id"]: json.loads(r["payload_json"])["status"]
        for r in rows
        if r["type"] == "pipeline_finished"
    }
    # p2 already had its closing event, so only p1 gets one.
    assert finished == {"p2": "critic_pass", "p1": "aborted_error"}
    assert "abandoned" in json.loads(
        next(r for r in rows if r["pipeline_id"] == "p1" and r["type"] == "pipeline_finished")[
            "payload_json"
        ]
    )["error"]


def test_a_legacy_ledger_with_two_holdouts_for_one_idea_is_migrated_not_refused(ledger):
    ledger.create_run("r1", "{}", 1.0)
    for idea in ("i1", "i2"):
        ledger.add_idea(
            idea_id=idea, run_id="r1", pipeline_id="p", name=f"idea {idea}",
            hypothesis="h" * 12, description="d" * 12, expected_holding_period_days=1,
        )
    for attempt, idea in (("a1", "i1"), ("a2", "i1"), ("a3", "i1"), ("a4", "i2")):
        ledger.add_attempt(
            attempt_id=attempt, idea_id=idea, run_id="r1", attempt_number=1,
            code="c", code_hash="h", status="critic_pass",
        )
    # Two holdouts for idea i1, written before idea_id existed, as the old schema allowed.
    for holdout_id, attempt in (("h1", "a1"), ("h2", "a2")):
        ledger.conn.execute(
            """INSERT INTO holdout_runs (holdout_id, attempt_id, run_id, metrics_json, status,
                                         created_at)
               VALUES (?, ?, 'r1', NULL, 'complete', ?)""",
            (holdout_id, attempt, now_iso()),
        )
    ledger.conn.commit()

    duplicates = ledger.ensure_holdout_key()

    assert [d["idea_id"] for d in duplicates] == ["i1"]
    indexes = {
        row["name"]
        for row in ledger.conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'holdout_runs'"
        )
    }
    assert "holdout_runs_one_per_idea" not in indexes  # the unique index cannot exist yet
    # A new idea can still be claimed, and the rule is enforced in code for the duplicated one.
    ledger.claim_holdouts([{"holdout_id": "h3", "attempt_id": "a4", "idea_id": "i2", "run_id": "r1"}])
    with pytest.raises(HoldoutConflict, match="already been run"):
        ledger.claim_holdouts(
            [{"holdout_id": "h4", "attempt_id": "a3", "idea_id": "i1", "run_id": "r1"}]
        )


def test_trials_cannot_be_replaced_with_insert_or_replace(ledger):
    ledger.create_run("r1", "{}", 1.0)
    add_idea_and_attempt(ledger, run_id="r1")
    ledger.add_trial("t1", "r1", "a1", "h1")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        ledger.conn.execute(
            """INSERT OR REPLACE INTO trials (trial_id, run_id, attempt_id, code_hash, created_at)
               VALUES ('t1', 'r1', 'a1', 'forged', ?)""",
            (now_iso(),),
        )
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        ledger.conn.execute(
            """INSERT INTO trials (trial_id, run_id, attempt_id, code_hash, created_at)
               VALUES ('t1', 'r1', 'a1', 'forged', ?)""",
            (now_iso(),),
        )
    ledger.conn.rollback()
    stored = ledger.conn.execute("SELECT code_hash FROM trials WHERE trial_id = 't1'").fetchone()
    assert stored["code_hash"] == "h1"
    assert ledger.count_trials() == 1


def test_finishing_a_holdout_that_was_never_claimed_is_an_error(ledger):
    ledger.create_run("r1", "{}", 1.0)
    add_idea_and_attempt(ledger, run_id="r1")
    with pytest.raises(ValueError, match="no holdout row"):
        ledger.finish_holdout("a1", "complete", {"sharpe": 1.0})


def test_the_read_only_ledger_recovers_a_crashed_writers_journal_and_refuses_writes(tmp_path):
    source = tmp_path / "ledger.db"
    writer = Ledger(source)
    writer.create_run("r1", "{}", 1.0)
    writer.close()
    # A writer that dies inside a transaction leaves a hot journal behind. A small page cache makes
    # the dirty pages reach the database file before the commit, so the rollback is really needed.
    # Copy the file and the journal while that transaction is still open.
    crashed = sqlite3.connect(source)
    crashed.execute("PRAGMA cache_size = 2")
    crashed.execute("BEGIN IMMEDIATE")
    crashed.execute("UPDATE runs SET status = 'crashed' WHERE run_id = 'r1'")
    for i in range(3000):
        crashed.execute(
            "INSERT INTO events (event_id, run_id, stage, type, timestamp, payload_json) "
            "VALUES (?, 'r1', 'run', 'run_started', ?, ?)",
            (f"e{i}", now_iso(), "x" * 300),
        )
    copy_dir = tmp_path / "copy"
    copy_dir.mkdir()
    shutil.copy(source, copy_dir / "ledger.db")
    shutil.copy(tmp_path / "ledger.db-journal", copy_dir / "ledger.db-journal")
    crashed.rollback()
    crashed.close()

    reader = Ledger(copy_dir / "ledger.db", read_only=True)
    assert reader.get_run("r1")["status"] == "running"  # the crashed update was rolled back
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        reader.conn.execute("UPDATE runs SET status = 'x'")
    reader.close()

    missing = tmp_path / "missing.db"
    with pytest.raises(sqlite3.OperationalError):
        Ledger(missing, read_only=True)
    assert not missing.exists()


def test_a_dead_run_reads_as_abandoned_before_the_reaper_runs(ledger):
    from datetime import UTC, datetime

    now = datetime.now(UTC)
    ledger.create_run("live", "{}", 1.0)
    ledger.add_event("e1", "live", None, None, None, None, "run", "run_started", now_iso(), {})
    ledger.create_run("dead", "{}", 1.0)
    old = (now - timedelta(hours=2)).isoformat(timespec="milliseconds")
    ledger.add_event("e2", "dead", None, None, None, None, "run", "run_started", old, {})
    ledger.create_run("done", "{}", 1.0)
    ledger.finish_run("done", "completed", 0.0)

    window = timedelta(minutes=30)
    assert ledger.effective_run_status("live", window) == "running"
    assert ledger.effective_run_status("dead", window) == "abandoned"
    assert ledger.effective_run_status("done", window) == "completed"
    assert ledger.effective_run_status("missing", window) is None
    # Reading does not write: the stored status is still running until the reaper runs.
    assert ledger.get_run("dead")["status"] == "running"


def test_a_ledger_with_the_old_events_table_is_rebuilt_with_a_seq_key(tmp_path):
    # Build a ledger the way it was before the seq key: events keyed by a text id.
    path = tmp_path / "old.db"
    old = Ledger(path)
    old.create_run("r1", "{}", 1.0)
    old.close()
    conn = sqlite3.connect(path)
    conn.execute("DROP TABLE events")
    conn.execute(
        """CREATE TABLE events (
               event_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id),
               pipeline_id TEXT, idea_id TEXT, attempt_number INTEGER, agent TEXT, stage TEXT,
               type TEXT NOT NULL, timestamp TEXT NOT NULL, payload_json TEXT NOT NULL)"""
    )
    for i in range(3):
        conn.execute(
            "INSERT INTO events VALUES (?, 'r1', NULL, NULL, NULL, NULL, 'run', 'run_started', ?, '{}')",
            (f"e{i}", f"2026-01-0{i + 1}T00:00:00.000+00:00"),
        )
    conn.commit()
    conn.close()

    migrated = Ledger(path)
    columns = [row["name"] for row in migrated.conn.execute("PRAGMA table_info(events)")]
    assert "seq" in columns
    rows = migrated.events_for_run("r1")
    assert [r["event_id"] for r in rows] == ["e0", "e1", "e2"]  # the rows and their order survive
    page = migrated.events_after(0, limit=2)
    assert [r["seq"] for r in page] == [1, 2]
    assert [r["event_id"] for r in migrated.events_after(page[-1]["seq"])] == ["e2"]
    migrated.close()


def test_a_dead_run_reports_the_sum_of_its_calls_before_any_reaper_runs(ledger):
    # The run row still says 'running' and its stored total is 0. The reader must not show that 0.
    ledger.create_run("dead", "{}", 2.0)
    ledger.add_llm_call(
        call_id="d1", run_id="dead", idea_id=None, agent="ideas", model="m",
        prompt_version="p", input_tokens=None, output_tokens=None, cache_read_tokens=None,
        cache_write_tokens=None, cost_usd=0.4, latency_ms=1, status="ok",
    )
    assert ledger.get_run("dead")["total_cost_usd"] == 0
    assert ledger.run_cost_usd("dead") == pytest.approx(0.4)


def test_a_finished_run_reports_its_stored_total(ledger):
    ledger.create_run("done", "{}", 2.0)
    ledger.finish_run("done", "completed", 0.9)
    assert ledger.run_cost_usd("done") == pytest.approx(0.9)
    assert ledger.run_cost_usd("missing") is None


def test_close_running_attempts_closes_only_this_runs_open_attempts(ledger):
    ledger.create_run("r1", "{}", 1.0)
    ledger.create_run("r2", "{}", 1.0)
    add_idea_and_attempt(ledger, run_id="r1", attempt_id="open-1", status="running")
    ledger.add_attempt(
        attempt_id="done-1", idea_id="i1", run_id="r1", attempt_number=2,
        code="x", code_hash="h", status="critic_pass",
    )
    ledger.add_idea(
        idea_id="i2", run_id="r2", pipeline_id="p2", name="Other", hypothesis="h",
        description="d", expected_holding_period_days=5,
    )
    ledger.add_attempt(
        attempt_id="other-open", idea_id="i2", run_id="r2", attempt_number=1,
        code="x", code_hash="h", status="running",
    )

    assert ledger.close_running_attempts("r1", "the run ended first") == 1
    statuses = {r["attempt_id"]: r["status"] for r in ledger.conn.execute("SELECT * FROM attempts")}
    assert statuses == {"open-1": "aborted_error", "done-1": "critic_pass", "other-open": "running"}


def test_a_holdout_outcome_outside_the_registry_is_refused(ledger):
    ledger.create_run("r1", "{}", 1.0)
    add_idea_and_attempt(ledger, run_id="r1")
    ledger.add_holdout("h1", "a1", "r1", "running", None)
    with pytest.raises(ValueError, match="holdout status"):
        ledger.finish_holdout("a1", "passed", {"sharpe": 1.0})
    assert ledger.get_holdout("a1")["status"] == "running"


def test_finishing_a_holdout_inside_a_transaction_does_not_undo_the_callers_writes(ledger):
    # finish_holdout's own failure is caught by the caller, which then keeps writing (R4-AGT-14).
    ledger.create_run("r1", "{}", 1.0)
    with ledger.transaction():
        ledger.add_llm_call(
            call_id="kept", run_id="r1", idea_id=None, agent="a", model="m",
            prompt_version="p", input_tokens=None, output_tokens=None, cache_read_tokens=None,
            cache_write_tokens=None, cost_usd=0.1, latency_ms=1, status="ok",
        )
        with pytest.raises(ValueError, match="no holdout row"):
            ledger.finish_holdout("never-claimed", "complete", None)
    assert ledger.llm_cost_total("r1") == pytest.approx(0.1)


def test_claiming_a_holdout_inside_a_transaction_is_refused_before_anything_is_committed(ledger):
    # claim_holdouts takes the write lock itself. Inside another transaction it would commit the
    # caller's pending writes, so it refuses to run there (R4-AGT-14).
    ledger.create_run("r1", "{}", 1.0)
    add_idea_and_attempt(ledger, run_id="r1")
    with pytest.raises(RuntimeError, match="transaction"):
        with ledger.transaction():
            ledger.add_llm_call(
                call_id="pending", run_id="r1", idea_id=None, agent="a", model="m",
                prompt_version="p", input_tokens=None, output_tokens=None,
                cache_read_tokens=None, cache_write_tokens=None, cost_usd=0.1,
                latency_ms=1, status="ok",
            )
            ledger.claim_holdouts(
                [{"holdout_id": "h1", "attempt_id": "a1", "idea_id": "i1", "run_id": "r1"}]
            )
    assert ledger.llm_cost_total("r1") == pytest.approx(0.0)
