"""SQLite ledger: every run, idea, attempt, LLM call, trial, event, validation and holdout run.

Standard library only. Nothing is ever deleted. Trials are counted here so that
the multiple-testing correction (deflated Sharpe) can use the full history.
"""

from __future__ import annotations

import json
import math
import numbers
import re
import sqlite3
import sys
import uuid
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from alpha_lab.config import PROJECT_ROOT
from alpha_lab.statuses import HOLDOUT_STATUSES, LLM_CALL_STATUSES

LEDGER_PATH = PROJECT_ROOT / "runs" / "ledger.db"


class HoldoutConflict(RuntimeError):
    """A holdout row for this idea or this attempt already exists. Raised by claim_holdouts."""


def _check_holdout_status(status: str) -> None:
    """Holdout rows take only the statuses in alpha_lab.statuses (DECISION-13, R4-WHOLE-05)."""
    if status not in HOLDOUT_STATUSES:
        raise ValueError(f"holdout status {status!r} is not one of {sorted(HOLDOUT_STATUSES)}")


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    config_json TEXT NOT NULL,
    budget_usd REAL NOT NULL,
    total_cost_usd REAL NOT NULL DEFAULT 0,
    status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ideas (
    idea_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    pipeline_id TEXT NOT NULL,
    name TEXT NOT NULL,
    hypothesis TEXT NOT NULL,
    description TEXT NOT NULL,
    expected_holding_period_days INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS attempts (
    attempt_id TEXT PRIMARY KEY,
    idea_id TEXT NOT NULL REFERENCES ideas(idea_id),
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    attempt_number INTEGER NOT NULL,
    code TEXT,
    code_hash TEXT,
    static_ok INTEGER,
    sandbox_ok INTEGER,
    lookahead_ok INTEGER,
    error TEXT,
    train_metrics_json TEXT,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS llm_calls (
    call_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    idea_id TEXT,
    agent TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cache_read_tokens INTEGER,
    cache_write_tokens INTEGER,
    cost_usd REAL,
    latency_ms INTEGER,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS trials (
    trial_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    attempt_id TEXT NOT NULL REFERENCES attempts(attempt_id),
    code_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL UNIQUE,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    pipeline_id TEXT,
    idea_id TEXT,
    attempt_number INTEGER,
    agent TEXT,
    stage TEXT,
    type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rejected_ideas (
    rejection_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    pipeline_id TEXT NOT NULL,
    name TEXT NOT NULL,
    similar_to_idea_id TEXT,
    similarity REAL NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS validations (
    validation_id TEXT PRIMARY KEY,
    attempt_id TEXT NOT NULL UNIQUE REFERENCES attempts(attempt_id),
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    metrics_json TEXT,
    passed INTEGER NOT NULL,
    error TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS holdout_runs (
    holdout_id TEXT PRIMARY KEY,
    attempt_id TEXT NOT NULL UNIQUE REFERENCES attempts(attempt_id),
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    metrics_json TEXT,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS trials_no_delete BEFORE DELETE ON trials
BEGIN
    SELECT RAISE(ABORT, 'trials are append-only: the deflated Sharpe counts them');
END;
CREATE TRIGGER IF NOT EXISTS trials_no_update BEFORE UPDATE ON trials
BEGIN
    SELECT RAISE(ABORT, 'trials are append-only: the deflated Sharpe counts them');
END;
-- INSERT OR REPLACE skips the delete trigger unless recursive_triggers is on, so inserts are guarded.
CREATE TRIGGER IF NOT EXISTS trials_no_replace BEFORE INSERT ON trials
WHEN EXISTS (SELECT 1 FROM trials WHERE trial_id = NEW.trial_id)
BEGIN
    SELECT RAISE(ABORT, 'trials are append-only: the deflated Sharpe counts them');
END;
"""

# Columns added after the first schema. Applied to existing ledgers without losing rows.
MIGRATIONS = {
    "ideas": {"theme": "TEXT"},
    "attempts": {
        "normalised_hash": "TEXT",
        "critic_json": "TEXT",
        "suspicion_level": "TEXT",
        "train_returns_json": "TEXT",
        "containment": "TEXT",  # the sandbox containment that ran: job-object, process-group, taskkill
    },
    "validations": {"returns_json": "TEXT"},
    # The SHA-256 of the prompt text that produced the call (WHOLE-15).
    "llm_calls": {"prompt_sha256": "TEXT"},
    # Holdout key and audit fields. idea_id gets its unique index in ensure_holdout_key().
    "holdout_runs": {"idea_id": "TEXT", "data_sha256": "TEXT", "invocation_json": "TEXT"},
}

# The events table definition, taken from SCHEMA so the rebuild uses the same columns.
_EVENTS_DDL = re.search(r"CREATE TABLE IF NOT EXISTS events \(.*?\n\);", SCHEMA, re.S).group(0)

# Ledgers written before the seq key have events keyed by a text id and paged by rowid. VACUUM may
# renumber rowids, so such a table is rebuilt once, with the same rows in the same order (BE-07).
_EVENTS_TABLE_OLD_NAME = "events_before_seq"

# Lookups by run and by code hash. Created after the migrations, so older ledgers get them too.
INDEXES = (
    "CREATE INDEX IF NOT EXISTS ideas_run ON ideas(run_id)",
    "CREATE INDEX IF NOT EXISTS attempts_run ON attempts(run_id)",
    "CREATE INDEX IF NOT EXISTS attempts_normalised_hash ON attempts(normalised_hash)",
    "CREATE INDEX IF NOT EXISTS llm_calls_run ON llm_calls(run_id)",
    "CREATE INDEX IF NOT EXISTS trials_run ON trials(run_id)",
    "CREATE INDEX IF NOT EXISTS events_run ON events(run_id)",
    "CREATE INDEX IF NOT EXISTS rejected_ideas_run ON rejected_ideas(run_id)",
    "CREATE INDEX IF NOT EXISTS validations_run ON validations(run_id)",
    "CREATE INDEX IF NOT EXISTS holdout_runs_run ON holdout_runs(run_id)",
    "CREATE INDEX IF NOT EXISTS attempts_idea ON attempts(idea_id)",
    "CREATE INDEX IF NOT EXISTS llm_calls_idea ON llm_calls(idea_id)",
)


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def json_ready(value: Any) -> Any:
    """A copy of `value` that strict JSON can hold. NaN and infinity become None."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):  # includes numpy floats, which json does not know
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def strict_json(value: Any) -> str:
    """JSON text for the ledger and the event stream, with no NaN or Infinity in it.

    Non-finite numbers become null first. allow_nan=False is a backstop: a value that gets
    past json_ready raises here, rather than writing the bare token NaN, which JSON.parse rejects.
    """
    return json.dumps(json_ready(value), default=str, allow_nan=False)


class Ledger:
    def __init__(self, path: Path = LEDGER_PATH, *, read_only: bool = False) -> None:
        """Open the ledger. With read_only=True the file must already exist and is never created, and
        no schema, migration or index statement runs. The connection is read-write with
        PRAGMA query_only = ON: application writes are refused, and SQLite can still roll back a
        crashed writer's hot journal, which a mode=ro connection cannot do (R3-BE-01)."""
        self.path = path
        self.read_only = read_only
        self._tx_depth = 0
        self._known_columns: dict[str, set[str]] = {}
        self._warned_duplicate_holdouts = False
        if read_only:
            self.conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=rw", uri=True)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA query_only = ON")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._migrate()
        for statement in INDEXES:
            self.conn.execute(statement)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def _rebuild_events_with_seq(self) -> None:
        """Rebuild an events table that has no seq key. The rows keep their order (BE-07)."""
        columns = {row["name"] for row in self.conn.execute("PRAGMA table_info(events)")}
        if "seq" in columns:
            return
        new_table = _EVENTS_DDL.replace("CREATE TABLE IF NOT EXISTS events", "CREATE TABLE events")
        self.conn.executescript(
            f"""BEGIN;
            ALTER TABLE events RENAME TO {_EVENTS_TABLE_OLD_NAME};
            {new_table};
            INSERT INTO events (event_id, run_id, pipeline_id, idea_id, attempt_number, agent,
                                stage, type, timestamp, payload_json)
              SELECT event_id, run_id, pipeline_id, idea_id, attempt_number, agent,
                     stage, type, timestamp, payload_json
              FROM {_EVENTS_TABLE_OLD_NAME} ORDER BY timestamp, rowid;
            DROP TABLE {_EVENTS_TABLE_OLD_NAME};
            COMMIT;"""
        )

    def _migrate(self) -> None:
        self._rebuild_events_with_seq()
        for table, columns in MIGRATIONS.items():
            existing = {row["name"] for row in self.conn.execute(f"PRAGMA table_info({table})")}
            for column, col_type in columns.items():
                if column not in existing:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")

    def _check_columns(self, table: str, names: Iterable[str]) -> None:
        """Refuse any column name the table does not have. Names are written into the SQL text."""
        if table not in self._known_columns:
            rows = self.conn.execute(f"PRAGMA table_info({table})").fetchall()
            self._known_columns[table] = {row["name"] for row in rows}
        unknown = sorted(set(names) - self._known_columns[table])
        if unknown:
            raise ValueError(f"{table} has no column named {', '.join(unknown)}")

    def _commit(self) -> None:
        if self._tx_depth == 0:
            self.conn.commit()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Group several writes into one commit. If the block raises, all of them are rolled back."""
        outermost = self._tx_depth == 0
        self._tx_depth += 1
        try:
            yield
        except BaseException:
            self._tx_depth -= 1
            if outermost:
                self.conn.rollback()
            raise
        self._tx_depth -= 1
        if outermost:
            self.conn.commit()

    def _insert(self, table: str, row: dict[str, Any]) -> None:
        self._check_columns(table, row)
        cols = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        self.conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", tuple(row.values()))
        self._commit()

    def _update(self, table: str, key: str, key_value: str, fields: dict[str, Any]) -> None:
        self._check_columns(table, [key, *fields])
        sets = ", ".join(f"{col} = ?" for col in fields)
        self.conn.execute(
            f"UPDATE {table} SET {sets} WHERE {key} = ?", (*fields.values(), key_value)
        )
        self._commit()

    # Runs
    def create_run(self, run_id: str, config_json: str, budget_usd: float) -> None:
        self._insert(
            "runs",
            {
                "run_id": run_id,
                "started_at": now_iso(),
                "config_json": config_json,
                "budget_usd": budget_usd,
                "status": "running",
            },
        )

    def finish_run(self, run_id: str, status: str, total_cost_usd: float) -> None:
        self._update(
            "runs",
            "run_id",
            run_id,
            {"finished_at": now_iso(), "status": status, "total_cost_usd": total_cost_usd},
        )

    def list_runs(self, limit: int = 20) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
        ).fetchall()

    def get_run(self, run_id: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()

    def effective_run_status(
        self, run_id: str, stale_after: timedelta, now: datetime | None = None
    ) -> str | None:
        """The status a reader should show for a run. Reads only, so it works on a read-only ledger.

        A run still marked 'running' whose newest event is older than `stale_after` is reported as
        'abandoned': its process has died, and the reaper has not run yet (R3-WHOLE-02). Returns None
        for an unknown run.
        """
        run = self.get_run(run_id)
        if run is None:
            return None
        if run["status"] != "running":
            return run["status"]
        newest = self.conn.execute(
            "SELECT MAX(timestamp) FROM events WHERE run_id = ?", (run_id,)
        ).fetchone()[0]
        last_seen = datetime.fromisoformat(newest or run["started_at"])
        cutoff = (now or datetime.now(UTC)) - stale_after
        return "running" if last_seen >= cutoff else "abandoned"

    def abandon_stale_runs(self, stale_after: timedelta) -> list[str]:
        """Mark 'running' runs 'abandoned' when their newest event is older than `stale_after`.

        A live run writes events all the time, so a run that has been silent this long belongs to a
        process that died. Its attempts still marked 'running' become 'abandoned'. Each pipeline it
        left open (pipeline_started with no pipeline_finished) gets a pipeline_finished event with
        status aborted_error. The caller (run.execute_run) writes run_finished. Its total_cost_usd is the sum of its
        llm_calls rows, as for a live run (DECISION-5).
        """
        cutoff = (datetime.now(UTC) - stale_after).isoformat(timespec="milliseconds")
        rows = self.conn.execute(
            """SELECT r.run_id FROM runs r
               WHERE r.status = 'running'
                 AND COALESCE((SELECT MAX(e.timestamp) FROM events e
                               WHERE e.run_id = r.run_id), r.started_at) < ?""",
            (cutoff,),
        ).fetchall()
        run_ids = [row["run_id"] for row in rows]
        finished = now_iso()
        with self.transaction():
            for run_id in run_ids:
                self._abandon_run(run_id, finished)
        return run_ids

    def _abandon_run(self, run_id: str, finished: str) -> None:
        """Close one dead run. Runs inside abandon_stale_runs' transaction."""
        open_pipelines = self.conn.execute(
            """SELECT s.pipeline_id, s.payload_json FROM events s
               WHERE s.run_id = ? AND s.type = 'pipeline_started'
                 AND NOT EXISTS (SELECT 1 FROM events f
                                 WHERE f.run_id = s.run_id AND f.pipeline_id = s.pipeline_id
                                   AND f.type = 'pipeline_finished')
               ORDER BY s.rowid""",
            (run_id,),
        ).fetchall()
        for pipeline in open_pipelines:
            pipeline_id = pipeline["pipeline_id"]
            attempts = self.conn.execute(
                "SELECT COALESCE(MAX(attempt_number), 0) FROM events WHERE pipeline_id = ?",
                (pipeline_id,),
            ).fetchone()[0]
            self.add_event(
                uuid.uuid4().hex,
                run_id,
                pipeline_id,
                None,
                None,
                None,
                "pipeline",
                "pipeline_finished",
                finished,
                {
                    "status": "aborted_error",
                    "attempts": attempts,
                    "error": "the run was abandoned: its process stopped before this pipeline finished",
                    "theme": json.loads(pipeline["payload_json"]).get("theme"),
                },
            )
        self.conn.execute(
            """UPDATE runs SET status = 'abandoned', finished_at = ?,
                   total_cost_usd = (SELECT COALESCE(SUM(c.cost_usd), 0.0) FROM llm_calls c
                                     WHERE c.run_id = runs.run_id)
               WHERE run_id = ?""",
            (finished, run_id),
        )
        self.conn.execute(
            "UPDATE attempts SET status = 'abandoned' WHERE run_id = ? AND status = 'running'",
            (run_id,),
        )

    # Ideas and attempts
    def add_idea(self, **fields: Any) -> str:
        fields.setdefault("created_at", now_iso())
        self._insert("ideas", fields)
        return fields["idea_id"]

    def recent_ideas(self, limit: int = 30) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT name, description FROM ideas ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()

    def all_ideas(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM ideas ORDER BY created_at").fetchall()

    def add_rejected_idea(self, **fields: Any) -> None:
        fields.setdefault("created_at", now_iso())
        self._insert("rejected_ideas", fields)

    def count_rejected_ideas(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM rejected_ideas").fetchone()[0]

    def add_attempt(self, **fields: Any) -> str:
        fields.setdefault("created_at", now_iso())
        self._insert("attempts", fields)
        return fields["attempt_id"]

    def update_attempt(self, attempt_id: str, **fields: Any) -> None:
        self._update("attempts", "attempt_id", attempt_id, fields)

    def close_running_attempts(self, run_id: str, error: str) -> int:
        """Close this run's attempts that are still 'running' as aborted_error. The run calls it when it
        ends, so an attempt whose own status write failed is not left open (R4-AGT-02). Returns the count."""
        cursor = self.conn.execute(
            "UPDATE attempts SET status = 'aborted_error', error = ? "
            "WHERE run_id = ? AND status = 'running'",
            (error, run_id),
        )
        self._commit()
        return cursor.rowcount

    def attempts_for_run(self, run_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM attempts WHERE run_id = ? ORDER BY created_at", (run_id,)
        ).fetchall()

    def all_attempts(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM attempts ORDER BY created_at").fetchall()

    def find_normalised_code(self, normalised_hash: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM attempts WHERE normalised_hash = ? LIMIT 1", (normalised_hash,)
        ).fetchone()

    def ideas_for_run(self, run_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM ideas WHERE run_id = ? ORDER BY created_at", (run_id,)
        ).fetchall()

    # LLM calls, trials and events
    def add_llm_call(self, **fields: Any) -> None:
        fields.setdefault("created_at", now_iso())
        self._insert("llm_calls", fields)

    def llm_cost_total(self, run_id: str) -> float:
        """What the run was charged: the sum of its llm_calls rows. Refused calls cost nothing."""
        return float(
            self.conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0.0) FROM llm_calls WHERE run_id = ?", (run_id,)
            ).fetchone()[0]
        )

    def run_cost_usd(self, run_id: str) -> float | None:
        """The cost a reader should show for a run, or None for an unknown run. A run whose row still
        says 'running' has no final total yet, so its cost is the sum of its llm_calls rows. That covers
        a run whose process died before the reaper ran (R4-AGT-05). Reads only."""
        run = self.get_run(run_id)
        if run is None:
            return None
        if run["status"] == "running":
            return self.llm_cost_total(run_id)
        return float(run["total_cost_usd"])

    def llm_cost_by_agent(self, run_id: str) -> list[sqlite3.Row]:
        """Per agent and model. Cost includes failed, discarded and cancelled calls, which were charged.
        `calls` counts the calls whose reply the harness used (status 'ok'). A discarded reply is not a call."""
        return self.conn.execute(
            """SELECT agent, model,
                      SUM(CASE WHEN status = 'ok' THEN 1 ELSE 0 END) AS calls,
                      SUM(input_tokens) AS input_tokens, SUM(output_tokens) AS output_tokens,
                      SUM(cost_usd) AS cost_usd
               FROM llm_calls WHERE run_id = ? AND status != 'refused'
               GROUP BY agent, model ORDER BY agent""",
            (run_id,),
        ).fetchall()

    def set_llm_call_status(self, call_id: str, status: str) -> None:
        """Change the status of one llm_calls row, for example to 'discarded' when its reply is unusable.
        The cost stays as it was: the call was billed."""
        if status not in LLM_CALL_STATUSES:
            raise ValueError(
                f"llm call status {status!r} is not one of {sorted(LLM_CALL_STATUSES)}"
            )
        self._update("llm_calls", "call_id", call_id, {"status": status})

    def add_trial(self, trial_id: str, run_id: str, attempt_id: str, code_hash: str) -> None:
        self._insert(
            "trials",
            {
                "trial_id": trial_id,
                "run_id": run_id,
                "attempt_id": attempt_id,
                "code_hash": code_hash,
                "created_at": now_iso(),
            },
        )

    def count_trials(self) -> int:
        """Rows in the trials table: one per attempt that was recorded as a trial before its sandbox run
        (single.py, _run_checks). Duplicates, static failures and rejected ideas add no row. Every
        revision that reaches the sandbox adds one. The deflated Sharpe uses this count as its number
        of trials. This method only counts rows. It does not show that the count is the right one."""
        return self.conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0]

    def add_validation(
        self,
        attempt_id: str,
        run_id: str,
        metrics: dict | None,
        passed: bool,
        error: str | None,
        returns: dict[str, float] | None = None,
    ) -> None:
        self._insert(
            "validations",
            {
                "validation_id": f"v_{attempt_id}",
                "attempt_id": attempt_id,
                "run_id": run_id,
                "metrics_json": None if metrics is None else strict_json(metrics),
                "passed": int(passed),
                "error": error,
                "returns_json": None if returns is None else strict_json(returns),
                "created_at": now_iso(),
            },
        )

    def validations(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            """SELECT v.*, a.idea_id, a.code, a.train_metrics_json, i.name, i.run_id AS idea_run,
                      i.theme, a.attempt_number
               FROM validations v
               JOIN attempts a ON a.attempt_id = v.attempt_id
               JOIN ideas i ON i.idea_id = a.idea_id
               ORDER BY v.created_at"""
        ).fetchall()

    def add_event(
        self,
        event_id: str,
        run_id: str,
        pipeline_id: str | None,
        idea_id: str | None,
        attempt_number: int | None,
        agent: str | None,
        stage: str,
        type_: str,
        timestamp: str,
        payload: dict[str, Any],
    ) -> None:
        self._insert(
            "events",
            {
                "event_id": event_id,
                "run_id": run_id,
                "pipeline_id": pipeline_id,
                "idea_id": idea_id,
                "attempt_number": attempt_number,
                "agent": agent,
                "stage": stage,
                "type": type_,
                "timestamp": timestamp,
                "payload_json": strict_json(payload),
            },
        )

    def events_for_run(self, run_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM events WHERE run_id = ? ORDER BY timestamp, seq", (run_id,)
        ).fetchall()

    def events_after(self, after_seq: int, limit: int = 500) -> list[sqlite3.Row]:
        """Events with a seq above `after_seq`, in seq order. seq is an AUTOINCREMENT key, so it is
        stable where rowid is not (BE-07)."""
        return self.conn.execute(
            "SELECT * FROM events WHERE seq > ? ORDER BY seq LIMIT ?",
            (after_seq, limit),
        ).fetchall()

    # Holdout runs. One row per idea, ever. Every eligible candidate is claimed in one
    # transaction before any holdout price is requested (see pipeline/holdout.py).
    def ensure_holdout_key(self) -> list[sqlite3.Row]:
        """Backfill idea_id on older holdout rows, then index it for the one-holdout-per-idea rule.

        Returns the ideas that already hold more than one holdout row. Ledgers written before the
        rule existed can have them. Then the unique index cannot be created, so a non-unique index
        is created instead, and claim_holdouts enforces the rule in code, inside BEGIN IMMEDIATE.
        A single transaction makes the backfill and the index all-or-nothing. Idempotent.
        """
        with self.transaction():
            self.conn.execute(
                """UPDATE holdout_runs SET idea_id = (
                       SELECT a.idea_id FROM attempts a WHERE a.attempt_id = holdout_runs.attempt_id)
                   WHERE idea_id IS NULL"""
            )
            duplicates = self.holdout_duplicates()
            if duplicates:
                self.conn.execute(
                    "CREATE INDEX IF NOT EXISTS holdout_runs_by_idea ON holdout_runs(idea_id)"
                )
            else:
                self.conn.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS holdout_runs_one_per_idea "
                    "ON holdout_runs(idea_id)"
                )
        if duplicates and not self._warned_duplicate_holdouts:
            self._warned_duplicate_holdouts = True
            ideas = ", ".join(str(row["idea_id"]) for row in duplicates)
            print(
                f"[ledger] {len(duplicates)} idea(s) hold more than one holdout row ({ideas}), "
                "written before the one-holdout-per-idea rule. The rule is enforced in code, not by "
                "the index. To inspect them, run: SELECT h.*, a.idea_id FROM holdout_runs h JOIN "
                "attempts a ON a.attempt_id = h.attempt_id WHERE a.idea_id IN (<the ids above>)",
                file=sys.stderr,
            )
        return duplicates

    def holdout_duplicates(self) -> list[sqlite3.Row]:
        """Ideas with more than one holdout row, with the attempts that hold them. Empty when the
        ledger follows the one-holdout-per-idea rule."""
        return self.conn.execute(
            """SELECT idea_id, COUNT(*) AS holdouts, GROUP_CONCAT(attempt_id, ' ') AS attempts
               FROM holdout_runs WHERE idea_id IS NOT NULL
               GROUP BY idea_id HAVING COUNT(*) > 1 ORDER BY idea_id"""
        ).fetchall()

    def holdout_claimed(self, idea_id: str, attempt_id: str) -> bool:
        """True when this idea, or this attempt, already has a holdout row in any status."""
        row = self.conn.execute(
            "SELECT 1 FROM holdout_runs WHERE idea_id = ? OR attempt_id = ? LIMIT 1",
            (idea_id, attempt_id),
        ).fetchone()
        return row is not None

    def claim_holdouts(
        self, rows: list[dict[str, Any]], invocation: dict[str, Any] | None = None
    ) -> None:
        """Insert every row in one transaction, or none. Raises HoldoutConflict on a clash.

        Each row needs holdout_id, attempt_id, idea_id and run_id. status (default 'running'),
        metrics and data_sha256 are optional. invocation (argv, user, time) is stored as JSON.

        This takes the write lock itself (BEGIN IMMEDIATE), so it must not run inside transaction():
        there it would commit the caller's pending writes (R4-AGT-14). It is refused there.
        """
        if self._tx_depth:
            raise RuntimeError(
                "claim_holdouts takes the write lock itself, so it cannot run inside transaction()"
            )
        for row in rows:
            _check_holdout_status(row.get("status", "running"))
        self.ensure_holdout_key()
        if self.conn.in_transaction:
            self.conn.commit()
        invocation_json = None if invocation is None else strict_json(invocation)
        created_at = now_iso()
        with self.transaction():  # commits on success, rolls back on any exception
            self.conn.execute("BEGIN IMMEDIATE")
            for row in rows:
                if self.holdout_claimed(row["idea_id"], row["attempt_id"]):
                    raise HoldoutConflict(
                        f"holdout has already been run for this idea (attempt {row['attempt_id']})"
                    )
            for row in rows:
                metrics = row.get("metrics")
                self.conn.execute(
                    """INSERT INTO holdout_runs (holdout_id, attempt_id, idea_id, run_id,
                           metrics_json, status, created_at, data_sha256, invocation_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        row["holdout_id"],
                        row["attempt_id"],
                        row["idea_id"],
                        row["run_id"],
                        None if metrics is None else strict_json(metrics),
                        row.get("status", "running"),
                        created_at,
                        row.get("data_sha256"),
                        invocation_json,
                    ),
                )

    def add_holdout(
        self,
        holdout_id: str,
        attempt_id: str,
        run_id: str,
        status: str,
        metrics: dict | None,
        data_sha256: str | None = None,
    ) -> None:
        """Insert one holdout row. A thin wrapper over claim_holdouts, so the one-per-idea rule holds."""
        found = self.conn.execute(
            "SELECT idea_id FROM attempts WHERE attempt_id = ?", (attempt_id,)
        ).fetchone()
        if found is None:
            raise ValueError(f"no attempt with id {attempt_id!r}")
        self.claim_holdouts(
            [
                {
                    "holdout_id": holdout_id,
                    "attempt_id": attempt_id,
                    "idea_id": found["idea_id"],
                    "run_id": run_id,
                    "status": status,
                    "metrics": metrics,
                    "data_sha256": data_sha256,
                }
            ]
        )

    def finish_holdout(
        self,
        attempt_id: str,
        status: str,
        metrics: dict | None,
        data_sha256: str | None = None,
    ) -> None:
        """Record the outcome. A hash already stored is kept if None is passed. Raises when no holdout
        row exists for the attempt, so a lost outcome is never reported as saved (R3-AGT-15).
        Uses transaction(), so inside an outer transaction the caller's writes are not rolled back by a
        failure here (R4-AGT-14)."""
        _check_holdout_status(status)
        with self.transaction():
            cursor = self.conn.execute(
                """UPDATE holdout_runs
                   SET status = ?, metrics_json = ?, data_sha256 = COALESCE(?, data_sha256)
                   WHERE attempt_id = ?""",
                (
                    status,
                    None if metrics is None else strict_json(metrics),
                    data_sha256,
                    attempt_id,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"no holdout row for attempt {attempt_id!r}; nothing was recorded")

    def update_holdout(
        self,
        attempt_id: str,
        status: str,
        metrics: dict | None,
        data_sha256: str | None = None,
    ) -> None:
        self.finish_holdout(attempt_id, status, metrics, data_sha256)

    def get_holdout(self, attempt_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM holdout_runs WHERE attempt_id = ?", (attempt_id,)
        ).fetchone()

    def holdouts(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM holdout_runs ORDER BY created_at").fetchall()
