"""SQLite ledger: every run, idea, attempt, LLM call, trial, event, validation and holdout run.

Standard library only. Nothing is ever deleted. Trials are counted here so that
the multiple-testing correction (deflated Sharpe) can use the full history.
"""

from __future__ import annotations

import json
import math
import numbers
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from alpha_lab.config import PROJECT_ROOT

LEDGER_PATH = PROJECT_ROOT / "runs" / "ledger.db"


class HoldoutConflict(RuntimeError):
    """A holdout row for this idea or this attempt already exists. Raised by claim_holdouts."""

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
    event_id TEXT PRIMARY KEY,
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
"""

# Columns added after the first schema. Applied to existing ledgers without losing rows.
MIGRATIONS = {
    "ideas": {"theme": "TEXT"},
    "attempts": {
        "normalised_hash": "TEXT",
        "critic_json": "TEXT",
        "suspicion_level": "TEXT",
        "train_returns_json": "TEXT",
    },
    "validations": {"returns_json": "TEXT"},
    # Holdout key and audit fields. idea_id gets its unique index in ensure_holdout_key().
    "holdout_runs": {"idea_id": "TEXT", "data_sha256": "TEXT", "invocation_json": "TEXT"},
}

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
        """Open the ledger. With read_only=True the file is opened with SQLite's mode=ro: it is never
        created, and no schema, migration or index statement runs, so a read cannot change the file."""
        self.path = path
        self.read_only = read_only
        self._tx_depth = 0
        self._known_columns: dict[str, set[str]] = {}
        if read_only:
            self.conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
            self.conn.row_factory = sqlite3.Row
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

    def _migrate(self) -> None:
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

    def abandon_stale_runs(self, stale_after: timedelta) -> list[str]:
        """Mark 'running' runs 'abandoned' when their newest event is older than `stale_after`.

        A live run writes events all the time, so a run that has been silent this long belongs
        to a process that died. Its attempts still marked 'running' become 'abandoned' too.
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
                self.conn.execute(
                    "UPDATE runs SET status = 'abandoned', finished_at = ? WHERE run_id = ?",
                    (finished, run_id),
                )
                self.conn.execute(
                    "UPDATE attempts SET status = 'abandoned' WHERE run_id = ? AND status = 'running'",
                    (run_id,),
                )
        return run_ids

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

    def llm_cost_by_agent(self, run_id: str) -> list[sqlite3.Row]:
        """Per agent and model. Cost includes failed and cancelled calls, which were charged.
        `calls` counts the calls that succeeded."""
        return self.conn.execute(
            """SELECT agent, model,
                      SUM(CASE WHEN status = 'ok' THEN 1 ELSE 0 END) AS calls,
                      SUM(input_tokens) AS input_tokens, SUM(output_tokens) AS output_tokens,
                      SUM(cost_usd) AS cost_usd
               FROM llm_calls WHERE run_id = ? AND status != 'refused'
               GROUP BY agent, model ORDER BY agent""",
            (run_id,),
        ).fetchall()

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
        """Attempts that reached the sandbox. Every executed candidate counts, so the correction stays conservative."""
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
            "SELECT * FROM events WHERE run_id = ? ORDER BY timestamp, rowid", (run_id,)
        ).fetchall()

    def events_after(self, rowid: int, limit: int = 500) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT rowid AS seq, * FROM events WHERE rowid > ? ORDER BY rowid LIMIT ?",
            (rowid, limit),
        ).fetchall()

    # Holdout runs. One row per idea, ever. Every eligible candidate is claimed in one
    # transaction before any holdout price is requested (see pipeline/holdout.py).
    def ensure_holdout_key(self) -> None:
        """Backfill idea_id on older holdout rows and create the one-holdout-per-idea index.

        Idempotent. claim_holdouts calls it first, so the index exists before any claim.
        """
        self.conn.execute(
            """UPDATE holdout_runs SET idea_id = (
                   SELECT a.idea_id FROM attempts a WHERE a.attempt_id = holdout_runs.attempt_id)
               WHERE idea_id IS NULL"""
        )
        self.conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS holdout_runs_one_per_idea ON holdout_runs(idea_id)"
        )
        self.conn.commit()

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
        """
        self.ensure_holdout_key()
        if self.conn.in_transaction:
            self.conn.commit()
        invocation_json = None if invocation is None else strict_json(invocation)
        created_at = now_iso()
        self.conn.execute("BEGIN IMMEDIATE")
        try:
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
            self.conn.commit()
        except BaseException:
            self.conn.rollback()
            raise

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
        """Record the outcome. A hash already stored is kept if None is passed."""
        self.conn.execute(
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
        self.conn.commit()

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
