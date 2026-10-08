"""SQLite ledger: every run, idea, attempt, LLM call, trial, event, validation and holdout run.

Standard library only. Nothing is ever deleted. Trials are counted here so that
the multiple-testing correction (deflated Sharpe) can use the full history.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from alpha_lab.config import PROJECT_ROOT

LEDGER_PATH = PROJECT_ROOT / "runs" / "ledger.db"

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
"""

# Columns added after the first schema. Applied to existing ledgers without losing rows.
MIGRATIONS = {
    "ideas": {"theme": "TEXT"},
    "attempts": {
        "normalised_hash": "TEXT",
        "critic_json": "TEXT",
        "suspicion_level": "TEXT",
    },
}


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class Ledger:
    def __init__(self, path: Path = LEDGER_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def _migrate(self) -> None:
        for table, columns in MIGRATIONS.items():
            existing = {row["name"] for row in self.conn.execute(f"PRAGMA table_info({table})")}
            for column, col_type in columns.items():
                if column not in existing:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")

    def _insert(self, table: str, row: dict[str, Any]) -> None:
        cols = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        self.conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", tuple(row.values()))
        self.conn.commit()

    def _update(self, table: str, key: str, key_value: str, fields: dict[str, Any]) -> None:
        sets = ", ".join(f"{col} = ?" for col in fields)
        self.conn.execute(
            f"UPDATE {table} SET {sets} WHERE {key} = ?", (*fields.values(), key_value)
        )
        self.conn.commit()

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

    def llm_cost_by_agent(self, run_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            """SELECT agent, model, COUNT(*) AS calls,
                      SUM(input_tokens) AS input_tokens, SUM(output_tokens) AS output_tokens,
                      SUM(cost_usd) AS cost_usd
               FROM llm_calls WHERE run_id = ? AND status = 'ok'
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
        self, attempt_id: str, run_id: str, metrics: dict | None, passed: bool, error: str | None
    ) -> None:
        self._insert(
            "validations",
            {
                "validation_id": f"v_{attempt_id}",
                "attempt_id": attempt_id,
                "run_id": run_id,
                "metrics_json": None if metrics is None else json.dumps(metrics),
                "passed": int(passed),
                "error": error,
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
                "payload_json": json.dumps(payload, default=str),
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
