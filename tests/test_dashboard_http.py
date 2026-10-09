"""The dashboard over real HTTP: uvicorn in a thread on a free port, requests made with urllib.

No httpx and no network. Each test that needs a server starts one on 127.0.0.1 and stops it on exit.
Every ledger is a temporary file. Bodies are parsed with a parser that rejects NaN and Infinity,
because a browser's JSON.parse rejects them too.
"""

import asyncio
import json
import shutil
import socket
import sqlite3
import threading
import time
import urllib.error
import urllib.request
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import uvicorn

from alpha_lab.dashboard import queries
from alpha_lab.dashboard.server import create_app
from alpha_lab.events.bus import EventBus
from alpha_lab.events.demo import DEMO_RUN_ID, run_demo
from alpha_lab.ledger.db import Ledger

STATIC = Path(__file__).resolve().parents[1] / "src" / "alpha_lab" / "dashboard" / "static"
NAN = float("nan")
# A holdout number that must never reach a response before its holdout run is complete.
SENTINEL = {"sharpe": 7.7777, "ann_return": 0.31415}
SENTINEL_TEXT = b"7.7777"
JSON_ROUTES = [
    "/api/runs",
    "/api/runs/{run}/summary",
    "/api/runs/{run}/events?after=0&limit=2000",
    "/api/runs/{run}/llm",
    "/api/runs/{run}/timeline",
    "/api/runs/{run}/research",
    "/api/runs/{run}/equity?top=5",
    "/api/runs/{run}/correlation",
    "/api/runs/{run}/benchmark?top=5",
]


def _refuse_constant(token: str):
    raise ValueError(f"not strict JSON: {token}")


def strict_loads(body: bytes):
    """Parse a body the way a browser does: NaN, Infinity and -Infinity are errors."""
    return json.loads(body.decode("utf-8"), parse_constant=_refuse_constant)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextmanager
def serve(app):
    """uvicorn on 127.0.0.1 in a background thread. Yields the base URL and stops the server on exit."""
    port = free_port()
    config = uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="warning", timeout_graceful_shutdown=2
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 20
    while not server.started:
        if not thread.is_alive() or time.monotonic() > deadline:
            raise RuntimeError("uvicorn did not start")
        time.sleep(0.02)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=20)
        assert not thread.is_alive(), "uvicorn did not stop"


def fetch(base: str, path: str, *, method: str = "GET", headers: dict | None = None):
    """Returns (status, content type, body). HTTP error statuses are returned, not raised."""
    request = urllib.request.Request(base + path, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.status, response.headers.get("Content-Type", ""), response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.headers.get("Content-Type", ""), error.read()


def read_stream(base: str, query: str, count: int) -> list[tuple[str, str]]:
    """Open /events/stream, read `count` events, then disconnect. Returns (id line, data line) pairs."""
    pairs = []
    pending_id = None
    with urllib.request.urlopen(base + "/events/stream?" + query, timeout=20) as response:
        assert response.status == 200
        assert response.headers.get_content_type() == "text/event-stream"
        while len(pairs) < count:
            raw = response.readline()
            if not raw:  # end of stream; a blank line is only the gap between two events
                break
            line = raw.decode("utf-8").rstrip("\r\n")
            if line.startswith("id: "):
                pending_id = line
            elif line.startswith("data: "):
                pairs.append((pending_id, line))
    return pairs


@contextmanager
def opened(path: Path):
    ledger = Ledger(path)
    try:
        yield ledger
    finally:
        ledger.close()


def copy_of(source: Path, tmp_path: Path) -> Path:
    target = tmp_path / source.name
    shutil.copy2(source, target)
    return target


def returns_between(start: str, end: str, mean: float, std: float, seed: int) -> dict[str, float]:
    days = pd.bdate_range(start, end)
    values = np.random.default_rng(seed).normal(mean, std, len(days))
    return {str(d.date()): float(v) for d, v in zip(days, values)}


def zero_returns(start: str, end: str) -> dict[str, float]:
    return {str(d.date()): 0.0 for d in pd.bdate_range(start, end)}


def metrics(sharpe: float, turnover: float = 0.1) -> dict:
    return {
        "n_days": 1000.0,
        "ann_return": 0.05,
        "ann_volatility": 0.1,
        "sharpe": sharpe,
        "max_drawdown": -0.1,
        "avg_daily_turnover": turnover,
        "hit_rate": 0.5,
    }


TRAIN = ("2018-01-02", "2018-06-29")
VALIDATION = ("2019-01-02", "2019-06-28")


@pytest.fixture(scope="module")
def demo_db(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("http_demo") / "demo.db"
    ledger = Ledger(path)
    asyncio.run(run_demo(ledger, EventBus(ledger), n_pipelines=20, pace=0.0, seed=3))
    ledger.close()
    return path


class NanLedger:
    """Builds the NaN fixture: NaN metrics the way the pipeline writes them (bare NaN tokens),
    zero-variance returns, a holdout with NaN numbers, and a legacy validation row without returns."""

    def __init__(self, path: Path) -> None:
        self.ledger = Ledger(path)
        self.attempts: dict[str, str] = {}

    def idea(self, run: str, name: str, theme: str) -> str:
        idea_id = uuid.uuid4().hex
        self.ledger.add_idea(
            idea_id=idea_id,
            run_id=run,
            pipeline_id=uuid.uuid4().hex,
            name=name,
            hypothesis="hyp",
            description="desc",
            expected_holding_period_days=21,
            theme=theme,
        )
        return idea_id

    def attempt(self, run, idea_id, status, train_metrics, train_returns, critic=None) -> str:
        attempt_id = uuid.uuid4().hex
        self.ledger.add_attempt(
            attempt_id=attempt_id,
            idea_id=idea_id,
            run_id=run,
            attempt_number=1,
            code="def signal(prices):\n    return prices * 0.0\n",
            code_hash=attempt_id,
            normalised_hash=attempt_id,
            status=status,
            static_ok=1,
            sandbox_ok=1,
            lookahead_ok=1,
            train_metrics_json=None if train_metrics is None else json.dumps(train_metrics),
            train_returns_json=None if train_returns is None else json.dumps(train_returns),
            critic_json=None if critic is None else json.dumps(critic),
            suspicion_level=None if critic is None else critic["suspicion_level"],
        )
        self.ledger.add_trial(uuid.uuid4().hex, run, attempt_id, attempt_id)
        return attempt_id

    def raw(self, sql: str, params: tuple) -> None:
        """Writes that bypass the ledger's normalisation, so the bare NaN token reaches the file, as in older ledgers."""
        self.ledger.conn.execute(sql, params)
        self.ledger.conn.commit()

    def event(self, run, pipeline_id, type_, stage, payload, idea_id=None) -> None:
        self.raw(
            "INSERT INTO events (event_id, run_id, pipeline_id, idea_id, attempt_number, agent, "
            "stage, type, timestamp, payload_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                uuid.uuid4().hex,
                run,
                pipeline_id,
                idea_id,
                1,
                None,
                stage,
                type_,
                "2026-01-01T00:00:00.000+00:00",
                json.dumps(payload),
            ),
        )

    def build(self) -> dict[str, str]:
        run = "nan_run"
        self.ledger.create_run(run, "{}", 5.0)
        self.event(
            run, None, "run_started", "run", {"pipelines": 4, "budget_usd": 5.0, "concurrency": 2}
        )

        pid = uuid.uuid4().hex
        self.event(run, pid, "pipeline_started", "pipeline", {"theme": "momentum"})
        i1 = self.idea(run, "Idea one NaN train", "momentum")
        self.attempts["failed_gate"] = self.attempt(
            run, i1, "failed_gate", metrics(NAN), returns_between(*TRAIN, 0.0003, 0.01, 1)
        )
        self.event(run, pid, "train_backtest", "train", metrics(NAN), idea_id=i1)
        self.event(
            run,
            pid,
            "llm_call_finished",
            "llm",
            {
                "call_id": "c1",
                "status": "ok",
                "cost_usd": 0.01,
                "latency_ms": 900,
                "input_tokens": 9,
                "output_tokens": 3,
            },
            idea_id=i1,
        )
        self.event(
            run,
            pid,
            "pipeline_finished",
            "pipeline",
            {"status": "failed_gate", "attempts": 1, "error": None, "theme": "momentum"},
        )
        self.ledger.add_llm_call(
            call_id=uuid.uuid4().hex,
            run_id=run,
            idea_id=i1,
            agent="coder",
            model="claude-haiku-5-5",
            prompt_version="coder_v1",
            input_tokens=9,
            output_tokens=3,
            cache_read_tokens=0,
            cache_write_tokens=0,
            cost_usd=0.01,
            latency_ms=900,
            status="ok",
        )

        i2 = self.idea(run, "Idea two NaN validation", "volatility")
        self.attempts["validation_fail"] = self.attempt(
            run, i2, "validation_fail", metrics(0.3), returns_between(*TRAIN, 0.0002, 0.01, 2)
        )
        # Legacy row: the validation metrics hold the bare NaN token.
        self.raw(
            "INSERT INTO validations (validation_id, attempt_id, run_id, metrics_json, passed, "
            "error, created_at, returns_json) VALUES (?, ?, ?, ?, 0, NULL, ?, ?)",
            (
                uuid.uuid4().hex,
                self.attempts["validation_fail"],
                run,
                json.dumps(metrics(NAN)),
                "2026-01-01T00:00:00.000+00:00",
                json.dumps(zero_returns(*VALIDATION)),
            ),
        )

        # New row: written through the ledger, which stores NaN as null.
        i5 = self.idea(run, "Idea five null validation", "trend filters")
        self.attempts["null_validation"] = self.attempt(
            run, i5, "validation_fail", metrics(0.2), returns_between(*TRAIN, 0.0002, 0.01, 9)
        )
        self.ledger.add_validation(
            self.attempts["null_validation"],
            run,
            metrics(NAN),
            False,
            None,
            returns_between(*VALIDATION, 0.0001, 0.01, 10),
        )

        i3 = self.idea(run, "Idea three survivor", "momentum")
        critic = {"concerns": [], "suspicion_level": "low", "suggested_change": "none"}
        self.attempts["survivor"] = self.attempt(
            run,
            i3,
            "validation_pass",
            metrics(0.5),
            returns_between(*TRAIN, 0.0003, 0.01, 3),
            critic,
        )
        self.ledger.add_validation(
            self.attempts["survivor"],
            run,
            metrics(0.8),
            True,
            None,
            returns_between(*VALIDATION, 0.0005, 0.01, 4),
        )
        self.ledger.add_holdout(
            uuid.uuid4().hex, self.attempts["survivor"], run, "complete", metrics(NAN)
        )

        i4 = self.idea(run, "Idea four flat survivor", "mean reversion")
        self.attempts["flat"] = self.attempt(
            run, i4, "validation_pass", metrics(0.5), returns_between(*TRAIN, 0.0003, 0.01, 5)
        )
        self.ledger.add_validation(
            self.attempts["flat"], run, metrics(0.0), True, None, zero_returns(*VALIDATION)
        )
        self.ledger.add_rejected_idea(
            rejection_id=uuid.uuid4().hex,
            run_id=run,
            pipeline_id=uuid.uuid4().hex,
            name="duplicate draft",
            similar_to_idea_id=i1,
            similarity=0.9,
        )
        self.ledger.finish_run(run, "completed", 0.5)

        # A second run with a validation row written before returns_json existed.
        legacy = "legacy_run"
        self.ledger.create_run(legacy, "{}", 1.0)
        il = self.idea(legacy, "Legacy idea", "trend filters")
        self.attempts["legacy"] = self.attempt(
            legacy, il, "validation_fail", metrics(0.2), returns_between(*TRAIN, 0.0002, 0.01, 8)
        )
        self.ledger.add_validation(self.attempts["legacy"], legacy, metrics(0.2), False, None, None)
        self.ledger.finish_run(legacy, "completed", 0.1)

        # A run with nothing in it, and a rejection that belongs to another run.
        self.ledger.create_run("empty_run", "{}", 1.0)
        self.ledger.finish_run("empty_run", "completed", 0.0)
        self.ledger.create_run("other_run", "{}", 1.0)
        self.ledger.add_rejected_idea(
            rejection_id=uuid.uuid4().hex,
            run_id="other_run",
            pipeline_id=uuid.uuid4().hex,
            name="elsewhere",
            similar_to_idea_id=None,
            similarity=0.95,
        )
        self.ledger.finish_run("other_run", "completed", 0.0)
        self.ledger.close()
        return self.attempts


@pytest.fixture(scope="module")
def nan_db(tmp_path_factory) -> tuple[Path, dict[str, str]]:
    path = tmp_path_factory.mktemp("http_nan") / "edge.db"
    attempts = NanLedger(path).build()
    return path, attempts


def attempt_ids_of(path: Path, run: str) -> list[str]:
    conn = sqlite3.connect(path)
    try:
        return [
            r[0] for r in conn.execute("SELECT attempt_id FROM attempts WHERE run_id = ?", (run,))
        ]
    finally:
        conn.close()


def test_the_nan_fixture_really_holds_non_finite_numbers(nan_db):
    path, _ = nan_db
    conn = sqlite3.connect(path)
    try:
        raw = [r[0] for r in conn.execute("SELECT train_metrics_json FROM attempts")]
    finally:
        conn.close()
    assert any("NaN" in text for text in raw if text), "the fixture must store bare NaN tokens"


def test_every_json_route_answers_200_with_strict_json_for_the_nan_fixture(nan_db):
    path, _ = nan_db
    with serve(create_app(ledger_path=path)) as base:
        for template in JSON_ROUTES:
            status, content_type, body = fetch(base, template.format(run="nan_run"))
            assert status == 200, template
            assert content_type.startswith("application/json"), template
            strict_loads(body)
        for attempt_id in attempt_ids_of(path, "nan_run"):
            status, _, body = fetch(base, f"/api/signals/{attempt_id}")
            assert status == 200
            strict_loads(body)


def test_non_finite_numbers_are_sent_as_null(nan_db):
    path, attempts = nan_db
    with serve(create_app(ledger_path=path)) as base:
        detail = strict_loads(fetch(base, f"/api/signals/{attempts['failed_gate']}")[2])
        assert detail["train_metrics"]["sharpe"] is None
        survivor = strict_loads(fetch(base, f"/api/signals/{attempts['survivor']}")[2])
        assert survivor["holdout"]["metrics"]["sharpe"] is None


def test_the_event_stream_sends_strict_json_for_the_nan_fixture(nan_db):
    path, _ = nan_db
    with opened(path) as ledger:
        count = len(ledger.events_for_run("nan_run"))
    with serve(create_app(ledger_path=path)) as base:
        pairs = read_stream(base, "run_id=nan_run&after=0", count)
    assert len(pairs) == count
    events = [strict_loads(data[len("data: ") :].encode()) for _, data in pairs]
    nan_event = next(e for e in events if e["type"] == "train_backtest")
    assert nan_event["payload"]["sharpe"] is None


def test_the_first_stream_chunk_carries_its_cursor_and_a_valid_event(demo_db):
    with opened(demo_db) as ledger:
        first = queries.events_after(ledger, DEMO_RUN_ID, 0, 1)[0]["seq"]
    with serve(create_app(ledger_path=demo_db)) as base:
        [(id_line, data_line)] = read_stream(base, f"run_id={DEMO_RUN_ID}&after=0", 1)
    event = strict_loads(data_line[len("data: ") :].encode())
    assert id_line == f"id: {first}"
    assert event["seq"] == first
    assert event["type"] == "run_started"


def test_the_stream_resumes_after_the_cursor(demo_db):
    with serve(create_app(ledger_path=demo_db)) as base:
        ((_, data_line),) = read_stream(base, f"run_id={DEMO_RUN_ID}&after=0", 1)
        first = strict_loads(data_line[len("data: ") :].encode())["seq"]
        ((_, next_line),) = read_stream(base, f"run_id={DEMO_RUN_ID}&after={first}", 1)
    assert strict_loads(next_line[len("data: ") :].encode())["seq"] > first


def test_no_holdout_numbers_are_sent_before_the_holdout_is_complete(demo_db, tmp_path):
    path = copy_of(demo_db, tmp_path)
    with opened(path) as ledger:
        attempt_id = queries.galaxy(ledger, DEMO_RUN_ID)["points"][0]["attempt_id"]
        # The worst case: the row holds the numbers while the run is still "running".
        ledger.add_holdout(uuid.uuid4().hex, attempt_id, DEMO_RUN_ID, "running", SENTINEL)
    with serve(create_app(ledger_path=path)) as base:
        for route in (
            f"/api/signals/{attempt_id}",
            f"/api/runs/{DEMO_RUN_ID}/research",
            f"/api/runs/{DEMO_RUN_ID}/equity?top=20",
        ):
            status, _, body = fetch(base, route)
            assert status == 200, route
            assert SENTINEL_TEXT not in body, route
        detail = strict_loads(fetch(base, f"/api/signals/{attempt_id}")[2])
        assert detail["holdout"] == {"status": "running"}
        curves = strict_loads(fetch(base, f"/api/runs/{DEMO_RUN_ID}/equity?top=20")[2])
        assert all(c["holdout"] == "locked" for c in curves)


def test_holdout_numbers_appear_only_once_the_holdout_is_complete(demo_db, tmp_path):
    path = copy_of(demo_db, tmp_path)
    with opened(path) as ledger:
        attempt_id = queries.galaxy(ledger, DEMO_RUN_ID)["points"][0]["attempt_id"]
        ledger.add_holdout(uuid.uuid4().hex, attempt_id, DEMO_RUN_ID, "running", None)
        ledger.update_holdout(attempt_id, "complete", SENTINEL)
    with serve(create_app(ledger_path=path)) as base:
        detail = strict_loads(fetch(base, f"/api/signals/{attempt_id}")[2])
        assert detail["holdout"]["metrics"] == SENTINEL
        curves = strict_loads(fetch(base, f"/api/runs/{DEMO_RUN_ID}/equity?top=20")[2])
        statuses = {c["attempt_id"]: c["holdout"] for c in curves}
        assert statuses[attempt_id] == "complete"
        assert SENTINEL_TEXT not in fetch(base, f"/api/runs/{DEMO_RUN_ID}/research")[2]


def test_a_validation_row_without_returns_does_not_fail_the_run(nan_db):
    path, _ = nan_db
    with serve(create_app(ledger_path=path)) as base:
        for template in (
            "/api/runs/legacy_run/summary",
            "/api/runs/legacy_run/research",
            "/api/runs/legacy_run/equity",
            "/api/runs/legacy_run/correlation",
        ):
            assert fetch(base, template)[0] == 200, template
        curves = strict_loads(fetch(base, "/api/runs/legacy_run/equity")[2])
        assert curves == []


def test_the_rejected_idea_count_is_per_run(nan_db):
    path, _ = nan_db
    with serve(create_app(ledger_path=path)) as base:
        summary = strict_loads(fetch(base, "/api/runs/nan_run/summary")[2])
    assert summary["rejected_ideas"] == 1


def test_an_empty_run_answers_with_empty_data(nan_db):
    path, _ = nan_db
    with serve(create_app(ledger_path=path)) as base:
        for template in (
            "/api/runs/empty_run/research",
            "/api/runs/empty_run/equity",
            "/api/runs/empty_run/correlation",
            "/api/runs/empty_run/timeline",
        ):
            status, _, body = fetch(base, template)
            assert status == 200, template
            strict_loads(body)


def test_unknown_runs_are_404_and_out_of_range_cursors_are_422(nan_db):
    path, _ = nan_db
    with serve(create_app(ledger_path=path)) as base:
        for template in (
            "/api/runs/nope/summary",
            "/api/runs/nope/research",
            "/api/runs/nope/equity",
            "/api/runs/nope/events",
        ):
            assert fetch(base, template)[0] == 404, template
        assert fetch(base, "/api/runs/nan_run/events?after=99999999999999999999999999")[0] == 422
        assert fetch(base, "/api/runs/nan_run/events?after=-1")[0] == 422
        assert (
            fetch(base, "/events/stream?run_id=nan_run&after=99999999999999999999999999")[0] == 422
        )


def test_the_server_is_read_only_and_refuses_foreign_hosts(nan_db, tmp_path):
    path = copy_of(nan_db[0], tmp_path)
    before = path.read_bytes()
    with serve(create_app(ledger_path=path)) as base:
        assert fetch(base, "/api/runs", method="POST")[0] == 405
        assert fetch(base, "/api/runs", headers={"Host": "evil.example"})[0] == 400
        assert fetch(base, "/api/runs")[0] == 200
        for template in JSON_ROUTES:
            fetch(base, template.format(run="nan_run"))
    assert path.read_bytes() == before, "reads must not change the ledger file"


def test_static_pages_are_served_and_live_is_routed(nan_db):
    path, _ = nan_db
    with serve(create_app(ledger_path=path)) as base:
        for page in ("/", "/research", "/signal", "/replay"):
            status, content_type, _ = fetch(base, page)
            assert status == 200, page
            assert content_type.startswith("text/html"), page
        status, content_type, body = fetch(base, "/live")
    assert status == 200
    assert content_type.startswith("text/html")
    assert body == (STATIC / "live.html").read_bytes()


def test_path_traversal_under_static_is_refused(nan_db):
    path, _ = nan_db
    with serve(create_app(ledger_path=path)) as base:
        assert fetch(base, "/static/%2e%2e/%2e%2e/pyproject.toml")[0] == 404
        assert fetch(base, "/static/..%2F..%2Fpyproject.toml")[0] == 404


def test_the_signal_response_names_its_run(demo_db):
    with opened(demo_db) as ledger:
        attempt_id = ledger.attempts_for_run(DEMO_RUN_ID)[0]["attempt_id"]
    with serve(create_app(ledger_path=demo_db)) as base:
        detail = strict_loads(fetch(base, f"/api/signals/{attempt_id}")[2])
    assert detail["run_id"] == DEMO_RUN_ID


def test_a_missing_ledger_is_listed_as_empty_and_never_created(tmp_path):
    path = tmp_path / "missing" / "ledger.db"
    with serve(create_app(ledger_path=path)) as base:
        status, _, body = fetch(base, "/api/runs")
        assert status == 200
        assert strict_loads(body) == []
        assert fetch(base, "/api/runs/x/summary")[0] == 503
    assert not path.exists()
    assert not path.parent.exists()


def test_the_demo_needs_its_own_ledger_path():
    with pytest.raises(ValueError, match="real ledger"):
        create_app(demo=True)


def test_stopping_the_demo_server_marks_the_demo_interrupted(tmp_path):
    path = tmp_path / "demo.db"
    app = create_app(ledger_path=path, demo=True, demo_pace=0.2)
    with serve(app) as base:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            status, _, body = fetch(base, f"/api/runs/{DEMO_RUN_ID}/events?after=0&limit=5")
            if status == 200 and len(strict_loads(body)) >= 3:
                break
            time.sleep(0.05)
        else:
            pytest.fail("the demo never wrote its first events")
    conn = sqlite3.connect(path)
    try:
        status = conn.execute(
            "SELECT status FROM runs WHERE run_id = ?", (DEMO_RUN_ID,)
        ).fetchone()[0]
    finally:
        conn.close()
    assert status == "interrupted"
