"""FastAPI app: JSON endpoints read from the ledger, plus a server-sent event stream that tails it.

The frontend never touches SQLite. A live `lab run` in another process writes the
ledger, and this server streams new rows, so the dashboard works for real and demo runs.

The server only reads. Each request opens its own read-only connection, every JSON body is
strict JSON (a non-finite number is sent as null), and only the loopback host names are accepted.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from alpha_lab.config import PROJECT_ROOT
from alpha_lab.dashboard import queries
from alpha_lab.events.bus import EventBus
from alpha_lab.events.demo import DEMO_RUN_ID, run_demo
from alpha_lab.ledger.db import LEDGER_PATH, Ledger, strict_json

STATIC = Path(__file__).parent / "static"
SSE_INTERVAL_SECONDS = 0.3
HEARTBEAT_SECONDS = 15
STREAM_PAGE = 500
MAX_SEQ = 2**63 - 1  # SQLite integers are signed 64-bit; a larger cursor raises OverflowError
ALLOWED_HOSTS = ["127.0.0.1", "localhost"]  # blocks DNS rebinding from other web pages


class StrictJSONResponse(JSONResponse):
    """Every JSON route uses this, so no route can send NaN or Infinity (ledger.db.strict_json)."""

    def render(self, content: Any) -> bytes:
        return strict_json(content).encode("utf-8")


def create_app(
    ledger_path: Path | None = None, demo: bool = False, demo_pace: float = 0.35
) -> FastAPI:
    """Build the dashboard. The demo writes a synthetic run, so it must be given its own ledger path."""
    if demo and ledger_path is None:
        raise ValueError(
            "demo=True needs an explicit ledger_path, so the demo never writes to the real ledger"
        )
    path = LEDGER_PATH if ledger_path is None else ledger_path

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        ledger = None
        task = None
        if demo:
            ledger = Ledger(path)
            task = asyncio.create_task(run_demo(ledger, EventBus(ledger), pace=demo_pace))
        try:
            yield
        finally:
            try:
                if task is not None:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            finally:
                if ledger is not None:
                    ledger.close()

    app = FastAPI(
        title="Alpha Lab dashboard",
        lifespan=lifespan,
        default_response_class=StrictJSONResponse,
    )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)

    @contextmanager
    def read_ledger() -> Iterator[Ledger]:
        """One read-only connection per request, closed afterwards. No DDL runs and no file is created."""
        if not path.exists():
            raise HTTPException(503, "no ledger yet: run `lab run` or `lab dashboard --demo`")
        ledger = Ledger(path, read_only=True)
        try:
            yield ledger
        finally:
            ledger.close()

    @contextmanager
    def run_ledger(run_id: str) -> Iterator[Ledger]:
        with read_ledger() as ledger:
            if ledger.get_run(run_id) is None:
                raise HTTPException(404, "no such run")
            yield ledger

    def events_since(run_id: str, cursor: int) -> list[dict]:
        """Runs in a worker thread, so a slow query never blocks the event loop."""
        if not path.exists():
            return []  # the run has not been written yet, so keep waiting
        with read_ledger() as ledger:
            return queries.events_after(ledger, run_id, cursor, STREAM_PAGE)

    @app.get("/api/runs")
    def runs():
        if not path.exists():
            return []  # an empty dashboard; nothing is created
        with read_ledger() as ledger:
            return queries.list_runs(ledger)

    @app.get("/api/runs/{run_id}/summary")
    def summary(run_id: str):
        with run_ledger(run_id) as ledger:
            data = queries.run_summary(ledger, run_id)
            if data is None:  # run_ledger already checked the run exists
                raise HTTPException(404, "no such run")
            data["funnel"] = queries.funnel(ledger, run_id)
            data["spend_by_agent"] = queries.spend_by_agent(ledger, run_id)
            return data

    @app.get("/api/runs/{run_id}/events")
    def events(
        run_id: str,
        after: int = Query(0, ge=0, le=MAX_SEQ),
        limit: int = Query(2000, ge=1, le=20000),
    ):
        with run_ledger(run_id) as ledger:
            return queries.events_after(ledger, run_id, after, limit)

    @app.get("/api/runs/{run_id}/llm")
    def llm(run_id: str):
        with run_ledger(run_id) as ledger:
            return queries.llm_series(ledger, run_id)

    @app.get("/api/runs/{run_id}/timeline")
    def timeline(run_id: str):
        with run_ledger(run_id) as ledger:
            return queries.timeline(ledger, run_id)

    @app.get("/api/runs/{run_id}/research")
    def research(run_id: str):
        with run_ledger(run_id) as ledger:
            return queries.research(ledger, run_id)

    @app.get("/api/runs/{run_id}/equity")
    def equity(run_id: str, top: int = Query(5, ge=1, le=20)):
        with run_ledger(run_id) as ledger:
            return queries.equity(ledger, run_id, top)

    @app.get("/api/runs/{run_id}/correlation")
    def correlation(run_id: str):
        with run_ledger(run_id) as ledger:
            return queries.correlation(ledger, run_id)

    @app.get("/api/runs/{run_id}/benchmark")
    def benchmark(run_id: str, top: int = Query(5, ge=1, le=20)):
        with run_ledger(run_id) as ledger:
            curves = queries.equity(ledger, run_id, top)
        dates = curves[0]["dates"] if curves else []
        return queries.equal_weight_benchmark(dates)

    @app.get("/api/signals/{attempt_id}")
    def signal(attempt_id: str):
        with read_ledger() as ledger:
            data = queries.signal_detail(ledger, attempt_id)
        if data is None:
            raise HTTPException(404, "no such signal")
        return data

    @app.get("/events/stream")
    async def stream(request: Request, run_id: str, after: int = Query(0, ge=0, le=MAX_SEQ)):
        # Not a 404 for an unknown run: the stream may be opened before the run's first row exists.
        async def generate():
            cursor = after
            idle = 0.0
            while not await request.is_disconnected():
                rows = await asyncio.to_thread(events_since, run_id, cursor)
                for event in rows:
                    cursor = event["seq"]
                    yield f"id: {cursor}\ndata: {strict_json(event)}\n\n"
                if rows:
                    idle = 0.0
                else:
                    idle += SSE_INTERVAL_SECONDS
                    if idle >= HEARTBEAT_SECONDS:
                        yield ": keep-alive\n\n"
                        idle = 0.0
                if (
                    len(rows) < STREAM_PAGE
                ):  # a full page means more rows are waiting: read on at once
                    await asyncio.sleep(SSE_INTERVAL_SECONDS)

        return StreamingResponse(
            generate(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
        )

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/research")
    def research_page():
        return FileResponse(STATIC / "research.html")

    @app.get("/signal")
    def signal_page():
        return FileResponse(STATIC / "signal.html")

    @app.get("/replay")
    def replay_page():
        return FileResponse(STATIC / "replay.html")

    @app.get("/live")
    def live_page():
        return FileResponse(STATIC / "live.html")

    app.state.demo_run_id = DEMO_RUN_ID if demo else None
    app.state.project_root = PROJECT_ROOT
    return app
