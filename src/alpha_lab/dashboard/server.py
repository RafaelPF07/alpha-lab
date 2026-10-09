"""FastAPI app: JSON endpoints read from the ledger, plus a server-sent event stream that tails it.

The frontend never touches SQLite. A live `lab run` in another process writes the
ledger, and this server streams new rows, so the dashboard works for real and demo runs.

The server never writes application data. Each request opens its own connection and closes it
afterwards, and never creates a file. SQLite may roll back a crashed writer's journal on first read,
which is the one write it can make (see README). A missing or unreadable ledger answers 503, and so
does a ledger with an older events table, which names the command that upgrades it. Every JSON body
is strict JSON (a non-finite number is sent as null), and only the loopback host names are accepted.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from collections.abc import Iterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from alpha_lab.dashboard import queries
from alpha_lab.events.bus import EventBus
from alpha_lab.events.demo import DEMO_RUN_ID, run_demo
from alpha_lab.ledger.db import LEDGER_PATH, Ledger, strict_json
from alpha_lab.pipeline.run import STALE_RUN_AFTER
from alpha_lab.statuses import RUN_STATUSES

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"
SSE_INTERVAL_SECONDS = 0.3
HEARTBEAT_SECONDS = 15
STREAM_PAGE = 500
MAX_SEQ = 2**63 - 1  # SQLite integers are signed 64-bit; a larger cursor raises OverflowError
ENDED_RUN_STATUSES = RUN_STATUSES - {"running"}  # a stream ends once no new rows are coming
# A stream for a run id that never appears in a readable ledger ends after this long (R4-BE-11).
UNKNOWN_RUN_WAIT_SECONDS = 60
ALLOWED_HOSTS = ["127.0.0.1", "localhost"]  # blocks DNS rebinding from other web pages
# Browsers ask for /favicon.ico on every page. This SVG is served there, so that request is not a 404.
FAVICON = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">'
    '<rect width="16" height="16" rx="3" fill="#1a1b26"/>'
    '<path d="M3 12 6.5 7 9 9.5 13 4" fill="none" stroke="#7aa2f7" stroke-width="1.8" '
    'stroke-linecap="round" stroke-linejoin="round"/></svg>'
)


def _log_demo_failure(task: asyncio.Task) -> None:
    """Logs a demo failure when it happens, not only when the server stops (R4-BE-04)."""
    if not task.cancelled() and task.exception() is not None:
        log.error(
            "the demo run stopped with an error; its run row is marked failed",
            exc_info=task.exception(),
        )


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
    # Refused here, before the demo opens any connection: a writer open would run schema DDL on the real ledger.
    if demo and path.resolve() == LEDGER_PATH.resolve():
        raise ValueError("the demo must not write to the real ledger; give it its own path")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        ledger = None
        task = None
        try:
            if demo:
                ledger = Ledger(path)
                task = asyncio.create_task(run_demo(ledger, EventBus(ledger), pace=demo_pace))
                task.add_done_callback(_log_demo_failure)
                await asyncio.sleep(
                    0
                )  # let the demo's first step run, so a refusal surfaces at startup
                if task.done() and not task.cancelled() and task.exception() is not None:
                    raise task.exception()
            yield
        finally:
            try:
                if task is not None:
                    task.cancel()
                    # Waits for the demo to finish. A failure is logged by _log_demo_failure, so it is not
                    # raised here, where it would stop the shutdown (R4-BE-04).
                    await asyncio.gather(task, return_exceptions=True)
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
        """One read-only connection per request, closed afterwards. No file is created.

        A file that is not a ledger (zero bytes, not SQLite, or without a runs table) answers 503, as
        does a read that fails. The error text says why, so the page can show it.
        """
        if not path.exists():
            raise HTTPException(503, "no ledger yet: run `lab run` or `lab dashboard --demo`")
        try:
            ledger = Ledger(path, read_only=True)
        except sqlite3.DatabaseError as error:
            raise HTTPException(503, f"ledger is unreadable: {error}") from None
        try:
            try:
                has_runs = ledger.conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'runs'"
                ).fetchone()
                if has_runs is None:
                    raise HTTPException(503, "the file has no runs table, so it is not a ledger")
                if not queries.has_current_schema(ledger):
                    raise HTTPException(
                        503,
                        "this ledger was written by an older version. Run `lab ledger migrate` once "
                        "to upgrade it, then reload the page",
                    )
                yield ledger
            except sqlite3.DatabaseError as error:
                raise HTTPException(503, f"ledger is unreadable: {error}") from None
        finally:
            ledger.close()

    @contextmanager
    def run_ledger(run_id: str) -> Iterator[Ledger]:
        with read_ledger() as ledger:
            if ledger.get_run(run_id) is None:
                raise HTTPException(404, "no such run")
            yield ledger

    def poll(run_id: str, cursor: int) -> tuple[list[dict], str | None]:
        """Runs in a worker thread, so a slow query never blocks the event loop.
        Returns the new rows and the run's status. Raises HTTPException when the ledger cannot be read
        now, with the reason. The status is read first: once it is final, the rows read after it are complete."""
        with read_ledger() as ledger:
            # A run that has gone silent for STALE_RUN_AFTER reads 'abandoned': its process died.
            status = ledger.effective_run_status(run_id, STALE_RUN_AFTER)
            rows = queries.events_after(ledger, run_id, cursor, STREAM_PAGE)
            return rows, status

    def closing_state(run_id: str, last_status: str | None) -> dict:
        """The status fields for the stream's closing frame. If the ledger cannot be read at that moment,
        the status from the last successful poll is sent."""
        try:
            with read_ledger() as ledger:
                return queries.run_state(ledger, run_id)
        except HTTPException:
            return {"status": last_status, "stored_status": None, "stale": False}

    @app.get("/api/runs")
    def runs():
        if not path.exists():
            return []  # an empty dashboard; nothing is created
        with read_ledger() as ledger:
            return queries.list_runs(ledger)

    @app.get("/api/runs/{run_id}/summary")
    def summary(run_id: str):
        with run_ledger(run_id) as ledger:
            data = queries.run_summary(ledger, run_id)  # run_ledger already checked the run exists
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

    @app.get("/api/limits")
    def limits():
        return queries.limits()

    @app.get("/api/statuses")
    def statuses():
        return queries.status_sets()

    @app.get("/events/stream")
    async def stream(request: Request, run_id: str, after: int = Query(0, ge=0, le=MAX_SEQ)):
        # Not a 404 for an unknown run: the stream may be opened before the run's first row exists.
        async def closing_frame(last_status: str | None) -> str:
            """The last frame of every stream that ends: the run's status, so the page can tell a run that
            stopped from a dropped connection (R4-BE-01). The status is read again, so it is current."""
            state = await asyncio.to_thread(closing_state, run_id, last_status)
            return f"event: status\ndata: {strict_json(state)}\n\n"

        async def generate():
            cursor = after
            idle = 0.0
            unknown = 0.0  # seconds a readable ledger has had no row for this run (R4-BE-11)
            reported = None  # the last ledger problem sent, so each one is sent once
            while not await request.is_disconnected():
                status = None
                try:
                    rows, status = await asyncio.to_thread(poll, run_id, cursor)
                    problem = None
                except HTTPException as error:  # no ledger yet, unreadable, or an older schema
                    rows, problem = [], str(error.detail)
                except Exception as error:  # a failure the page should see; the stream stays open
                    rows, problem = [], f"the ledger could not be read: {error}"
                    if (
                        problem != reported
                    ):  # log the traceback once per distinct failure, not every poll
                        log.exception("the event stream could not read run %s", run_id)
                if problem is not None:
                    if problem != reported:
                        yield f"event: ledger_error\ndata: {strict_json({'error': problem})}\n\n"
                        reported = problem
                else:
                    reported = None
                for event in rows:
                    cursor = event["seq"]
                    yield f"id: {cursor}\ndata: {strict_json(event)}\n\n"
                    if event["type"] == "run_finished":
                        # The run is over and nothing more is written, so the stream ends here.
                        yield await closing_frame(status)
                        return
                if not rows and status in ENDED_RUN_STATUSES:
                    # The run was stopped or abandoned, and no new rows are coming. This also covers a run
                    # that ended without a run_finished row. A client that still wants more reconnects.
                    yield await closing_frame(status)
                    return
                if problem is None and status is None:
                    unknown += SSE_INTERVAL_SECONDS
                    if unknown >= UNKNOWN_RUN_WAIT_SECONDS:
                        yield await closing_frame(None)  # the run never appeared: say so, then end
                        return
                else:
                    unknown = 0.0
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

    @app.get("/favicon.ico")
    def favicon():
        return Response(FAVICON, media_type="image/svg+xml")

    app.state.demo_run_id = DEMO_RUN_ID if demo else None
    return app
