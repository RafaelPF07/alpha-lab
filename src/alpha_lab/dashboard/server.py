"""FastAPI app: JSON endpoints read from the ledger, plus a server-sent event stream that tails it.

The frontend never touches SQLite. A live `lab run` in another process writes the
ledger, and this server streams new rows, so the dashboard works for real and demo runs.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from alpha_lab.config import PROJECT_ROOT
from alpha_lab.dashboard import queries
from alpha_lab.events.bus import EventBus
from alpha_lab.events.demo import DEMO_RUN_ID, run_demo
from alpha_lab.ledger.db import LEDGER_PATH, Ledger

STATIC = Path(__file__).parent / "static"
SSE_INTERVAL_SECONDS = 0.3
HEARTBEAT_SECONDS = 15


def create_app(
    ledger_path: Path = LEDGER_PATH, demo: bool = False, demo_pace: float = 0.35
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        task = None
        if demo:
            ledger = Ledger(ledger_path)
            task = asyncio.create_task(run_demo(ledger, EventBus(ledger), pace=demo_pace))
        yield
        if task is not None:
            task.cancel()

    app = FastAPI(title="Alpha Lab dashboard", lifespan=lifespan)

    def open_ledger() -> Ledger:
        return Ledger(ledger_path)

    @app.get("/api/runs")
    def runs():
        ledger = open_ledger()
        try:
            return queries.list_runs(ledger)
        finally:
            ledger.close()

    @app.get("/api/runs/{run_id}/summary")
    def summary(run_id: str):
        ledger = open_ledger()
        try:
            data = queries.run_summary(ledger, run_id)
            if data is None:
                raise HTTPException(404, "no such run")
            data["funnel"] = queries.funnel(ledger, run_id)
            data["spend_by_agent"] = queries.spend_by_agent(ledger, run_id)
            return data
        finally:
            ledger.close()

    @app.get("/api/runs/{run_id}/events")
    def events(run_id: str, after: int = Query(0, ge=0), limit: int = Query(2000, ge=1, le=20000)):
        ledger = open_ledger()
        try:
            return queries.events_after(ledger, run_id, after, limit)
        finally:
            ledger.close()

    @app.get("/api/runs/{run_id}/llm")
    def llm(run_id: str):
        ledger = open_ledger()
        try:
            return queries.llm_series(ledger, run_id)
        finally:
            ledger.close()

    @app.get("/api/runs/{run_id}/timeline")
    def timeline(run_id: str):
        ledger = open_ledger()
        try:
            return queries.timeline(ledger, run_id)
        finally:
            ledger.close()

    @app.get("/api/runs/{run_id}/research")
    def research(run_id: str):
        ledger = open_ledger()
        try:
            return {
                "galaxy": queries.galaxy(ledger, run_id),
                "themes": queries.themes(ledger, run_id),
                "failures": queries.failures(ledger, run_id),
                "economics": queries.economics(ledger),
            }
        finally:
            ledger.close()

    @app.get("/api/runs/{run_id}/equity")
    def equity(run_id: str, top: int = Query(5, ge=1, le=20)):
        ledger = open_ledger()
        try:
            return queries.equity(ledger, run_id, top)
        finally:
            ledger.close()

    @app.get("/api/runs/{run_id}/correlation")
    def correlation(run_id: str):
        ledger = open_ledger()
        try:
            return queries.correlation(ledger, run_id)
        finally:
            ledger.close()

    @app.get("/api/runs/{run_id}/benchmark")
    def benchmark(run_id: str, top: int = Query(5, ge=1, le=20)):
        ledger = open_ledger()
        try:
            curves = queries.equity(ledger, run_id, top)
        finally:
            ledger.close()
        dates = curves[0]["dates"] if curves else []
        return queries.equal_weight_benchmark(dates)

    @app.get("/api/signals/{attempt_id}")
    def signal(attempt_id: str):
        ledger = open_ledger()
        try:
            data = queries.signal_detail(ledger, attempt_id)
            if data is None:
                raise HTTPException(404, "no such signal")
            return data
        finally:
            ledger.close()

    @app.get("/events/stream")
    async def stream(request: Request, run_id: str, after: int = 0):
        async def generate():
            cursor = after
            idle = 0.0
            ledger = open_ledger()
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    rows = queries.events_after(ledger, run_id, cursor)
                    for event in rows:
                        cursor = event["seq"]
                        yield f"id: {cursor}\ndata: {json.dumps(event)}\n\n"
                    if rows:
                        idle = 0.0
                    else:
                        idle += SSE_INTERVAL_SECONDS
                        if idle >= HEARTBEAT_SECONDS:
                            yield ": keep-alive\n\n"
                            idle = 0.0
                    await asyncio.sleep(SSE_INTERVAL_SECONDS)
            finally:
                ledger.close()

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

    app.state.demo_run_id = DEMO_RUN_ID if demo else None
    app.state.project_root = PROJECT_ROOT
    return app
