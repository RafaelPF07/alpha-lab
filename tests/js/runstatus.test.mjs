// The run status a page shows comes from the server (GET /api/runs): status, stale and started_at (R4-F2).
// The page does not work out its own stale rule from event times.
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { loadStatic } from "./_static.mjs";

const { statusLabel, RunStatuses, clockOf } = await loadStatic("core/runstatus.js");

const at = (hh, mm) => new Date(2026, 9, 9, hh, mm, 0).getTime();
const info = (status, extra = {}) => ({ status, stale: false, startedMs: at(0, 5), ...extra });

describe("statusLabel shows the server's status", () => {
  it("shows stale with the start time when the server says the run is stale", () => {
    assert.deepEqual(statusLabel(info("running", { stale: true }), "live"), { text: "stale · started 00:05", tone: "warn" });
  });

  it("a run the server reports as abandoned shows that status, even if the stale flag is also set", () => {
    assert.deepEqual(statusLabel(info("abandoned", { stale: true }), "live"), { text: "abandoned", tone: "warn" });
  });

  it("stale wins over a reconnect, because the server's verdict is the run's state", () => {
    assert.equal(statusLabel(info("running", { stale: true }), "reconnecting", 4000).text, "stale · started 00:05");
  });

  it("shows the final status the server reports, and a finished run is not live", () => {
    assert.deepEqual(statusLabel(info("completed"), "finished"), { text: "completed", tone: "done" });
    assert.deepEqual(statusLabel(info("budget_stopped"), "live"), { text: "budget stopped", tone: "warn" });
  });

  it("shows live for a run the server says is running", () => {
    assert.deepEqual(statusLabel(info("running"), "live"), { text: "live", tone: "live" });
  });

  it("shows reconnecting while a running run's stream is retried", () => {
    assert.deepEqual(statusLabel(info("running"), "reconnecting", 2000), { text: "reconnecting in 2 s", tone: "warn" });
  });

  it("says connecting or waiting when the server has not answered for this run", () => {
    assert.equal(statusLabel(null, "connecting").text, "connecting…");
    assert.equal(statusLabel(null, "live").text, "waiting for events");
  });
});

describe("RunStatuses keeps the server's answer", () => {
  it("reads status, stale and started_at from /api/runs", async () => {
    const original = globalThis.fetch;
    globalThis.fetch = async () => ({ ok: true, json: async () => [
      { run_id: "r1", status: "running", stale: true, started_at: "2026-10-09T00:05:00+00:00" },
      { run_id: "r2", status: "completed", stale: false, started_at: "2026-10-08T00:00:00+00:00" },
    ] });
    try {
      const runs = new RunStatuses();
      await runs.refresh();
      assert.deepEqual(runs.get("r1"), { status: "running", stale: true, startedMs: Date.parse("2026-10-09T00:05:00+00:00") });
      assert.equal(runs.get("r2").status, "completed");
      assert.equal(runs.get("missing"), null);
      assert.equal(runs.list.length, 2);
    } finally {
      globalThis.fetch = original;
    }
  });

  it("clockOf formats local HH:MM", () => {
    assert.equal(clockOf(at(7, 9)), "07:09");
  });
});
