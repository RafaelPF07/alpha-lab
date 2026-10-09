// The live page with no runs yet (R4-FE-04). The pill and the hint must still read "no runs yet" after the
// 15-second run-list refresh, not change to "connecting…" for good.
import { describe, it, after } from "node:test";
import assert from "node:assert/strict";
import { FIXTURE_LIMITS, FIXTURE_STATUSES } from "./_static.mjs";
import { installPage, loadPage, settle } from "./_page.mjs";

const realSetInterval = globalThis.setInterval;

const page = installPage({
  path: "/live",
  routes: {
    "/api/statuses": { status: 200, body: FIXTURE_STATUSES },
    "/api/limits": { status: 200, body: FIXTURE_LIMITS },
    "/api/runs": { status: 200, body: [] },
  },
});
await loadPage("pages/live.js");
await settle();

after(() => { globalThis.setInterval = realSetInterval; });

describe("the live page with no runs (R4-FE-04)", () => {
  it("the pill says no runs yet, and the summary line says how to start one", () => {
    assert.equal(page.doc.elements.get("live-state").textContent, "no runs yet");
    assert.match(page.doc.elements.get("stage-summary").textContent, /uv run lab run --pipelines 5 --budget 0.50/);
  });

  it("the 15-second refresh keeps the message, and does not change it to connecting", async () => {
    assert.equal(page.intervals.length, 1, "the page has one refresh timer");
    page.intervals[0]();
    await settle();
    assert.equal(page.doc.elements.get("live-state").textContent, "no runs yet");
  });
});
