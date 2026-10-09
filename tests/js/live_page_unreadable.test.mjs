// The live page, with a ledger the server cannot read (R4-FE-01). The stream sends only a comment line, which the
// browser never passes on, so the page must ask with a normal request and say what is wrong and how to fix it.
import { describe, it, after } from "node:test";
import assert from "node:assert/strict";
import { FIXTURE_LIMITS, FIXTURE_STATUSES } from "./_static.mjs";
import { installPage, loadPage, settle } from "./_page.mjs";

const RUN = "637b469e9b82";
const REASON = "ledger is unreadable: no such column: seq";
const realSetInterval = globalThis.setInterval;

const page = installPage({
  path: "/live",
  routes: {
    "/api/statuses": { status: 200, body: FIXTURE_STATUSES },
    "/api/limits": { status: 200, body: FIXTURE_LIMITS },
    "/api/runs": { status: 200, body: [{ run_id: RUN, status: "completed", stale: false, started_at: "2026-10-09T10:00:00+00:00" }] },
    [`/api/runs/${RUN}/events*`]: { status: 503, body: { detail: REASON } },
  },
});
await loadPage("pages/live.js");
await settle();

after(() => { globalThis.setInterval = realSetInterval; });

describe("the live page on an unreadable ledger (R4-FE-01)", () => {
  it("asks the server whether the run's events can be read", () => {
    assert.ok(page.requested.includes(`/api/runs/${RUN}/events?after=0&limit=1`));
  });

  it("shows the server's reason and the command that migrates the ledger, in the error banner", () => {
    const box = page.doc.elements.get("page-error");
    assert.ok(box && box.hidden === false, "the banner is shown");
    assert.match(box.textContent, /no such column: seq/);
    assert.match(box.textContent, /uv run lab ledger show/);
  });

  it("the pill says the ledger is unreadable, not 'completed' over zero pipelines", () => {
    assert.equal(page.doc.elements.get("live-state").textContent, "ledger unreadable");
  });

  it("the page still opens the stream, so it recovers when the ledger is fixed", () => {
    assert.equal(page.streams.length, 1);
    assert.ok(page.streams[0].url.includes(`run_id=${RUN}`));
  });
});
