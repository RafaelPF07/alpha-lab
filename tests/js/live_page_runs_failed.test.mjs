// The live page when the first run-list request fails (R4-FE-05). The run select must still switch runs, so its
// change listener is attached before the first request, not after it.
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
    "/api/runs": { status: 500, body: { detail: "boom" } },
  },
});
await loadPage("pages/live.js");
await settle();

after(() => { globalThis.setInterval = realSetInterval; });

describe("the live page when the run list fails first (R4-FE-05)", () => {
  it("shows that the run list could not load", () => {
    const box = page.doc.elements.get("page-error");
    assert.ok(box && box.hidden === false);
    assert.match(box.textContent, /Could not load the run list/);
  });

  it("the run select still has its change listener, so choosing a run works", () => {
    const select = page.doc.elements.get("run-select");
    assert.equal(typeof select.listeners.change, "function", "the change listener is attached");
  });
});
