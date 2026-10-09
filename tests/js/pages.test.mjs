// Small pieces the pages rely on: the demo rule (FE-06) and the newest-request guard (FE-04).
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { loadStatic } from "./_static.mjs";

describe("demo status", () => {
  it("a run is demo only when its id is demo", async () => {
    const runs = await loadStatic("core/runs.js");
    assert.equal(runs.isDemoRun("demo"), true);
    assert.equal(runs.isDemoRun("637b469e9b82"), false);
    assert.equal(runs.isDemoRun(undefined), false);
  });

  it("a real signal with a name is not demo (FE-06)", async () => {
    const runs = await loadStatic("core/runs.js");
    const real = { run_id: "637b469e9b82", idea: { name: "Twelve month winners" }, attempts: [] };
    assert.equal(runs.signalIsDemo(real), false);
  });

  it("a signal from the demo run is demo, and a signal with no run id is not", async () => {
    const runs = await loadStatic("core/runs.js");
    assert.equal(runs.signalIsDemo({ run_id: "demo", idea: { name: "x" } }), true);
    assert.equal(runs.signalIsDemo({ idea: { name: "x" } }), false);
    assert.equal(runs.signalIsDemo({ run_id: "637b469e9b82", demo: true }), true, "an explicit flag is honoured");
  });
});

describe("LatestRequest", () => {
  it("only the most recent request may write to the page (FE-04)", async () => {
    const { LatestRequest } = await loadStatic("core/api.js");
    const latest = new LatestRequest();
    const first = latest.begin();
    const second = latest.begin();
    assert.equal(latest.isCurrent(first), false, "a slow earlier response is stale");
    assert.equal(latest.isCurrent(second), true);
  });
});

describe("fetchJson", () => {
  it("throws an error that names the URL and the HTTP status (FE-12)", async () => {
    const { fetchJson } = await loadStatic("core/api.js");
    globalThis.fetch = async () => ({ ok: false, status: 404, json: async () => ({}) });
    await assert.rejects(fetchJson("/api/signals/nope"), (err) => {
      assert.match(err.message, /\/api\/signals\/nope/);
      assert.match(err.message, /404/);
      assert.equal(err.status, 404);
      return true;
    });
    delete globalThis.fetch;
  });
});
