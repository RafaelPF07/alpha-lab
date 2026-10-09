// Pure helpers the pages use (R4-FE-10): the summary throttle (R4-FE-06), holdout and call-table wording
// (R4-FE-07, R4-FE-12), research wording (R4-FE-08, R4-WHOLE-01) and the holdout band copy (R4-FE-03, R4-WHOLE-04).
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { installFakeEcharts, fakeElement, loadStatic, useFixtureContract } from "./_static.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const STATIC_DIR = join(HERE, "..", "..", "src", "alpha_lab", "dashboard", "static");

// A clock that the test moves by hand, and a timer list the test can fire.
function fakeClock() {
  const timers = [];
  const state = { now: 0 };
  return {
    state,
    now: () => state.now,
    schedule: (fn, ms) => {
      const timer = { fn, due: state.now + ms, live: true };
      timers.push(timer);
      return timer;
    },
    cancel: (timer) => { if (timer) timer.live = false; },
    pending: () => timers.filter((t) => t.live),
  };
}

describe("trailingThrottle (R4-FE-06)", () => {
  it("runs at once, holds back updates inside the interval, and runs once more when it ends", async () => {
    const { trailingThrottle } = await loadStatic("core/throttle.js");
    const c = fakeClock();
    const calls = [];
    const throttle = trailingThrottle((at) => calls.push(at), 5000, c);
    throttle.request();
    c.state.now = 1000;
    throttle.request();
    c.state.now = 2000;
    throttle.request();
    assert.deepEqual(calls, [0], "the two updates inside the interval are held back");
    assert.equal(c.pending().length, 1, "one trailing update is scheduled, not one per event");
    assert.equal(c.pending()[0].due, 5000);
    c.state.now = 5000;
    const [timer] = c.pending();
    timer.live = false;
    timer.fn();
    assert.deepEqual(calls, [0, 5000], "the last batch is not lost when the interval ends");
  });

  it("a forced request runs at once and drops the pending trailing update", async () => {
    const { trailingThrottle } = await loadStatic("core/throttle.js");
    const c = fakeClock();
    const calls = [];
    const throttle = trailingThrottle((at) => calls.push(at), 5000, c);
    throttle.request();
    c.state.now = 1000;
    throttle.request();
    c.state.now = 1500;
    throttle.request(true);
    assert.deepEqual(calls, [0, 1500]);
    assert.equal(c.pending().length, 0, "the forced update cancels the trailing one");
  });

  it("reset lets the next request run at once, for a new run", async () => {
    const { trailingThrottle } = await loadStatic("core/throttle.js");
    const c = fakeClock();
    const calls = [];
    const throttle = trailingThrottle((at) => calls.push(at), 5000, c);
    throttle.request();
    c.state.now = 1000;
    throttle.request();
    throttle.reset();
    assert.equal(c.pending().length, 0);
    throttle.request();
    assert.deepEqual(calls, [0, 1000]);
  });
});

describe("holdoutNote says what happened to the holdout (R4-FE-07)", () => {
  it("a failed holdout says it ran and cannot be run again, not 'not run'", async () => {
    const { holdoutNote } = await loadStatic("core/signal_text.js");
    const text = holdoutNote({ status: "error" });
    assert.match(text, /failed/i);
    assert.match(text, /cannot be run again/);
    assert.doesNotMatch(text, /not run/i);
  });

  it("a locked holdout is 'not run', and a recorded one says so", async () => {
    const { holdoutNote } = await loadStatic("core/signal_text.js");
    assert.match(holdoutNote({ status: "locked" }), /Holdout not run for this signal/);
    assert.equal(holdoutNote({ status: "complete" }), "Holdout run recorded.");
  });

  it("a stale running holdout is marked stale", async () => {
    const { holdoutNote } = await loadStatic("core/signal_text.js");
    const text = holdoutNote({ status: "running", stale: true, started_at: "2026-10-08T10:00:00+00:00" });
    assert.match(text, /stale/);
  });
});

describe("callsTable escapes every ledger value (R4-FE-12)", () => {
  it("escapes token counts as well as text, and shows n/a for a missing count", async () => {
    const { callsTable } = await loadStatic("core/signal_text.js");
    const html = callsTable([{ agent: "coder", model: "m", input_tokens: "<img src=x onerror=1>", output_tokens: null, cost_usd: 0.01 }]);
    assert.doesNotMatch(html, /<img/, "a token count is never parsed as markup");
    assert.match(html, /&lt;img src=x onerror=1&gt;/);
    assert.match(html, /n\/a/);
  });
});

describe("research wording: reached validation, passed validation (R4-FE-08, R4-WHOLE-01)", () => {
  const points = [
    { name: "a", passed: true },
    { name: "b", passed: true },
    { name: "c", passed: false },
  ];

  it("counts reached and passed from each point's passed flag", async () => {
    const { validationCounts } = await loadStatic("core/research_text.js");
    assert.deepEqual(validationCounts(points), { reached: 3, passed: 2 });
  });

  it("the run note reads '3 reached validation, 2 passed', never 'validated'", async () => {
    const { runNote } = await loadStatic("core/research_text.js");
    const note = runNote(points, 40, "");
    assert.equal(note, "3 reached validation, 2 passed · 40 trials in the ledger");
    assert.doesNotMatch(note, /validated/i);
    assert.match(runNote(points, 40, "too few days"), /deflated Sharpe not defined: too few days/);
  });

  it("labels each signal by its own passed flag", async () => {
    const { signalOutcome } = await loadStatic("core/research_text.js");
    assert.equal(signalOutcome(points[0]), "passed validation");
    assert.equal(signalOutcome(points[2]), "failed validation");
  });

  it("chart text counts signals that reached validation", async () => {
    const { reachedText } = await loadStatic("core/research_text.js");
    assert.equal(reachedText(1), "1 signal that reached validation");
    assert.equal(reachedText(3), "3 signals that reached validation");
  });
});

describe("the holdout window is not drawn, and the copy says so (R4-FE-03, R4-WHOLE-04)", () => {
  it("the research page no longer says the holdout is shaded", () => {
    const html = readFileSync(join(STATIC_DIR, "research.html"), "utf-8");
    assert.doesNotMatch(html, /Shaded by period: train, validation, and the holdout window/);
    assert.match(html, /The holdout window is not drawn/);
  });

  it("an equity chart with no holdout dates draws no holdout band", async () => {
    const charts = installFakeEcharts();
    const equity = await loadStatic("viz/equity.js");
    equity.init(fakeElement());
    equity.update([{ name: "signal", dates: ["2026-01-02", "2026-01-05"], equity: [1, 1.1], train_end: "2025-12-31",
      validation_end: "2026-01-05", holdout: "locked", dsr_reason: "" }], null);
    const areas = charts[0].option.series[0].markArea.data;
    const names = areas.map((pair) => pair[0]?.name ?? "");
    assert.ok(!names.some((n) => /holdout/.test(n)), `markArea names: ${names.join(", ")}`);
  });
});

// R5-F4: galaxy points carry the attempt's status, and equity curves carry passed. The page counts from the
// status and the server's sets (DECISION-13), and the flag is only a fallback for an older payload.
describe("research counts from the galaxy point's status (R5-F4)", () => {
  it("counts reached and passed from each point's status and the server's sets", async () => {
    await useFixtureContract();
    const { validationCounts } = await loadStatic("core/research_text.js");
    const points = [{ status: "validation_pass" }, { status: "validation_fail" }, { status: "failed_gate" }];
    assert.deepEqual(validationCounts(points), { reached: 2, passed: 1 });
  });

  it("the status wins over a disagreeing passed flag", async () => {
    await useFixtureContract();
    const { signalOutcome, validationCounts } = await loadStatic("core/research_text.js");
    const point = { status: "validation_fail", passed: true };
    assert.equal(signalOutcome(point), "failed validation");
    assert.deepEqual(validationCounts([point]), { reached: 1, passed: 0 });
  });

  it("falls back to the passed flag only for a payload with no status", async () => {
    const { validationCounts } = await loadStatic("core/research_text.js");
    assert.deepEqual(validationCounts([{ passed: true }, { passed: false }]), { reached: 2, passed: 1 });
  });
});

describe("equity curves name whether each signal passed (R5-F4)", () => {
  it("labels each curve with its passed flag", async () => {
    const charts = installFakeEcharts();
    const equity = await loadStatic("viz/equity.js");
    equity.init(fakeElement());
    const curve = (name, passed) => ({ name, passed, dates: ["2020-01-01", "2020-01-02"], equity: [1, 1.1], drawdown: [0, 0],
      train_end: "2020-01-01", validation_end: "2020-01-02", holdout: "locked" });
    equity.update([curve("alpha", true), curve("beta", false)], null);
    const names = charts[0].option.series.map((s) => s.name);
    assert.ok(names.includes("alpha (passed)"), `series names: ${names}`);
    assert.ok(names.includes("beta (failed)"), `series names: ${names}`);
  });
});
