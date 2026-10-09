// Pipeline theatre, canvas half (static/js/viz/theatre.js), checked with a fake canvas: text fitting,
// resizing, opening a card by click (touch has no hover), and the reduced-motion setting.
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { loadStatic, media, useFixtureContract } from "./_static.mjs";

const { PipelineTheatre, fitText, CANVAS_MUTED } = await loadStatic("viz/theatre.js");
await useFixtureContract();

function fakeParent(width, height) {
  return {
    clientWidth: width,
    clientHeight: height,
    getBoundingClientRect() { return { left: 0, top: 0, width: this.clientWidth, height: this.clientHeight }; },
  };
}

// Every drawing call is a no-op. Only measureText has a meaningful result.
function fakeContext() {
  const gradient = () => ({ addColorStop() {} });
  const special = {
    measureText: (text) => ({ width: String(text).length * 6 }),
    createRadialGradient: gradient,
    createLinearGradient: gradient,
  };
  return new Proxy({}, {
    get(target, prop) {
      if (prop in target) return target[prop];
      if (prop in special) return special[prop];
      return () => undefined;
    },
    set(target, prop, value) {
      target[prop] = value;
      return true;
    },
  });
}

function fakeCanvas(parent) {
  const ctx = fakeContext();
  return {
    width: 0,
    height: 0,
    style: {},
    parentElement: parent,
    listeners: {},
    addEventListener(type, handler) { this.listeners[type] = handler; },
    getBoundingClientRect() { return parent.getBoundingClientRect(); },
    getContext() { return ctx; },
  };
}

let seq = 0;
const ev = (type, pid, extra = {}) => ({
  seq: ++seq,
  type,
  pipeline_id: pid,
  idea_id: extra.idea_id ?? null,
  agent: extra.agent ?? null,
  stage: "",
  attempt_number: null,
  timestamp: "2026-10-09T00:00:00.000+00:00",
  payload: extra.payload ?? {},
});

describe("fitText", () => {
  const ctx = { measureText: (text) => ({ width: String(text).length * 6 }) };

  it("keeps short text unchanged", () => {
    assert.equal(fitText(ctx, "CODE", 60), "CODE");
  });

  it("truncates long text with an ellipsis so it fits its space", () => {
    const fitted = fitText(ctx, "validation passed", 60);
    assert.ok(ctx.measureText(fitted).width <= 60, `"${fitted}" is wider than 60`);
    assert.ok(fitted.endsWith("…"));
  });

  it("returns nothing when no text can fit", () => {
    assert.equal(fitText(ctx, "anything", 0), "");
  });
});

describe("PipelineTheatre canvas", () => {
  it("resizes the canvas with its container and asks for a redraw (after a resize)", () => {
    const parent = fakeParent(600, 500);
    const canvas = fakeCanvas(parent);
    const theatre = new PipelineTheatre(canvas, {});
    assert.equal(canvas.width, 600);
    parent.clientWidth = 390;
    parent.clientHeight = 640;
    theatre.resize();
    assert.equal(canvas.width, 390);
    assert.equal(canvas.height, 640);
    assert.equal(canvas.style.width, "390px");
    assert.equal(theatre.dirty, true, "a resize asks for a redraw");
    theatre.destroy();
  });

  it("opens the card under a click without a prior mouse move, so touch works (phone)", () => {
    const canvas = fakeCanvas(fakeParent(900, 600));
    const opened = [];
    const theatre = new PipelineTheatre(canvas, { onOpen: (id) => opened.push(id) });
    theatre.ingest([ev("pipeline_started", "p1"), ev("idea_created", "p1", { idea_id: "idea-1", payload: { name: "Momentum" } })]);
    const rect = theatre.cardRect(0, theatre.layout());
    canvas.listeners.click({ clientX: rect.x + 30, clientY: rect.y + 30 });
    assert.deepEqual(opened, ["idea-1"]);
    canvas.listeners.click({ clientX: 2, clientY: 2 });
    assert.deepEqual(opened, ["idea-1"], "a click outside any card opens nothing");
    theatre.destroy();
  });

  it("reads the reduced-motion setting and keeps it current", () => {
    media.reduceMotion = true;
    try {
      const theatre = new PipelineTheatre(fakeCanvas(fakeParent(600, 500)), {});
      assert.equal(theatre.reduceMotion, true);
      theatre.destroy();
    } finally {
      media.reduceMotion = false;
    }
  });
});

// A run of n pipelines whose events interleave, the way a concurrent run arrives. Some pipelines are
// revised, some fail the critic, some pass. Returned in one array, so it can be fed as one batch.
function interleavedRun(n) {
  const perPipeline = [];
  for (let i = 0; i < n; i++) {
    const pid = `pipe-${String(i).padStart(2, "0")}`;
    const steps = [
      ev("pipeline_started", pid, { payload: { theme: "momentum" } }),
      ev("llm_call_started", pid, { agent: "ideas" }),
      ev("llm_call_finished", pid, { agent: "ideas", payload: { status: "ok", cost_usd: 0.001, input_tokens: 900, output_tokens: 200 } }),
      ev("idea_created", pid, { idea_id: `idea-${i}`, payload: { name: `Idea ${i}` } }),
      ev("llm_call_started", pid, { agent: "coder" }),
      ev("llm_call_finished", pid, { agent: "coder", payload: { status: "ok", cost_usd: 0.002, input_tokens: 3000, output_tokens: 900 } }),
      ev("code_written", pid, { agent: "coder" }),
    ];
    if (i % 4 === 0) {
      steps.push(ev("static_check", pid, { payload: { ok: false, errors: ["import of 'os' is not allowed"] } }));
      steps.push(ev("revision_requested", pid, { payload: { reason: "failed_static" } }));
      steps.push(ev("code_written", pid, { agent: "coder" }));
    }
    steps.push(ev("static_check", pid, { payload: { ok: true, errors: [] } }));
    steps.push(ev("sandbox_run", pid, { payload: { status: "ok" } }));
    steps.push(ev("lookahead_check", pid, { payload: { test: "truncation", passed: true } }));
    steps.push(ev("lookahead_check", pid, { payload: { test: "perturbation", passed: true } }));
    steps.push(ev("train_backtest", pid, { payload: { sharpe: 0.5 + (i % 3) / 10 } }));
    if (i % 3 === 0) {
      steps.push(ev("critic_review", pid, { payload: { suspicion_level: "high" } }));
      steps.push(ev("pipeline_finished", pid, { payload: { status: "failed_critic" } }));
    } else {
      steps.push(ev("critic_review", pid, { payload: { suspicion_level: "low" } }));
      steps.push(ev("validation_result", pid, { payload: { passed: i % 2 === 0 } }));
      steps.push(ev("pipeline_finished", pid, { payload: { status: i % 2 === 0 ? "validation_pass" : "validation_fail" } }));
    }
    perPipeline.push(steps);
  }
  const out = [];
  const longest = Math.max(...perPipeline.map((s) => s.length));
  for (let k = 0; k < longest; k++) {
    for (const steps of perPipeline) if (k < steps.length) out.push(steps[k]);
  }
  out.forEach((event, index) => { event.seq = index + 1; });
  return out;
}

// Lets every queued effect come due, the way frames do in a browser.
function settle(theatre) {
  theatre.frame(performance.now() + 60000);
}

describe("PipelineTheatre draws what the model knows (R3-FE-01)", () => {
  it("draws every pipeline of a finished run that arrives as one backlog", () => {
    const canvas = fakeCanvas(fakeParent(1036, 666));
    const theatre = new PipelineTheatre(canvas, {});
    theatre.ingest(interleavedRun(40));
    settle(theatre);
    assert.equal(theatre.vcards.size, 40, "every pipeline has a visual card");
    assert.equal(theatre.model.stats().done, 40);
    for (const card of theatre.model.cards.values()) {
      const v = theatre.vcards.get(card.id);
      assert.ok(v, `${card.id} has no visual card`);
      assert.equal(v.status, card.status, `${card.id} status`);
      assert.deepEqual(v.nodes.map((n) => n.state), card.stations.map((s) => s.state), `${card.id} station states`);
      assert.deepEqual(v.nodes.map((n) => n.text), card.stations.map((s) => s.text), `${card.id} station text`);
    }
    theatre.destroy();
  });

  it("keeps the newest state when a backlog is followed by a live batch", () => {
    const theatre = new PipelineTheatre(fakeCanvas(fakeParent(1036, 666)), {});
    const run = interleavedRun(30);
    const cut = run.length - 10;
    theatre.ingest(run.slice(0, cut));
    theatre.ingest(run.slice(cut));
    settle(theatre);
    for (const card of theatre.model.cards.values()) {
      assert.equal(theatre.vcards.get(card.id).status, card.status, `${card.id} status`);
    }
    theatre.destroy();
  });
});

describe("PipelineTheatre idles when nothing moves (R3-FE-02)", () => {
  it("does not clear and redraw a finished run with no effects queued", () => {
    const theatre = new PipelineTheatre(fakeCanvas(fakeParent(1036, 666)), {});
    theatre.ingest(interleavedRun(40));
    settle(theatre);
    // The backlog's floats, bursts and packets play for about a second after they start. Let them finish.
    const calm = performance.now() + 60000 + 2000;
    for (let k = 0; k < 130; k++) theatre.frame(calm - 2000 + k * 16);
    let clears = 0;
    theatre.ctx.clearRect = () => { clears += 1; };
    for (let k = 0; k < 60; k++) theatre.frame(calm + k * 16);
    assert.equal(clears, 0, "a finished run that nothing moves in must not be redrawn");
    theatre.destroy();
  });
});

describe("PipelineTheatre at 40 pipelines (R3-FE-03)", () => {
  it("draws every pipeline on a desktop stage, with compact cards and finished ones still visible", () => {
    const theatre = new PipelineTheatre(fakeCanvas(fakeParent(1036, 666)), {});
    theatre.ingest(interleavedRun(40));
    settle(theatre);
    const drawn = [];
    const original = theatre.drawCard.bind(theatre);
    theatre.drawCard = (card, v, rect, now) => { drawn.push(card.status); original(card, v, rect, now); };
    theatre.frame(performance.now() + 60100);
    assert.equal(drawn.length, 40, "all 40 pipelines are drawn");
    assert.equal(theatre.hiddenCount, 0, "nothing is reported as not shown");
    assert.ok(drawn.some((s) => s === "validation_pass"), "a finished pass stays visible");
    assert.ok(drawn.some((s) => s === "failed_critic"), "a finished failure stays visible");
    assert.equal(theatre.layout().compact, true, "40 pipelines need the compact card size");
    theatre.destroy();
  });

  it("says how many pipelines a phone stage cannot show", () => {
    const theatre = new PipelineTheatre(fakeCanvas(fakeParent(346, 590)), {});
    theatre.ingest(interleavedRun(40));
    settle(theatre);
    const drawn = [];
    const original = theatre.drawCard.bind(theatre);
    theatre.drawCard = (card, v, rect, now) => { drawn.push(card.id); original(card, v, rect, now); };
    theatre.frame(performance.now() + 60100);
    assert.ok(drawn.length > 0 && drawn.length < 40, `drew ${drawn.length}`);
    assert.equal(drawn.length + theatre.hiddenCount, 40, "every pipeline is either drawn or counted as not shown");
    theatre.destroy();
  });
});

// Records every font the canvas is given, so a test can read the sizes that are drawn.
function recordFonts(theatre) {
  const fonts = [];
  let value = "";
  Object.defineProperty(theatre.ctx, "font", {
    get() { return value; },
    set(v) { fonts.push(String(v)); value = v; },
    configurable: true,
  });
  return fonts;
}

describe("PipelineTheatre text and motion (R3-FE-02, R3-FE-06)", () => {
  it("draws no canvas text smaller than 10 px, on normal and compact cards", () => {
    const minimumPx = 10;
    for (const [width, height, count] of [[1036, 666, 4], [1036, 666, 40], [346, 590, 40]]) {
      const theatre = new PipelineTheatre(fakeCanvas(fakeParent(width, height)), {});
      theatre.ingest(interleavedRun(count));
      settle(theatre);
      const fonts = recordFonts(theatre);
      theatre.dirty = true;
      theatre.frame(performance.now() + 60100);
      assert.ok(fonts.length > 0, "the stage drew some text");
      for (const font of fonts) {
        const px = Number(/(\d+(?:\.\d+)?)px/.exec(font)?.[1]);
        assert.ok(px >= minimumPx, `${count} pipelines at ${width}px drew "${font}"`);
      }
      theatre.destroy();
    }
  });

  it("keeps the idle station label readable on the card body (4.5:1 or better)", () => {
    const channel = (c) => { const v = c / 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; };
    const luminance = (hex) => {
      const [r, g, b] = [1, 3, 5].map((i) => channel(parseInt(hex.slice(i, i + 2), 16)));
      return 0.2126 * r + 0.7152 * g + 0.0722 * b;
    };
    const ratio = (a, b) => {
      const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
      return (hi + 0.05) / (lo + 0.05);
    };
    const body = "#161d2f"; // the top of the card body gradient in theatre.js
    assert.ok(ratio(CANVAS_MUTED, body) >= 4.5, `ratio ${ratio(CANVAS_MUTED, body).toFixed(2)}`);
  });

  it("redraws a live run at most about ten times a second when nothing changes", () => {
    const theatre = new PipelineTheatre(fakeCanvas(fakeParent(1036, 666)), {});
    theatre.ingest([ev("run_started", null, { payload: { pipelines: 1, budget_usd: 1 } })]);
    const t0 = performance.now() + 60000;
    theatre.frame(t0); // the run start is applied; its hub message plays for 2.6 s
    for (let k = 1; k <= 200; k++) theatre.frame(t0 + k * 16);
    let clears = 0;
    theatre.ctx.clearRect = () => { clears += 1; };
    const calm = t0 + 3300;
    for (let k = 0; k < 63; k++) theatre.frame(calm + k * 16); // about one second
    assert.ok(clears >= 8 && clears <= 11, `a live idle run drew ${clears} times in a second`);
    theatre.destroy();
  });
});
