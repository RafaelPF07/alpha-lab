// Pipeline theatre, canvas half (static/js/viz/theatre.js), checked with a fake canvas: text fitting,
// resizing, opening a card by click (touch has no hover), and the reduced-motion setting.
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { loadStatic, media } from "./_static.mjs";

const { PipelineTheatre, fitText } = await loadStatic("viz/theatre.js");

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
