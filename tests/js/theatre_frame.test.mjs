// The theatre's draw loop when one card fails to draw (R4-FE-09): the other cards still draw, and the frame
// does not throw. The canvas stand-in is the one the page tests use.
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { loadStatic, useFixtureContract } from "./_static.mjs";
import { fakeStageCanvas } from "./_page.mjs";

const { PipelineTheatre } = await loadStatic("viz/theatre.js");
await useFixtureContract();

let seq = 0;
const ev = (type, pid, payload = {}) => ({
  seq: ++seq,
  type,
  pipeline_id: pid,
  idea_id: null,
  agent: null,
  stage: "",
  attempt_number: null,
  timestamp: "2026-10-09T00:00:00.000+00:00",
  payload,
});

describe("PipelineTheatre keeps drawing when one card fails (R4-FE-09)", () => {
  it("a card whose draw throws is skipped, and the other cards are still drawn", () => {
    const theatre = new PipelineTheatre(fakeStageCanvas(), {});
    theatre.ingest([
      ev("pipeline_started", "a", { theme: "momentum" }),
      ev("pipeline_started", "b", { theme: "value" }),
      ev("pipeline_started", "c", { theme: "carry" }),
    ]);
    const drawn = [];
    const draw = theatre.drawCard.bind(theatre);
    theatre.drawCard = (card, v, rect, now) => {
      if (card.id === "b") throw new TypeError("a card with bad data");
      drawn.push(card.id);
      return draw(card, v, rect, now);
    };
    const logged = [];
    const originalError = console.error;
    console.error = (...args) => logged.push(args);
    try {
      assert.doesNotThrow(() => theatre.frame(performance.now() + 60000));
      theatre.dirty = true; // a second frame with nothing new is skipped, so ask for a redraw
      theatre.frame(performance.now() + 60000);
    } finally {
      console.error = originalError;
    }
    assert.deepEqual(drawn, ["a", "c", "a", "c"], "the card after the bad one still draws, on every frame");
    assert.equal(logged.length, 1, "the bad card is reported once, not on every frame");
    theatre.destroy();
  });
});
