// Theatre model regressions: duplicates counted once per pipeline (R4-FE-02), the revision stations for every
// failure status the server writes (R4-WHOLE-06), and payload values coerced to text at the model boundary (R4-FE-09).
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { loadStatic, useFixtureContract } from "./_static.mjs";

const { TheatreModel, FAIL } = await loadStatic("viz/theatre_model.js");
await useFixtureContract();

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

describe("duplicates caught are counted once per pipeline (R4-FE-02)", () => {
  it("two pipelines that finished rejected_duplicate_code, with no duplicate_code event, count two", () => {
    const m = new TheatreModel();
    for (const pid of ["d1", "d2"]) {
      m.apply(ev("pipeline_started", pid, { payload: { theme: "momentum" } }));
      m.apply(ev("pipeline_finished", pid, { payload: { status: "rejected_duplicate_code" } }));
    }
    assert.equal(m.stats().duplicates, 2, "the headline agrees with the 'rejected duplicate' cards");
  });

  it("a pipeline with a duplicate_code event and a rejected_duplicate_code finish counts once", () => {
    const m = new TheatreModel();
    m.apply(ev("pipeline_started", "d3", { payload: { theme: "momentum" } }));
    m.apply(ev("duplicate_code", "d3", { payload: { earlier_attempt_id: "a1" } }));
    m.apply(ev("pipeline_finished", "d3", { payload: { status: "rejected_duplicate_code" } }));
    assert.equal(m.stats().duplicates, 1);
  });

  it("the same pipeline reported twice by duplicate_code counts once", () => {
    const m = new TheatreModel();
    m.apply(ev("pipeline_started", "d4", { payload: {} }));
    m.apply(ev("duplicate_code", "d4", { payload: {} }));
    m.apply(ev("duplicate_code", "d4", { payload: {} }));
    assert.equal(m.stats().duplicates, 1);
  });
});

describe("every failure status the server writes has a revision station (R4-WHOLE-06)", () => {
  it("a failed_backtest revision marks the train station red", () => {
    const m = new TheatreModel();
    m.apply(ev("pipeline_started", "r1", { payload: {} }));
    const arc = m.apply(ev("revision_requested", "r1", { payload: { reason: "failed_backtest" } }))
      .find((e) => e.type === "arc");
    assert.equal(arc.from, 5, "the revision arc starts at the train station");
    assert.equal(m.cards.get("r1").stations[5].state, FAIL, "the train station stays red");
  });

  it("a failed_critic_output revision marks the critic station red", () => {
    const m = new TheatreModel();
    m.apply(ev("pipeline_started", "r2", { payload: {} }));
    const effects = m.apply(ev("revision_requested", "r2", { payload: { reason: "failed_critic_output" } }));
    assert.equal(effects.find((e) => e.type === "arc").from, 6);
    assert.equal(m.cards.get("r2").stations[6].state, FAIL);
  });

  it("a finished failed_backtest pipeline marks the train station red", () => {
    const m = new TheatreModel();
    m.apply(ev("pipeline_started", "r3", { payload: {} }));
    m.apply(ev("pipeline_finished", "r3", { payload: { status: "failed_backtest" } }));
    assert.equal(m.cards.get("r3").stations[5].state, FAIL);
  });
});

describe("events the page does not draw leave the state alone (R4-WHOLE-06)", () => {
  it("llm_call_refused, llm_call_retrying and budget_overrun change no stat or card", () => {
    const m = new TheatreModel();
    m.apply(ev("run_started", null, { payload: { pipelines: 2, budget_usd: 1 } }));
    m.apply(ev("pipeline_started", "u1", { payload: { theme: "momentum" } }));
    const before = JSON.stringify({ stats: m.stats(), card: m.cards.get("u1") });
    for (const type of ["llm_call_refused", "llm_call_retrying", "budget_overrun", "not_a_real_type"]) {
      const effects = m.apply(ev(type, "u1", { payload: { spent_usd: 99, limit_usd: 1 } }));
      assert.deepEqual(effects.filter((e) => e.type !== "hubtext"), [], `${type} has no visual effect`);
    }
    assert.equal(JSON.stringify({ stats: m.stats(), card: m.cards.get("u1") }), before);
  });
});

describe("payload values are coerced to text at the model boundary (R4-FE-09)", () => {
  it("a numeric theme and status become text, so the canvas can call string methods on them", () => {
    const m = new TheatreModel();
    m.apply(ev("pipeline_started", "c1", { payload: { theme: 42 } }));
    assert.equal(m.cards.get("c1").theme, "42");
    m.apply(ev("pipeline_finished", "c1", { payload: { status: 7 } }));
    assert.equal(m.cards.get("c1").status, "7");
    assert.equal(typeof m.cards.get("c1").status, "string");
  });

  it("a null theme and status fall back to the empty theme and 'finished'", () => {
    const m = new TheatreModel();
    m.apply(ev("pipeline_started", "c2", { payload: { theme: null } }));
    assert.equal(m.cards.get("c2").theme, "");
    m.apply(ev("pipeline_finished", "c2", { payload: { status: null } }));
    assert.equal(m.cards.get("c2").status, "finished");
  });
});
