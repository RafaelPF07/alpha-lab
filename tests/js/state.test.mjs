// Run state (static/js/core/state.js): the numbers every mission-control chart reads.
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { loadStatic } from "./_static.mjs";

const { RunState } = await loadStatic("core/state.js");

let seq = 0;
const ev = (type, { pid = null, idea = null, agent = null, payload = {}, run, ts } = {}) => ({
  seq: ++seq,
  type,
  pipeline_id: pid,
  idea_id: idea,
  agent,
  stage: "",
  attempt_number: 1,
  timestamp: ts ?? new Date(Date.UTC(2026, 9, 9, 0, 0, 0) + seq * 1000).toISOString(),
  payload,
  ...(run === undefined ? {} : { run_id: run }),
});

describe("RunState funnel", () => {
  it("counts distinct ideas per stage, not events (FE-05)", () => {
    const s = new RunState();
    s.ingest(ev("idea_created", { pid: "p1", idea: "a" }));
    s.ingest(ev("code_written", { pid: "p1", idea: "a" }));
    s.ingest(ev("static_check", { pid: "p1", idea: "a", payload: { ok: false, errors: ["x"] } }));
    s.ingest(ev("revision_requested", { pid: "p1", idea: "a", payload: { reason: "failed_static" } }));
    s.ingest(ev("code_written", { pid: "p1", idea: "a" }));
    s.ingest(ev("static_check", { pid: "p1", idea: "a", payload: { ok: true } }));
    s.ingest(ev("sandbox_run", { pid: "p1", idea: "a", payload: { status: "ok" } }));
    s.ingest(ev("lookahead_check", { pid: "p1", idea: "a", payload: { test: "perturbation", passed: true } }));
    s.ingest(ev("train_backtest", { pid: "p1", idea: "a", payload: { sharpe: 0.5 } }));
    s.ingest(ev("critic_review", { pid: "p1", idea: "a", payload: { suspicion_level: "low" } }));
    s.ingest(ev("validation_result", { pid: "p1", idea: "a", payload: { passed: true } }));
    s.ingest(ev("idea_created", { pid: "p2", idea: "b" }));
    s.ingest(ev("code_written", { pid: "p2", idea: "b" }));
    s.ingest(ev("static_check", { pid: "p2", idea: "b", payload: { ok: false, errors: ["y"] } }));
    const { funnel } = s.snapshot();
    assert.deepEqual(funnel, {
      generated: 2, coded: 2, static: 1, sandbox: 1, lookahead: 1, train: 1, critic: 1, validation: 1,
    });
  });

  it("clamps every stage to the stage before it (FE-05)", () => {
    const s = new RunState();
    s.ingest(ev("static_check", { pid: "p9", idea: "x", payload: { ok: true } }));
    s.ingest(ev("sandbox_run", { pid: "p9", idea: "x", payload: { status: "ok" } }));
    const { funnel } = s.snapshot();
    assert.equal(funnel.coded, 0);
    assert.equal(funnel.static, 0, "a stage cannot exceed the stage before it");
    assert.equal(funnel.sandbox, 0);
  });

  it("maps events without idea_id to their pipeline's idea", () => {
    const s = new RunState();
    s.ingest(ev("idea_created", { pid: "p1", idea: "a" }));
    s.ingest(ev("code_written", { pid: "p1" }));
    s.ingest(ev("static_check", { pid: "p1", payload: { ok: true } }));
    const { funnel } = s.snapshot();
    assert.equal(funnel.coded, 1);
    assert.equal(funnel.static, 1);
  });

  it("counts one generated idea once even if its creation is repeated", () => {
    const s = new RunState();
    s.ingest(ev("idea_created", { pid: "p1", idea: "a" }));
    s.ingest(ev("idea_created", { pid: "p1", idea: "a" }));
    assert.equal(s.snapshot().ideas, 1);
    assert.equal(s.snapshot().funnel.generated, 1);
  });
});

describe("RunState ticker, runs and money", () => {
  it("keeps the most recent 5000 events, so a filter can reach older ones (FE-07)", () => {
    const s = new RunState();
    const oldest = ev("validation_result", { pid: "old", payload: { passed: true } });
    s.ingest(oldest);
    for (let i = 0; i < 6000; i++) s.ingest(ev("llm_call_started", { pid: "p" }));
    const { ticker } = s.snapshot();
    assert.equal(ticker.length, 5000);
    assert.equal(ticker.at(-1).seq - ticker[0].seq, 4999, "the kept events are consecutive and the newest");
    assert.ok(ticker[0].seq > oldest.seq, "the oldest event is no longer kept");
  });

  it("drops events that belong to another run (run_id check)", () => {
    const s = new RunState("run-a");
    s.ingest(ev("idea_created", { pid: "p1", idea: "a", run: "run-b" }));
    s.ingest(ev("idea_created", { pid: "p2", idea: "b", run: "run-a" }));
    s.ingest(ev("idea_created", { pid: "p3", idea: "c" }));
    assert.equal(s.snapshot().ideas, 2);
  });

  it("reports the run status from run_finished, so failed runs are not shown as completed", () => {
    const s = new RunState();
    s.ingest(ev("run_finished", { payload: { status: "failed" } }));
    const snap = s.snapshot();
    assert.equal(snap.finished, true);
    assert.equal(snap.runStatus, "failed");
  });

  it("exposes the first and last timestamps of the run", () => {
    const s = new RunState();
    const first = ev("run_started", { payload: { budget_usd: 1 } });
    const last = ev("run_finished", { payload: { status: "completed" } });
    s.ingest(first);
    s.ingest(last);
    const snap = s.snapshot();
    assert.equal(snap.runStart, first.timestamp);
    assert.equal(snap.runEnd, last.timestamp);
  });

  it("keeps the budget unknown until a run or budget event states it (FE-15)", () => {
    const s = new RunState();
    s.ingest(ev("llm_call_finished", { agent: "ideas", pid: "p1", payload: { cost_usd: 0.01 } }));
    assert.equal(s.snapshot().budget, null);
    assert.equal(s.snapshot().projectedExhaustion, null);
    s.ingest(ev("run_started", { payload: { budget_usd: 2 } }));
    assert.equal(s.snapshot().budget, 2);
  });

  it("projects no budget exhaustion for a finished run, which spends nothing more", () => {
    const s = new RunState();
    s.ingest(ev("run_started", { payload: { budget_usd: 1 }, ts: "2026-10-09T00:00:00.000+00:00" }));
    s.ingest(ev("llm_call_finished", { agent: "ideas", pid: "p1", payload: { cost_usd: 0.1 },
      ts: "2026-10-09T00:10:00.000+00:00" }));
    assert.notEqual(s.snapshot().projectedExhaustion, null, "a running run is projected");
    s.ingest(ev("run_finished", { payload: { status: "completed" }, ts: "2026-10-09T00:10:00.000+00:00" }));
    assert.equal(s.snapshot().projectedExhaustion, null);
  });

  it("does not let a non-numeric cost poison the spend total", () => {
    const s = new RunState();
    s.ingest(ev("llm_call_finished", { agent: "coder", pid: "p1", payload: { cost_usd: "abc" } }));
    s.ingest(ev("llm_call_finished", { agent: "coder", pid: "p1", payload: { cost_usd: 0.25 } }));
    assert.equal(s.snapshot().spent, 0.25);
  });

  it("remembers each pipeline's idea so a click can open it (pipeline-select)", () => {
    const s = new RunState();
    s.ingest(ev("pipeline_started", { pid: "p1", payload: { theme: "momentum" } }));
    s.ingest(ev("idea_created", { pid: "p1", idea: "a", payload: { name: "Twelve month winners" } }));
    assert.equal(s.pipelines.get("p1").ideaId, "a");
  });
});
