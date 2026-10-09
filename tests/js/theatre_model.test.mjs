import assert from "node:assert/strict";
import { loadStatic } from "./_static.mjs";

const { TheatreModel, IDLE, OK, FAIL, ACTIVE, describe: describeEvent } = await loadStatic("viz/theatre_model.js");

let seq = 0;
const ev = (type, pid, extra = {}) => ({
  seq: ++seq,
  type,
  pipeline_id: pid,
  idea_id: extra.idea_id ?? null,
  agent: extra.agent ?? null,
  stage: extra.stage ?? "",
  attempt_number: extra.attempt_number ?? null,
  timestamp: "2026-10-09T00:00:00.000+00:00",
  payload: extra.payload ?? {},
});

function passingPipeline(m, pid) {
  m.apply(ev("pipeline_started", pid, { payload: { theme: "momentum" } }));
  m.apply(ev("llm_call_started", pid, { agent: "ideas" }));
  m.apply(ev("llm_call_finished", pid, { agent: "ideas", payload: { status: "ok", cost_usd: 0.001, input_tokens: 900, output_tokens: 200 } }));
  m.apply(ev("idea_created", pid, { idea_id: "idea-1", payload: { name: "Twelve month winners", theme: "momentum" } }));
  m.apply(ev("llm_call_started", pid, { agent: "coder" }));
  m.apply(ev("llm_call_finished", pid, { agent: "coder", payload: { status: "ok", cost_usd: 0.002, input_tokens: 3000, output_tokens: 900 } }));
  m.apply(ev("code_written", pid, { agent: "coder" }));
  m.apply(ev("static_check", pid, { payload: { ok: true, errors: [] } }));
  m.apply(ev("sandbox_run", pid, { payload: { status: "ok" } }));
  m.apply(ev("lookahead_check", pid, { payload: { test: "truncation", passed: true } }));
  m.apply(ev("lookahead_check", pid, { payload: { test: "perturbation", passed: true } }));
  m.apply(ev("train_backtest", pid, { payload: { sharpe: 0.84 } }));
  m.apply(ev("llm_call_started", pid, { agent: "critic" }));
  m.apply(ev("llm_call_finished", pid, { agent: "critic", payload: { status: "ok", cost_usd: 0.007, input_tokens: 3000, output_tokens: 500 } }));
  m.apply(ev("critic_review", pid, { payload: { suspicion_level: "low" } }));
  m.apply(ev("validation_result", pid, { payload: { passed: true } }));
  m.apply(ev("pipeline_finished", pid, { payload: { status: "validation_pass" } }));
}

// 1. A full passing pipeline.
{
  const m = new TheatreModel();
  m.apply(ev("run_started", null, { payload: { pipelines: 2, budget_usd: 1 } }));
  passingPipeline(m, "p1");
  const card = m.cards.get("p1");
  assert.equal(card.status, "validation_pass");
  assert.deepEqual(card.stations.map((s) => s.state), [OK, OK, OK, OK, OK, OK, OK, OK]);
  assert.equal(card.idea, "Twelve month winners");
  assert.equal(card.ideaId, "idea-1");
  assert.equal(card.stations[5].text, "SR 0.84");
  assert.ok(Math.abs(card.cost - 0.010) < 1e-9, `cost was ${card.cost}`);
  const s = m.stats();
  assert.equal(s.passed, 1);
  assert.equal(s.done, 1);
  assert.equal(s.running, 0);
  assert.equal(s.total, 2);
  console.log("ok: passing pipeline");
}

// 2. A static check fails, the coder is asked to revise, and the second attempt passes.
{
  const m = new TheatreModel();
  const pid = "p2";
  m.apply(ev("pipeline_started", pid, { payload: { theme: "volatility" } }));
  m.apply(ev("idea_created", pid, { idea_id: "idea-2", payload: { name: "Vol carry" } }));
  m.apply(ev("code_written", pid, { agent: "coder" }));
  const failing = m.apply(ev("static_check", pid, { payload: { ok: false, errors: ["import of 'os' is not allowed"] } }));
  assert.equal(m.cards.get(pid).stations[2].state, FAIL);
  assert.ok(failing.some((e) => e.type === "packet" && e.from === 1 && e.to === 2), "packet should travel from code to static");
  const effects = m.apply(ev("revision_requested", pid, { payload: { reason: "failed_static" } }));
  const card = m.cards.get(pid);
  assert.equal(card.attempt, 2);
  assert.equal(card.stations[1].state, ACTIVE);
  assert.equal(card.stations[2].state, FAIL, "the failing station stays red until it is re-run");
  assert.equal(card.stations[3].state, IDLE, "later stations reset for the new attempt");
  assert.ok(effects.some((e) => e.type === "arc" && e.from === 2 && e.to === 1));
  console.log("ok: static failure and revision");
}

// 3. The training gate fails. The failure is shown on the train station when the revision is requested.
{
  const m = new TheatreModel();
  const pid = "p3";
  m.apply(ev("pipeline_started", pid, { payload: { theme: "trend filters" } }));
  m.apply(ev("static_check", pid, { payload: { ok: true, errors: [] } }));
  m.apply(ev("sandbox_run", pid, { payload: { status: "ok" } }));
  m.apply(ev("lookahead_check", pid, { payload: { test: "truncation", passed: true } }));
  m.apply(ev("lookahead_check", pid, { payload: { test: "perturbation", passed: true } }));
  m.apply(ev("train_backtest", pid, { payload: { sharpe: -0.4 } }));
  m.apply(ev("revision_requested", pid, { payload: { reason: "failed_gate" } }));
  assert.equal(m.cards.get(pid).stations[5].state, FAIL);
  m.apply(ev("pipeline_finished", pid, { payload: { status: "failed_gate" } }));
  assert.equal(m.stats().passed, 0);
  assert.equal(m.stats().done, 1);
  console.log("ok: training gate failure");
}

// 4. A duplicate code rejection is recognised from the static check message and counted.
{
  const m = new TheatreModel();
  const pid = "p4";
  m.apply(ev("pipeline_started", pid, { payload: { theme: "sector rotation" } }));
  m.apply(ev("static_check", pid, { payload: { ok: false, errors: ["Duplicate code: this is the same program"] } }));
  assert.equal(m.cards.get(pid).stations[2].text, "dup");
  m.apply(ev("idea_rejected_duplicate", pid, { payload: { name: "copy" } }));
  assert.equal(m.stats().duplicates, 1);
  console.log("ok: duplicate detection");
}

// 5. A backlog replayed silently produces the same state with no effects.
{
  const silent = new TheatreModel();
  let effects = 0;
  seq = 100;
  const fake = [];
  const record = (type, pid, extra) => fake.push(ev(type, pid, extra));
  record("pipeline_started", "p5", { payload: { theme: "momentum" } });
  record("validation_result", "p5", { payload: { passed: true } });
  record("pipeline_finished", "p5", { payload: { status: "validation_pass" } });
  for (const e of fake) effects += silent.apply(e, false).length;
  assert.equal(effects, 0);
  assert.equal(silent.stats().passed, 1);
  console.log("ok: silent backlog");
}

// 6. Revision after a failed critic moves the loop back to the coder and keeps the critic failure visible.
{
  const m = new TheatreModel();
  const pid = "p6";
  m.apply(ev("pipeline_started", pid, { payload: { theme: "pairs and relative value" } }));
  m.apply(ev("train_backtest", pid, { payload: { sharpe: 0.9 } }));
  m.apply(ev("critic_review", pid, { payload: { suspicion_level: "high" } }));
  assert.equal(m.cards.get(pid).stations[6].state, FAIL);
  m.apply(ev("revision_requested", pid, { payload: { reason: "failed_critic" } }));
  assert.equal(m.cards.get(pid).stations[6].state, FAIL);
  assert.equal(m.cards.get(pid).stations[5].state, OK);
  console.log("ok: critic failure revision");
}

// 7. The feed text never shows undefined or NaN, whatever a payload or an agent field contains (QA).
{
  const payloads = [{}, { status: null, ok: "x", cost_usd: "abc", input_tokens: "y", sharpe: "NaN", errors: null,
    spent_usd: "abc", name: undefined, suspicion_level: undefined, test: undefined, reason: undefined }];
  const types = ["run_started", "run_finished", "pipeline_started", "idea_created", "llm_call_started",
    "llm_call_finished", "code_written", "static_check", "sandbox_run", "lookahead_check", "train_backtest",
    "critic_review", "validation_result", "revision_requested", "pipeline_finished", "budget_update",
    "idea_rejected_duplicate", "not_a_real_type"];
  for (const type of types) {
    for (const payload of payloads) {
      for (const agent of [null, undefined]) {
        const text = describeEvent({ seq: 1, type, agent, pipeline_id: "p", idea_id: null, payload,
          timestamp: "2026-10-09T00:00:00.000+00:00" });
        assert.doesNotMatch(text, /undefined|NaN/, `${type}: "${text}"`);
      }
    }
  }
  console.log("ok: feed text is never undefined or NaN");
}

// 8. Non-numeric payload values never reach the theatre's numbers or its float text.
{
  const m = new TheatreModel();
  m.apply(ev("run_started", null, { payload: { budget_usd: "lots" } }));
  assert.equal(m.stats().budget, null, "a non-numeric budget is treated as unknown");
  m.apply(ev("pipeline_started", "p8", { payload: {} }));
  const effects = m.apply(ev("llm_call_finished", "p8", { agent: "coder",
    payload: { status: "ok", cost_usd: "abc", input_tokens: 10, output_tokens: 5 } }));
  for (const e of effects.filter((x) => x.type === "float")) {
    assert.doesNotMatch(e.text, /NaN|undefined/, `float text was "${e.text}"`);
  }
  m.apply(ev("budget_update", null, { payload: { spent_usd: "abc" } }));
  assert.equal(m.stats().spent, 0, "a non-numeric spend does not poison the total");
  m.apply(ev("train_backtest", "p8", { payload: { sharpe: "NaN" } }));
  assert.equal(m.cards.get("p8").stations[5].text, "SR n/a");
  console.log("ok: non-numeric payloads");
}

console.log("all theatre model checks passed");
