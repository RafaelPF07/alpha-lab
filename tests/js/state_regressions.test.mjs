// Mission-control state regressions: duplicates are counted the same way the theatre counts them (R4-WHOLE-06).
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { loadStatic, useFixtureContract } from "./_static.mjs";

const { RunState } = await loadStatic("core/state.js");
await useFixtureContract();

let seq = 0;
const ev = (type, pid, payload = {}) => ({
  seq: ++seq,
  type,
  pipeline_id: pid,
  idea_id: null,
  agent: null,
  stage: "",
  attempt_number: 1,
  timestamp: new Date(Date.UTC(2026, 9, 9, 0, 0, 0) + seq * 1000).toISOString(),
  payload,
});

describe("RunState counts duplicates as the theatre does (R4-WHOLE-06)", () => {
  it("counts a duplicate_code event once per pipeline", () => {
    const s = new RunState();
    s.ingest(ev("pipeline_started", "p1", { theme: "momentum" }));
    s.ingest(ev("duplicate_code", "p1", {}));
    s.ingest(ev("duplicate_code", "p1", {}));
    assert.equal(s.snapshot().duplicates, 1);
  });

  it("counts a pipeline that finished rejected_duplicate_code, and not twice when it also had a duplicate_code event", () => {
    const s = new RunState();
    s.ingest(ev("pipeline_started", "p2", {}));
    s.ingest(ev("duplicate_code", "p2", {}));
    s.ingest(ev("pipeline_finished", "p2", { status: "rejected_duplicate_code" }));
    s.ingest(ev("pipeline_started", "p3", {}));
    s.ingest(ev("pipeline_finished", "p3", { status: "rejected_duplicate_code" }));
    assert.equal(s.snapshot().duplicates, 2);
  });

  it("an unknown event type leaves the duplicate count alone", () => {
    const s = new RunState();
    s.ingest(ev("llm_call_retrying", "p4", {}));
    assert.equal(s.snapshot().duplicates, 0);
  });
});
