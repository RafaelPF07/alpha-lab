// Event stream client (static/js/core/api.js). Uses a fake EventSource and mocked timers, so the
// reconnect and stop behaviour can be checked step by step without a server.
import { describe, it, beforeEach, afterEach, mock } from "node:test";
import assert from "node:assert/strict";
import { loadStatic } from "./_static.mjs";

const { connectStream } = await loadStatic("core/api.js");

class FakeEventSource {
  static instances = [];

  constructor(url) {
    this.url = url;
    this.closed = false;
    FakeEventSource.instances.push(this);
  }

  close() { this.closed = true; }

  open() { this.onopen?.({}); }

  message(data) { this.onmessage?.({ data: typeof data === "string" ? data : JSON.stringify(data) }); }

  fail() { this.onerror?.({}); }
}

const ev = (seq, type = "idea_created") => ({ seq, type, timestamp: "2026-10-09T00:00:00+00:00", payload: {} });
const sources = () => FakeEventSource.instances;
const last = () => sources().at(-1);

let batches;
const seqs = () => batches.flatMap((b) => b.batch.map((e) => e.seq));
const onBatch = (batch, meta) => batches.push({ batch, meta });

beforeEach(() => {
  FakeEventSource.instances = [];
  globalThis.EventSource = FakeEventSource;
  batches = [];
  mock.timers.enable({ apis: ["setTimeout"] });
});

afterEach(() => {
  mock.timers.reset();
  delete globalThis.EventSource;
});

describe("connectStream", () => {
  it("opens one stream, delivers each event once and tags the batch with its run", () => {
    const stop = connectStream("run-a", onBatch, { flushMs: 10 });
    assert.equal(sources().length, 1);
    assert.match(sources()[0].url, /run_id=run-a/);
    assert.match(sources()[0].url, /after=0/);
    last().message(ev(1));
    last().message(ev(2));
    mock.timers.tick(10);
    assert.equal(batches.length, 1);
    assert.deepEqual(seqs(), [1, 2]);
    assert.equal(batches[0].meta.runId, "run-a");
    stop();
  });

  it("after an error keeps one open stream and resumes after the last sequence number", () => {
    const stop = connectStream("run-a", onBatch, { flushMs: 10, minRetryMs: 1000 });
    last().message(ev(1));
    last().message(ev(2));
    mock.timers.tick(10);
    last().fail();
    assert.equal(sources()[0].closed, true, "the failed source is closed");
    mock.timers.tick(999);
    assert.equal(sources().length, 1, "no reconnect before the retry delay");
    mock.timers.tick(1);
    assert.equal(sources().length, 2);
    assert.match(sources()[1].url, /after=2/);
    sources()[1].message(ev(3));
    mock.timers.tick(10);
    assert.deepEqual(seqs(), [1, 2, 3], "no duplicates and no gaps");
    assert.equal(sources().filter((s) => !s.closed).length, 1, "exactly one stream is open");
    stop();
  });

  // The server closes a stale or finished run's stream at once. Each open used to reset the failure count,
  // so the page reopened it every second. The delay must now grow, as it does for any other failure.
  it("a stream the server closes at once is retried with a growing delay, not every second", () => {
    const stop = connectStream("run-stale", onBatch, { minRetryMs: 1000, maxRetryMs: 30000 });
    const gaps = [];
    for (let i = 0; i < 4; i++) {
      last().open();
      last().fail();
      const before = sources().length;
      let waited = 0;
      while (sources().length === before && waited < 60000) {
        mock.timers.tick(100);
        waited += 100;
      }
      gaps.push(waited);
    }
    assert.deepEqual(gaps, [1000, 2000, 4000, 8000]);
    stop();
  });

  it("a stream that stays open a while resets the delay for the next failure", () => {
    const stop = connectStream("run-a", onBatch, { minRetryMs: 1000, maxRetryMs: 30000 });
    last().fail();
    mock.timers.tick(1000);
    last().open();
    mock.timers.tick(5000);
    last().fail();
    const before = sources().length;
    mock.timers.tick(999);
    assert.equal(sources().length, before, "the delay is back to the first step");
    mock.timers.tick(1);
    assert.equal(sources().length, before + 1);
    stop();
  });

  it("stop during the retry window never opens another stream (FE-01)", () => {
    const stop = connectStream("run-a", onBatch, { minRetryMs: 1000 });
    last().fail();
    stop();
    mock.timers.tick(60000);
    assert.equal(sources().length, 1, "no stream is opened after stop");
  });

  it("stop closes the open stream, discards pending events and ignores later messages", () => {
    const stop = connectStream("run-a", onBatch, { flushMs: 250 });
    const first = last();
    first.message(ev(1));
    stop();
    assert.equal(first.closed, true);
    first.message(ev(2));
    mock.timers.tick(1000);
    assert.deepEqual(seqs(), [], "pending events from the stopped run are not delivered");
  });

  it("several errors and then stop leave no open stream and no timers that reopen one", () => {
    const stop = connectStream("run-a", onBatch, { minRetryMs: 1000 });
    last().fail();
    mock.timers.tick(1000);
    last().fail();
    mock.timers.tick(2000);
    last().fail();
    stop();
    mock.timers.tick(100000);
    assert.equal(sources().length, 3);
    assert.ok(sources().every((s) => s.closed));
  });

  it("doubles the retry delay after each failure and stops at the cap", () => {
    const stop = connectStream("run-a", onBatch, { minRetryMs: 1000, maxRetryMs: 4000 });
    for (const delay of [1000, 2000, 4000, 4000]) {
      const before = sources().length;
      last().fail();
      mock.timers.tick(delay - 1);
      assert.equal(sources().length, before, `no reconnect before ${delay} ms`);
      mock.timers.tick(1);
      assert.equal(sources().length, before + 1, `reconnect at ${delay} ms`);
    }
    stop();
  });

  it("reports connecting, live and reconnecting through onStatus", () => {
    const states = [];
    const stop = connectStream("run-a", onBatch, {
      minRetryMs: 1000,
      onStatus: (state, detail) => states.push([state, detail?.delay]),
    });
    last().open();
    last().fail();
    mock.timers.tick(1000);
    last().open();
    stop();
    assert.deepEqual(states.map((s) => s[0]), ["connecting", "live", "reconnecting", "connecting", "live"]);
    assert.equal(states[2][1], 1000);
  });

  it("closes the stream on run_finished, delivers the batch and never reconnects", () => {
    const stop = connectStream("run-a", onBatch, { flushMs: 10 });
    last().message(ev(1));
    last().message(ev(2, "run_finished"));
    mock.timers.tick(10);
    assert.equal(last().closed, true);
    assert.deepEqual(seqs(), [1, 2]);
    sources()[0].fail();
    mock.timers.tick(60000);
    assert.equal(sources().length, 1, "a finished run is not reopened");
    stop();
  });

  it("skips malformed event data and keeps streaming", () => {
    const stop = connectStream("run-a", onBatch, { flushMs: 10 });
    last().message("not json{");
    last().message('{"seq": 1, "type": "llm_call_finished", "payload": {"cost_usd": NaN}}');
    last().message(ev(2));
    mock.timers.tick(10);
    assert.deepEqual(seqs(), [2]);
    stop();
  });

  it("can be stopped twice", () => {
    const stop = connectStream("run-a", onBatch);
    stop();
    stop();
    assert.equal(sources().length, 1);
  });
});
