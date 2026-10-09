// The ledger problem a page shows (R4-FE-01): the server's reason reaches the page, the fix is named, and an
// unreadable ledger is never shown as an empty run. Also the idle wording once the server has answered (R4-FE-04).
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { loadStatic } from "./_static.mjs";

const { fetchJson } = await loadStatic("core/api.js");
const { ledgerProblem, checkLedger, loadFailure, MIGRATE_HINT } = await loadStatic("core/ledgerprobe.js");
const { statusLabel, idleLabel, RunStatuses } = await loadStatic("core/runstatus.js");

const REASON = "ledger is unreadable: no such column: seq";

function reply(status, body) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => {
      if (body === undefined) throw new SyntaxError("not JSON");
      return body;
    },
  };
}

async function withFetch(handler, fn) {
  const original = globalThis.fetch;
  globalThis.fetch = async (url) => handler(url);
  try {
    return await fn();
  } finally {
    globalThis.fetch = original;
  }
}

describe("fetchJson keeps the server's reason (R4-FE-01)", () => {
  it("puts the 503 detail in the message and on the error", async () => {
    await withFetch(() => reply(503, { detail: REASON }), async () => {
      await assert.rejects(fetchJson("/api/runs/x/events"), (err) => {
        assert.equal(err.status, 503);
        assert.equal(err.detail, REASON);
        assert.match(err.message, /\/api\/runs\/x\/events -> 503: ledger is unreadable: no such column: seq/);
        return true;
      });
    });
  });

  it("still names the URL and status when the error body is not JSON", async () => {
    await withFetch(() => reply(500, undefined), async () => {
      await assert.rejects(fetchJson("/api/x"), (err) => {
        assert.equal(err.message, "/api/x -> 500");
        assert.equal(err.detail, "");
        return true;
      });
    });
  });
});

describe("ledgerProblem and loadFailure name the fix (R4-FE-01)", () => {
  const error = Object.assign(new Error(`/api -> 503: ${REASON}`), { status: 503, detail: REASON });

  it("gives the server's reason and the command that migrates the ledger", () => {
    const text = ledgerProblem(error);
    assert.match(text, /no such column: seq/);
    assert.match(text, /uv run lab ledger show/);
    assert.ok(text.includes(MIGRATE_HINT));
  });

  it("is null for errors that are not a 503 from the ledger", () => {
    assert.equal(ledgerProblem(Object.assign(new Error("x"), { status: 404 })), null);
    assert.equal(ledgerProblem(new Error("could not be reached")), null);
    assert.equal(ledgerProblem(null), null);
  });

  it("loadFailure says what failed, then the reason and the fix", () => {
    const text = loadFailure("Could not load run 637b469e9b82", error);
    assert.match(text, /^Could not load run 637b469e9b82\. /);
    assert.match(text, /no such column: seq/);
    assert.match(text, /lab ledger show/);
    assert.equal(loadFailure("Could not load the run list", new Error("boom")), "Could not load the run list. boom");
  });
});

describe("checkLedger asks the server whether the events can be read (R4-FE-01)", () => {
  it("returns the problem when the events request answers 503", async () => {
    const asked = [];
    const get = async (url) => {
      asked.push(url);
      throw Object.assign(new Error(`${url} -> 503`), { status: 503, detail: REASON });
    };
    const problem = await checkLedger("637b469e9b82", get);
    assert.deepEqual(asked, ["/api/runs/637b469e9b82/events?after=0&limit=1"]);
    assert.match(problem, /no such column: seq/);
  });

  it("is null when the events can be read, and when the server cannot be reached", async () => {
    assert.equal(await checkLedger("r", async () => []), null);
    assert.equal(await checkLedger("r", async () => { throw new Error("could not be reached"); }), null);
  });
});

describe("the pill says the ledger is unreadable, not 'completed' over no data (R4-FE-01)", () => {
  it("statusLabel shows the problem before the run status", () => {
    const info = { status: "completed", stale: false, startedMs: null };
    assert.deepEqual(statusLabel(info, "live", 0, "ledger problem"), { text: "ledger unreadable", tone: "warn" });
    assert.equal(statusLabel(info, "live", 0, null).text, "completed");
  });
});

describe("the idle label does not say 'connecting' once the server has answered (R4-FE-04)", () => {
  it("says connecting until the list has been read, then no runs yet, then waiting for a run", () => {
    assert.equal(idleLabel(false, 0).text, "connecting…");
    assert.equal(idleLabel(true, 0).text, "no runs yet");
    assert.equal(idleLabel(true, 3).text, "waiting for a run");
  });

  it("RunStatuses records whether the server has answered, and a failed list is not an answer", async () => {
    const runs = new RunStatuses();
    assert.equal(runs.answered, false);
    await withFetch(() => reply(500, undefined), async () => {
      await assert.rejects(runs.refresh());
    });
    assert.equal(runs.answered, false, "a failed request is not an answer");
    await withFetch(() => reply(200, []), async () => {
      await runs.refresh();
    });
    assert.equal(runs.answered, true);
    assert.equal(runs.list.length, 0);
  });
});
