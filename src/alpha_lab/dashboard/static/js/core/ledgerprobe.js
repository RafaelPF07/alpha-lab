// What a page says when the ledger cannot be read (R4-FE-01). The server answers 503 with the reason in `detail`,
// for example a ledger written before the events table had a seq column. The page shows that reason and the
// command that fixes it, rather than an empty run with no error.
import { fetchJson } from "./api.js";

export const MIGRATE_HINT = "If the reason names a missing column, the ledger was written by an older version. "
  + "Run `uv run lab ledger show` once (it opens the ledger for writing and migrates it), then reload this page.";

// The message for a 503 from the ledger, or null for any other error.
export function ledgerProblem(error) {
  if (!error || error.status !== 503) return null;
  const reason = String(error.detail || "").replace(/\.$/, "");
  const sentence = reason ? `${reason.charAt(0).toUpperCase()}${reason.slice(1)}` : "The ledger cannot be read";
  return `${sentence}. ${MIGRATE_HINT}`;
}

// Asks the server whether a run's events can be read. The stream cannot say so: an unreadable ledger sends only a
// comment line, and the browser does not pass comments on. So the page makes one normal request. Returns the
// problem text, or null when the events can be read or the server cannot be reached.
export async function checkLedger(runId, get = fetchJson) {
  try {
    await get(`/api/runs/${encodeURIComponent(runId)}/events?after=0&limit=1`);
    return null;
  } catch (error) {
    return ledgerProblem(error);
  }
}

// The line a page shows when a load fails: what failed, then the reason (and the fix, when the ledger is the cause).
export function loadFailure(what, error) {
  return `${what}. ${ledgerProblem(error) ?? error.message}`;
}
