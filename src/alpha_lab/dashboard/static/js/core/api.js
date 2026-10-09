// Network helpers. The page never talks to SQLite, only to these endpoints.

export async function fetchJson(url) {
  let res;
  try {
    res = await fetch(url, { headers: { Accept: "application/json" } });
  } catch (err) {
    throw new Error(`${url} could not be reached (${err.message})`);
  }
  if (!res.ok) {
    const error = new Error(`${url} -> ${res.status}`);
    error.status = res.status;
    throw error;
  }
  return res.json();
}

// Guards a load that a newer request can supersede (a run switch, for example). Only the newest
// request may write to the page, so a slow earlier response cannot overwrite a newer run.
export class LatestRequest {
  constructor() { this.current = 0; }

  begin() {
    this.current += 1;
    return this.current;
  }

  isCurrent(token) { return token === this.current; }
}

// Streams new events for one run. Events arrive in batches, about four times a second, so the
// charts redraw a few times a second rather than once per event.
//
// There is one connection at a time. After an error the failed connection is closed, and a new one
// opens after a delay that doubles up to a cap. It resumes after the last sequence number seen, so
// nothing is skipped or delivered twice.
//
// The returned stop function closes the connection, cancels any pending retry, and discards events
// not yet delivered. After it returns, onBatch is never called again.
//
// onBatch(batch, { runId }) receives events in order. onStatus(state, { runId, delay? }) reports
// "connecting", "live", "reconnecting" (with the delay in ms) and "finished" (run_finished arrived).
export function connectStream(runId, onBatch, options = {}) {
  const { after = 0, flushMs = 250, minRetryMs = 1000, maxRetryMs = 30000, onStatus = () => {} } = options;
  const url = `/events/stream?run_id=${encodeURIComponent(runId)}`;
  let cursor = after;
  let pending = [];
  let flushTimer = null;
  let retryTimer = null;
  let source = null;
  let stopped = false;
  let finished = false;
  let failures = 0;

  const report = (state, detail = {}) => onStatus(state, { runId, ...detail });

  const flush = () => {
    flushTimer = null;
    if (stopped || pending.length === 0) return;
    const batch = pending;
    pending = [];
    onBatch(batch, { runId });
  };

  const closeSource = () => {
    if (source) {
      source.close();
      source = null;
    }
  };

  const open = () => {
    retryTimer = null;
    if (stopped || finished) return;
    report("connecting");
    const current = new EventSource(`${url}&after=${cursor}`);
    source = current;

    current.onopen = () => {
      if (source !== current) return;
      failures = 0;
      report("live");
    };

    current.onmessage = (msg) => {
      if (source !== current || stopped) return;
      let event;
      try {
        event = JSON.parse(msg.data);
      } catch {
        return; // a malformed event is skipped; the next one still arrives
      }
      if (!event || typeof event !== "object") return;
      cursor = Math.max(cursor, Number(event.seq) || 0);
      pending.push(event);
      if (flushTimer === null) flushTimer = setTimeout(flush, flushMs);
      if (event.type === "run_finished") {
        finished = true;
        closeSource();
        report("finished");
      }
    };

    current.onerror = () => {
      if (source !== current) return;
      closeSource();
      if (stopped || finished) return;
      failures += 1;
      const delay = Math.min(maxRetryMs, minRetryMs * 2 ** (failures - 1));
      report("reconnecting", { delay, failures });
      retryTimer = setTimeout(open, delay);
    };
  };

  const stop = () => {
    if (stopped) return;
    stopped = true;
    closeSource();
    clearTimeout(retryTimer);
    clearTimeout(flushTimer);
    retryTimer = null;
    flushTimer = null;
    pending = [];
  };

  open();
  return stop;
}
