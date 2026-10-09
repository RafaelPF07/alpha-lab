// Network helpers. The page never talks to SQLite, only to these endpoints.

export async function fetchJson(url) {
  const res = await fetch(url, { headers: { Accept: "application/json" } });
  if (!res.ok) throw new Error(`${url} -> ${res.status}`);
  return res.json();
}

// Streams new events for a run. Events are delivered in batches, about four times a second,
// so the charts redraw a few times a second rather than once per event.
export function connectStream(runId, onBatch, { after = 0, flushMs = 250 } = {}) {
  let cursor = after;
  let pending = [];
  let timer = null;
  const source = new EventSource(`/events/stream?run_id=${encodeURIComponent(runId)}&after=${cursor}`);

  const flush = () => {
    timer = null;
    if (pending.length) {
      const batch = pending;
      pending = [];
      onBatch(batch);
    }
  };

  source.onmessage = (msg) => {
    const event = JSON.parse(msg.data);
    cursor = Math.max(cursor, event.seq);
    pending.push(event);
    if (timer === null) timer = setTimeout(flush, flushMs);
  };

  source.onerror = () => {
    // EventSource reconnects by itself. Resume from the last cursor seen, so nothing is missed.
    source.close();
    setTimeout(() => connectStream(runId, onBatch, { after: cursor, flushMs }), 2000);
  };

  return () => {
    source.close();
    if (timer !== null) clearTimeout(timer);
    flush();
  };
}
