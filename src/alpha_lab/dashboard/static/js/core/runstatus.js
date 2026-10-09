// The run status a page shows. It comes from the server (GET /api/runs): the status, whether the run is stale,
// and when it started. The page does not work out its own stale rule from event times (R4-F2). The event
// stream only says how the page's connection is doing.
import { fetchJson } from "./api.js";

// Local HH:MM for an epoch time. Used only to show a start time.
export function clockOf(ms) {
  const d = new Date(ms);
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}

// The latest /api/runs answer for every run.
export class RunStatuses {
  constructor() {
    this.list = [];
    this.byRun = new Map();
    this.answered = false; // true once the server has answered the run list (a failed request does not count)
  }

  async refresh() {
    const runs = await fetchJson("/api/runs");
    this.list = runs;
    this.byRun = new Map(runs.map((r) => [r.run_id, {
      status: String(r.status),
      stale: Boolean(r.stale),
      startedMs: Date.parse(r.started_at || "") || null,
    }]));
    this.answered = true;
    return runs;
  }

  // { status, stale, startedMs } for a run, or null before the server has answered for it.
  get(runId) {
    return this.byRun.get(runId) ?? null;
  }
}

// The line a page shows when no run is selected: connecting until the server has listed the runs, then no runs yet,
// or waiting for a run to be chosen. Once the server has answered, this never says connecting (R4-FE-04).
export function idleLabel(answered, runCount) {
  if (!answered) return { text: "connecting…", tone: "muted" };
  return runCount === 0 ? { text: "no runs yet", tone: "muted" } : { text: "waiting for a run", tone: "muted" };
}

// The one line a page shows for a run. info is RunStatuses.get(); connection is the stream's state
// ("connecting", "live", "reconnecting", "finished"); delay is the retry delay in ms. problem is the text
// of a ledger the server cannot read (ledgerprobe.js), which takes precedence: the status is unknown then.
// Order: the ledger problem, a final status the server reports, then its stale verdict (only for a run it still
// says is running), then the page's own connection, then running.
export function statusLabel(info, connection = "connecting", delay = 0, problem = null) {
  if (problem) return { text: "ledger unreadable", tone: "warn" };
  if (info && info.status !== "running") {
    return { text: info.status.replace(/_/g, " "), tone: info.status === "completed" ? "done" : "warn" };
  }
  if (info?.stale) {
    return { text: `stale · started ${info.startedMs ? clockOf(info.startedMs) : "unknown"}`, tone: "warn" };
  }
  if (connection === "reconnecting") return { text: `reconnecting in ${Math.round(delay / 1000)} s`, tone: "warn" };
  if (info) return { text: "live", tone: "live" };
  return { text: connection === "connecting" ? "connecting…" : "waiting for events", tone: "muted" };
}
