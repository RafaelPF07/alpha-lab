// Mission control: live view of a run (or the demo run). Every chart reads the same aggregated state.
import { connectStream } from "../core/api.js";
import { loadContract, contract } from "../core/contract.js";
import { RunStatuses, idleLabel, statusLabel } from "../core/runstatus.js";
import { RunState } from "../core/state.js";
import { isDemoRun } from "../core/runs.js";
import { showError, clearError, requireCharts } from "../core/page.js";
import { checkLedger, loadFailure } from "../core/ledgerprobe.js";
import * as counters from "../viz/counters.js";
import * as funnel from "../viz/funnel.js";
import * as swimlanes from "../viz/swimlanes.js";
import * as gauge from "../viz/budget_gauge.js";
import * as spend from "../viz/budget_spend.js";
import * as stack from "../viz/budget_stack.js";
import * as context from "../viz/context.js";
import * as ticker from "../viz/ticker.js";
import * as swarm from "../viz/swarm.js";

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);

let state = new RunState();
let stop = null;
let scheduled = false;
let currentRun = null;
let connection = { state: "connecting", delay: 0 };
let problem = null; // the reason the server gives when it cannot read the current run's events (R4-FE-01)
const runs = new RunStatuses(); // each run's status as the server reports it
const NO_RUNS_HINT = "no runs yet. Start one with: uv run lab run --pipelines 5 --budget 0.50";

// The status line shows the run's status as the server reports it. While the server says the run is live,
// the line also counts the running pipelines from the events.
function updateStatusLine() {
  const el = $("run-status");
  // With no run selected, the line says whether the server has listed any runs yet (R4-FE-04).
  if (!currentRun) {
    const idle = idleLabel(runs.answered, runs.list.length).text;
    el.textContent = runs.answered && runs.list.length === 0 ? NO_RUNS_HINT : idle;
    return;
  }
  const label = statusLabel(runs.get(currentRun), connection.state, connection.delay, problem);
  el.textContent = label.text === "live" ? `live · ${state.runningPipelines} running` : label.text;
}

async function refreshRuns() {
  try {
    await runs.refresh();
    renderRunSelect();
  } catch {
    // Keep the last answer. The status line still shows the stream's state.
  }
  updateStatusLine();
}

// Asks the server whether the current run's events can be read. A stream cannot say so, so the page must ask (R4-FE-01).
async function checkRun(runId) {
  const found = await checkLedger(runId);
  if (runId !== currentRun) return;
  if (found) {
    problem = found;
    showError(found);
  } else if (problem) {
    if (document.getElementById("page-error")?.textContent === problem) clearError();
    problem = null;
  }
  updateStatusLine();
}

function runOptionText(runId) {
  const info = runs.get(runId);
  return `${runId} · ${info?.status ?? "unknown"}${info?.stale && info.status === "running" ? " · stale" : ""}`;
}

// The run select is rebuilt from the server's list, keeping the run on screen selected.
function renderRunSelect() {
  const select = $("run-select");
  select.replaceChildren(...runs.list.map((r) => {
    const option = document.createElement("option");
    option.value = r.run_id;
    option.textContent = runOptionText(r.run_id);
    return option;
  }));
  if (currentRun && runs.get(currentRun)) select.value = currentRun;
}

function render() {
  scheduled = false;
  const snap = state.snapshot();
  counters.update(snap);
  funnel.update(snap);
  swimlanes.update(snap);
  gauge.update(snap);
  spend.update(snap);
  stack.update(snap);
  context.update(snap);
  const info = ticker.update(snap.ticker);
  $("ticker-note").textContent = info.matches === info.total
    ? `showing the newest ${info.shown} of the last ${info.total} events`
    : `${info.matches} of the last ${info.total} events match`;
  swarm.update(snap);
  updateStatusLine();
}

function scheduleRender() {
  if (!scheduled) {
    scheduled = true;
    requestAnimationFrame(render);
  }
}

function attach(runId) {
  if (stop) stop();
  currentRun = runId;
  problem = null;
  state = new RunState(runId);
  connection = { state: "connecting", delay: 0 };
  $("demo-badge").hidden = !isDemoRun(runId);
  stop = connectStream(runId, (batch, meta) => {
    // A batch that belongs to another run is dropped, so it can never reach this run's state (FE-01).
    if (meta.runId !== currentRun) return;
    for (const event of batch) state.ingest(event);
    scheduleRender();
  }, {
    onStatus: (status, detail) => {
      if (detail.runId !== currentRun) return;
      connection = { state: status, delay: detail.delay ?? 0 };
      if (status === "finished") refreshRuns();
      updateStatusLine();
    },
  });
  scheduleRender();
  checkRun(runId);
}

async function init() {
  // The status sets and the thresholds come from the server before any chart is drawn (DECISION-13).
  try {
    await loadContract();
  } catch (err) {
    showError(`Could not load the dashboard settings. ${err.message}`);
    return;
  }
  try {
    requireCharts();
    counters.init($("counters"));
    funnel.init($("funnel"));
    swimlanes.init($("swimlanes"));
    gauge.init($("gauge"));
    spend.init($("spend"));
    stack.init($("spend-stack"));
    // Thresholds of the pricing tier and the context limit, as set in config/settings.toml, served by the API.
    const limits = contract();
    context.init($("context"), { pricingThreshold: limits.pricingTierTokens, contextLimit: limits.contextLimitTokens });
    ticker.init($("ticker"));
    swarm.init($("swarm"));
  } catch (err) {
    showError(err.message);
    return;
  }

  $("ticker-filter").addEventListener("input", (e) => {
    ticker.setFilter(e.target.value);
    scheduleRender();
  });
  $("autoscroll").addEventListener("change", (e) => {
    swimlanes.setAutoscroll(e.target.checked);
    scheduleRender();
  });
  document.addEventListener("pipeline-select", (e) => {
    const pl = state.pipelines.get(e.detail);
    if (pl && pl.ideaId) {
      location.href = `/signal?id=${encodeURIComponent(pl.ideaId)}`;
      return;
    }
    $("run-status").textContent = `pipeline ${String(e.detail).slice(0, 8)} has no idea yet`;
  });

  // The run select is wired before the first request, so a failed run list still leaves run switching working (R4-FE-05).
  const select = $("run-select");
  select.addEventListener("change", () => attach(select.value));
  try {
    await runs.refresh();
    renderRunSelect();
    const chosen = params.get("run") || (runs.list[0] && runs.list[0].run_id);
    if (chosen) {
      select.value = chosen;
      attach(chosen);
    }
  } catch (err) {
    showError(loadFailure("Could not load the run list", err));
  }
  updateStatusLine();
}

// The run list is refreshed from the server every few seconds, so the status line follows the server, and the
// run's readability is checked again (R4-FE-01).
setInterval(() => {
  refreshRuns();
  if (currentRun) checkRun(currentRun);
}, 15000);

init();
