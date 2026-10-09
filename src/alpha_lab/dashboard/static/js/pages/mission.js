// Mission control: live view of a run (or the demo run). Every chart reads the same aggregated state.
import { fetchJson, connectStream } from "../core/api.js";
import { RunState } from "../core/state.js";
import { isDemoRun } from "../core/runs.js";
import { showError, requireCharts } from "../core/page.js";
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

function updateStatusLine() {
  const el = $("run-status");
  if (connection.state === "reconnecting") {
    el.textContent = `connection lost, retrying in ${Math.round(connection.delay / 1000)} s`;
  } else if (state.finished) {
    el.textContent = `finished · ${state.runStatus || "done"}`;
  } else if (connection.state === "connecting") {
    el.textContent = "connecting…";
  } else {
    el.textContent = `live · ${state.runningPipelines} running`;
  }
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
      updateStatusLine();
    },
  });
  scheduleRender();
}

async function init() {
  try {
    requireCharts();
    counters.init($("counters"));
    funnel.init($("funnel"));
    swimlanes.init($("swimlanes"));
    gauge.init($("gauge"));
    spend.init($("spend"));
    stack.init($("spend-stack"));
    // Thresholds of the pricing tier and the context limit, as set in config/settings.toml.
    context.init($("context"), { pricingThreshold: 100000, contextLimit: 90000 });
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

  try {
    const runs = await fetchJson("/api/runs");
    const select = $("run-select");
    for (const r of runs) {
      const opt = document.createElement("option");
      opt.value = r.run_id;
      opt.textContent = `${r.run_id} · ${r.status}`;
      select.appendChild(opt);
    }
    const chosen = params.get("run") || (runs[0] && runs[0].run_id);
    if (chosen) {
      select.value = chosen;
      attach(chosen);
    } else {
      $("run-status").textContent = "no runs yet. Start one with: uv run lab run --pipelines 5 --budget 0.50";
    }
    select.addEventListener("change", () => attach(select.value));
  } catch (err) {
    showError(`Could not load the run list. ${err.message}`);
  }
}

init();
