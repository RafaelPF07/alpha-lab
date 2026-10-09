// Mission control: live view of a run (or the demo run). Every chart reads the same aggregated state.
import { fetchJson, connectStream } from "../core/api.js";
import { RunState } from "../core/state.js";
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
  ticker.update(snap.ticker);
  swarm.update(snap);
  $("run-status").textContent = snap.finished ? "finished" : `live · ${snap.runningPipelines} running`;
}

function scheduleRender() {
  if (!scheduled) {
    scheduled = true;
    requestAnimationFrame(render);
  }
}

function attach(runId) {
  if (stop) stop();
  state = new RunState();
  stop = connectStream(runId, (batch) => {
    for (const event of batch) state.ingest(event);
    scheduleRender();
  });
}

async function init() {
  counters.init($("counters"));
  funnel.init($("funnel"));
  swimlanes.init($("swimlanes"));
  gauge.init($("gauge"));
  spend.init($("spend"));
  stack.init($("spend-stack"));
  context.init($("context"), { pricingThreshold: 100000, contextLimit: 90000 });
  ticker.init($("ticker"));
  swarm.init($("swarm"));

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
    const idea = state.events.find((ev) => ev.pipeline_id === e.detail && ev.idea_id)?.idea_id;
    if (idea) location.href = `/signal?id=${encodeURIComponent(idea)}`;
    else if (pl) alert(`pipeline ${e.detail.slice(0, 8)} has no idea yet`);
  });

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
    if (chosen === "demo") $("demo-badge").hidden = false;
    attach(chosen);
  } else {
    $("run-status").textContent = "no runs yet. Start one with: uv run lab run --pipelines 5 --budget 0.50";
  }
  select.addEventListener("change", () => {
    const run = select.value;
    $("demo-badge").hidden = run !== "demo";
    attach(run);
  });
}

init();
