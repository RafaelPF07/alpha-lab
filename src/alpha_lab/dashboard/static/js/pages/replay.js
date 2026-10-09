// Replay: rebuilds a finished run from its stored events, on the same visuals as mission control.
// The clock moves at 1×, 10× or 100× real time, and the scrubber jumps anywhere in the run.
import { fetchJson } from "../core/api.js";
import { RunState } from "../core/state.js";
import * as counters from "../viz/counters.js";
import * as funnel from "../viz/funnel.js";
import * as swimlanes from "../viz/swimlanes.js";
import * as gauge from "../viz/budget_gauge.js";
import * as spend from "../viz/budget_spend.js";
import * as stack from "../viz/budget_stack.js";

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const PAGE = 2000;

let events = [];
let state = new RunState();
let cursor = 0;
let clock = 0;
let speed = 10;
let playing = false;
let lastFrame = null;
let lastRender = 0;

function reset() {
  state = new RunState();
  cursor = 0;
  clock = events.length ? Date.parse(events[0].timestamp) : 0;
}

function advanceTo(time) {
  while (cursor < events.length && Date.parse(events[cursor].timestamp) <= time) {
    state.ingest(events[cursor]);
    cursor += 1;
  }
}

function render() {
  const snap = state.snapshot();
  counters.update(snap);
  funnel.update(snap);
  swimlanes.update(snap);
  gauge.update(snap);
  spend.update(snap);
  stack.update(snap);
  $("scrub").value = String(cursor);
  const last = events[cursor - 1];
  $("clock").textContent = last ? `${last.timestamp.slice(11, 19)} · event ${cursor}/${events.length}` : "start";
}

function frame(now) {
  if (playing && events.length) {
    const dt = lastFrame == null ? 0 : now - lastFrame;
    clock += dt * speed;
    advanceTo(clock);
    if (cursor >= events.length) playing = false;
    $("play").textContent = playing ? "pause" : "play";
  }
  lastFrame = now;
  if (now - lastRender > 250) {
    lastRender = now;
    render();
  }
  requestAnimationFrame(frame);
}

async function loadAll(runId) {
  const all = [];
  let after = 0;
  for (;;) {
    const page = await fetchJson(`/api/runs/${encodeURIComponent(runId)}/events?after=${after}&limit=${PAGE}`);
    all.push(...page);
    if (page.length < PAGE) break;
    after = page[page.length - 1].seq;
  }
  return all;
}

async function openRun(runId) {
  playing = false;
  events = await loadAll(runId);
  $("scrub").max = String(events.length);
  reset();
  advanceTo(clock);
  render();
}

async function init() {
  counters.init($("counters"));
  funnel.init($("funnel"));
  swimlanes.init($("swimlanes"));
  gauge.init($("gauge"));
  spend.init($("spend"));
  stack.init($("spend-stack"));

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
    $("demo-badge").hidden = chosen !== "demo";
    await openRun(chosen);
  }
  select.addEventListener("change", () => { $("demo-badge").hidden = select.value !== "demo"; openRun(select.value); });

  $("play").addEventListener("click", () => {
    if (cursor >= events.length) { reset(); advanceTo(clock); }
    playing = !playing;
    $("play").textContent = playing ? "pause" : "play";
  });
  document.querySelectorAll(".speed").forEach((btn) => btn.addEventListener("click", () => {
    speed = Number(btn.dataset.speed);
  }));
  $("scrub").addEventListener("input", (e) => {
    playing = false;
    $("play").textContent = "play";
    reset();
    cursor = 0;
    const target = Number(e.target.value);
    while (cursor < target && cursor < events.length) { state.ingest(events[cursor]); cursor += 1; }
    clock = cursor > 0 ? Date.parse(events[cursor - 1].timestamp) : clock;
    render();
  });
  requestAnimationFrame(frame);
}

init();
