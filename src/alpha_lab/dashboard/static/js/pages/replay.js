// Replay: rebuilds a finished run from its stored events, on the same visuals as mission control.
// The clock moves at 1×, 10× or 100× real time, and the scrubber jumps anywhere in the run.
import { fetchJson, LatestRequest } from "../core/api.js";
import { loadContract } from "../core/contract.js";
import { RunState } from "../core/state.js";
import { isDemoRun } from "../core/runs.js";
import { showError, clearError, requireCharts } from "../core/page.js";
import { loadFailure } from "../core/ledgerprobe.js";
import * as counters from "../viz/counters.js";
import * as funnel from "../viz/funnel.js";
import * as swimlanes from "../viz/swimlanes.js";
import * as gauge from "../viz/budget_gauge.js";
import * as spend from "../viz/budget_spend.js";
import * as stack from "../viz/budget_stack.js";

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const PAGE = 2000;
const latest = new LatestRequest(); // only the newest run selection may write to the page (FE-04)

let events = [];
let state = new RunState();
let cursor = 0;
let clock = 0;
let speed = 10;
let playing = false;
let lastFrame = null;
let lastRender = 0;
let dirty = true; // something changed since the last draw; a paused replay draws nothing otherwise
let pendingSeek = null; // the scrubber position to jump to on the next frame

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

// Rebuilds the state up to a position. Scrubbing is coalesced to one rebuild per frame.
function seek(target) {
  reset();
  while (cursor < target && cursor < events.length) {
    state.ingest(events[cursor]);
    cursor += 1;
  }
  clock = cursor > 0 ? Date.parse(events[cursor - 1].timestamp) : clock;
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
  $("clock").textContent = last ? `${(last.timestamp || "").slice(11, 19)} · event ${cursor}/${events.length}` : "start";
  dirty = false;
}

function setPlaying(on) {
  playing = on;
  $("play").textContent = playing ? "pause" : "play";
}

function frame(now) {
  if (pendingSeek !== null) {
    const target = pendingSeek;
    pendingSeek = null;
    seek(target);
    dirty = true;
  }
  if (playing && events.length) {
    const dt = lastFrame == null ? 0 : now - lastFrame;
    clock += dt * speed;
    advanceTo(clock);
    if (cursor >= events.length) setPlaying(false);
    dirty = true;
  }
  lastFrame = now;
  if (dirty && (!playing || now - lastRender > 250)) {
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

async function choose(runId) {
  const token = latest.begin();
  $("demo-badge").hidden = !isDemoRun(runId);
  setPlaying(false);
  try {
    const loaded = await loadAll(runId);
    if (!latest.isCurrent(token)) return; // a newer run was chosen while this one loaded
    events = loaded;
    $("scrub").max = String(events.length);
    reset();
    advanceTo(clock);
    dirty = true;
    clearError();
  } catch (err) {
    if (latest.isCurrent(token)) showError(loadFailure(`Could not load run ${runId}`, err));
  }
}

function markSpeed(button) {
  for (const btn of document.querySelectorAll(".speed")) {
    btn.setAttribute("aria-pressed", String(btn === button));
  }
}

async function init() {
  try {
    await loadContract();
    requireCharts();
    counters.init($("counters"));
    funnel.init($("funnel"));
    swimlanes.init($("swimlanes"));
    gauge.init($("gauge"));
    spend.init($("spend"));
    stack.init($("spend-stack"));
  } catch (err) {
    showError(err.message);
    return;
  }

  // The controls are wired before the first request, so they work even if the run list fails (FE-12).
  $("play").addEventListener("click", () => {
    if (cursor >= events.length) { reset(); advanceTo(clock); }
    setPlaying(!playing);
  });
  for (const btn of document.querySelectorAll(".speed")) {
    btn.addEventListener("click", () => {
      speed = Number(btn.dataset.speed);
      markSpeed(btn);
    });
  }
  markSpeed(document.querySelector(`.speed[data-speed="${speed}"]`));
  $("scrub").addEventListener("input", (e) => {
    setPlaying(false);
    pendingSeek = Number(e.target.value);
  });
  const select = $("run-select");
  select.addEventListener("change", () => choose(select.value));
  requestAnimationFrame(frame);

  try {
    const runs = await fetchJson("/api/runs");
    for (const r of runs) {
      const opt = document.createElement("option");
      opt.value = r.run_id;
      opt.textContent = `${r.run_id} · ${r.status}${r.stale && r.status === "running" ? " · stale" : ""}`;
      select.appendChild(opt);
    }
    const chosen = params.get("run") || (runs[0] && runs[0].run_id);
    if (chosen) {
      select.value = chosen;
      await choose(chosen);
    }
  } catch (err) {
    showError(loadFailure("Could not load the run list", err));
  }
}

init();
