// Live pipelines page. Connects the event stream to the pipeline theatre, the stats strip and the feed.
import { fetchJson, connectStream } from "../core/api.js";
import { isDemoRun } from "../core/runs.js";
import { showError } from "../core/page.js";
import { PipelineTheatre } from "../viz/theatre.js";
import { STAGES, describe } from "../viz/theatre_model.js";

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const FEED_LIMIT = 60;
const FEED_KEEP = 400;
const FLUSH_MS = 80; // how quickly a new event reaches the screen

const FEED_GROUPS = {
  llm: new Set(["llm_call_started", "llm_call_finished"]),
  checks: new Set(["static_check", "sandbox_run", "lookahead_check", "critic_review", "code_written"]),
  results: new Set(["train_backtest", "validation_result", "pipeline_finished", "run_finished", "run_started", "idea_created"]),
};

const TYPE_STAGE = {
  pipeline_started: 0, idea_created: 0, llm_call_started: 0, llm_call_finished: 0,
  code_written: 1, revision_requested: 1,
  static_check: 2, sandbox_run: 3, lookahead_check: 4, train_backtest: 5,
  critic_review: 6, validation_result: 7, pipeline_finished: 7,
  run_started: 0, run_finished: 7, budget_update: 0, idea_rejected_duplicate: 0,
};

let theatre = null;
let stop = null;
let filter = "all";
let feedItems = [];
let currentRun = null;
let connection = { state: "connecting", delay: 0 };

function stageColor(index) {
  const key = STAGES[index]?.color || "--stage-idea";
  return getComputedStyle(document.documentElement).getPropertyValue(key).trim() || "#7aa2f7";
}

// The pill says what the connection is doing, and then what the run is doing.
function renderStatus(s) {
  const state = $("live-state");
  if (connection.state === "reconnecting") {
    state.textContent = `reconnecting in ${Math.round(connection.delay / 1000)} s`;
    state.className = "pill pill-warn";
  } else if (s.runStatus === "running") {
    state.textContent = "live";
    state.className = "pill pill-live";
  } else if (s.runStatus === "waiting") {
    state.textContent = connection.state === "connecting" ? "connecting…" : "waiting for events";
    state.className = "pill pill-muted";
  } else {
    state.textContent = String(s.runStatus).replace(/_/g, " ");
    state.className = "pill pill-done";
  }
}

function renderStats() {
  const s = theatre.model.stats();
  $("stat-running").textContent = String(s.running);
  $("stat-done").textContent = s.total ? `${s.done}/${s.total}` : String(s.done);
  $("stat-passed").textContent = String(s.passed);
  $("stat-dupes").textContent = String(s.duplicates);
  $("stat-spent").textContent = `$${s.spent.toFixed(4)}`;
  $("stat-budget").textContent = s.budget ? `of $${s.budget.toFixed(2)} budget` : "spent";
  $("progress-fill").style.width = `${Math.round((s.progress ?? 0) * 100)}%`;
  renderStatus(s);
}

function groupOf(item) {
  if (FEED_GROUPS.llm.has(item.type)) return "llm";
  if (FEED_GROUPS.checks.has(item.type)) return "checks";
  if (FEED_GROUPS.results.has(item.type)) return "results";
  return "all";
}

function renderFeed() {
  const list = $("feed");
  const visible = feedItems.filter((item) => filter === "all" || groupOf(item) === filter).slice(-FEED_LIMIT);
  const fragment = document.createDocumentFragment();
  for (const item of visible.slice().reverse()) {
    const li = document.createElement("li");
    const time = document.createElement("time");
    time.textContent = item.time;
    const chip = document.createElement("span");
    chip.className = "chip";
    const stage = TYPE_STAGE[item.type];
    chip.style.background = stage !== undefined ? stageColor(stage) : "#8a94a8";
    chip.textContent = item.type.replace(/_/g, " ");
    const body = document.createElement("span");
    body.className = "txt";
    const who = document.createElement("span");
    who.className = "who";
    who.textContent = item.pid ? `#${item.pid} ` : "";
    body.appendChild(who);
    body.appendChild(document.createTextNode(item.text));
    li.appendChild(time);
    li.appendChild(chip);
    if (item.ideaId) {
      const link = document.createElement("a");
      link.href = `/signal?id=${encodeURIComponent(item.ideaId)}`;
      link.appendChild(body);
      li.appendChild(link);
    } else {
      li.appendChild(body);
    }
    fragment.appendChild(li);
  }
  list.replaceChildren(fragment);
}

function ingest(batch, meta) {
  // A batch that belongs to another run is dropped (FE-01).
  if (meta.runId !== currentRun) return;
  theatre.ingest(batch);
  for (const event of batch) {
    feedItems.push({
      seq: event.seq,
      type: String(event.type ?? "event"),
      time: (event.timestamp || "").slice(11, 19),
      pid: event.pipeline_id ? event.pipeline_id.slice(0, 6) : "",
      text: describe(event),
      ideaId: event.idea_id || null,
    });
  }
  if (feedItems.length > FEED_KEEP) feedItems = feedItems.slice(-FEED_KEEP);
  renderStats();
  renderFeed();
}

function attach(run) {
  if (stop) stop();
  currentRun = run;
  feedItems = [];
  theatre.reset();
  $("feed").replaceChildren();
  connection = { state: "connecting", delay: 0 };
  $("demo-badge").hidden = !isDemoRun(run);
  stop = connectStream(run, ingest, {
    flushMs: FLUSH_MS,
    onStatus: (status, detail) => {
      if (detail.runId !== currentRun) return;
      connection = { state: status, delay: detail.delay ?? 0 };
      renderStats();
    },
  });
  renderStats();
}

async function init() {
  theatre = new PipelineTheatre($("stage"), {
    onOpen: (ideaId) => { location.href = `/signal?id=${encodeURIComponent(ideaId)}`; },
  });

  document.querySelectorAll(".seg button").forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.filter === filter));
    button.addEventListener("click", () => {
      filter = button.dataset.filter;
      document.querySelectorAll(".seg button").forEach((b) => {
        const on = b === button;
        b.classList.toggle("seg-on", on);
        b.setAttribute("aria-pressed", String(on));
      });
      renderFeed();
    });
  });

  try {
    const runs = await fetchJson("/api/runs");
    const select = $("run-select");
    for (const r of runs) {
      const option = document.createElement("option");
      option.value = r.run_id;
      option.textContent = `${r.run_id} · ${r.status}`;
      select.appendChild(option);
    }
    const chosen = params.get("run") || (runs[0] && runs[0].run_id) || null;
    if (chosen) {
      select.value = chosen;
      attach(chosen);
    } else {
      $("live-state").textContent = "no runs yet. Start one with: uv run lab run --pipelines 5 --budget 0.50";
    }
    select.addEventListener("change", () => attach(select.value));
  } catch (err) {
    showError(`Could not load the run list. ${err.message}`);
  }
}

init();
