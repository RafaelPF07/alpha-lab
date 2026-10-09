// Live pipelines page. Connects the event stream to the pipeline theatre, the stats strip, the feed, and a
// text list of the same pipelines for keyboard and screen-reader users (the canvas is not reachable by keyboard).
import { connectStream } from "../core/api.js";
import { loadContract } from "../core/contract.js";
import { isDemoRun } from "../core/runs.js";
import { showError, clearError } from "../core/page.js";
import { checkLedger, loadFailure } from "../core/ledgerprobe.js";
import { RunStatuses, idleLabel, statusLabel } from "../core/runstatus.js";
import { trailingThrottle } from "../core/throttle.js";
import { PipelineTheatre } from "../viz/theatre.js";
import { STAGES, describe } from "../viz/theatre_model.js";

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const FEED_LIMIT = 60;
const FEED_KEEP = 400;
const FLUSH_MS = 80; // how quickly a new event reaches the screen
const CLOCK_MS = 15000; // how often the run list is refreshed from the server
const SUMMARY_MS = 5000; // how often the screen-reader summary and text list are refreshed, not per event
const NO_RUNS_HINT = "No runs yet. Start one with: uv run lab run --pipelines 5 --budget 0.50";
const LIST_MAX = 60; // pipelines in the text list (the newest with an idea)
const PILL = { live: "pill-live", done: "pill-done", warn: "pill-warn", muted: "pill-muted" };

const FEED_GROUPS = {
  llm: new Set(["llm_call_started", "llm_call_finished"]),
  checks: new Set(["static_check", "duplicate_code", "sandbox_run", "lookahead_check", "critic_review", "code_written"]),
  results: new Set(["train_backtest", "validation_result", "pipeline_finished", "run_finished", "run_started", "idea_created"]),
};

const TYPE_STAGE = {
  pipeline_started: 0, idea_created: 0, llm_call_started: 0, llm_call_finished: 0,
  code_written: 1, revision_requested: 1,
  static_check: 2, duplicate_code: 2, sandbox_run: 3, lookahead_check: 4, train_backtest: 5,
  critic_review: 6, validation_result: 7, pipeline_finished: 7,
  run_started: 0, run_finished: 7, budget_update: 0, idea_rejected_duplicate: 0,
};

let theatre = null;
let stop = null;
let filter = "all";
let feedItems = [];
let currentRun = null;
let connection = { state: "connecting", delay: 0 };
let problem = null; // the reason the server gives when it cannot read the current run's events (R4-FE-01)
const runs = new RunStatuses(); // the run list and each run's status, as the server reports them
let rovingIndex = 0; // which feed link is in the tab order
let feedDirty = false; // the feed changed while a link in it had focus, so it waits for focus to leave
let listDirty = false;
// The screen-reader summary is refreshed at most every SUMMARY_MS, and the last batch is never held back (R4-FE-06).
const summary = trailingThrottle(() => writeAccessibleStage(), SUMMARY_MS);

function stageColor(index) {
  const key = STAGES[index]?.color || "--stage-idea";
  return getComputedStyle(document.documentElement).getPropertyValue(key).trim() || "#7aa2f7";
}

// The pill shows the run's status as the server reports it. The stream's state only matters while the
// server still says the run is going (see statusLabel). With no run selected it says so (R4-FE-04).
function renderStatus() {
  const state = $("live-state");
  const label = currentRun
    ? statusLabel(runs.get(currentRun), connection.state, connection.delay, problem)
    : idleLabel(runs.answered, runs.list.length);
  state.textContent = label.text;
  state.className = `pill ${PILL[label.tone] || "pill-muted"}`;
}

// With no run to show, the stage summary carries the command that starts one (R4-FE-04).
function renderHint() {
  if (currentRun || !runs.answered || runs.list.length) return;
  $("stage-summary").textContent = NO_RUNS_HINT;
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
  renderStatus();
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

async function refreshRuns() {
  try {
    await runs.refresh();
    renderRunSelect();
  } catch {
    // Keep the last answer. The pill still shows the stream's state.
  }
  renderStats();
  renderHint();
}

function renderStats() {
  if (!theatre) return;
  const s = theatre.model.stats();
  $("stat-running").textContent = String(s.running);
  $("stat-done").textContent = s.total ? `${s.done}/${s.total}` : String(s.done);
  $("stat-passed").textContent = `${s.passed} / ${s.validated}`;
  $("stat-dupes").textContent = String(s.duplicates);
  $("stat-spent").textContent = `$${s.spent.toFixed(4)}`;
  $("stat-budget").textContent = s.budget ? `of $${s.budget.toFixed(2)} budget` : "spent";
  $("progress-fill").style.width = `${Math.round((s.progress ?? 0) * 100)}%`;
  renderStatus();
}

// A text version of the stage: one summary line, and a list of links to each pipeline's signal page.
// Updated every few seconds, not per event, so a screen reader is not flooded.
function renderAccessibleStage(force = false) {
  summary.request(force);
}

function writeAccessibleStage() {
  if (!theatre || !currentRun) return;
  const s = theatre.model.stats();
  const total = s.total ? `${s.done} of ${s.total}` : String(s.done);
  // Counted from the layout, not from the last frame, so a fresh batch is described correctly at once.
  const shown = Math.min(theatre.model.order.length, theatre.layout().capacity);
  const notShown = theatre.model.order.length - shown;
  const hidden = notShown > 0 ? ` ${notShown} are not shown on the stage. The list below has the newest pipelines with an idea.` : "";
  $("stage-summary").textContent = `${total} pipelines finished, ${s.running} running, ${s.passed} passed the critic, ${s.validated} passed validation.${hidden}`;
  renderPipelineList();
}

function renderPipelineList() {
  const list = $("pipeline-list");
  // A list with keyboard focus inside is left alone, or the focused link would be removed.
  if (list.contains(document.activeElement)) {
    listDirty = true;
    return;
  }
  listDirty = false;
  const cards = theatre.model.order.slice(-LIST_MAX).map((id) => theatre.model.cards.get(id)).filter((c) => c && c.ideaId);
  list.replaceChildren(...cards.map((card) => {
    const li = document.createElement("li");
    const link = document.createElement("a");
    link.href = `/signal?id=${encodeURIComponent(card.ideaId)}`;
    link.textContent = `#${card.short} ${card.idea || "idea pending"} · ${card.status === "running" ? "running" : card.status.replace(/_/g, " ")}`;
    li.appendChild(link);
    return li;
  }));
}

function groupOf(item) {
  if (FEED_GROUPS.llm.has(item.type)) return "llm";
  if (FEED_GROUPS.checks.has(item.type)) return "checks";
  if (FEED_GROUPS.results.has(item.type)) return "results";
  return "all";
}

function feedLinks() {
  return [...$("feed").querySelectorAll("a")];
}

// Only one feed link is in the tab order (roving tabindex), so Tab leaves the feed in one stop.
// Arrow keys move between the links.
function applyRoving() {
  feedLinks().forEach((a, i) => { a.tabIndex = i === rovingIndex ? 0 : -1; });
}

function renderFeed() {
  const feed = $("feed");
  if (feed.contains(document.activeElement)) {
    feedDirty = true; // redrawn when focus leaves the feed
    return;
  }
  feedDirty = false;
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
  feed.replaceChildren(fragment);
  if (rovingIndex >= feedLinks().length) rovingIndex = 0;
  applyRoving();
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
  renderAccessibleStage();
}

function attach(run) {
  if (stop) stop();
  currentRun = run;
  problem = null;
  feedItems = [];
  theatre.reset();
  $("feed").replaceChildren();
  $("pipeline-list").replaceChildren();
  connection = { state: "connecting", delay: 0 };
  rovingIndex = 0;
  feedDirty = false;
  listDirty = false;
  summary.reset();
  $("demo-badge").hidden = !isDemoRun(run);
  stop = connectStream(run, ingest, {
    flushMs: FLUSH_MS,
    onStatus: (status, detail) => {
      if (detail.runId !== currentRun) return;
      connection = { state: status, delay: detail.delay ?? 0 };
      // When the stream ends, ask the server what the run's status now is, and write the final summary now.
      if (status === "finished") {
        refreshRuns();
        renderAccessibleStage(true);
      }
      renderStats();
    },
  });
  renderStats();
  checkRun(run);
}

// Feed keyboard support: arrows move between links, and focus leaving the feed applies any deferred update.
function setupFeedKeys() {
  const feed = $("feed");
  feed.addEventListener("focusin", (e) => {
    const links = feedLinks();
    const index = links.indexOf(e.target);
    if (index >= 0 && index !== rovingIndex) {
      rovingIndex = index;
      applyRoving();
    }
  });
  feed.addEventListener("focusout", (e) => {
    if (feed.contains(e.relatedTarget)) return;
    if (feedDirty) renderFeed();
  });
  feed.addEventListener("keydown", (e) => {
    if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
    const links = feedLinks();
    const at = links.indexOf(document.activeElement);
    if (at < 0) return;
    const next = links[Math.min(links.length - 1, Math.max(0, at + (e.key === "ArrowDown" ? 1 : -1)))];
    if (!next) return;
    e.preventDefault();
    next.focus();
  });
  $("pipeline-list").addEventListener("focusout", () => {
    if (listDirty && !$("pipeline-list").contains(document.activeElement)) renderPipelineList();
  });
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
  setupFeedKeys();

  // The status sets and thresholds come from the server before anything is drawn (DECISION-13).
  try {
    await loadContract();
  } catch (err) {
    showError(`Could not load the dashboard settings. ${err.message}`);
    return;
  }

  // The run select is wired before the first request, so a failed run list still leaves run switching working (R4-FE-05).
  const select = $("run-select");
  select.addEventListener("change", () => attach(select.value));
  try {
    await runs.refresh();
    renderRunSelect();
    const chosen = params.get("run") || (runs.list[0] && runs.list[0].run_id) || null;
    if (chosen) {
      select.value = chosen;
      attach(chosen);
    }
  } catch (err) {
    showError(loadFailure("Could not load the run list", err));
  }
  renderStats();
  renderHint();

  // The run list and the run's readability are checked again, and the text list is refreshed when it was held back.
  setInterval(() => {
    if (!theatre) return;
    refreshRuns();
    if (currentRun) checkRun(currentRun);
    renderAccessibleStage(listDirty);
  }, CLOCK_MS);
}

init();
