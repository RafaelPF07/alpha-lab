// Event ticker: a scrolling, filterable feed of the latest events. A row is a button, so it can be reached
// and opened with the keyboard. Activating a row selects its pipeline in the other views.
import { agentColor, statusColor } from "./common.js";
import { esc } from "../core/escape.js";

// Tag background colours. Each one keeps dark tag text above 4.5:1 contrast.
const TYPE_COLORS = new Map([
  ["pipeline_started", "#7aa2f7"], ["idea_created", "#7aa2f7"], ["idea_rejected_duplicate", "#e0af68"],
  ["llm_call_finished", "#9aa5ce"], ["code_written", "#9ece6a"], ["static_check", "#e0af68"],
  ["sandbox_run", "#ff9e64"], ["lookahead_check", "#f7768e"], ["train_backtest", "#bb9af7"],
  ["critic_review", "#7dcfff"], ["revision_requested", "#e0af68"], ["validation_result", "#73daca"],
  ["pipeline_finished", "#9ece6a"], ["budget_update", "#ff9e64"], ["run_started", "#6ea8fe"], ["run_finished", "#6ea8fe"],
]);
const DEFAULT_TAG = "#8a94a8";
const SHOWN = 200;

let container = null;
let filter = "";

export function init(el) {
  container = el;
  container.classList.add("ticker");
  container.setAttribute("aria-label", "Event ticker, newest first");
  container.addEventListener("click", (e) => {
    const row = e.target.closest("[data-pipeline]");
    if (row && row.dataset.pipeline) {
      container.dispatchEvent(new CustomEvent("pipeline-select", { detail: row.dataset.pipeline, bubbles: true }));
    }
  });
  return container;
}

export function setFilter(text) { filter = String(text ?? "").trim().toLowerCase(); }

// Every value is coerced to a readable form, so a missing field can never print "undefined" or "NaN".
const usd = (x) => (Number.isFinite(Number(x)) ? Number(x).toFixed(4) : "n/a");
const num = (x, digits = 3) => (Number.isFinite(Number(x)) ? Number(x).toFixed(digits) : "n/a");

function describe(e) {
  const p = e.payload || {};
  switch (e.type) {
    case "llm_call_finished": return `${e.agent || "agent"} call, $${usd(p.cost_usd)}, ${p.latency_ms ?? "?"} ms`;
    case "pipeline_finished": return `pipeline ${p.status || "finished"}`;
    case "static_check": return p.ok ? "static check passed" : `static check failed: ${(p.errors || [])[0] || ""}`;
    case "sandbox_run": return `sandbox ${p.status || "unknown"}`;
    case "lookahead_check": return `${p.test || "lookahead"} lookahead ${p.passed ? "passed" : "FAILED"}`;
    case "train_backtest": return `train Sharpe ${num(p.sharpe)}`;
    case "critic_review": return `critic suspicion ${p.suspicion_level || "unknown"}`;
    case "validation_result": return `validation ${p.passed ? "passed" : "failed"}`;
    case "idea_created": return `idea: ${p.name || "unnamed"}`;
    case "revision_requested": return `revision requested (${p.reason || "unknown"})`;
    case "budget_update": return `spent $${usd(p.spent_usd)}`;
    default: return e.type || "event";
  }
}

export function update(events) {
  if (!container) return { shown: 0, matches: 0, total: 0 };
  const matching = filter
    ? events.filter((e) => `${e.type} ${e.stage || ""} ${describe(e)}`.toLowerCase().includes(filter))
    : events;
  const shown = matching.slice(-SHOWN).reverse();
  const rows = shown.map((e) => {
    const type = String(e.type ?? "");
    const failed = type.includes("failed") || e.payload?.passed === false || e.payload?.ok === false;
    const color = failed ? statusColor("failed") : TYPE_COLORS.get(type) || DEFAULT_TAG;
    const pipe = e.pipeline_id ? e.pipeline_id.slice(0, 6) : "-";
    const body = `<span class="time">${esc((e.timestamp || "").slice(11, 23))}</span>`
      + `<span class="tag" style="background:${esc(color)}">${esc(type)}</span>`
      + `<span class="agent" style="--c:${esc(agentColor(e.agent))}">${esc(e.agent || e.stage || "")}</span>`
      + `<span class="pipe">p:${esc(pipe)}</span>`
      + `<span class="text">${esc(describe(e))}</span>`;
    return e.pipeline_id
      ? `<button type="button" class="row" data-pipeline="${esc(e.pipeline_id)}">${body}</button>`
      : `<div class="row">${body}</div>`;
  });
  container.innerHTML = rows.join("");
  return { shown: shown.length, matches: matching.length, total: events.length };
}
