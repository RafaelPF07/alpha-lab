// Event ticker: a scrolling, filterable feed. Clicking a row selects its pipeline in the other views.
import { agentColor, statusColor } from "./common.js";

const TYPE_COLORS = {
  pipeline_started: "#7aa2f7", idea_created: "#7aa2f7", idea_rejected_duplicate: "#e0af68",
  llm_call_finished: "#565f89", code_written: "#9ece6a", static_check: "#e0af68",
  sandbox_run: "#ff9e64", lookahead_check: "#f7768e", train_backtest: "#bb9af7",
  critic_review: "#7dcfff", revision_requested: "#e0af68", validation_result: "#73daca",
  pipeline_finished: "#9ece6a", budget_update: "#ff9e64", run_started: "#6ea8fe", run_finished: "#6ea8fe",
};

let container = null;
let filter = "";

export function init(el) {
  container = el;
  container.classList.add("ticker");
  container.addEventListener("click", (e) => {
    const row = e.target.closest(".row");
    if (row && row.dataset.pipeline) {
      container.dispatchEvent(new CustomEvent("pipeline-select", { detail: row.dataset.pipeline, bubbles: true }));
    }
  });
  return container;
}

export function setFilter(text) { filter = text.trim().toLowerCase(); }

function describe(e) {
  const p = e.payload || {};
  if (e.type === "llm_call_finished") return `${e.agent} call, $${Number(p.cost_usd || 0).toFixed(4)}, ${p.latency_ms} ms`;
  if (e.type === "pipeline_finished") return `pipeline ${p.status}`;
  if (e.type === "static_check") return p.ok ? "static check passed" : `static check failed: ${(p.errors || [])[0] || ""}`;
  if (e.type === "sandbox_run") return `sandbox ${p.status}`;
  if (e.type === "lookahead_check") return `${p.test} lookahead ${p.passed ? "passed" : "FAILED"}`;
  if (e.type === "train_backtest") return `train Sharpe ${Number(p.sharpe ?? 0).toFixed(3)}`;
  if (e.type === "critic_review") return `critic suspicion ${p.suspicion_level}`;
  if (e.type === "validation_result") return `validation ${p.passed ? "passed" : "failed"}`;
  if (e.type === "idea_created") return `idea: ${p.name}`;
  if (e.type === "revision_requested") return `revision requested (${p.reason})`;
  if (e.type === "budget_update") return `spent $${Number(p.spent_usd || 0).toFixed(4)}`;
  return e.type;
}

export function update(events) {
  if (!container) return;
  const shown = events.filter((e) => !filter || `${e.type} ${e.stage} ${describe(e)}`.toLowerCase().includes(filter));
  const rows = shown.slice(-200).reverse().map((e) => {
    const color = e.type.includes("failed") || e.payload?.passed === false || e.payload?.ok === false
      ? statusColor("failed")
      : TYPE_COLORS[e.type] || "#8a94a8";
    const pipe = e.pipeline_id ? e.pipeline_id.slice(0, 6) : "-";
    return `<div class="row" data-pipeline="${e.pipeline_id || ""}">
      <span class="mono" style="color:#8a94a8">${(e.timestamp || "").slice(11, 23)}</span>
      <span class="tag" style="background:${color}">${e.type}</span>
      <span style="color:${agentColor(e.agent)}">${e.agent || e.stage || ""}</span>
      <span class="mono" style="color:#8a94a8">p:${pipe}</span>
      <span>${describe(e)}</span></div>`;
  });
  container.innerHTML = rows.join("");
}
