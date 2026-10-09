// Signal inspector: one idea, its attempts, the code diff between attempts, checks, critic and holdout status.
import { fetchJson } from "../core/api.js";
import { COLORS, fmtNum, fmtPct, fmtUsd, statusColor } from "../viz/common.js";

const $ = (id) => document.getElementById(id);
const id = new URLSearchParams(location.search).get("id");

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// Line diff against the previous attempt. Added lines green, removed red. A simple LCS keeps it honest.
function lineDiff(a, b) {
  const x = (a || "").split("\n");
  const y = (b || "").split("\n");
  const n = x.length;
  const m = y.length;
  const dp = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
  for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--)
    dp[i][j] = x[i] === y[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
  const out = [];
  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (x[i] === y[j]) { out.push(["same", x[i]]); i++; j++; }
    else if (dp[i + 1][j] >= dp[i][j + 1]) { out.push(["del", x[i]]); i++; }
    else { out.push(["add", y[j]]); j++; }
  }
  while (i < n) out.push(["del", x[i++]]);
  while (j < m) out.push(["add", y[j++]]);
  return out;
}

function renderCode(attempt, previous) {
  const rows = previous ? lineDiff(previous.code, attempt.code) : attempt.code.split("\n").map((l) => ["same", l]);
  const html = rows.map(([kind, line]) => {
    const style = kind === "add" ? "background:rgba(158,206,106,0.18)" : kind === "del" ? "background:rgba(247,118,142,0.18)" : "";
    const sign = kind === "add" ? "+" : kind === "del" ? "-" : " ";
    return `<div style="${style}">${sign} ${esc(line)}</div>`;
  }).join("");
  $("diff").innerHTML = `<div class="note">attempt ${attempt.attempt_number}${previous ? ` · diff against attempt ${previous.attempt_number}` : ""} · ${esc(attempt.status)}</div><pre class="code">${html}</pre>`;
}

function renderTree(attempts, onPick) {
  const children = attempts.map((a) => ({
    name: `#${a.attempt_number} ${a.status}`,
    value: a.attempt_number,
    itemStyle: { color: statusColor(a.status) },
    attempt: a,
  }));
  const root = { name: "idea", children, itemStyle: { color: COLORS.text } };
  const chart = echarts.init($("tree"));
  chart.setOption({
    tooltip: { trigger: "item" },
    series: [{ type: "tree", data: [root], top: "8%", bottom: "8%", left: "10%", right: "20%",
      symbol: "circle", symbolSize: 12, orient: "LR",
      label: { position: "right", color: COLORS.text, fontSize: 12 },
      leaves: { label: { position: "right" } }, expandAndCollapse: false }],
  });
  chart.on("click", (p) => { if (p.data?.attempt) onPick(p.data.attempt); });
  return chart;
}

function metricsTable(train, validation, holdout) {
  const rows = [];
  const row = (label, t, v, h) => rows.push(`<tr><td>${label}</td><td class="num">${t}</td><td class="num">${v}</td><td class="num">${h}</td></tr>`);
  const t = train || {};
  const v = validation?.metrics || {};
  const h = holdout.status === "complete" ? holdout.metrics || {} : null;
  const cell = (obj, key, fmt) => (obj && obj[key] != null ? fmt(obj[key]) : "n/a");
  row("Sharpe (annualised)", cell(t, "sharpe", (x) => fmtNum(x)), cell(v, "sharpe", (x) => fmtNum(x)), h ? cell(h, "sharpe", (x) => fmtNum(x)) : `<span class="locked">locked</span>`);
  row("annual return", cell(t, "ann_return", fmtPct), cell(v, "ann_return", fmtPct), h ? cell(h, "ann_return", fmtPct) : `<span class="locked">locked</span>`);
  row("max drawdown", cell(t, "max_drawdown", fmtPct), cell(v, "max_drawdown", fmtPct), h ? cell(h, "max_drawdown", fmtPct) : `<span class="locked">locked</span>`);
  row("avg daily turnover", cell(t, "avg_daily_turnover", (x) => fmtNum(x, 4)), cell(v, "avg_daily_turnover", (x) => fmtNum(x, 4)), h ? cell(h, "avg_daily_turnover", (x) => fmtNum(x, 4)) : `<span class="locked">locked</span>`);
  return `<table class="data"><thead><tr><th>metric</th><th class="num">train</th><th class="num">validation</th><th class="num">holdout</th></tr></thead><tbody>${rows.join("")}</tbody></table>
    <p class="note">${holdout.status === "complete" ? "Holdout run recorded." : "Holdout not run for this signal. Its numbers stay hidden until the official holdout run."}</p>`;
}

async function init() {
  if (!id) { $("title").textContent = "Signal inspector: no id given"; return; }
  const data = await fetchJson(`/api/signals/${encodeURIComponent(id)}`);
  if (data.demo || data.idea?.name) $("demo-badge").hidden = false;
  $("title").textContent = data.idea ? data.idea.name : "Signal";
  $("idea").innerHTML = data.idea
    ? `<h2>Idea</h2><p><b>${esc(data.idea.name)}</b> · theme ${esc(data.idea.theme || "n/a")}</p>
       <p>${esc(data.idea.hypothesis)}</p><p class="note">${esc(data.idea.description)}</p>`
    : "<p>no idea recorded</p>";

  const attempts = data.attempts;
  const pick = (attempt) => {
    const previous = attempts.find((a) => a.attempt_number === attempt.attempt_number - 1);
    renderCode(attempt, previous);
  };
  renderTree(attempts, pick);
  const latest = attempts[attempts.length - 1];
  if (latest) pick(latest);

  $("metrics").innerHTML = metricsTable(data.train_metrics, data.validation, data.holdout);

  if (data.critic) {
    const concerns = (data.critic.concerns || []).map((c) => `<li>${esc(c)}</li>`).join("") || "<li>none</li>";
    $("critic").innerHTML = `<p>suspicion: <b>${esc(data.critic.suspicion_level)}</b></p><ul>${concerns}</ul>
      <p class="note">suggested change: ${esc(data.critic.suggested_change)}</p>`;
  } else {
    $("critic").innerHTML = `<p class="note">The critic did not review this attempt (it did not pass the training gate).</p>`;
  }

  const calls = data.llm_calls.map((c) => `<tr><td>${esc(c.agent)}</td><td class="mono">${esc(c.model)}</td>
    <td class="num">${c.input_tokens ?? "n/a"}</td><td class="num">${c.output_tokens ?? "n/a"}</td><td class="num">${fmtUsd(c.cost_usd)}</td></tr>`).join("");
  $("calls").innerHTML = `<table class="data"><thead><tr><th>agent</th><th>model</th><th class="num">in</th><th class="num">out</th><th class="num">cost</th></tr></thead><tbody>${calls}</tbody></table>`;
}

init();
