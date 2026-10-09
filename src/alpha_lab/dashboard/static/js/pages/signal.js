// Signal inspector: one idea, its attempts, the code diff between attempts, checks, critic and holdout status.
import { fetchJson } from "../core/api.js";
import { esc } from "../core/escape.js";
import { signalIsDemo } from "../core/runs.js";
import { showError, requireCharts } from "../core/page.js";
import { COLORS, fmtNum, fmtPct, fmtUsd, statusColor, track } from "../viz/common.js";

const $ = (id) => document.getElementById(id);
const id = new URLSearchParams(location.search).get("id");
// The line diff table has (n+1)*(m+1) cells. Above this size the diff is skipped, to keep the page responsive.
const MAX_DIFF_CELLS = 250000;

// Line diff against the previous attempt. Added lines green, removed red. A simple LCS keeps it honest.
// Returns null when the files are too large to compare.
function lineDiff(a, b) {
  const x = (a || "").split("\n");
  const y = (b || "").split("\n");
  const n = x.length;
  const m = y.length;
  if ((n + 1) * (m + 1) > MAX_DIFF_CELLS) return null;
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
  const code = attempt.code || "";
  let rows = code.split("\n").map((l) => ["same", l]);
  let note = "";
  if (previous) {
    const diff = lineDiff(previous.code, code);
    if (diff) rows = diff;
    else note = " · diff skipped, the code is too large to compare";
  }
  const html = rows.map(([kind, line]) => {
    const style = kind === "add" ? "background:rgba(158,206,106,0.18)" : kind === "del" ? "background:rgba(247,118,142,0.18)" : "";
    const sign = kind === "add" ? "+" : kind === "del" ? "-" : " ";
    return `<div style="${style}">${sign} ${esc(line)}</div>`;
  }).join("");
  $("diff").innerHTML = `<div class="note">attempt ${esc(attempt.attempt_number)}${previous ? ` · diff against attempt ${esc(previous.attempt_number)}` : ""} · ${esc(attempt.status)}${note}</div><pre class="code">${html}</pre>`;
}

// Shows one attempt's code, and marks its button in the list as pressed.
function showAttempt(attempts, index) {
  const attempt = attempts[index];
  const previous = attempts.find((a) => a.attempt_number === attempt.attempt_number - 1);
  renderCode(attempt, previous);
  for (const btn of document.querySelectorAll("#attempts button")) {
    btn.setAttribute("aria-pressed", String(Number(btn.dataset.index) === index));
  }
}

function renderTree(attempts, onPick) {
  const children = attempts.map((a) => ({
    name: `#${a.attempt_number} ${a.status}`,
    value: a.attempt_number,
    itemStyle: { color: statusColor(a.status) },
    attempt: a,
  }));
  const root = { name: "idea", children, itemStyle: { color: COLORS.text } };
  const chart = track(echarts.init($("tree")));
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

// The same attempts as a list of buttons, so the code can be opened from the keyboard (FE-08).
function renderAttemptList(attempts, onPick) {
  const list = $("attempts");
  list.innerHTML = attempts.map((a, i) => `<li><button type="button" data-index="${i}" aria-pressed="false">`
    + `#${esc(a.attempt_number)} · ${esc(a.status)}</button></li>`).join("");
  list.addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-index]");
    if (btn) onPick(Number(btn.dataset.index));
  });
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

function renderSignal(data) {
  // The demo badge comes from the run the signal belongs to, never from the idea's name (FE-06).
  $("demo-badge").hidden = !signalIsDemo(data);
  $("title").textContent = data.idea ? data.idea.name : "Signal";
  $("idea").innerHTML = data.idea
    ? `<h2>Idea</h2><p><b>${esc(data.idea.name)}</b> · theme ${esc(data.idea.theme || "n/a")}</p>
       <p>${esc(data.idea.hypothesis)}</p><p class="note">${esc(data.idea.description)}</p>`
    : "<p>no idea recorded</p>";

  const attempts = data.attempts;
  if (attempts.length) {
    renderTree(attempts, (attempt) => showAttempt(attempts, attempts.indexOf(attempt)));
    renderAttemptList(attempts, (index) => showAttempt(attempts, index));
    showAttempt(attempts, attempts.length - 1);
  } else {
    $("diff").textContent = "No attempts recorded for this idea.";
  }

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

async function init() {
  if (!id) { $("title").textContent = "Signal inspector: no id given"; return; }
  try {
    requireCharts();
    renderSignal(await fetchJson(`/api/signals/${encodeURIComponent(id)}`));
  } catch (err) {
    $("title").textContent = "Signal inspector";
    showError(`Could not load this signal. ${err.message}`);
  }
}

init();
