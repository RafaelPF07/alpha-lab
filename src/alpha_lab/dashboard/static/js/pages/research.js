// Research views: post-run analysis of a finished run. Everything comes from the JSON endpoints.
import { fetchJson, LatestRequest } from "../core/api.js";
import { loadContract } from "../core/contract.js";
import { esc } from "../core/escape.js";
import { isDemoRun } from "../core/runs.js";
import { showError, clearError, requireCharts } from "../core/page.js";
import { loadFailure } from "../core/ledgerprobe.js";
import { runNote, signalOutcome } from "../core/research_text.js";
import * as galaxy from "../viz/galaxy.js";
import * as luck from "../viz/luck.js";
import * as equity from "../viz/equity.js";
import * as drawdown from "../viz/drawdown.js";
import * as correlation from "../viz/correlation.js";
import * as themes from "../viz/themes.js";
import * as failures from "../viz/failures.js";
import * as economics from "../viz/economics.js";

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const latest = new LatestRequest(); // only the newest run selection may write to the page (FE-04)

let current = null;
let research = null; // the last research payload for the current run, reused when only the colour changes

const fmt2 = (x) => (Number.isFinite(x) ? x.toFixed(2) : "n/a");

// A plain, keyboard-reachable list of the signals that reached validation, so every point can be opened without
// the chart. Each entry says whether its own validation passed (R4-FE-08).
function renderSignalList(points) {
  $("signal-count").textContent = String(points.length);
  $("signal-list").innerHTML = points.length
    ? points.map((p) => `<li><a href="/signal?id=${encodeURIComponent(p.attempt_id)}">${esc(p.name || p.attempt_id)}</a>`
      + ` <span class="note">${esc(signalOutcome(p))} · train ${fmt2(p.train_sharpe)} · validation ${fmt2(p.validation_sharpe)}</span></li>`).join("")
    : "<li>No signal reached validation in this run.</li>";
}

async function load(runId) {
  const token = latest.begin();
  const base = `/api/runs/${encodeURIComponent(runId)}`;
  const [res, eq, corr, bench] = await Promise.all([
    fetchJson(`${base}/research`),
    fetchJson(`${base}/equity?top=5`),
    fetchJson(`${base}/correlation`),
    fetchJson(`${base}/benchmark?top=5`),
  ]);
  if (!latest.isCurrent(token)) return; // a newer run was chosen while this one loaded
  research = res;
  galaxy.update(res.galaxy);
  luck.update(res.galaxy);
  equity.update(eq, bench);
  drawdown.update(eq);
  correlation.update(corr);
  themes.update(res.themes);
  failures.update(res.failures);
  economics.update(res.economics);
  renderSignalList(res.galaxy.points);
  // The deflated Sharpe is run-level: when it is not defined, the note says why (DECISION-12).
  $("run-note").textContent = runNote(res.galaxy.points, res.galaxy.luck.n_trials, eq[0]?.dsr_reason);
  clearError();
}

async function choose(runId) {
  current = runId;
  $("demo-badge").hidden = !isDemoRun(runId);
  try {
    await load(runId);
  } catch (err) {
    if (current === runId) showError(loadFailure(`Could not load run ${runId}`, err));
  }
}

async function init() {
  try {
    // The status sets come from the server before the research counts are worded (DECISION-13).
    await loadContract();
    requireCharts();
    galaxy.init($("galaxy"));
    luck.init($("luck"));
    equity.init($("equity"));
    drawdown.init($("drawdown"));
    correlation.init($("correlation"));
    themes.init($("themes"));
    failures.init($("failures"));
    economics.init($("economics"));
  } catch (err) {
    showError(err.message);
    return;
  }

  // Listeners are attached before the first request, so a change made while a slow load is running is never lost.
  const select = $("run-select");
  select.addEventListener("change", () => choose(select.value));
  $("galaxy-color").addEventListener("change", (e) => {
    galaxy.setColorBy(e.target.value);
    if (research) galaxy.update(research.galaxy);
  });
  document.addEventListener("signal-select", (e) => {
    location.href = `/signal?id=${encodeURIComponent(e.detail)}`;
  });

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
