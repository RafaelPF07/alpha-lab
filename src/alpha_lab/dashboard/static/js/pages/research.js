// Research views: post-run analysis of a finished run. Everything comes from the JSON endpoints.
import { fetchJson, LatestRequest } from "../core/api.js";
import { esc } from "../core/escape.js";
import { isDemoRun } from "../core/runs.js";
import { showError, clearError, requireCharts } from "../core/page.js";
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

// A plain, keyboard-reachable list of the validated signals, so every point can be opened without the chart.
function renderSignalList(points) {
  $("signal-count").textContent = String(points.length);
  $("signal-list").innerHTML = points.length
    ? points.map((p) => `<li><a href="/signal?id=${encodeURIComponent(p.attempt_id)}">${esc(p.name || p.attempt_id)}</a>`
      + ` <span class="note">train ${fmt2(p.train_sharpe)} · validation ${fmt2(p.validation_sharpe)}</span></li>`).join("")
    : "<li>No validated signals in this run.</li>";
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
  $("run-note").textContent = `${res.galaxy.points.length} validated signal(s) · ${res.galaxy.luck.n_trials} trials in the ledger`;
  clearError();
}

async function choose(runId) {
  current = runId;
  $("demo-badge").hidden = !isDemoRun(runId);
  try {
    await load(runId);
  } catch (err) {
    if (current === runId) showError(`Could not load run ${runId}. ${err.message}`);
  }
}

async function init() {
  try {
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
      opt.textContent = `${r.run_id} · ${r.status}`;
      select.appendChild(opt);
    }
    const chosen = params.get("run") || (runs[0] && runs[0].run_id);
    if (chosen) {
      select.value = chosen;
      await choose(chosen);
    }
  } catch (err) {
    showError(`Could not load the run list. ${err.message}`);
  }
}

init();
