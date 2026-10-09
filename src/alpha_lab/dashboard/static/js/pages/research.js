// Research views: post-run analysis of a finished run. Everything comes from the JSON endpoints.
import { fetchJson } from "../core/api.js";
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

const charts = {
  galaxy: galaxy.init($("galaxy")),
  luck: luck.init($("luck")),
  equity: equity.init($("equity")),
  drawdown: drawdown.init($("drawdown")),
  correlation: correlation.init($("correlation")),
  themes: themes.init($("themes")),
  failures: failures.init($("failures")),
  economics: economics.init($("economics")),
};

let current = null;

async function load(runId) {
  const [research, eq, corr, bench, econ] = await Promise.all([
    fetchJson(`/api/runs/${encodeURIComponent(runId)}/research`),
    fetchJson(`/api/runs/${encodeURIComponent(runId)}/equity?top=5`),
    fetchJson(`/api/runs/${encodeURIComponent(runId)}/correlation`),
    fetchJson(`/api/runs/${encodeURIComponent(runId)}/benchmark?top=5`),
    fetchJson(`/api/runs/${encodeURIComponent(runId)}/research`).then((r) => r.economics),
  ]);
  galaxy.update(research.galaxy);
  luck.update(research.galaxy);
  equity.update(eq, bench);
  drawdown.update(eq);
  correlation.update(corr);
  themes.update(research.themes);
  failures.update(research.failures);
  economics.update(econ);
  $("run-note").textContent = `${research.galaxy.points.length} validated signal(s), ${research.galaxy.luck.n_trials} trials`;
}

async function init() {
  const runs = await fetchJson("/api/runs");
  const select = $("run-select");
  for (const r of runs) {
    const opt = document.createElement("option");
    opt.value = r.run_id;
    opt.textContent = `${r.run_id} · ${r.status}`;
    select.appendChild(opt);
  }
  current = params.get("run") || (runs[0] && runs[0].run_id);
  if (current) {
    select.value = current;
    $("demo-badge").hidden = current !== "demo";
    await load(current);
  }
  select.addEventListener("change", async () => {
    current = select.value;
    $("demo-badge").hidden = current !== "demo";
    await load(current);
  });
  $("galaxy-color").addEventListener("change", (e) => {
    galaxy.setColorBy(e.target.value);
    fetchJson(`/api/runs/${encodeURIComponent(current)}/research`).then((r) => galaxy.update(r.galaxy));
  });
  document.addEventListener("signal-select", (e) => {
    location.href = `/signal?id=${encodeURIComponent(e.detail)}`;
  });
}

init();
