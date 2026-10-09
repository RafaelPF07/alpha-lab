// Headline counters. Numbers ease to their new value so changes are visible on a screen recording.
import { fmtUsd } from "./common.js";

const DEFS = [
  { key: "runningPipelines", label: "pipelines running", fmt: (v) => Math.round(v) },
  { key: "donePipelines", label: "pipelines done", fmt: (v) => Math.round(v) },
  { key: "ideas", label: "ideas generated", fmt: (v) => Math.round(v) },
  { key: "passedPipelines", label: "train passes", fmt: (v) => Math.round(v) },
  { key: "validationPasses", label: "validation passes", fmt: (v) => Math.round(v) },
  { key: "spent", label: "spend", fmt: (v) => fmtUsd(v) },
  { key: "spendRate", label: "spend rate per min", fmt: (v) => fmtUsd(v) },
];

let refs = null;
const shown = {};

export function init(container) {
  container.innerHTML = "";
  container.style.display = "grid";
  container.style.gridTemplateColumns = "repeat(auto-fit, minmax(150px, 1fr))";
  container.style.gap = "12px";
  refs = {};
  for (const def of DEFS) {
    const cell = document.createElement("div");
    cell.innerHTML = `<div class="counter" data-k="${def.key}">0</div><div class="counter-label">${def.label}</div>`;
    container.appendChild(cell);
    refs[def.key] = { el: cell.querySelector(".counter"), fmt: def.fmt };
    shown[def.key] = 0;
  }
}

function ease(key, target) {
  const start = shown[key] ?? 0;
  const t0 = performance.now();
  const step = (now) => {
    const k = Math.min(1, (now - t0) / 350);
    const value = start + (target - start) * (1 - Math.pow(1 - k, 3));
    shown[key] = value;
    refs[key].el.textContent = refs[key].fmt(value);
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

export function update(snapshot) {
  if (!refs) return;
  for (const def of DEFS) {
    const target = Number(snapshot[def.key] ?? 0);
    if (Math.abs(target - (shown[def.key] ?? 0)) > 1e-9) ease(def.key, target);
  }
}
