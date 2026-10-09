// Signal galaxy: every validated signal, train Sharpe against validation Sharpe. Most points should
// fall below the y = x line, which is the visual evidence of overfitting decay.
import { COLORS, baseAxis, baseText, describeChart, plural, track } from "./common.js";
import { esc } from "../core/escape.js";

let chart = null;
let container = null;
let colorBy = "theme";

export function init(el) {
  container = el;
  chart = track(echarts.init(el, null, { renderer: "canvas" }));
  chart.on("click", (p) => {
    if (p.data?.attempt_id) {
      el.dispatchEvent(new CustomEvent("signal-select", { detail: p.data.attempt_id, bubbles: true }));
    }
  });
  return chart;
}

export function setColorBy(mode) { colorBy = mode; }

const PALETTE = ["#7aa2f7", "#9ece6a", "#ff9e64", "#f7768e", "#bb9af7", "#7dcfff", "#e0af68", "#73daca"];

// A missing number shows as n/a. Number(null) is 0, which would print a false 0.000.
const fmt3 = (x) => (Number.isFinite(x) ? x.toFixed(3) : "n/a");

export function update(galaxy) {
  if (!chart) return;
  const points = galaxy.points.filter((p) => p.train_sharpe != null && p.validation_sharpe != null);
  const groups = new Map();
  for (const p of points) {
    const key = colorBy === "revisions" ? `revisions ${p.revisions}` : p.theme;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(p);
  }
  const series = [...groups.entries()].map(([name, pts], i) => ({
    name,
    type: "scatter",
    data: pts.map((p) => ({
      value: [p.train_sharpe, p.validation_sharpe],
      symbolSize: Math.max(6, Math.min(30, 60 * (p.turnover || 0))),
      attempt_id: p.attempt_id,
      name: p.name,
      deflated: p.deflated_sharpe,
    })),
    itemStyle: { color: PALETTE[i % PALETTE.length], opacity: 0.85 },
  }));
  // Loops, not spread arguments: a spread of many values can exceed the engine's argument limit.
  let lo = Infinity;
  let hi = -Infinity;
  for (const p of points) {
    lo = Math.min(lo, p.train_sharpe, p.validation_sharpe);
    hi = Math.max(hi, p.train_sharpe, p.validation_sharpe);
  }
  if (!points.length) { lo = -1; hi = 1; }
  series.push({
    name: "y = x",
    type: "line",
    data: [[lo, lo], [hi, hi]],
    symbol: "none",
    lineStyle: { color: COLORS.muted, type: "dashed" },
  });

  const below = points.filter((p) => p.validation_sharpe < p.train_sharpe).length;
  describeChart(container, `Signal galaxy: ${plural(points.length, "validated signal")}, ${below} with lower validation `
    + "Sharpe than train Sharpe. Each one can be opened from the list below the chart.");

  chart.setOption({
    ...baseText(),
    grid: { left: 70, right: 20, top: 56, bottom: 44 },
    legend: { type: "scroll", textStyle: { color: COLORS.muted }, top: 0 },
    tooltip: {
      formatter: (p) => {
        const d = p.data || {};
        if (!d.name) return esc(p.seriesName);
        const deflated = d.deflated == null ? null : Number(d.deflated);
        return `${esc(d.name)}<br>train ${fmt3(d.value[0])} · validation ${fmt3(d.value[1])}<br>deflated ${fmt3(deflated)}`;
      },
    },
    xAxis: { type: "value", name: "train Sharpe (annualised)", nameLocation: "middle", nameGap: 26, ...baseAxis() },
    yAxis: { type: "value", name: "validation Sharpe (annualised)", ...baseAxis() },
    series,
  }, true);
}
