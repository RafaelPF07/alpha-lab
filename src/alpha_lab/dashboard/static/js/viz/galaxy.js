// Signal galaxy: every validated signal, train Sharpe against validation Sharpe. Most points should
// fall below the y = x line, which is the visual evidence of overfitting decay.
import { COLORS, baseAxis, baseText, track } from "./common.js";

let chart = null;
let colorBy = "theme";

export function init(container) {
  chart = track(echarts.init(container, null, { renderer: "canvas" }));
  chart.on("click", (p) => {
    if (p.data?.attempt_id) {
      container.dispatchEvent(new CustomEvent("signal-select", { detail: p.data.attempt_id, bubbles: true }));
    }
  });
  return chart;
}

export function setColorBy(mode) { colorBy = mode; }

const PALETTE = ["#7aa2f7", "#9ece6a", "#ff9e64", "#f7768e", "#bb9af7", "#7dcfff", "#e0af68", "#73daca"];

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
  const all = points.flatMap((p) => [p.train_sharpe, p.validation_sharpe]);
  const lo = all.length ? Math.min(...all) : -1;
  const hi = all.length ? Math.max(...all) : 1;
  series.push({
    name: "y = x",
    type: "line",
    data: [[lo, lo], [hi, hi]],
    symbol: "none",
    lineStyle: { color: COLORS.muted, type: "dashed" },
  });
  chart.setOption({
    ...baseText(),
    grid: { left: 70, right: 20, top: 56, bottom: 44 },
    legend: { type: "scroll", textStyle: { color: COLORS.muted }, top: 0 },
    tooltip: {
      formatter: (p) => {
        const d = p.data || {};
        if (!d.name) return p.seriesName;
        return `${d.name}<br>train ${d.value[0].toFixed(3)} · validation ${d.value[1].toFixed(3)}<br>deflated ${Number(d.deflated).toFixed(3)}`;
      },
    },
    xAxis: { type: "value", name: "train Sharpe (annualised)", nameLocation: "middle", nameGap: 26, ...baseAxis() },
    yAxis: { type: "value", name: "validation Sharpe (annualised)", ...baseAxis() },
    series,
  }, true);
}
