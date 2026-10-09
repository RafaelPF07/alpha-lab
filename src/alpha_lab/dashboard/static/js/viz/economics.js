// Model economics: cost and success rate by agent and model. Future runs with other models line up beside these.
import { COLORS, baseAxis, baseText, track } from "./common.js";

let chart = null;

export function init(container) {
  chart = track(echarts.init(container, null, { renderer: "canvas" }));
  return chart;
}

export function update(rows) {
  if (!chart) return;
  const labels = rows.map((r) => `${r.agent} · ${r.model}`);
  chart.setOption({
    ...baseText(),
    grid: { left: 60, right: 60, top: 40, bottom: 80 },
    legend: { textStyle: { color: COLORS.muted }, top: 0 },
    tooltip: { trigger: "axis" },
    xAxis: { type: "category", data: labels, axisLabel: { color: COLORS.muted, rotate: 20 } },
    yAxis: [
      { type: "value", name: "USD", ...baseAxis() },
      { type: "value", name: "success rate", min: 0, max: 1, ...baseAxis() },
    ],
    series: [
      { name: "cost (USD)", type: "bar", data: rows.map((r) => r.cost_usd), itemStyle: { color: "#ff9e64" } },
      { name: "success rate", type: "line", yAxisIndex: 1, data: rows.map((r) => r.success_rate), itemStyle: { color: "#9ece6a" } },
    ],
  }, true);
}
