// Cumulative spend over time, with a dashed projection to the moment the budget runs out.
import { COLORS, baseAxis, baseText, track } from "./common.js";

let chart = null;

export function init(container) {
  chart = track(echarts.init(container, null, { renderer: "canvas" }));
  return chart;
}

export function update(snapshot) {
  if (!chart) return;
  const data = snapshot.spendSeries.map((p) => [Date.parse(p.t), p.spent]);
  const marks = [];
  if (snapshot.budget) {
    marks.push({ yAxis: snapshot.budget, name: "budget", lineStyle: { color: COLORS.status.budget(), type: "solid" } });
  }
  const projected = [];
  if (snapshot.projectedExhaustion && data.length) {
    const last = data[data.length - 1];
    projected.push(last, [Date.parse(snapshot.projectedExhaustion), snapshot.budget]);
  }
  chart.setOption({
    ...baseText(),
    animation: false,
    grid: { left: 60, right: 24, top: 20, bottom: 30 },
    tooltip: { trigger: "axis" },
    xAxis: { type: "time", ...baseAxis() },
    yAxis: { type: "value", name: "USD", nameTextStyle: { color: COLORS.muted }, ...baseAxis() },
    series: [
      {
        name: "spent",
        type: "line",
        showSymbol: false,
        data,
        lineStyle: { width: 2, color: COLORS.status.passed() },
        markLine: { symbol: "none", data: marks, label: { color: COLORS.muted } },
      },
      {
        name: "projected",
        type: "line",
        showSymbol: false,
        data: projected,
        lineStyle: { type: "dashed", color: COLORS.status.budget(), width: 2 },
      },
    ],
  }, true);
}
