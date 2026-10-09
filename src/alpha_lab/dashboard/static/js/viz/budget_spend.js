// Cumulative spend over time, with a dashed projection to the moment the budget runs out. The
// projection stops at most one run length past the run's end, so a slow spend rate cannot stretch the
// time axis over days (FE-15). Past that point the dashed line shows the spend rate, not the budget.
import { COLORS, baseAxis, baseText, describeChart, track } from "./common.js";

let chart = null;
let container = null;

export function init(el) {
  container = el;
  chart = track(echarts.init(el, null, { renderer: "canvas" }));
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
  if (snapshot.projectedExhaustion && snapshot.budget && data.length) {
    const last = data[data.length - 1];
    const start = snapshot.runStart ? Date.parse(snapshot.runStart) : last[0];
    const end = snapshot.runEnd ? Date.parse(snapshot.runEnd) : last[0];
    const horizon = end + Math.max(60000, end - start);
    const exhaustion = Date.parse(snapshot.projectedExhaustion);
    if (exhaustion <= horizon) {
      projected.push(last, [exhaustion, snapshot.budget]);
    } else {
      const minutes = Math.max(0, (horizon - last[0]) / 60000);
      projected.push(last, [horizon, last[1] + (snapshot.spendRate || 0) * minutes]);
    }
  }
  const spent = Number(snapshot.spent) || 0;
  describeChart(container, snapshot.budget
    ? `Cumulative spend: $${spent.toFixed(3)} of $${Number(snapshot.budget).toFixed(2)} budget.`
    : `Cumulative spend: $${spent.toFixed(3)}. No budget is on record.`);
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
