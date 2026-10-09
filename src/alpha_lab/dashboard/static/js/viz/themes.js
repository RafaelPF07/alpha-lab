// Theme leaderboard: per theme, how many ideas it produced, how far they got, and median Sharpe ratios.
import { COLORS, baseAxis, baseText, describeChart, plural, track } from "./common.js";

let chart = null;
let container = null;

export function init(el) {
  container = el;
  chart = track(echarts.init(el, null, { renderer: "canvas" }));
  return chart;
}

export function update(themes) {
  if (!chart) return;
  const rows = themes.filter((t) => t.theme);
  describeChart(container, `Theme leaderboard: ${plural(rows.length, "theme")}, median train and validation Sharpe for each.`);
  chart.setOption({
    ...baseText(),
    grid: { left: 170, right: 30, top: 40, bottom: 30 },
    legend: { textStyle: { color: COLORS.muted }, top: 0 },
    tooltip: { trigger: "axis" },
    xAxis: { type: "value", name: "median Sharpe (annualised)", nameLocation: "middle", nameGap: 24, ...baseAxis() },
    yAxis: { type: "category", data: rows.map((t) => `${t.theme} (${t.ideas})`), ...baseAxis() },
    series: [
      { name: "median train SR", type: "bar", data: rows.map((t) => (Number.isFinite(t.median_train_sharpe) ? t.median_train_sharpe : null)), itemStyle: { color: "#7aa2f7" } },
      { name: "median validation SR", type: "bar", data: rows.map((t) => (Number.isFinite(t.median_validation_sharpe) ? t.median_validation_sharpe : null)), itemStyle: { color: "#9ece6a" } },
    ],
  }, true);
}
