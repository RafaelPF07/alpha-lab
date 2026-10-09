// Correlation map: daily-return correlation between validation survivors. Clusters above 0.7 are
// probably the same signal in disguise.
import { COLORS, baseText, track } from "./common.js";

let chart = null;

export function init(container) {
  chart = track(echarts.init(container, null, { renderer: "canvas" }));
  return chart;
}

export function update(corr) {
  if (!chart) return;
  if (!corr || !corr.names || corr.names.length < 2) {
    chart.clear();
    chart.setOption({ ...baseText(), title: { text: "fewer than two validation survivors", textStyle: { color: COLORS.muted, fontSize: 13 }, left: "center", top: "middle" } });
    return;
  }
  const data = [];
  corr.names.forEach((a, i) => corr.names.forEach((b, j) => data.push([i, j, Number(corr.matrix[i][j]).toFixed(3)])));
  chart.setOption({
    ...baseText(),
    grid: { left: 140, right: 60, top: 20, bottom: 100 },
    tooltip: { formatter: (p) => `${corr.names[p.data[0]]} vs ${corr.names[p.data[1]]}: ${p.data[2]}` },
    xAxis: { type: "category", data: corr.names, axisLabel: { color: COLORS.muted, rotate: 40 } },
    yAxis: { type: "category", data: corr.names, axisLabel: { color: COLORS.muted } },
    visualMap: { min: -1, max: 1, calculable: true, orient: "vertical", right: 0, top: "middle",
      inRange: { color: ["#f7768e", "#1a2130", "#9ece6a"] }, textStyle: { color: COLORS.muted } },
    series: [{ type: "heatmap", data, label: { show: corr.names.length <= 8, color: COLORS.text, formatter: (p) => p.data[2] } }],
  }, true);
}
