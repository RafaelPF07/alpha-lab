// Correlation map: daily-return correlation between validation survivors. Clusters above 0.7 are
// probably the same signal in disguise.
import { COLORS, baseText, describeChart, track } from "./common.js";
import { esc } from "../core/escape.js";

let chart = null;
let container = null;

const shorten = (name) => (String(name).length > 24 ? `${String(name).slice(0, 23)}…` : String(name));

export function init(el) {
  container = el;
  chart = track(echarts.init(el, null, { renderer: "canvas" }));
  return chart;
}

export function update(corr) {
  if (!chart) return;
  if (!corr || !corr.names || corr.names.length < 2) {
    describeChart(container, "Correlation map: fewer than two validation survivors.");
    chart.clear();
    chart.setOption({ ...baseText(), title: { text: "fewer than two validation survivors", textStyle: { color: COLORS.muted, fontSize: 13 }, left: "center", top: "middle" } });
    return;
  }
  const data = [];
  corr.names.forEach((a, i) => corr.names.forEach((b, j) => data.push([i, j, Number(corr.matrix[i][j]).toFixed(3)])));

  let best = null;
  corr.names.forEach((a, i) => corr.names.forEach((b, j) => {
    const v = Number(corr.matrix[i][j]);
    if (i < j && Number.isFinite(v) && (best === null || v > best.v)) best = { v, a, b };
  }));
  describeChart(container, best
    ? `Correlation map of ${corr.names.length} survivors. Highest pair: ${best.a} and ${best.b}, correlation ${best.v.toFixed(2)}.`
    : `Correlation map of ${corr.names.length} survivors.`);

  chart.setOption({
    ...baseText(),
    grid: { left: 150, right: 60, top: 20, bottom: 100 },
    tooltip: { formatter: (p) => `${esc(corr.names[p.data[0]])} vs ${esc(corr.names[p.data[1]])}: ${esc(p.data[2])}` },
    // Long signal names are shortened on the axes; the tooltip above shows the full names.
    xAxis: { type: "category", data: corr.names, axisLabel: { color: COLORS.muted, rotate: 40, formatter: shorten } },
    yAxis: { type: "category", data: corr.names, axisLabel: { color: COLORS.muted, formatter: shorten } },
    visualMap: { min: -1, max: 1, calculable: true, orient: "vertical", right: 0, top: "middle",
      inRange: { color: ["#f7768e", COLORS.panel2 || "#1a2130", "#9ece6a"] }, textStyle: { color: COLORS.muted } },
    series: [{ type: "heatmap", data, label: { show: corr.names.length <= 8, color: COLORS.text, formatter: (p) => p.data[2] } }],
  }, true);
}
