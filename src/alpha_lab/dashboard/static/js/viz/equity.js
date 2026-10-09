// Equity curves for the top signals by deflated Sharpe, shaded by period. The holdout stays grey and
// labelled "locked" until its official run has happened. Dashed line is the equal-weight benchmark.
import { COLORS, baseAxis, baseText, track } from "./common.js";

let chart = null;
const PALETTE = ["#7aa2f7", "#9ece6a", "#ff9e64", "#bb9af7", "#7dcfff"];

export function init(container) {
  chart = track(echarts.init(container, null, { renderer: "canvas" }));
  return chart;
}

export function update(curves, benchmark = null) {
  if (!chart || !curves.length) return;
  const first = curves[0];
  const trainEnd = first.train_end;
  const valEnd = first.validation_end;
  const series = curves.map((c, i) => ({
    name: c.name,
    type: "line",
    showSymbol: false,
    data: c.dates.map((d, j) => [d, c.equity[j]]),
    lineStyle: { width: 2, color: PALETTE[i % PALETTE.length] },
  }));
  if (benchmark) {
    series.push({ name: "equal weight", type: "line", showSymbol: false,
      data: benchmark.dates.map((d, j) => [d, benchmark.equity[j]]),
      lineStyle: { type: "dashed", color: COLORS.muted, width: 1.5 } });
  }
  const areas = [];
  if (trainEnd) areas.push([{ xAxis: first.dates[0], name: "train", itemStyle: { color: "rgba(122,162,247,0.06)" } }, { xAxis: trainEnd }]);
  if (valEnd && trainEnd) areas.push([{ xAxis: trainEnd, name: "validation", itemStyle: { color: "rgba(158,206,106,0.07)" } }, { xAxis: valEnd }]);
  const holdoutLocked = curves.some((c) => c.holdout !== "complete");
  if (holdoutLocked && valEnd) {
    const last = first.dates[first.dates.length - 1];
    areas.push([{ xAxis: valEnd, name: "holdout: locked", itemStyle: { color: "rgba(86,95,137,0.25)" },
      label: { color: COLORS.locked, position: "insideTop" } }, { xAxis: last }]);
  }
  series[0].markArea = { silent: true, data: areas };
  chart.setOption({
    ...baseText(),
    grid: { left: 60, right: 20, top: 40, bottom: 40 },
    legend: { textStyle: { color: COLORS.muted }, top: 0 },
    tooltip: { trigger: "axis" },
    xAxis: { type: "time", ...baseAxis() },
    yAxis: { type: "value", name: "growth of 1.0", ...baseAxis() },
    series,
  }, true);
}
