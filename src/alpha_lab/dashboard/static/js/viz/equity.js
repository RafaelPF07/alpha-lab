// Equity curves for the top signals by deflated Sharpe, shaded by period. The holdout is shaded only
// when the payload gives its dates (holdout_start and holdout_end). Dashed line is the equal-weight benchmark.
import { COLORS, baseAxis, baseText, describeChart, plural, track } from "./common.js";

let chart = null;
let container = null;
const PALETTE = ["#7aa2f7", "#9ece6a", "#ff9e64", "#bb9af7", "#7dcfff"];

export function init(el) {
  container = el;
  chart = track(echarts.init(el, null, { renderer: "canvas" }));
  return chart;
}

export function update(curves, benchmark = null) {
  if (!chart) return;
  if (!curves || !curves.length) {
    // Clear the previous run's curves, so the chart and the run note never disagree (FE-03).
    describeChart(container, "Equity curves: no validated signals in this run.");
    chart.clear();
    chart.setOption({ ...baseText(), title: { text: "no validated signals in this run", left: "center", top: "middle",
      textStyle: { color: COLORS.muted, fontSize: 13 } } });
    return;
  }
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
  if (holdoutLocked && first.holdout_start && first.holdout_end) {
    areas.push([{ xAxis: first.holdout_start, name: "holdout: locked", itemStyle: { color: "rgba(86,95,137,0.25)" },
      label: { color: COLORS.muted, position: "insideTop" } }, { xAxis: first.holdout_end }]);
  }
  series[0].markArea = { silent: true, data: areas };

  const ends = curves.map((c) => {
    const last = c.equity?.at(-1);
    return `${c.name} ${Number.isFinite(last) ? last.toFixed(2) : "n/a"}`;
  });
  describeChart(container, `Equity curves for ${plural(curves.length, "validated signal")}, growth of 1.0 at the end: ${ends.join(", ")}.`);
  chart.setOption({
    ...baseText(),
    // Room for a two-row legend above the plot, so the legend does not sit on the period labels.
    grid: { left: 60, right: 20, top: 64, bottom: 40 },
    legend: { textStyle: { color: COLORS.muted }, top: 0 },
    tooltip: { trigger: "axis" },
    xAxis: { type: "time", ...baseAxis() },
    yAxis: { type: "value", name: "growth of 1.0", ...baseAxis() },
    series,
  }, true);
}
