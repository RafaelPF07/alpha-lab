// Underwater plot: how far each top signal sits below its own peak, over the same dates as the equity curves.
import { COLORS, baseAxis, baseText, describeChart, track } from "./common.js";
import { reachedText } from "../core/research_text.js";

let chart = null;
let container = null;
const PALETTE = ["#7aa2f7", "#9ece6a", "#ff9e64", "#bb9af7", "#7dcfff"];

export function init(el) {
  container = el;
  chart = track(echarts.init(el, null, { renderer: "canvas" }));
  return chart;
}

export function update(curves) {
  if (!chart) return;
  if (!curves || !curves.length) {
    // Clear the previous run's curves, so the chart and the run note never disagree (FE-03).
    describeChart(container, "Drawdown: no signal reached validation in this run.");
    chart.clear();
    chart.setOption({ ...baseText(), title: { text: "no signal reached validation in this run", left: "center", top: "middle",
      textStyle: { color: COLORS.muted, fontSize: 13 } } });
    return;
  }
  describeChart(container, `Drawdown for ${reachedText(curves.length)}, shown from each signal's own peak.`);
  chart.setOption({
    ...baseText(),
    grid: { left: 60, right: 20, top: 40, bottom: 40 },
    legend: { textStyle: { color: COLORS.muted }, top: 0 },
    tooltip: { trigger: "axis", valueFormatter: (v) => `${(v * 100).toFixed(2)}%` },
    xAxis: { type: "time", ...baseAxis() },
    yAxis: { type: "value", name: "drawdown from peak", axisLabel: { formatter: (v) => `${(v * 100).toFixed(0)}%` }, ...baseAxis() },
    series: curves.map((c, i) => ({
      name: c.name,
      type: "line",
      showSymbol: false,
      areaStyle: { opacity: 0.12 },
      lineStyle: { width: 1.5, color: PALETTE[i % PALETTE.length] },
      itemStyle: { color: PALETTE[i % PALETTE.length] },
      data: c.dates.map((d, j) => [d, c.drawdown[j]]),
    })),
  }, true);
}
