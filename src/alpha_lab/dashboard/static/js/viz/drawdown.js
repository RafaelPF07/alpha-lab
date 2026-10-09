// Underwater plot: how far each top signal sits below its own peak, over the same dates as the equity curves.
import { baseAxis, baseText, track } from "./common.js";

let chart = null;
const PALETTE = ["#7aa2f7", "#9ece6a", "#ff9e64", "#bb9af7", "#7dcfff"];

export function init(container) {
  chart = track(echarts.init(container, null, { renderer: "canvas" }));
  return chart;
}

export function update(curves) {
  if (!chart || !curves.length) return;
  chart.setOption({
    ...baseText(),
    grid: { left: 60, right: 20, top: 40, bottom: 40 },
    legend: { textStyle: { color: "#8a94a8" }, top: 0 },
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
