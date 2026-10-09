// Cost by agent over time, as a stacked area. Shows which agent is driving the spend.
import { COLORS, agentColor, baseAxis, baseText, track } from "./common.js";

let chart = null;
const BINS = 40;

export function init(container) {
  chart = track(echarts.init(container, null, { renderer: "canvas" }));
  return chart;
}

export function update(snapshot) {
  if (!chart) return;
  const agents = Object.keys(snapshot.agentSpend);
  const all = agents.flatMap((a) => snapshot.agentSpend[a].map((p) => Date.parse(p.t)));
  if (!all.length) { chart.clear(); return; }
  const lo = Math.min(...all);
  const hi = Math.max(...all, lo + 1);
  const width = (hi - lo) / BINS;
  const xs = Array.from({ length: BINS }, (_, i) => lo + (i + 0.5) * width);
  const series = agents.map((agent) => {
    const bins = new Array(BINS).fill(0);
    for (const p of snapshot.agentSpend[agent]) {
      const i = Math.min(BINS - 1, Math.floor((Date.parse(p.t) - lo) / width));
      bins[i] += p.cost;
    }
    let running = 0;
    const cumulative = bins.map((v) => (running += v));
    return {
      name: agent,
      type: "line",
      stack: "cost",
      areaStyle: { opacity: 0.55 },
      showSymbol: false,
      itemStyle: { color: agentColor(agent) },
      data: xs.map((x, i) => [x, cumulative[i]]),
    };
  });
  chart.setOption({
    ...baseText(),
    animation: false,
    grid: { left: 60, right: 20, top: 30, bottom: 30 },
    legend: { textStyle: { color: COLORS.muted }, top: 0 },
    tooltip: { trigger: "axis" },
    xAxis: { type: "time", ...baseAxis() },
    yAxis: { type: "value", name: "USD", nameTextStyle: { color: COLORS.muted }, ...baseAxis() },
    series,
  }, true);
}
