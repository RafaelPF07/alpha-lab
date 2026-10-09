// Cost by agent over time, as a stacked area. Shows which agent is driving the spend.
import { COLORS, agentColor, baseAxis, baseText, describeChart, plural, track } from "./common.js";

let chart = null;
let container = null;
const BINS = 40;

export function init(el) {
  container = el;
  chart = track(echarts.init(el, null, { renderer: "canvas" }));
  return chart;
}

export function update(snapshot) {
  if (!chart) return;
  const agents = Object.keys(snapshot.agentSpend);
  // Loops, not spread arguments: a spread of many calls can exceed the engine's argument limit (FE-20).
  let lo = Infinity;
  let hi = -Infinity;
  let calls = 0;
  for (const agent of agents) {
    for (const p of snapshot.agentSpend[agent]) {
      const t = Date.parse(p.t);
      calls += 1;
      if (t < lo) lo = t;
      if (t > hi) hi = t;
    }
  }
  if (!Number.isFinite(lo)) { describeChart(container, "Cost by agent: no calls yet."); chart.clear(); return; }
  hi = Math.max(hi, lo + 1);
  const width = (hi - lo) / BINS;
  const xs = Array.from({ length: BINS }, (_, i) => lo + (i + 0.5) * width);
  const series = agents.map((agent) => {
    const bins = new Array(BINS).fill(0);
    for (const p of snapshot.agentSpend[agent]) {
      const i = Math.min(BINS - 1, Math.max(0, Math.floor((Date.parse(p.t) - lo) / width)));
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
  describeChart(container, `Cost by agent: ${plural(calls, "LLM call")} across ${plural(agents.length, "agent")}.`);
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
