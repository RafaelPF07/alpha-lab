// Context-size monitor: input tokens for every LLM call over time. Shows whether prompts are creeping up.
import { COLORS, agentColor, baseAxis, baseText, describeChart, plural, track } from "./common.js";

let chart = null;
let container = null;

export function init(el, { pricingThreshold = 100000, contextLimit = 90000 } = {}) {
  container = el;
  chart = track(echarts.init(el, null, { renderer: "canvas" }));
  chart.__thresholds = { pricingThreshold, contextLimit };
  return chart;
}

export function update(snapshot) {
  if (!chart) return;
  const { pricingThreshold, contextLimit } = chart.__thresholds;
  const agents = [...new Set(snapshot.context.map((c) => c.agent))];
  let largest = 0;
  for (const c of snapshot.context) if (c.tokens > largest) largest = c.tokens;
  describeChart(container, `Context size: ${plural(snapshot.context.length, "LLM call")}, largest ${largest} input tokens. `
    + `Marked: the pricing tier at ${pricingThreshold} tokens and the context limit at ${contextLimit}.`);
  const series = agents.map((agent) => ({
    name: agent,
    type: "scatter",
    symbolSize: 7,
    itemStyle: { color: agentColor(agent), opacity: 0.85 },
    data: snapshot.context.filter((c) => c.agent === agent).map((c) => [Date.parse(c.t), c.tokens]),
  }));
  series.push({
    name: "limits",
    type: "line",
    data: [],
    markLine: {
      symbol: "none",
      label: { color: COLORS.muted },
      data: [
        { yAxis: pricingThreshold, name: "100K pricing tier", lineStyle: { color: COLORS.status.budget(), type: "solid" } },
        { yAxis: contextLimit, name: "context limit", lineStyle: { color: COLORS.status.failed(), type: "dashed" } },
      ],
    },
  });
  chart.setOption({
    ...baseText(),
    animation: false,
    grid: { left: 70, right: 20, top: 30, bottom: 30 },
    legend: { textStyle: { color: COLORS.muted }, top: 0, data: agents },
    tooltip: { trigger: "item" },
    xAxis: { type: "time", ...baseAxis() },
    yAxis: { type: "value", name: "input tokens", min: 0, max: Math.ceil(pricingThreshold * 1.1),
      nameTextStyle: { color: COLORS.muted }, ...baseAxis() },
    series,
  }, true);
}
