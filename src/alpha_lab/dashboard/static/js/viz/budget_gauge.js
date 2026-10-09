// Budget gauge: spent against the run's budget, with a colour that warns as it fills.
import { COLORS, baseText, track } from "./common.js";

let chart = null;

export function init(container) {
  chart = track(echarts.init(container, null, { renderer: "canvas" }));
  return chart;
}

export function update(snapshot) {
  if (!chart) return;
  const budget = snapshot.budget || 1;
  const pct = Math.min(100, (100 * snapshot.spent) / budget);
  chart.setOption({
    ...baseText(),
    series: [{
      type: "gauge",
      min: 0,
      max: 100,
      progress: { show: true, width: 14 },
      axisLine: { lineStyle: { width: 14, color: [[0.7, COLORS.status.passed()], [0.9, COLORS.status.budget()], [1, COLORS.status.failed()]] } },
      pointer: { show: false },
      axisTick: { show: false },
      splitLine: { show: false },
      axisLabel: { show: false },
      detail: {
        valueAnimation: true,
        formatter: (v) => `$${snapshot.spent.toFixed(3)}\nof $${budget.toFixed(2)}`,
        color: COLORS.text,
        fontSize: 16,
        offsetCenter: [0, "10%"],
      },
      data: [{ value: pct }],
    }],
  }, true);
}
