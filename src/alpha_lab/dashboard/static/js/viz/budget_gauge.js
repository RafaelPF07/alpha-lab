// Budget gauge: spent against the run's budget, with a colour that warns as it fills.
// With no budget on record it says so, instead of showing a made-up limit (FE-15).
import { COLORS, baseText, describeChart, track } from "./common.js";

let chart = null;
let container = null;

export function init(el) {
  container = el;
  chart = track(echarts.init(el, null, { renderer: "canvas" }));
  return chart;
}

export function update(snapshot) {
  if (!chart) return;
  const spent = Number(snapshot.spent) || 0;
  const limit = Number(snapshot.budget);
  const known = snapshot.budget != null && Number.isFinite(limit) && limit > 0;
  const pct = known ? Math.min(100, (100 * spent) / limit) : 0;
  const spentText = `$${spent.toFixed(3)}`;
  describeChart(container, known
    ? `Budget: ${spentText} of $${limit.toFixed(2)} spent, ${pct.toFixed(0)} percent.`
    : `Budget: ${spentText} spent. No budget is on record for this run.`);
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
        formatter: () => (known ? `${spentText}\nof $${limit.toFixed(2)}` : `${spentText}\nbudget unknown`),
        color: COLORS.text,
        fontSize: 16,
        offsetCenter: [0, "10%"],
      },
      data: [{ value: pct }],
    }],
  }, true);
}
