// Luck distribution: train Sharpe of every trial, with the expected maximum from pure noise marked.
// Survivors are marked on the same axis, so you can see how much of the top tail is luck.
import { COLORS, baseAxis, baseText, describeChart, plural, track } from "./common.js";

let chart = null;
let container = null;

export function init(el) {
  container = el;
  chart = track(echarts.init(el, null, { renderer: "canvas" }));
  return chart;
}

export function update(galaxy) {
  if (!chart) return;
  const sharpes = galaxy.luck.train_sharpes;
  const expected = galaxy.luck.expected_max;
  const trials = galaxy.luck.n_trials ?? "?";
  const survivors = galaxy.points.filter((p) => p.passed && p.train_sharpe != null).map((p) => p.train_sharpe);
  // Loops, not spread arguments: a spread of a large array can exceed the engine's argument limit (FE-20).
  let lo = 0;
  let hi = Math.max(0, expected ?? 0);
  for (const s of sharpes) { if (s < lo) lo = s; if (s > hi) hi = s; }
  for (const s of survivors) { if (s < lo) lo = s; if (s > hi) hi = s; }
  lo -= 0.1;
  hi += 0.1;
  const bins = 24;
  const width = (hi - lo) / bins || 1;
  const counts = new Array(bins).fill(0);
  const indexOf = (x) => Math.min(bins - 1, Math.max(0, Math.floor((x - lo) / width)));
  for (const s of sharpes) counts[indexOf(s)] += 1;
  const labels = counts.map((_, i) => (lo + (i + 0.5) * width).toFixed(2));

  describeChart(container, `Luck distribution: ${plural(sharpes.length, "trial")}, `
    + `${plural(survivors.length, "survivor")}. Expected best from noise with ${trials} trials: `
    + `${expected == null ? "n/a" : expected.toFixed(2)}.`);
  chart.setOption({
    ...baseText(),
    grid: { left: 60, right: 20, top: 40, bottom: 50 },
    tooltip: { trigger: "axis" },
    xAxis: { type: "category", data: labels, name: "train Sharpe (annualised)", nameLocation: "middle", nameGap: 32,
      axisLabel: { color: COLORS.muted, interval: 3 }, ...baseAxis() },
    yAxis: { type: "value", name: "trials", nameTextStyle: { color: COLORS.muted }, ...baseAxis() },
    series: [
      {
        name: "trials",
        type: "bar",
        data: counts,
        barCategoryGap: "10%",
        itemStyle: { color: "rgba(122,162,247,0.6)" },
        markLine: {
          symbol: "none",
          // Inside the plot, to the left of the line, so a line near the right edge does not clip its label.
          label: { color: COLORS.muted, position: "insideEndTop", rotate: 0, formatter: (p) => p.name },
          data: expected == null ? [] : [{
            xAxis: indexOf(expected),
            name: `noise max (N=${trials}): ${expected.toFixed(2)}`,
            lineStyle: { color: COLORS.status.failed(), type: "dashed", width: 2 },
          }],
        },
        markPoint: {
          symbol: "pin",
          symbolSize: 26,
          // The pins are the survivors. The note on the research page explains them, so the pin carries no text.
          label: { show: false },
          itemStyle: { color: COLORS.status.passed() },
          data: survivors.map((s) => ({ coord: [indexOf(s), counts[indexOf(s)]], value: "survivor" })),
        },
      },
    ],
  }, true);
}
