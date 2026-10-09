// Luck distribution: train Sharpe of every trial, with the expected maximum from pure noise marked.
// Survivors are marked on the same axis, so you can see how much of the top tail is luck.
import { COLORS, baseAxis, baseText, track } from "./common.js";

let chart = null;

export function init(container) {
  chart = track(echarts.init(container, null, { renderer: "canvas" }));
  return chart;
}

export function update(galaxy) {
  if (!chart) return;
  const sharpes = galaxy.luck.train_sharpes;
  const expected = galaxy.luck.expected_max;
  const survivors = galaxy.points.filter((p) => p.passed && p.train_sharpe != null).map((p) => p.train_sharpe);
  const bins = 24;
  const lo = Math.min(...sharpes, ...survivors, 0) - 0.1;
  const hi = Math.max(...sharpes, ...survivors, expected ?? 0, 0) + 0.1;
  const width = (hi - lo) / bins || 1;
  const counts = new Array(bins).fill(0);
  for (const s of sharpes) counts[Math.min(bins - 1, Math.max(0, Math.floor((s - lo) / width)))] += 1;
  const labels = counts.map((_, i) => (lo + (i + 0.5) * width).toFixed(2));
  const indexOf = (x) => Math.min(bins - 1, Math.max(0, Math.floor((x - lo) / width)));

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
          label: { color: COLORS.muted, formatter: (p) => p.name },
          data: expected == null ? [] : [{
            xAxis: indexOf(expected),
            name: `expected max from noise, N=${galaxy.luck.n_trials}: ${expected.toFixed(2)}`,
            lineStyle: { color: COLORS.status.failed(), type: "dashed", width: 2 },
          }],
        },
        markPoint: {
          symbol: "pin",
          symbolSize: 34,
          itemStyle: { color: COLORS.status.passed() },
          data: survivors.map((s) => ({ coord: [indexOf(s), counts[indexOf(s)]], value: "survivor" })),
        },
      },
    ],
  }, true);
}
