// Failure anatomy: where candidates die, sized by how many attempts ended that way.
import { COLORS, baseText, track } from "./common.js";

let chart = null;

export function init(container) {
  chart = track(echarts.init(container, null, { renderer: "canvas" }));
  return chart;
}

export function update(failures) {
  if (!chart) return;
  const data = failures.map((f) => ({ name: f.status, value: f.count, itemStyle: { color: f.status.includes("pass") ? COLORS.status.passed() : f.status.includes("rejected") ? COLORS.status.rejected() : COLORS.status.failed() } }));
  chart.setOption({
    ...baseText(),
    tooltip: { formatter: (p) => `${p.name}: ${p.value} attempts` },
    series: [{ type: "treemap", roam: false, breadcrumb: { show: false }, data, label: { color: "#111", fontSize: 12 } }],
  }, true);
}
