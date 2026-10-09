// Failure anatomy: where candidates die, sized by how many attempts ended that way.
import { COLORS, baseText, describeChart, track } from "./common.js";
import { esc } from "../core/escape.js";

let chart = null;
let container = null;

export function init(el) {
  container = el;
  chart = track(echarts.init(el, null, { renderer: "canvas" }));
  return chart;
}

export function update(failures) {
  if (!chart) return;
  const data = failures.map((f) => {
    const status = String(f.status ?? "unknown");
    return {
      name: status,
      value: f.count,
      itemStyle: { color: status.includes("pass") ? COLORS.status.passed() : status.includes("rejected") ? COLORS.status.rejected() : COLORS.status.failed() },
    };
  });
  const total = failures.reduce((sum, f) => sum + (Number(f.count) || 0), 0);
  describeChart(container, `Failure anatomy: ${total} attempts ended early, in ${failures.length} kinds.`);
  chart.setOption({
    ...baseText(),
    tooltip: { formatter: (p) => `${esc(p.name)}: ${esc(p.value)} attempts` },
    series: [{ type: "treemap", roam: false, breadcrumb: { show: false }, data, label: { color: "#111", fontSize: 12 } }],
  }, true);
}
