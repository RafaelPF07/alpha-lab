// Pipeline swimlanes: one row per pipeline. LLM calls are coloured segments, checks and
// backtests are markers, so concurrency, bottlenecks and revisions are visible at a glance.
import { COLORS, agentColor, baseAxis, baseText, statusColor, track } from "./common.js";

let chart = null;
let autoscroll = true;

export function init(container) {
  chart = track(echarts.init(container, null, { renderer: "canvas" }));
  return chart;
}

export function setAutoscroll(on) { autoscroll = on; }

export function update(snapshot) {
  if (!chart) return;
  const lanes = snapshot.pipelines.slice(-50); // the most recent fifty, so the chart stays readable
  const now = Date.now();
  const segments = [];
  const markers = [];
  lanes.forEach((pl, row) => {
    const start = Date.parse(pl.started);
    const end = pl.finished ? Date.parse(pl.finished) : now;
    segments.push({ value: [row, start, end, "pipeline", pl.status, pl.theme || ""] });
    for (const s of pl.segments) {
      const t = Date.parse(s.t);
      segments.push({ value: [row, t - (s.latency || 0), t, s.agent, s.status, s.agent] });
    }
    for (const m of pl.markers) {
      markers.push({ value: [Date.parse(m.t), row], stage: m.stage });
    }
  });

  chart.setOption({
    ...baseText(),
    animation: false,
    grid: { left: 110, right: 20, top: 16, bottom: 36 },
    tooltip: { formatter: (p) => `${p.seriesName}: ${p.value?.[5] ?? ""}` },
    xAxis: { type: "time", ...baseAxis(), axisLabel: { color: COLORS.muted, formatter: (v) => new Date(v).toISOString().slice(11, 19) } },
    yAxis: {
      type: "category",
      data: lanes.map((pl) => `${pl.id.slice(0, 6)}${pl.theme ? " · " + pl.theme : ""}`),
      inverse: true,
      ...baseAxis(),
      axisLabel: { color: COLORS.muted, fontSize: 10 },
    },
    series: [
      {
        name: "segments",
        type: "custom",
        renderItem: (params, api) => {
          const row = api.value(0);
          const start = api.coord([api.value(1), row]);
          const end = api.coord([api.value(2), row]);
          const kind = api.value(3);
          const color = kind === "pipeline" ? "rgba(128,140,170,0.25)" : agentColor(kind);
          const height = kind === "pipeline" ? 12 : 8;
          return {
            type: "rect",
            shape: { x: start[0], y: start[1] - height / 2, width: Math.max(2, end[0] - start[0]), height },
            style: { fill: color, opacity: kind === "pipeline" ? 1 : 0.9 },
          };
        },
        encode: { x: [1, 2], y: 0 },
        data: segments,
      },
      {
        name: "stage",
        type: "scatter",
        symbolSize: 7,
        itemStyle: { color: COLORS.text },
        data: markers.map((m) => m.value),
      },
    ],
    dataZoom: autoscroll ? [{ type: "inside", xAxisIndex: 0, start: 0, end: 100 }] : [],
  }, true);
}
