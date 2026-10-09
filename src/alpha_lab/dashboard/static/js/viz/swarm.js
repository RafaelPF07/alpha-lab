// Agent swarm graph. An orchestrator node connects to one cluster per pipeline. Each cluster holds
// the three agents and the three harness stages. Node colour is status, node size is tokens used.
import { COLORS, track } from "./common.js";

let chart = null;

export function init(container) {
  chart = track(echarts.init(container, null, { renderer: "canvas" }));
  chart.on("click", (params) => {
    if (params.dataType === "node" && params.data?.pipelineId) {
      container.dispatchEvent(new CustomEvent("pipeline-select", { detail: params.data.pipelineId, bubbles: true }));
    }
  });
  return chart;
}

function nodeColor(status) {
  if (status === "running") return COLORS.status.running();
  if (status === "validation_pass" || status === "critic_pass") return COLORS.status.passed();
  if (!status) return COLORS.muted;
  return status.includes("budget") ? COLORS.status.budget() : COLORS.status.failed();
}

export function update(snapshot) {
  if (!chart) return;
  const pipelines = snapshot.pipelines.slice(-30); // keep the graph legible
  const nodes = [{ name: "orchestrator", symbolSize: 34, itemStyle: { color: COLORS.text }, fixed: false }];
  const links = [];
  const maxTokens = Math.max(1, ...pipelines.map((p) => p.tokens));
  for (const pl of pipelines) {
    const id = pl.id.slice(0, 6);
    const size = 12 + (22 * pl.tokens) / maxTokens;
    nodes.push({ name: `p:${id}`, pipelineId: pl.id, symbolSize: size, itemStyle: { color: nodeColor(pl.status) },
      label: { show: false }, tooltip: { formatter: `${id} · ${pl.theme || ""} · ${pl.status}` } });
    links.push({ source: "orchestrator", target: `p:${id}` });
    for (const stage of ["ideas", "coder", "critic", "checks", "sandbox", "backtest"]) {
      const name = `${stage}:${id}`;
      nodes.push({ name, symbolSize: 6, itemStyle: { color: stage === "checks" || stage === "sandbox" || stage === "backtest" ? COLORS.stage.static() : COLORS.agent[stage] ? COLORS.agent[stage]() : COLORS.muted }, label: { show: false } });
      links.push({ source: `p:${id}`, target: name });
    }
    links.push({ source: `ideas:${id}`, target: `coder:${id}` }, { source: `coder:${id}`, target: `checks:${id}` },
      { source: `checks:${id}`, target: `sandbox:${id}` }, { source: `sandbox:${id}`, target: `backtest:${id}` },
      { source: `backtest:${id}`, target: `critic:${id}` });
  }
  chart.setOption({
    animation: false,
    tooltip: {},
    series: [{
      type: "graph",
      layout: "force",
      roam: true,
      draggable: false,
      data: nodes,
      links,
      force: { repulsion: 120, edgeLength: [30, 90], gravity: 0.1 },
      lineStyle: { color: COLORS.border, opacity: 0.8, width: 1 },
      emphasis: { focus: "adjacency" },
    }],
  }, true);
}
