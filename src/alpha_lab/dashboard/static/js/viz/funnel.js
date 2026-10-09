// Live funnel: flows from ideas generated through each stage, with drop-offs branching off.
import { COLORS, baseText, track } from "./common.js";

const STAGES = [
  ["generated", "ideas generated", "idea"],
  ["coded", "coded", "code"],
  ["static", "passed static", "static"],
  ["sandbox", "ran in sandbox", "sandbox"],
  ["lookahead", "passed lookahead", "lookahead"],
  ["train", "train gate", "train"],
  ["critic", "passed critic", "critic"],
  ["validation", "validation pass", "validation"],
];

let chart = null;

export function init(container) {
  chart = track(echarts.init(container, null, { renderer: "canvas" }));
  return chart;
}

export function buildGraph(funnel, failures) {
  const nodes = [];
  const links = [];
  const seen = new Set();
  const addNode = (name, color) => {
    if (!seen.has(name)) { seen.add(name); nodes.push({ name, itemStyle: { color } }); }
  };
  const stageColor = (key) => COLORS.stage[key]?.() ?? COLORS.muted;
  const count = (k) => Number(funnel[k] ?? 0);
  for (let i = 0; i < STAGES.length; i++) {
    const [key, label, color] = STAGES[i];
    addNode(label, stageColor(color));
    if (i === 0) continue;
    const [prevKey, prevLabel] = STAGES[i - 1];
    const reached = count(key);
    const from = count(prevKey);
    if (reached > 0) links.push({ source: prevLabel, target: label, value: reached });
    const drop = Math.max(0, from - reached);
    if (drop > 0) {
      const dropName = `${prevLabel} → dropped`;
      addNode(dropName, COLORS.status.failed());
      links.push({ source: prevLabel, target: dropName, value: drop });
    }
  }
  return { nodes, links, failures };
}

export function update(snapshot) {
  if (!chart) return;
  const { nodes, links } = buildGraph(snapshot.funnel, snapshot.failures);
  chart.setOption({
    ...baseText(),
    tooltip: { trigger: "item" },
    series: [{
      type: "sankey",
      left: 8,
      right: 170,
      emphasis: { focus: "adjacency" },
      nodeGap: 14,
      nodeWidth: 14,
      data: nodes,
      links,
      lineStyle: { color: "gradient", curveness: 0.45, opacity: 0.35 },
      label: { color: COLORS.text, fontSize: 11 },
    }],
  }, true);
}
