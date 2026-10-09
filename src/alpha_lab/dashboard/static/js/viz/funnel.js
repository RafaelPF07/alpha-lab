// Live funnel: distinct ideas through each stage, with drop-offs branching off in red.
import { COLORS, baseText, describeChart, track } from "./common.js";

// One word per stage, so the labels do not collide on a narrow funnel. The note above the chart says what is counted.
const STAGES = [
  ["generated", "ideas", "idea"],
  ["coded", "coded", "code"],
  ["static", "static", "static"],
  ["sandbox", "sandbox", "sandbox"],
  ["lookahead", "lookahead", "lookahead"],
  ["train", "train", "train"],
  ["critic", "critic", "critic"],
  ["validation", "validated", "validation"],
];

let chart = null;
let container = null;

export function init(el) {
  container = el;
  chart = track(echarts.init(el, null, { renderer: "canvas" }));
  return chart;
}

// Counts are distinct ideas. Each stage is clamped to the stage before it, so a flow never grows.
export function buildGraph(funnel, failures) {
  const nodes = [];
  const links = [];
  const seen = new Set();
  const addNode = (name, color) => {
    if (!seen.has(name)) { seen.add(name); nodes.push({ name, itemStyle: { color } }); }
  };
  const stageColor = (key) => COLORS.stage[key]?.() ?? COLORS.muted;
  const count = (k) => Number(funnel[k] ?? 0);
  let previous = count(STAGES[0][0]);
  for (let i = 0; i < STAGES.length; i++) {
    const [key, label, color] = STAGES[i];
    addNode(label, stageColor(color));
    if (i === 0) continue;
    const [, prevLabel] = STAGES[i - 1];
    const reached = Math.min(count(key), previous);
    if (reached > 0) links.push({ source: prevLabel, target: label, value: reached });
    const drop = Math.max(0, previous - reached);
    if (drop > 0) {
      const dropName = `${prevLabel} → dropped`;
      addNode(dropName, COLORS.status.failed());
      links.push({ source: prevLabel, target: dropName, value: drop });
    }
    previous = reached;
  }
  return { nodes, links, failures };
}

export function update(snapshot) {
  if (!chart) return;
  const f = snapshot.funnel;
  const { nodes, links } = buildGraph(f, snapshot.failures);
  describeChart(container, `Funnel of distinct ideas: ${f.generated} generated, ${f.coded} coded, ${f.static} passed static checks, `
    + `${f.sandbox} ran in the sandbox, ${f.lookahead} passed lookahead, ${f.train} reached the train gate, `
    + `${f.critic} passed the critic, ${f.validation} passed validation.`);
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
