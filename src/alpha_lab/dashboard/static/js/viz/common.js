// Shared helpers for every chart: colours come from the design tokens, numbers are formatted once.

const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

export const COLORS = {
  get text() { return cssVar("--text"); },
  get muted() { return cssVar("--muted"); },
  get border() { return cssVar("--border"); },
  get panel() { return cssVar("--panel"); },
  get locked() { return cssVar("--stage-holdout"); },
  stage: {
    idea: () => cssVar("--stage-idea"),
    code: () => cssVar("--stage-code"),
    static: () => cssVar("--stage-static"),
    sandbox: () => cssVar("--stage-sandbox"),
    lookahead: () => cssVar("--stage-lookahead"),
    train: () => cssVar("--stage-train"),
    critic: () => cssVar("--stage-critic"),
    validation: () => cssVar("--stage-validation"),
    holdout: () => cssVar("--stage-holdout"),
  },
  agent: {
    ideas: () => cssVar("--agent-ideas"),
    coder: () => cssVar("--agent-coder"),
    critic: () => cssVar("--agent-critic"),
  },
  status: {
    running: () => cssVar("--status-running"),
    passed: () => cssVar("--status-passed"),
    failed: () => cssVar("--status-failed"),
    rejected: () => cssVar("--status-rejected"),
    budget: () => cssVar("--status-budget"),
  },
};

export function agentColor(agent) {
  if (agent === "ideas") return COLORS.agent.ideas();
  if (agent === "coder") return COLORS.agent.coder();
  if (agent === "critic") return COLORS.agent.critic();
  return COLORS.muted;
}

export function statusColor(status) {
  if (!status) return COLORS.status.running();
  if (status === "running") return COLORS.status.running();
  if (status === "validation_pass" || status === "critic_pass") return COLORS.status.passed();
  if (status === "budget_stopped" || status === "interrupted") return COLORS.status.budget();
  if (status.startsWith("rejected")) return COLORS.status.rejected();
  return COLORS.status.failed();
}

export const fmtUsd = (x) => (x == null || !Number.isFinite(x) ? "n/a" : `$${x.toFixed(4)}`);
export const fmtPct = (x) => (x == null || !Number.isFinite(x) ? "n/a" : `${(x * 100).toFixed(2)}%`);
export const fmtNum = (x, d = 3) => (x == null || !Number.isFinite(x) ? "n/a" : x.toFixed(d));

export const baseAxis = () => ({
  axisLine: { lineStyle: { color: COLORS.border } },
  axisLabel: { color: COLORS.muted },
  splitLine: { lineStyle: { color: COLORS.border, opacity: 0.5 } },
});

export const baseText = () => ({ textStyle: { color: COLORS.muted, fontFamily: cssVar("--font-ui") } });

const charts = new Set();
export function track(chart) {
  charts.add(chart);
  return chart;
}
window.addEventListener("resize", () => charts.forEach((c) => c.resize()));
