// Turns the event stream into the numbers every mission-control chart needs.
// Used by the live view and by replay, so both always show the same thing.

const TRAIN_PASS = new Set(["critic_pass", "validation_pass", "validation_fail"]);

export class RunState {
  constructor() {
    this.events = [];
    this.pipelines = new Map(); // pipeline_id -> {id, started, finished, status, theme, tokens, segments}
    this.ideas = 0;
    this.rejectedIdeas = 0;
    this.spent = 0;
    this.budget = null;
    this.spendSeries = []; // {t, spent}
    this.agentSpend = {}; // agent -> [{t, cost}]
    this.context = []; // {t, agent, tokens}
    this.funnel = { generated: 0, coded: 0, static: 0, sandbox: 0, lookahead: 0, train: 0, critic: 0, validation: 0 };
    this.failures = {}; // stage -> count
    this.statusCounts = {};
    this.finished = false;
    this.lastSeq = 0;
    this.runStart = null;
    this.runEnd = null;
  }

  ingest(event) {
    this.events.push(event);
    this.lastSeq = Math.max(this.lastSeq, event.seq ?? 0);
    const t = event.timestamp;
    const p = event.payload || {};
    if (!this.runStart) this.runStart = t;
    this.runEnd = t;

    switch (event.type) {
      case "run_started":
        this.budget = p.budget_usd ?? this.budget;
        break;
      case "run_finished":
        this.finished = true;
        break;
      case "pipeline_started": {
        this.pipelines.set(event.pipeline_id, { id: event.pipeline_id, started: t, finished: null,
          status: "running", theme: p.theme || null, tokens: 0, segments: [], markers: [] });
        break;
      }
      case "pipeline_finished": {
        const pl = this.pipelines.get(event.pipeline_id);
        if (pl) { pl.finished = t; pl.status = p.status || "done"; }
        this.statusCounts[p.status] = (this.statusCounts[p.status] || 0) + 1;
        break;
      }
      case "idea_created":
        this.ideas += 1;
        this.funnel.generated += 1;
        if (p.theme) {
          const pl = this.pipelines.get(event.pipeline_id);
          if (pl && !pl.theme) pl.theme = p.theme;
        }
        break;
      case "idea_rejected_duplicate":
        this.rejectedIdeas += 1;
        break;
      case "llm_call_started":
        break;
      case "llm_call_finished": {
        const cost = Number(p.cost_usd || 0);
        const tokens = Number(p.input_tokens || 0) + Number(p.output_tokens || 0);
        this.spent += cost;
        this.agentSpend[event.agent] = this.agentSpend[event.agent] || [];
        this.agentSpend[event.agent].push({ t, cost });
        this.spendSeries.push({ t, spent: this.spent });
        if (p.input_tokens != null) this.context.push({ t, agent: event.agent, tokens: p.input_tokens });
        const pl = this.pipelines.get(event.pipeline_id);
        if (pl) {
          pl.tokens += tokens;
          pl.segments.push({ t, agent: event.agent, latency: Number(p.latency_ms || 0), status: p.status });
        }
        break;
      }
      case "budget_update":
        if (p.spent_usd != null) this.spent = Math.max(this.spent, Number(p.spent_usd));
        this.budget = p.limit_usd ?? this.budget;
        break;
      case "code_written":
        this.funnel.coded += 1;
        break;
      case "static_check":
        if (p.ok) this.funnel.static += 1; else this.failures.static = (this.failures.static || 0) + 1;
        this.markPipeline(event, "static");
        break;
      case "sandbox_run":
        if (p.status === "ok") this.funnel.sandbox += 1;
        else this.failures.sandbox = (this.failures.sandbox || 0) + 1;
        this.markPipeline(event, "sandbox");
        break;
      case "lookahead_check":
        if (p.passed && p.test === "perturbation") this.funnel.lookahead += 1;
        if (!p.passed) this.failures.lookahead = (this.failures.lookahead || 0) + 1;
        this.markPipeline(event, "lookahead");
        break;
      case "train_backtest":
        this.funnel.train += 1;
        this.markPipeline(event, "train");
        break;
      case "critic_review":
        if (p.suspicion_level !== "high") this.funnel.critic += 1;
        else this.failures.critic = (this.failures.critic || 0) + 1;
        this.markPipeline(event, "critic");
        break;
      case "validation_result":
        if (p.passed) this.funnel.validation += 1;
        else this.failures.validation = (this.failures.validation || 0) + 1;
        this.markPipeline(event, "validation");
        break;
      case "revision_requested":
        this.markPipeline(event, "revision");
        break;
      default:
        break;
    }
  }

  markPipeline(event, stage) {
    const pl = this.pipelines.get(event.pipeline_id);
    if (pl) pl.markers.push({ t: event.timestamp, stage, attempt: event.attempt_number });
  }

  get runningPipelines() {
    let n = 0;
    for (const pl of this.pipelines.values()) if (pl.status === "running") n += 1;
    return n;
  }

  get donePipelines() {
    return [...this.pipelines.values()].filter((pl) => pl.status !== "running").length;
  }

  get passedPipelines() {
    return [...this.pipelines.values()].filter((pl) => TRAIN_PASS.has(pl.status)).length;
  }

  get validationPasses() {
    return this.statusCounts.validation_pass || 0;
  }

  spendRatePerMinute() {
    if (!this.runStart || !this.runEnd || this.runStart === this.runEnd) return 0;
    const minutes = (Date.parse(this.runEnd) - Date.parse(this.runStart)) / 60000;
    return minutes > 0 ? this.spent / minutes : 0;
  }

  // Projected time at which the budget runs out, at the current rate. Null when unknown.
  projectedExhaustion() {
    const rate = this.spendRatePerMinute();
    if (!this.budget || rate <= 0 || this.spent >= this.budget) return null;
    const minutesLeft = (this.budget - this.spent) / rate;
    return new Date(Date.parse(this.runEnd) + minutesLeft * 60000).toISOString();
  }

  snapshot() {
    return {
      runningPipelines: this.runningPipelines,
      donePipelines: this.donePipelines,
      totalPipelines: this.pipelines.size,
      passedPipelines: this.passedPipelines,
      validationPasses: this.validationPasses,
      ideas: this.ideas,
      rejectedIdeas: this.rejectedIdeas,
      spent: this.spent,
      budget: this.budget,
      spendRate: this.spendRatePerMinute(),
      projectedExhaustion: this.projectedExhaustion(),
      spendSeries: this.spendSeries,
      agentSpend: this.agentSpend,
      context: this.context,
      funnel: this.funnel,
      failures: this.failures,
      statusCounts: this.statusCounts,
      pipelines: [...this.pipelines.values()],
      ticker: this.events.slice(-200),
      finished: this.finished,
      costPerPass: this.passedPipelines ? this.spent / this.passedPipelines : null,
    };
  }
}
