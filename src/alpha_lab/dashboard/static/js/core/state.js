// Turns the event stream into the numbers every mission-control chart needs.
// Used by the live view and by replay, so both always show the same thing.
import { isPassedCritic, isPassedValidation } from "./contract.js";

// Funnel stages in order. Each stage counts distinct ideas (not events), and never exceeds the stage before it.
export const FUNNEL = ["generated", "coded", "static", "sandbox", "lookahead", "train", "critic", "validation"];

// The ticker and its filter see only the most recent events. Older events are not kept.
export const RECENT_LIMIT = 5000;

// Time series (spend, agent spend, context) keep at most this many points. When full, every other point is
// dropped, so the whole run stays on the chart at half the resolution (FE-14).
export const SERIES_MAX = 4000;

// Adds one point to a series, thinning the series when it is full. The newest point is always kept.
export function pushSeries(list, item) {
  list.push(item);
  if (list.length <= SERIES_MAX) return;
  const kept = list.filter((_, i) => i % 2 === 0);
  if (kept[kept.length - 1] !== item) kept.push(item);
  list.splice(0, list.length, ...kept);
}

function finiteOr(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function finiteOrNull(value) {
  if (value == null) return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

export class RunState {
  // runId is optional. When given, events that carry a different run_id are ignored.
  constructor(runId = null) {
    this.runId = runId;
    this.recent = []; // the latest events, trimmed to about RECENT_LIMIT
    this.pipelines = new Map(); // pipeline_id -> {id, started, finished, status, theme, ideaId, tokens, segments, markers}
    this.pipelineIdea = new Map(); // pipeline_id -> idea_id, for events that carry no idea_id
    this.stageIds = Object.fromEntries(FUNNEL.map((key) => [key, new Set()])); // distinct idea keys per stage
    this.rejectedIdeas = 0; // idea duplicates, one per idea_rejected_duplicate event
    this.duplicatePipelines = new Set(); // pipelines caught as duplicate code, counted once each (as theatre_model.js)
    this.spent = 0;
    this.budget = null; // null until a run or budget event states it
    this.spendSeries = []; // {t, spent}
    this.agentSpend = {}; // agent -> [{t, cost}]
    this.context = []; // {t, agent, tokens}
    this.llmCalls = 0; // every LLM call with an input size, not only the points kept in context
    this.largestContext = 0;
    this.failures = {}; // stage -> count of attempts that ended there
    this.statusCounts = {};
    this.finished = false;
    this.runStatus = null;
    this.runStart = null;
    this.runEnd = null;
  }

  get ideas() {
    return this.stageIds.generated.size;
  }

  // Duplicates caught, counted the way the theatre counts them (R4-WHOLE-06).
  get duplicates() {
    return this.duplicatePipelines.size + this.rejectedIdeas;
  }

  ingest(event) {
    if (this.runId && event.run_id && event.run_id !== this.runId) return;
    this.recent.push(event);
    if (this.recent.length > 2 * RECENT_LIMIT) this.recent = this.recent.slice(-RECENT_LIMIT);

    const t = event.timestamp;
    const p = event.payload || {};
    if (t) {
      if (!this.runStart) this.runStart = t;
      this.runEnd = t;
    }

    const pid = event.pipeline_id;
    if (pid && event.idea_id) this.pipelineIdea.set(pid, event.idea_id);
    const idea = event.idea_id || (pid && this.pipelineIdea.get(pid)) || (pid ? `pipeline:${pid}` : null);
    const mark = (stage) => { if (idea) this.stageIds[stage].add(idea); };

    switch (event.type) {
      case "run_started":
        this.budget = finiteOrNull(p.budget_usd) ?? this.budget;
        break;
      case "run_finished":
        this.finished = true;
        this.runStatus = p.status || "finished";
        break;
      case "pipeline_started": {
        this.pipelines.set(pid, { id: pid, started: t, finished: null, status: "running",
          theme: p.theme == null || p.theme === "" ? null : String(p.theme),
          ideaId: null, tokens: 0, segments: [], markers: [] });
        break;
      }
      case "pipeline_finished": {
        const status = String(p.status || "done");
        if (status === "rejected_duplicate_code" && pid) this.duplicatePipelines.add(pid);
        const pl = this.pipelines.get(pid);
        if (pl) { pl.finished = t; pl.status = status; }
        this.statusCounts[status] = (this.statusCounts[status] || 0) + 1;
        break;
      }
      case "idea_created": {
        mark("generated");
        const pl = this.pipelines.get(pid);
        if (pl) {
          if (!pl.theme && p.theme) pl.theme = p.theme;
          if (event.idea_id) pl.ideaId = event.idea_id;
        }
        break;
      }
      case "idea_rejected_duplicate":
        this.rejectedIdeas += 1;
        break;
      case "duplicate_code":
        if (pid) this.duplicatePipelines.add(pid);
        else this.rejectedIdeas += 1;
        break;
      case "llm_call_started":
        break;
      case "llm_call_finished": {
        const cost = finiteOr(p.cost_usd, 0);
        const tokens = finiteOr(p.input_tokens, 0) + finiteOr(p.output_tokens, 0);
        const agent = event.agent || "unknown";
        this.spent += cost;
        this.agentSpend[agent] = this.agentSpend[agent] || [];
        pushSeries(this.agentSpend[agent], { t, cost });
        pushSeries(this.spendSeries, { t, spent: this.spent });
        if (p.input_tokens != null) {
          const input = finiteOr(p.input_tokens, 0);
          this.llmCalls += 1;
          this.largestContext = Math.max(this.largestContext, input);
          pushSeries(this.context, { t, agent, tokens: input });
        }
        const pl = this.pipelines.get(pid);
        if (pl) {
          pl.tokens += tokens;
          pl.segments.push({ t, agent, latency: finiteOr(p.latency_ms, 0), status: p.status });
        }
        break;
      }
      case "budget_update": {
        const spent = Number(p.spent_usd);
        if (p.spent_usd != null && Number.isFinite(spent)) this.spent = Math.max(this.spent, spent);
        this.budget = finiteOrNull(p.limit_usd) ?? this.budget;
        break;
      }
      case "code_written":
        mark("coded");
        break;
      case "static_check":
        if (p.ok) mark("static"); else this.failures.static = (this.failures.static || 0) + 1;
        this.markPipeline(event, "static");
        break;
      case "sandbox_run":
        if (p.status === "ok") mark("sandbox"); else this.failures.sandbox = (this.failures.sandbox || 0) + 1;
        this.markPipeline(event, "sandbox");
        break;
      case "lookahead_check":
        if (p.passed && p.test === "perturbation") mark("lookahead");
        if (!p.passed) this.failures.lookahead = (this.failures.lookahead || 0) + 1;
        this.markPipeline(event, "lookahead");
        break;
      case "train_backtest":
        mark("train");
        this.markPipeline(event, "train");
        break;
      case "critic_review":
        if (p.suspicion_level !== "high") mark("critic"); else this.failures.critic = (this.failures.critic || 0) + 1;
        this.markPipeline(event, "critic");
        break;
      case "validation_result":
        if (p.passed) mark("validation"); else this.failures.validation = (this.failures.validation || 0) + 1;
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

  // Pipelines that passed the critic gate, by the server's status set (DECISION-13).
  get passedPipelines() {
    return [...this.pipelines.values()].filter((pl) => isPassedCritic(pl.status)).length;
  }

  // Pipelines that passed validation, by the server's status set.
  get validationPasses() {
    let n = 0;
    for (const [status, count] of Object.entries(this.statusCounts)) if (isPassedValidation(status)) n += count;
    return n;
  }

  // Each stage's distinct-idea count, clamped so no stage exceeds the one before it.
  funnelCounts() {
    const counts = {};
    let previous = Infinity;
    for (const key of FUNNEL) {
      const n = Math.min(this.stageIds[key].size, previous);
      counts[key] = n;
      previous = n;
    }
    return counts;
  }

  spendRatePerMinute() {
    if (!this.runStart || !this.runEnd || this.runStart === this.runEnd) return 0;
    const minutes = (Date.parse(this.runEnd) - Date.parse(this.runStart)) / 60000;
    return minutes > 0 ? this.spent / minutes : 0;
  }

  // Projected time at which the budget runs out, at the current rate. Null when unknown, or when the run
  // has finished: a finished run spends nothing more, so there is nothing to project.
  projectedExhaustion() {
    if (this.finished) return null;
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
      duplicates: this.duplicates,
      spent: this.spent,
      budget: this.budget,
      spendRate: this.spendRatePerMinute(),
      projectedExhaustion: this.projectedExhaustion(),
      spendSeries: this.spendSeries,
      agentSpend: this.agentSpend,
      context: this.context,
      llmCalls: this.llmCalls,
      largestContext: this.largestContext,
      funnel: this.funnelCounts(),
      failures: this.failures,
      statusCounts: this.statusCounts,
      pipelines: [...this.pipelines.values()],
      ticker: this.recent.slice(-RECENT_LIMIT),
      finished: this.finished,
      runStatus: this.runStatus,
      runStart: this.runStart,
      runEnd: this.runEnd,
      costPerPass: this.passedPipelines ? this.spent / this.passedPipelines : null,
    };
  }
}
