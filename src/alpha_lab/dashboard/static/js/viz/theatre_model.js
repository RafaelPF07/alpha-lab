// Pipeline theatre: the logic half. Turns ledger events into visual effects and keeps the
// logical state of each pipeline. Pure JavaScript with no DOM or canvas, so it can be tested.
//
// Every pipeline is a card with eight stations. Events move packets between stations and set
// each station's state. Effects carry a delay (`at`, in milliseconds) so the renderer can play
// them in order even when a burst of events arrives at once.
//
// Which statuses count as passed comes from the server (core/contract.js, DECISION-13), never from this file.

import { isPassedCritic, isPassedOutcome, isPassedValidation } from "../core/contract.js";

export const STAGES = [
  { key: "idea", label: "idea", short: "IDEA", color: "--stage-idea" },
  { key: "code", label: "code", short: "CODE", color: "--stage-code" },
  { key: "static", label: "static", short: "STATIC", color: "--stage-static" },
  { key: "sandbox", label: "sandbox", short: "SAND", color: "--stage-sandbox" },
  { key: "lookahead", label: "lookahead", short: "LOOK", color: "--stage-lookahead" },
  { key: "train", label: "train", short: "TRAIN", color: "--stage-train" },
  { key: "critic", label: "critic", short: "CRIT", color: "--stage-critic" },
  { key: "validation", label: "validation", short: "VALID", color: "--stage-validation" },
];

export const IDLE = "idle";
export const ACTIVE = "active";
export const OK = "ok";
export const FAIL = "fail";

const AGENT_STATION = { ideas: 0, coder: 1, critic: 6 };
// The station a revision (or a failed finish) marks red. Every failure status the server writes has one (R4-WHOLE-06).
const REVISION_STATION = {
  failed_static: 2, rejected_duplicate_code: 2, failed_sandbox: 3, failed_lookahead: 4,
  failed_gate: 5, failed_backtest: 5, failed_critic: 6, failed_critic_output: 6,
};
const TIMING = { hubToCard: 520, packet: 420, node: 260, float: 0 };

// The lookahead stage runs two tests. Only the perturbation test marks the station as passed.
const LOOKAHEAD_TESTS = 2;

function freshStations() {
  return STAGES.map(() => ({ state: IDLE, text: "" }));
}

export function shortId(id) {
  return (id || "").slice(0, 6);
}

export function truncate(text, max) {
  if (!text) return "";
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}

// Numbers from the stream may be missing or malformed. These helpers keep them out of the text (QA: no NaN).
export function finiteOr(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

// Text from a payload value. A missing value gives the fallback; any other value is turned into text, so a number
// or an object never reaches a string method on the canvas (R4-FE-09).
export function textOf(value, fallback = "") {
  return value == null || value === "" ? fallback : String(value);
}

export function finiteOrNull(value) {
  if (value == null) return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

export function compactCount(n) {
  return n < 1000 ? String(Math.round(n)) : `${(n / 1000).toFixed(1)}k`;
}

const money = (x) => (finiteOrNull(x) == null ? "n/a" : `$${finiteOrNull(x).toFixed(4)}`);

export class TheatreModel {
  constructor() {
    this.cards = new Map(); // pipeline_id -> card
    this.order = []; // pipeline ids in arrival order
    this.total = null;
    this.budget = null;
    this.spent = 0;
    this.running = 0;
    this.done = 0;
    this.passed = 0; // passed the critic gate
    this.validated = 0; // passed validation
    this.duplicatePipelines = new Set(); // pipelines that were caught as duplicate code
    this.duplicateIdeas = 0; // idea duplicates, one per idea_rejected_duplicate event
    this.runStatus = "waiting";
    this.feed = []; // newest last, capped
    this.lastSeq = 0;
  }

  // Duplicates caught: each pipeline that was caught as duplicate code counts once, however many events said so
  // (R4-FE-02), plus each idea the ideas agent rejected as a duplicate.
  get duplicates() {
    return this.duplicatePipelines.size + this.duplicateIdeas;
  }

  // Apply one event. Returns effects. With animate=false the state updates but no effects are
  // produced, which is how a large backlog (a finished run being opened) is fast-forwarded.
  apply(event, animate = true) {
    if (event.seq != null) this.lastSeq = Math.max(this.lastSeq, event.seq);
    const effects = [];
    const emit = (effect) => { if (animate) effects.push(effect); };
    const p = event.payload || {};
    const pid = event.pipeline_id;

    this.pushFeed(event);

    switch (event.type) {
      case "run_started": {
        this.total = finiteOrNull(p.pipelines) ?? this.total;
        this.budget = finiteOrNull(p.budget_usd) ?? this.budget;
        this.runStatus = "running";
        emit({ type: "hubtext", text: `run started · ${this.total ?? "?"} pipelines`, tone: "info", at: 0 });
        break;
      }
      case "run_finished": {
        this.runStatus = p.status || "finished";
        emit({ type: "hubtext", text: `run ${this.runStatus}`, tone: this.runStatus === "completed" ? "ok" : "warn", at: 0 });
        break;
      }
      case "budget_update": {
        const spent = finiteOrNull(p.spent_usd);
        if (spent != null) this.spent = Math.max(this.spent, spent);
        this.budget = finiteOrNull(p.limit_usd) ?? this.budget;
        break;
      }
      case "pipeline_started": {
        if (!pid || this.cards.has(pid)) break;
        const card = {
          id: pid,
          short: shortId(pid),
          theme: textOf(p.theme),
          idea: null,
          ideaId: null,
          status: "running",
          attempt: 1,
          stations: freshStations(),
          lookaheadPassed: 0,
          lastSharpe: null,
          cost: 0,
          tokens: 0,
        };
        this.cards.set(pid, card);
        this.order.push(pid);
        this.running += 1;
        emit({ type: "packet", pid, from: "hub", to: "card", color: "--stage-idea", dur: TIMING.hubToCard, at: 0 });
        emit({ type: "spawn", pid, at: TIMING.hubToCard - 80 });
        break;
      }
      case "idea_created": {
        const card = this.cards.get(pid);
        if (!card) break;
        if (p.name != null && p.name !== "") card.idea = String(p.name);
        card.ideaId = event.idea_id || card.ideaId;
        card.theme = card.theme || textOf(p.theme);
        this.setStation(card, 0, OK, "", emit);
        break;
      }
      case "idea_rejected_duplicate": {
        this.duplicateIdeas += 1;
        emit({ type: "hubtext", text: "duplicate idea rejected", tone: "warn", at: 0 });
        break;
      }
      case "llm_call_started": {
        const card = this.cards.get(pid);
        const station = AGENT_STATION[event.agent];
        if (card && station !== undefined) this.setStation(card, station, ACTIVE, card.stations[station].text, emit);
        break;
      }
      case "llm_call_finished": {
        const card = this.cards.get(pid);
        if (!card) break;
        const tokens = finiteOr(p.input_tokens, 0) + finiteOr(p.output_tokens, 0);
        const cost = finiteOr(p.cost_usd, 0);
        card.tokens += tokens;
        card.cost += cost;
        const station = AGENT_STATION[event.agent];
        if (station !== undefined) {
          const ok = p.status === "ok";
          if (ok && station !== 6) this.setStation(card, station, OK, card.stations[station].text, emit);
          if (!ok) this.setStation(card, station, FAIL, "api error", emit);
          emit({ type: "float", pid, index: station, text: `${compactCount(tokens)} tok`, tone: ok ? "muted" : "fail", at: 0 });
        }
        break;
      }
      case "code_written": {
        const card = this.cards.get(pid);
        if (!card) break;
        this.setStation(card, 1, OK, "", emit);
        break;
      }
      case "static_check": {
        const card = this.cards.get(pid);
        if (!card) break;
        // A duplicate of an earlier program is reported by duplicate_code, not here (WHOLE-08).
        this.passStation(card, 2, p.ok ? OK : FAIL, p.ok ? "clean" : "reject", emit, 1);
        break;
      }
      case "duplicate_code": {
        // The backend rejects a program that repeats an earlier attempt, before the static check runs.
        if (pid) this.duplicatePipelines.add(pid);
        else this.duplicateIdeas += 1;
        const card = this.cards.get(pid);
        if (card) this.passStation(card, 2, FAIL, "dup", emit, 1);
        emit({ type: "hubtext", text: "duplicate code rejected", tone: "warn", at: 0 });
        break;
      }
      case "sandbox_run": {
        const card = this.cards.get(pid);
        if (!card) break;
        const ok = p.status === "ok";
        this.passStation(card, 3, ok ? OK : FAIL, ok ? "ran" : p.status || "error", emit, 2);
        break;
      }
      case "lookahead_check": {
        const card = this.cards.get(pid);
        if (!card) break;
        if (!p.passed) {
          this.passStation(card, 4, FAIL, "failed", emit, 3);
        } else {
          card.lookaheadPassed += 1;
          if (card.lookaheadPassed === 1) this.passStation(card, 4, ACTIVE, "1/2", emit, 3);
          if (card.lookaheadPassed >= LOOKAHEAD_TESTS) this.passStation(card, 4, OK, "2/2", emit);
        }
        break;
      }
      case "train_backtest": {
        const card = this.cards.get(pid);
        if (!card) break;
        card.lastSharpe = Number(p.sharpe);
        this.passStation(card, 5, OK, Number.isFinite(card.lastSharpe) ? `SR ${card.lastSharpe.toFixed(2)}` : "SR n/a", emit, 4);
        break;
      }
      case "critic_review": {
        const card = this.cards.get(pid);
        if (!card) break;
        const high = p.suspicion_level === "high";
        this.passStation(card, 6, high ? FAIL : OK, p.suspicion_level || "?", emit, 5);
        break;
      }
      case "validation_result": {
        const card = this.cards.get(pid);
        if (!card) break;
        this.passStation(card, 7, p.passed ? OK : FAIL, p.passed ? "passed" : "failed", emit, 6);
        break;
      }
      case "revision_requested": {
        const card = this.cards.get(pid);
        if (!card) break;
        const station = REVISION_STATION[p.reason];
        if (station !== undefined && card.stations[station].state !== FAIL) {
          this.setStation(card, station, FAIL, card.stations[station].text, emit);
        }
        card.attempt += 1;
        // Stations from the previous attempt stay visible but marked stale, so the card still
        // shows the history until the new attempt overwrites each one. The station that failed
        // stays red, so the reason for the revision remains visible.
        for (let i = 1; i < STAGES.length; i++) {
          if (i === station) continue;
          const old = card.stations[i];
          card.stations[i] = old.state === IDLE ? old : { ...old, stale: true };
          if (old.state !== IDLE) emit({ type: "stale", pid, index: i, at: 0 });
        }
        // The coder station restarts with the revision.
        this.setStation(card, 1, ACTIVE, "redo", emit);
        emit({ type: "arc", pid, from: station ?? 5, to: 1, color: "--status-rejected", at: 0 });
        emit({ type: "attempt", pid, n: card.attempt, at: 0 });
        break;
      }
      case "pipeline_finished": {
        const status = textOf(p.status, "finished");
        // Counted from the finish too, so a duplicate the backend reports only as a status is still counted (R4-FE-02).
        if (status === "rejected_duplicate_code" && pid) this.duplicatePipelines.add(pid);
        const card = this.cards.get(pid);
        if (!card) break;
        card.status = status;
        this.running = Math.max(0, this.running - 1);
        this.done += 1;
        if (isPassedCritic(status)) this.passed += 1;
        const passed = isPassedOutcome(status);
        if (isPassedValidation(status)) this.validated += 1;
        const failedStation = REVISION_STATION[status];
        if (failedStation !== undefined && card.stations[failedStation].state !== FAIL) {
          this.setStation(card, failedStation, FAIL, card.stations[failedStation].text, emit);
        }
        emit({ type: "burst", pid, tone: passed ? "ok" : "fail", at: 0 });
        if (!passed) emit({ type: "shake", pid, at: 0 });
        emit({ type: "status", pid, status, at: 0 });
        break;
      }
      default:
        break;
    }
    return effects;
  }

  // Sets a station immediately in the model, and emits the matching effects with a delay.
  setStation(card, index, state, text, emit) {
    card.stations[index] = { state, text };
    emit({ type: "node", pid: card.id, index, state, text, at: 0 });
  }

  // A stage result arrives as a packet from the previous station, then the station settles.
  passStation(card, index, state, text, emit, previous) {
    card.stations[index] = { state, text };
    if (previous !== undefined && previous >= 0) {
      emit({ type: "packet", pid: card.id, from: previous, to: index, color: STAGES[index].color, dur: TIMING.packet, at: 0 });
    }
    emit({ type: "node", pid: card.id, index, state, text, at: TIMING.packet * 0.9 });
    if (state === FAIL) emit({ type: "float", pid: card.id, index, text: "✕", tone: "fail", at: TIMING.packet });
    if (state === OK && index !== 0) emit({ type: "float", pid: card.id, index, text: "✓", tone: "ok", at: TIMING.packet });
  }

  pushFeed(event) {
    this.feed.push({
      seq: event.seq,
      type: event.type,
      agent: event.agent || event.stage || "",
      pid: event.pipeline_id ? shortId(event.pipeline_id) : "",
      text: describe(event),
      time: (event.timestamp || "").slice(11, 19),
    });
    if (this.feed.length > 80) this.feed.splice(0, this.feed.length - 80);
  }

  visibleCards(limit) {
    const ids = this.order.slice(-limit);
    return ids.map((id) => this.cards.get(id));
  }

  stats() {
    const pct = this.total ? Math.min(1, this.done / this.total) : null;
    return {
      running: this.running,
      done: this.done,
      passed: this.passed,
      validated: this.validated,
      total: this.total,
      spent: this.spent,
      budget: this.budget,
      duplicates: this.duplicates,
      runStatus: this.runStatus,
      progress: pct,
    };
  }
}

export function describe(event) {
  const p = event.payload || {};
  const agent = event.agent || event.stage || "agent";
  const sharpe = finiteOrNull(p.sharpe);
  switch (event.type) {
    case "pipeline_started": return `pipeline started${p.theme ? ` · ${p.theme}` : ""}`;
    case "idea_created": return `idea · ${truncate(String(p.name || "unnamed"), 48)}`;
    case "llm_call_started": return `${agent} thinking`;
    case "llm_call_finished": return `${agent} replied · ${money(p.cost_usd)}`;
    case "code_written": return "code written";
    case "static_check": return p.ok ? "static check passed" : "static check failed";
    case "sandbox_run": return `sandbox ${p.status || "unknown"}`;
    case "lookahead_check": return `${p.test || "lookahead"} check ${p.passed ? "passed" : "failed"}`;
    case "train_backtest": return `train Sharpe ${sharpe == null ? "n/a" : sharpe.toFixed(2)}`;
    case "critic_review": return `critic: ${p.suspicion_level || "unknown"} suspicion`;
    case "validation_result": return `validation ${p.passed ? "passed" : "failed"}`;
    case "revision_requested": return `revising (${p.reason || "unknown"})`;
    case "pipeline_finished": return `finished · ${p.status || "unknown"}`;
    case "budget_update": return `spent ${money(p.spent_usd)}`;
    case "run_started": return "run started";
    case "run_finished": return `run ${p.status || "finished"}`;
    case "idea_rejected_duplicate": return "duplicate idea rejected";
    case "duplicate_code": return "duplicate code rejected";
    default: return event.type ? String(event.type) : "event";
  }
}

export { TIMING };
