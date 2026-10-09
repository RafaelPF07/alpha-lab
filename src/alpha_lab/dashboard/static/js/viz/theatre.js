// Pipeline theatre: the canvas half. Plays the effects produced by theatre_model.js in real time.
//
// Every pipeline is drawn as a card with eight stations. Packets travel between stations as stages
// complete, stations light up or fail, revision loops arc back to the coder, and each finished pipeline
// bursts green (passed) or red (failed). Finished cards are dimmed, never removed.
//
// A page opened in the middle of a run gets the backlog in one batch. Those events update each card's
// visual state at once, without animation, so every pipeline is drawn in its current state (R3-FE-01).
//
// The canvas redraws only when something changes. While a run is live, the ambient motion (hub pulse,
// card breathing, stars) runs at up to ten frames a second. With prefers-reduced-motion it stops, and the
// stage still redraws on every state change (R3-FE-02).
//
// When a run has more pipelines than the normal card grid holds, the stage switches to compact cards.
// Pipelines that still do not fit are counted in the header, not drawn (R3-FE-03).
import { STAGES, IDLE, ACTIVE, OK, FAIL, TIMING, TheatreModel, truncate } from "./theatre_model.js";
import { prefersReducedMotion } from "./common.js";
import { isPassedOutcome } from "../core/contract.js";

const HEADER_H = 92;
const PAD = 22;
const MIN_WIDTH = 280;
const MIN_HEIGHT = 240;
const MAX_LIVE_EVENTS = 60; // a backlog larger than this is applied without animation
const IDLE_FRAME_MS = 100; // ambient motion while the run is live: at most ten frames a second
const FINISHED_ALPHA = 0.6; // finished cards are dimmed, not removed
export const CANVAS_MUTED = "#9aa4b8"; // readable on the card body (about 6:1); used for all canvas labels
const TEXT_MIN_PX = 10; // no canvas text is smaller than this (R3-FE-06)
const FIT_CACHE_MAX = 4000;

// Two card sizes. The normal size is used while every pipeline fits. The compact size is used when the run
// has more pipelines than the normal grid holds.
const NORMAL = { compact: false, cardH: 146, gap: 18, minW: 252, maxW: 320, trackY: 92 };
const COMPACT = { compact: true, cardH: 64, gap: 10, minW: 140, maxW: 200, trackY: 34 };

function cssVar(name, fallback) {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}

function hexToRgba(hex, alpha) {
  const h = hex.replace("#", "");
  const full = h.length === 3 ? h.split("").map((c) => c + c).join("") : h;
  const n = parseInt(full, 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
}

const easeInOut = (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);

// Shortens text with an ellipsis until it fits maxWidth. Returns "" when not even one character fits.
// Used for every station value, so text never runs into its neighbours (no squeezing).
export function fitText(ctx, text, maxWidth) {
  const value = String(text ?? "");
  if (!value) return "";
  if (ctx.measureText(value).width <= maxWidth) return value;
  for (let n = value.length - 1; n > 0; n--) {
    const candidate = `${value.slice(0, n)}…`;
    if (ctx.measureText(candidate).width <= maxWidth) return candidate;
  }
  return "";
}

function watchMedia(query, handler) {
  if (typeof matchMedia !== "function") return;
  const mq = matchMedia(query);
  if (mq.addEventListener) mq.addEventListener("change", handler);
  else if (mq.addListener) mq.addListener(handler);
}

export class PipelineTheatre {
  constructor(canvas, { onOpen = () => {} } = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.onOpen = onOpen;
    this.model = new TheatreModel();
    this.vcards = new Map();
    this.queue = [];
    this.packets = [];
    this.arcs = [];
    this.floats = [];
    this.bursts = [];
    this.hubText = null;
    this.hover = null;
    this.mouse = null;
    this.width = 0;
    this.height = 0;
    this.dpr = window.devicePixelRatio || 1;
    this.colors = {};
    this.dirty = true; // something changed since the last draw
    this.lastDraw = 0;
    this.hiddenCount = 0; // pipelines in the run that the stage cannot show
    this.frameIndex = new Map(); // pipeline id -> card rectangle, rebuilt on every draw
    this.fitCache = new Map();
    this.reduceMotion = prefersReducedMotion();
    this.stars = Array.from({ length: 70 }, (_, i) => ({
      x: ((i * 9301 + 49297) % 233280) / 233280,
      y: ((i * 49297 + 9301) % 233280) / 233280,
      s: 0.4 + (((i * 7919) % 100) / 100) * 1.2,
      p: ((i * 131) % 628) / 100,
    }));

    this.resize();
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas.parentElement || canvas);
    watchMedia("(prefers-reduced-motion: reduce)", () => { this.reduceMotion = prefersReducedMotion(); this.dirty = true; });
    watchMedia("(prefers-color-scheme: light)", () => this.readColors());

    canvas.addEventListener("pointermove", (e) => this.onMove(e));
    canvas.addEventListener("pointerleave", () => { this.mouse = null; this.hover = null; this.dirty = true; });
    // Opening works from the click position itself, so it also works on touch, where there is no hover.
    canvas.addEventListener("click", (e) => {
      const card = this.cardAt(this.pointOf(e));
      if (card && card.ideaId) this.onOpen(card.ideaId);
    });

    this.running = true;
    this.raf = requestAnimationFrame((t) => this.frame(t));
  }

  // Forget everything shown so far, so the page can switch to another run.
  reset() {
    this.model = new TheatreModel();
    this.vcards = new Map();
    this.queue = [];
    this.packets = [];
    this.arcs = [];
    this.floats = [];
    this.bursts = [];
    this.hubText = null;
    this.hover = null;
    this.hiddenCount = 0;
    this.frameIndex = new Map();
    this.dirty = true;
  }

  destroy() {
    this.running = false;
    cancelAnimationFrame(this.raf);
    this.resizeObserver.disconnect();
  }

  readColors() {
    this.colors = {
      text: cssVar("--text", "#e6eaf2"),
      muted: cssVar("--muted", "#8a94a8"),
      border: cssVar("--border", "#242c3d"),
      panel: cssVar("--panel", "#131821"),
      ok: cssVar("--status-passed", "#9ece6a"),
      fail: cssVar("--status-failed", "#f7768e"),
      warn: cssVar("--status-rejected", "#e0af68"),
      budget: cssVar("--status-budget", "#ff9e64"),
      accent: cssVar("--accent", "#6ea8fe"),
    };
    this.stageColor = STAGES.map((s) => cssVar(s.color, "#7aa2f7"));
    this.dirty = true;
  }

  // Sizes the canvas to its container. Uses the content box, so the border is not drawn over.
  resize() {
    const parent = this.canvas.parentElement || this.canvas;
    const rect = parent.getBoundingClientRect();
    this.width = Math.max(MIN_WIDTH, Math.floor(parent.clientWidth || rect.width));
    this.height = Math.max(MIN_HEIGHT, Math.floor(parent.clientHeight || rect.height));
    this.dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.floor(this.width * this.dpr);
    this.canvas.height = Math.floor(this.height * this.dpr);
    this.canvas.style.width = `${this.width}px`;
    this.canvas.style.height = `${this.height}px`;
    this.readColors();
    this.dirty = true;
  }

  // Feed a batch of events from the stream.
  //
  // Only the last MAX_LIVE_EVENTS are animated. Older events in the batch (a backlog) are applied to the
  // model and copied into each card's visual state at once, so no pipeline is missing from the stage.
  // Animations still queued from earlier batches are applied first, so they cannot overwrite newer state.
  ingest(events) {
    const now = performance.now();
    const cut = Math.max(0, events.length - MAX_LIVE_EVENTS);
    if (cut > 0) this.flushQueue(now);
    for (let i = 0; i < events.length; i++) {
      if (i < cut) {
        this.model.apply(events[i], false);
        this.syncVisual(events[i].pipeline_id, now);
        continue;
      }
      const effects = this.model.apply(events[i], true);
      for (const effect of effects) this.queue.push({ due: now + (effect.at || 0), effect });
    }
    this.dirty = true;
  }

  // Applies every queued effect that changes state (spawn, node, stale, status) at once, and drops the
  // motion effects (packets, floats, bursts, arcs, shakes). The state effects are then overridden by
  // syncVisual, so this only decides how a backlog starts.
  flushQueue(now) {
    const pending = this.queue;
    this.queue = [];
    for (const item of pending) {
      const { type } = item.effect;
      if (type === "spawn" || type === "node" || type === "stale" || type === "status") this.applyEffect(item.effect, now);
    }
  }

  // Copies the model's current state for one pipeline into its visual card, with no animation.
  syncVisual(pid, now) {
    const card = pid ? this.model.cards.get(pid) : null;
    if (!card) return;
    // A card created here is treated as already grown, so it does not play its entrance animation.
    const v = this.ensureVCard(pid, now - 1000);
    v.nodes = card.stations.map((s) => ({ state: s.state, text: s.text || "", flashAt: -1, stale: Boolean(s.stale) }));
    v.status = card.status;
  }

  // ----- layout ---------------------------------------------------------------------------

  // The normal grid when every pipeline fits in it, otherwise the compact grid.
  layout() {
    const normal = this.gridFor(NORMAL);
    if (this.model.order.length <= normal.capacity) return normal;
    return this.gridFor(COMPACT);
  }

  // Cards stretch to fill the width, within the size's minimum and maximum width.
  gridFor(size) {
    const usableW = this.width - PAD * 2;
    const cols = Math.max(1, Math.floor((usableW + size.gap) / (size.minW + size.gap)));
    const cardW = Math.min(size.maxW, (usableW - (cols - 1) * size.gap) / cols);
    const rows = Math.max(1, Math.floor((this.height - HEADER_H - PAD + size.gap) / (size.cardH + size.gap)));
    const gridW = cols * cardW + (cols - 1) * size.gap;
    const left = PAD + Math.max(0, (usableW - gridW) / 2);
    return {
      compact: size.compact, cols, rows, left, top: HEADER_H, capacity: cols * rows, cardW,
      cardH: size.cardH, gap: size.gap, trackY: size.trackY,
    };
  }

  cardRect(index, layout) {
    const col = index % layout.cols;
    const row = Math.floor(index / layout.cols);
    return {
      x: layout.left + col * (layout.cardW + layout.gap),
      y: layout.top + row * (layout.cardH + layout.gap),
      w: layout.cardW,
      h: layout.cardH,
      compact: layout.compact,
      trackY: layout.trackY,
    };
  }

  stationPos(rect, index) {
    const inner = rect.w - 48;
    return { x: rect.x + 24 + (index * inner) / (STAGES.length - 1), y: rect.y + rect.trackY };
  }

  hubPos() {
    return { x: 46, y: 44 };
  }

  pointOf(event) {
    const r = this.canvas.getBoundingClientRect();
    return { x: event.clientX - r.left, y: event.clientY - r.top };
  }

  // The card under a point, found from the layout, so it does not depend on a previous hover.
  cardAt(point) {
    const layout = this.layout();
    const cards = this.model.visibleCards(layout.capacity);
    for (let i = 0; i < cards.length; i++) {
      if (cards[i] && this.inside(point, this.cardRect(i, layout))) return cards[i];
    }
    return null;
  }

  // Text is measured once per font and width. The cache is cleared when it grows too large.
  fit(text, maxWidth) {
    const key = `${this.ctx.font}|${maxWidth}|${text}`;
    let value = this.fitCache.get(key);
    if (value === undefined) {
      if (this.fitCache.size >= FIT_CACHE_MAX) this.fitCache.clear();
      value = fitText(this.ctx, text, maxWidth);
      this.fitCache.set(key, value);
    }
    return value;
  }

  // Whether the ambient motion runs: only while the run is live, and not with reduced motion.
  ambient() {
    return !this.reduceMotion && this.model.runStatus === "running";
  }

  // ----- effects -------------------------------------------------------------------------

  ensureVCard(pid, now) {
    let v = this.vcards.get(pid);
    if (!v) {
      v = {
        nodes: STAGES.map(() => ({ state: IDLE, text: "", flashAt: -1 })),
        status: "running",
        spawnAt: now,
        shakeAt: -1,
        burstAt: -1,
        burstTone: "ok",
        finalAt: -1,
      };
      this.vcards.set(pid, v);
    }
    return v;
  }

  applyEffect(effect, now) {
    switch (effect.type) {
      case "spawn": {
        this.ensureVCard(effect.pid, now);
        break;
      }
      case "node": {
        const v = this.ensureVCard(effect.pid, now);
        v.nodes[effect.index] = { state: effect.state, text: effect.text || "", flashAt: now };
        break;
      }
      case "packet": {
        this.packets.push({ pid: effect.pid, from: effect.from, to: effect.to, color: effect.color, start: now, dur: effect.dur || TIMING.packet });
        break;
      }
      case "stale": {
        const v = this.ensureVCard(effect.pid, now);
        if (v.nodes[effect.index]) v.nodes[effect.index] = { ...v.nodes[effect.index], stale: true };
        break;
      }
      case "float": {
        this.floats.push({ pid: effect.pid, index: effect.index, text: effect.text, tone: effect.tone, start: now });
        break;
      }
      case "arc": {
        this.arcs.push({ pid: effect.pid, from: effect.from, to: effect.to, start: now, dur: 900 });
        break;
      }
      case "burst": {
        const v = this.ensureVCard(effect.pid, now);
        v.burstAt = now;
        v.burstTone = effect.tone;
        this.bursts.push({ pid: effect.pid, tone: effect.tone, start: now });
        break;
      }
      case "shake": {
        this.ensureVCard(effect.pid, now).shakeAt = now;
        break;
      }
      case "status": {
        const v = this.ensureVCard(effect.pid, now);
        v.status = effect.status;
        v.finalAt = now;
        break;
      }
      case "hubtext": {
        this.hubText = { text: effect.text, tone: effect.tone, start: now };
        break;
      }
      default:
        break;
    }
  }

  // ----- drawing --------------------------------------------------------------------------

  frame(now) {
    if (!this.running) return;
    this.raf = requestAnimationFrame((t) => this.frame(t));
    if (window.devicePixelRatio && window.devicePixelRatio !== this.dpr) this.resize();
    if (!this.colors.text) this.readColors();

    // Apply due effects in order.
    if (this.queue.length) {
      const due = [];
      this.queue = this.queue.filter((item) => {
        if (item.due <= now) { due.push(item.effect); return false; }
        return true;
      });
      for (const effect of due) this.applyEffect(effect, now);
      if (due.length) this.dirty = true;
    }
    this.prune(now);

    const busy = this.queue.length > 0 || this.packets.length > 0 || this.arcs.length > 0
      || this.floats.length > 0 || this.bursts.length > 0 || Boolean(this.hubText && now - this.hubText.start < 2600);
    // Nothing changed: skip the draw. A live run keeps its ambient motion, at a capped rate.
    if (!busy && !this.dirty) {
      if (!this.ambient() || now - this.lastDraw < IDLE_FRAME_MS) return;
    }
    this.lastDraw = now;
    this.dirty = false;

    const ctx = this.ctx;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, this.width, this.height);
    this.drawBackground(now);

    const layout = this.layout();
    const cards = this.model.visibleCards(layout.capacity);
    this.hiddenCount = this.model.order.length - cards.length;
    this.frameIndex = new Map();
    this.hover = null;
    cards.forEach((card, index) => {
      const rect = this.cardRect(index, layout);
      this.frameIndex.set(card.id, rect);
      const v = this.vcards.get(card.id);
      if (!v) return;
      try {
        this.drawCard(card, v, rect, now);
      } catch (err) {
        // One card with bad data is skipped, so the others still draw and the frame is not lost (R4-FE-09).
        // The canvas state is restored in case the card stopped inside its own save().
        this.ctx.restore();
        if (!this.cardErrorLogged) {
          this.cardErrorLogged = true;
          console.error("A pipeline card could not be drawn and was skipped. Further card errors are not logged.", err);
        }
      }
      if (this.mouse && this.inside(this.mouse, rect)) this.hover = { ...card, rect };
    });
    this.drawHub(now, this.hiddenCount);

    this.drawPackets(now);
    this.drawArcs(now);
    this.drawFloats(now);
    this.drawBursts(now);
    if (this.hover) this.drawTooltip(this.hover);
  }

  prune(now) {
    this.packets = this.packets.filter((p) => now - p.start < p.dur + 60);
    this.arcs = this.arcs.filter((a) => now - a.start < a.dur + 60);
    this.floats = this.floats.filter((f) => now - f.start < 1100);
    this.bursts = this.bursts.filter((b) => now - b.start < 900);
  }

  inside(point, rect) {
    return point.x >= rect.x && point.x <= rect.x + rect.w && point.y >= rect.y && point.y <= rect.y + rect.h;
  }

  onMove(event) {
    this.mouse = this.pointOf(event);
    this.dirty = true;
    this.canvas.style.cursor = this.cardAt(this.mouse)?.ideaId ? "pointer" : "default";
  }

  drawBackground(now) {
    const ctx = this.ctx;
    const g = ctx.createRadialGradient(this.width * 0.15, 0, 40, this.width * 0.3, this.height * 0.6, Math.max(this.width, this.height));
    g.addColorStop(0, "#111a30");
    g.addColorStop(0.55, "#0a0e1a");
    g.addColorStop(1, "#06080f");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, this.width, this.height);

    // Dot grid.
    ctx.fillStyle = "rgba(122,162,247,0.07)";
    for (let x = 14; x < this.width; x += 28) {
      for (let y = 14; y < this.height; y += 28) ctx.fillRect(x, y, 1.2, 1.2);
    }
    // Stars that twinkle slowly while the run is live. Otherwise they stay still.
    const twinkle = this.ambient();
    for (const s of this.stars) {
      const wave = twinkle ? Math.sin(now / 900 + s.p) : 0;
      const alpha = 0.12 + 0.12 * wave;
      ctx.fillStyle = `rgba(200,215,255,${alpha.toFixed(3)})`;
      ctx.beginPath();
      ctx.arc(s.x * this.width, s.y * this.height, s.s, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  drawHub(now, hidden) {
    const ctx = this.ctx;
    const { x, y } = this.hubPos();
    const stats = this.model.stats();
    const live = this.ambient();
    const pulse = live ? 0.5 + 0.5 * Math.sin(now / 420) : 0.5;

    // Halo.
    const halo = ctx.createRadialGradient(x, y, 6, x, y, 58);
    halo.addColorStop(0, hexToRgba(this.stageColor[0], 0.28 + 0.12 * pulse));
    halo.addColorStop(1, hexToRgba(this.stageColor[0], 0));
    ctx.fillStyle = halo;
    ctx.beginPath();
    ctx.arc(x, y, 58, 0, Math.PI * 2);
    ctx.fill();

    // Orbiting dots, one per running pipeline up to twelve. They move only while the run is live.
    const orbiting = Math.min(12, stats.running);
    const turn = live ? now / 1100 : 0;
    for (let i = 0; i < orbiting; i++) {
      const a = turn + (i / Math.max(1, orbiting)) * Math.PI * 2;
      ctx.fillStyle = hexToRgba(this.stageColor[0], 0.9);
      ctx.beginPath();
      ctx.arc(x + Math.cos(a) * 31, y + Math.sin(a) * 31, 2.4, 0, Math.PI * 2);
      ctx.fill();
    }

    // Core.
    const core = ctx.createLinearGradient(x - 18, y - 18, x + 18, y + 18);
    core.addColorStop(0, "#2a3a66");
    core.addColorStop(1, "#141d36");
    ctx.fillStyle = core;
    ctx.beginPath();
    ctx.arc(x, y, 20, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = hexToRgba(this.stageColor[0], 0.6 + 0.3 * pulse);
    ctx.lineWidth = 1.5;
    ctx.stroke();

    ctx.fillStyle = this.colors.text;
    ctx.font = "600 14px ui-monospace, Consolas, monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(String(stats.running), x, y);
    ctx.fillStyle = CANVAS_MUTED;
    ctx.font = `${TEXT_MIN_PX}px system-ui, sans-serif`;
    ctx.fillText("running", x, y + 40);

    // Headline text next to the hub. Every line is fitted to the space to the right of the hub.
    const avail = Math.max(40, this.width - 86 - PAD);
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
    ctx.fillStyle = this.colors.text;
    ctx.font = "600 15px system-ui, sans-serif";
    const total = stats.total ?? "?";
    ctx.fillText(this.fit(`${stats.done} of ${total} pipelines`, avail), 86, 34);
    ctx.font = "12px system-ui, sans-serif";
    ctx.fillStyle = CANVAS_MUTED;
    const tail = [`${stats.passed} passed critic · ${stats.validated} passed validation`,
      `$${stats.spent.toFixed(3)}${stats.budget ? ` of $${stats.budget.toFixed(2)}` : ""}`];
    if (stats.duplicates) tail.push(`${stats.duplicates} duplicate ideas rejected`);
    if (hidden > 0) tail.push(`${hidden} not shown on the stage`);
    ctx.fillText(this.fit(tail.join("  ·  "), avail), 86, 54);

    // Progress bar across the header.
    const barW = Math.max(120, this.width - 86 - PAD);
    ctx.fillStyle = "rgba(122,162,247,0.12)";
    ctx.fillRect(86, 64, barW, 4);
    const fraction = stats.progress ?? 0;
    const grad = ctx.createLinearGradient(86, 0, 86 + barW, 0);
    grad.addColorStop(0, this.stageColor[0]);
    grad.addColorStop(1, this.colors.ok);
    ctx.fillStyle = grad;
    ctx.fillRect(86, 64, barW * fraction, 4);

    // Hub message, on the top line at the right, clear of the headline below it.
    if (this.hubText && now - this.hubText.start < 2600) {
      const alpha = 1 - (now - this.hubText.start) / 2600;
      ctx.globalAlpha = Math.max(0, alpha);
      ctx.fillStyle = this.toneColor(this.hubText.tone);
      ctx.font = "600 12px system-ui, sans-serif";
      ctx.textAlign = "right";
      ctx.fillText(this.fit(this.hubText.text, avail), this.width - PAD, 18);
      ctx.globalAlpha = 1;
      ctx.textAlign = "left";
    }
  }

  toneColor(tone) {
    if (tone === "ok") return this.colors.ok;
    if (tone === "fail") return this.colors.fail;
    if (tone === "warn") return this.colors.warn;
    return this.colors.muted;
  }

  // The border and its soft outer line. Running cards breathe; passed and failed finished cards are fixed.
  cardBorder(v, finished, passed, now) {
    if (!finished) {
      const breathe = this.reduceMotion ? 0.5 : 0.5 + 0.5 * Math.sin(now / 600 + (v.spawnAt % 97));
      return { border: hexToRgba(this.stageColor[0], 0.3 + 0.35 * breathe), glow: hexToRgba(this.stageColor[0], 0.14 * breathe) };
    }
    if (passed) return { border: hexToRgba(this.colors.ok, 0.75), glow: hexToRgba(this.colors.ok, 0.18) };
    return { border: hexToRgba(this.colors.fail, 0.45), glow: null };
  }

  // Shake after a failure, and the entrance animation. Both stop with reduced motion.
  cardMotion(v, now) {
    const appear = this.reduceMotion ? 1 : Math.min(1, Math.max(0, (now - v.spawnAt) / 380));
    const eased = 1 - Math.pow(1 - appear, 3);
    let dx = 0;
    if (!this.reduceMotion && v.shakeAt >= 0 && now - v.shakeAt < 520) {
      const t = (now - v.shakeAt) / 520;
      dx = 7 * Math.sin(t * 42) * Math.exp(-t * 5);
    }
    return { eased, dx };
  }

  drawCard(card, v, rect, now) {
    if (rect.compact) {
      this.drawCompactCard(card, v, rect, now);
      return;
    }
    const ctx = this.ctx;
    const { eased, dx } = this.cardMotion(v, now);
    const finished = v.status !== "running";
    const passed = finished && isPassedOutcome(v.status);
    const alpha = (finished ? FINISHED_ALPHA : 1) * eased;
    const lift = (1 - eased) * 14;
    const x = rect.x + dx;
    const y = rect.y + lift;

    ctx.save();
    ctx.globalAlpha = alpha;

    // Card body.
    const body = ctx.createLinearGradient(x, y, x, y + rect.h);
    body.addColorStop(0, "#161d2f");
    body.addColorStop(1, "#0d1220");
    ctx.fillStyle = body;
    roundRect(ctx, x, y, rect.w, rect.h, 14);
    ctx.fill();

    // Border, with a wider faint line under it for the glow (no shadow blur: that is expensive per frame).
    const { border, glow } = this.cardBorder(v, finished, passed, now);
    if (glow) {
      ctx.strokeStyle = glow;
      ctx.lineWidth = 6;
      roundRect(ctx, x, y, rect.w, rect.h, 14);
      ctx.stroke();
    }
    ctx.strokeStyle = border;
    ctx.lineWidth = 1.2;
    roundRect(ctx, x, y, rect.w, rect.h, 14);
    ctx.stroke();

    // Header: short id, theme pill, attempt.
    ctx.fillStyle = CANVAS_MUTED;
    ctx.font = `500 ${TEXT_MIN_PX}px ui-monospace, Consolas, monospace`;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(`#${card.short}`, x + 14, y + 17);
    if (card.theme) {
      ctx.font = `600 ${TEXT_MIN_PX}px system-ui, sans-serif`;
      const label = this.fit(card.theme.toUpperCase(), rect.w - 110);
      const tw = ctx.measureText(label).width + 14;
      const tx = x + rect.w - 14 - tw;
      ctx.fillStyle = hexToRgba(this.stageColor[0], 0.14);
      roundRect(ctx, tx, y + 9, tw, 16, 8);
      ctx.fill();
      ctx.fillStyle = hexToRgba(this.stageColor[0], 0.95);
      ctx.textAlign = "center";
      ctx.fillText(label, tx + tw / 2, y + 17);
    }
    ctx.textAlign = "left";
    if (card.attempt > 1) {
      ctx.fillStyle = this.colors.warn;
      ctx.font = `600 ${TEXT_MIN_PX}px system-ui, sans-serif`;
      ctx.fillText(`try ${card.attempt}`, x + 14 + 56, y + 17);
    }

    // Idea name.
    ctx.fillStyle = this.colors.text;
    ctx.font = "600 12.5px system-ui, sans-serif";
    ctx.fillText(this.fit(card.idea || "generating idea…", rect.w - 28), x + 14, y + 40);

    // Track.
    const stations = STAGES.map((_, i) => this.stationPos(rect, i));
    for (let i = 0; i < STAGES.length - 1; i++) {
      const a = v.nodes[i];
      const b = v.nodes[i + 1];
      const lit = (a.state === OK || a.state === ACTIVE) && b.state !== IDLE;
      ctx.strokeStyle = lit ? hexToRgba(this.stageColor[i + 1], 0.85) : "rgba(120,130,160,0.22)";
      ctx.lineWidth = lit ? 2 : 1.4;
      ctx.beginPath();
      ctx.moveTo(stations[i].x + dx, stations[i].y - (rect.y - y));
      ctx.lineTo(stations[i + 1].x + dx, stations[i + 1].y - (rect.y - y));
      ctx.stroke();
    }

    // Stations. Each value is fitted to the gap between neighbours, so it never runs into them.
    const spacing = (rect.w - 48) / (STAGES.length - 1);
    STAGES.forEach((stage, i) => {
      const node = v.nodes[i];
      ctx.globalAlpha = (node.stale ? 0.38 : 1) * alpha;
      const sx = stations[i].x + dx;
      const sy = stations[i].y - (rect.y - y);
      const color = this.stageColor[i];
      const popAge = node.flashAt >= 0 ? (now - node.flashAt) / 320 : 1;
      const pop = !this.reduceMotion && popAge < 1 ? 1 + 0.6 * (1 - popAge) * Math.sin(popAge * Math.PI) : 1;
      if (node.state === ACTIVE) {
        const ring = this.reduceMotion ? 10 : 6.5 + 4 + 3 * Math.sin(now / 180);
        ctx.strokeStyle = hexToRgba(color, 0.55);
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.arc(sx, sy, ring, 0, Math.PI * 2);
        ctx.stroke();
      }
      // A soft disc behind lit stations stands in for a shadow blur.
      const halo = haloFor(node.state, color, this.colors.fail);
      if (halo) {
        ctx.fillStyle = hexToRgba(halo.color, halo.alpha);
        ctx.beginPath();
        ctx.arc(sx, sy, halo.radius * pop, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.beginPath();
      ctx.arc(sx, sy, 6.5 * pop, 0, Math.PI * 2);
      if (node.state === OK || node.state === ACTIVE) ctx.fillStyle = color;
      else if (node.state === FAIL) ctx.fillStyle = this.colors.fail;
      else ctx.fillStyle = "#0e1424";
      ctx.fill();
      ctx.strokeStyle = node.state === IDLE ? "rgba(120,130,160,0.5)" : "rgba(255,255,255,0.18)";
      ctx.lineWidth = 1;
      ctx.stroke();

      // Station value above, label below. Both are at least TEXT_MIN_PX.
      if (node.text) {
        ctx.fillStyle = node.state === FAIL ? this.colors.fail : CANVAS_MUTED;
        ctx.font = `600 ${TEXT_MIN_PX}px system-ui, sans-serif`;
        ctx.textAlign = "center";
        const value = this.fit(truncate(node.text, 12), spacing - 1);
        if (value) ctx.fillText(value, sx, sy - 14);
      }
      ctx.fillStyle = node.state === IDLE ? CANVAS_MUTED : node.state === FAIL ? this.colors.fail : color;
      ctx.font = `600 ${TEXT_MIN_PX}px system-ui, sans-serif`;
      ctx.textAlign = "center";
      const label = this.fit(stage.short, spacing - 1);
      if (label) ctx.fillText(label, sx, sy + 20);
      ctx.globalAlpha = alpha;
    });
    ctx.textAlign = "left";

    // Result badge in the footer when the pipeline has finished.
    if (finished) {
      const label = card.status.replace(/_/g, " ");
      ctx.font = `600 ${TEXT_MIN_PX}px system-ui, sans-serif`;
      const bw = ctx.measureText(label).width + 16;
      const bx = x + 14;
      const by = y + rect.h - 22;
      ctx.fillStyle = passed ? hexToRgba(this.colors.ok, 0.18) : hexToRgba(this.colors.fail, 0.16);
      roundRect(ctx, bx, by, bw, 15, 7.5);
      ctx.fill();
      ctx.fillStyle = passed ? this.colors.ok : this.colors.fail;
      ctx.textBaseline = "middle";
      ctx.fillText(label, bx + 8, by + 8);
    }
    ctx.restore();
  }

  // A small card for large runs: id and status, a row of station dots, and the idea name. No station text.
  drawCompactCard(card, v, rect, now) {
    const ctx = this.ctx;
    const { eased, dx } = this.cardMotion(v, now);
    const finished = v.status !== "running";
    const passed = finished && isPassedOutcome(v.status);
    const alpha = (finished ? FINISHED_ALPHA : 1) * eased;
    const lift = (1 - eased) * 10;
    const x = rect.x + dx;
    const y = rect.y + lift;

    ctx.save();
    ctx.globalAlpha = alpha;
    const body = ctx.createLinearGradient(x, y, x, y + rect.h);
    body.addColorStop(0, "#161d2f");
    body.addColorStop(1, "#0d1220");
    ctx.fillStyle = body;
    roundRect(ctx, x, y, rect.w, rect.h, 10);
    ctx.fill();
    const { border } = this.cardBorder(v, finished, passed, now);
    ctx.strokeStyle = border;
    ctx.lineWidth = 1.2;
    roundRect(ctx, x, y, rect.w, rect.h, 10);
    ctx.stroke();

    // Header: id on the left, status (or "running") on the right.
    ctx.textBaseline = "middle";
    ctx.font = `500 ${TEXT_MIN_PX}px ui-monospace, Consolas, monospace`;
    ctx.fillStyle = CANVAS_MUTED;
    ctx.textAlign = "left";
    ctx.fillText(`#${card.short}`, x + 10, y + 13);
    ctx.font = `600 ${TEXT_MIN_PX}px system-ui, sans-serif`;
    ctx.textAlign = "right";
    if (finished) {
      ctx.fillStyle = passed ? this.colors.ok : this.colors.fail;
      ctx.fillText(this.fit(card.status.replace(/_/g, " "), rect.w * 0.55), x + rect.w - 10, y + 13);
    } else {
      ctx.fillStyle = this.stageColor[0];
      ctx.fillText("running", x + rect.w - 10, y + 13);
    }

    // Station dots, no labels. Lit stations are filled and carry a halo.
    const stations = STAGES.map((_, i) => this.stationPos(rect, i));
    const shift = rect.y - y;
    for (let i = 0; i < STAGES.length - 1; i++) {
      const a = v.nodes[i];
      const b = v.nodes[i + 1];
      const lit = (a.state === OK || a.state === ACTIVE) && b.state !== IDLE;
      ctx.strokeStyle = lit ? hexToRgba(this.stageColor[i + 1], 0.85) : "rgba(120,130,160,0.22)";
      ctx.lineWidth = lit ? 2 : 1.2;
      ctx.beginPath();
      ctx.moveTo(stations[i].x + dx, stations[i].y - shift);
      ctx.lineTo(stations[i + 1].x + dx, stations[i + 1].y - shift);
      ctx.stroke();
    }
    STAGES.forEach((_, i) => {
      const node = v.nodes[i];
      const sx = stations[i].x + dx;
      const sy = stations[i].y - shift;
      const color = this.stageColor[i];
      const halo = haloFor(node.state, color, this.colors.fail);
      if (halo) {
        ctx.fillStyle = hexToRgba(halo.color, halo.alpha);
        ctx.beginPath();
        ctx.arc(sx, sy, halo.radius * 0.8, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.beginPath();
      ctx.arc(sx, sy, 4, 0, Math.PI * 2);
      if (node.state === OK || node.state === ACTIVE) ctx.fillStyle = color;
      else if (node.state === FAIL) ctx.fillStyle = this.colors.fail;
      else ctx.fillStyle = "#0e1424";
      ctx.fill();
    });

    // The idea name, fitted to the card.
    ctx.textBaseline = "alphabetic";
    ctx.textAlign = "left";
    ctx.fillStyle = this.colors.text;
    ctx.font = `600 ${TEXT_MIN_PX}px system-ui, sans-serif`;
    ctx.fillText(this.fit(card.idea || "generating idea…", rect.w - 20), x + 10, y + 52);
    ctx.restore();
  }

  drawPackets(now) {
    const ctx = this.ctx;
    for (const p of this.packets) {
      const t = easeInOut(Math.min(1, Math.max(0, (now - p.start) / p.dur)));
      const rect = this.findRect(p.pid);
      if (!rect) continue;
      const from = p.from === "hub" ? this.hubPos() : this.stationPos(rect.rect, p.from);
      const to = p.to === "card" ? { x: rect.rect.x + rect.rect.w / 2, y: rect.rect.y + 6 } : this.stationPos(rect.rect, p.to);
      const bow = p.from === "hub" ? -40 : -12;
      const x = from.x + (to.x - from.x) * t;
      const y = from.y + (to.y - from.y) * t + Math.sin(Math.PI * t) * bow;
      const color = p.color.startsWith("--") ? cssVar(p.color, "#7aa2f7") : p.color;
      // Trail.
      for (let k = 1; k <= 6; k++) {
        const tt = Math.max(0, t - k * 0.035);
        const tx = from.x + (to.x - from.x) * tt;
        const ty = from.y + (to.y - from.y) * tt + Math.sin(Math.PI * tt) * bow;
        ctx.fillStyle = hexToRgba(color, 0.35 * (1 - k / 7));
        ctx.beginPath();
        ctx.arc(tx, ty, 3.2 * (1 - k / 8), 0, Math.PI * 2);
        ctx.fill();
      }
      // The packet, with a soft disc behind it.
      ctx.fillStyle = hexToRgba(color, 0.25);
      ctx.beginPath();
      ctx.arc(x, y, 7, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, y, 3.8, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // Revision loops are drawn on the normal cards only. A compact card shows the failed station as a red dot.
  drawArcs(now) {
    const ctx = this.ctx;
    for (const a of this.arcs) {
      const rect = this.findRect(a.pid);
      if (!rect || rect.rect.compact) continue;
      const from = this.stationPos(rect.rect, a.from);
      const to = this.stationPos(rect.rect, a.to);
      const t = easeInOut(Math.min(1, Math.max(0, (now - a.start) / a.dur)));
      const cx = (from.x + to.x) / 2;
      const cy = Math.max(from.y, to.y) + 60;
      const fade = 1 - Math.max(0, (now - a.start - a.dur * 0.6) / (a.dur * 0.4));
      // The loop itself, dashed.
      ctx.strokeStyle = hexToRgba(this.colors.warn, 0.7 * Math.max(0, fade));
      ctx.lineWidth = 1.6;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(from.x, from.y + 8);
      ctx.quadraticCurveTo(cx, cy, to.x, to.y + 8);
      ctx.stroke();
      ctx.setLineDash([]);
      // The packet riding the loop back to the coder.
      const px = (1 - t) * (1 - t) * from.x + 2 * (1 - t) * t * cx + t * t * to.x;
      const py = (1 - t) * (1 - t) * (from.y + 8) + 2 * (1 - t) * t * cy + t * t * (to.y + 8);
      ctx.fillStyle = hexToRgba(this.colors.warn, 0.25);
      ctx.beginPath();
      ctx.arc(px, py, 7, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = this.colors.warn;
      ctx.beginPath();
      ctx.arc(px, py, 3.6, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // Floats rise a little from the station's value row. They are not drawn on compact cards.
  drawFloats(now) {
    const ctx = this.ctx;
    ctx.textAlign = "center";
    for (const f of this.floats) {
      const rect = this.findRect(f.pid);
      if (!rect || rect.rect.compact) continue;
      const t = (now - f.start) / 1100;
      const pos = f.index >= 0 ? this.stationPos(rect.rect, f.index) : { x: rect.rect.x + rect.rect.w / 2, y: rect.rect.y };
      ctx.globalAlpha = Math.max(0, 1 - t);
      ctx.fillStyle = this.toneColor(f.tone);
      ctx.font = `600 ${TEXT_MIN_PX}px ui-monospace, Consolas, monospace`;
      const text = this.fit(f.text, (rect.rect.w - 48) / (STAGES.length - 1) * 2 - 4);
      if (text) ctx.fillText(text, pos.x, pos.y - 30 - t * 12);
    }
    ctx.globalAlpha = 1;
    ctx.textAlign = "left";
  }

  drawBursts(now) {
    const ctx = this.ctx;
    for (const b of this.bursts) {
      const rect = this.findRect(b.pid);
      if (!rect) continue;
      const t = (now - b.start) / 900;
      const cx = rect.rect.x + rect.rect.w / 2;
      const cy = rect.rect.y + rect.rect.h / 2;
      const color = b.tone === "ok" ? this.colors.ok : this.colors.fail;
      ctx.strokeStyle = hexToRgba(color, 0.7 * (1 - t));
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(cx, cy, 20 + t * rect.rect.w * 0.6, 0, Math.PI * 2);
      ctx.stroke();
      if (b.tone === "ok" && !this.reduceMotion) {
        for (let k = 0; k < 12; k++) {
          const a = (k / 12) * Math.PI * 2;
          const r = 16 + t * 90;
          ctx.fillStyle = hexToRgba(color, 0.8 * (1 - t));
          ctx.beginPath();
          ctx.arc(cx + Math.cos(a) * r, cy + Math.sin(a) * r, 2.2 * (1 - t) + 0.6, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }
  }

  drawTooltip(card) {
    const ctx = this.ctx;
    const lines = [
      card.idea || "generating idea…",
      `theme: ${card.theme || "n/a"}   status: ${card.status.replace(/_/g, " ")}`,
      `attempt ${card.attempt}   cost $${(card.cost || 0).toFixed(4)}`,
    ];
    if (card.lastSharpe != null && Number.isFinite(card.lastSharpe)) lines.push(`last train Sharpe ${card.lastSharpe.toFixed(2)}`);
    if (card.ideaId) lines.push("click to open the signal inspector");
    ctx.font = "12px system-ui, sans-serif";
    const w = Math.min(340, Math.max(...lines.map((l) => ctx.measureText(l).width))) + 24;
    const h = lines.length * 18 + 16;
    let x = this.mouse.x + 16;
    let y = this.mouse.y + 16;
    if (x + w > this.width - 8) x = this.mouse.x - w - 16;
    if (y + h > this.height - 8) y = this.mouse.y - h - 16;
    ctx.fillStyle = "rgba(10,14,24,0.94)";
    ctx.strokeStyle = "rgba(122,162,247,0.35)";
    ctx.lineWidth = 1;
    roundRect(ctx, x, y, w, h, 10);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = this.colors.text;
    ctx.textBaseline = "middle";
    lines.forEach((line, i) => {
      ctx.fillStyle = i === 0 ? this.colors.text : this.colors.muted;
      ctx.font = i === 0 ? "600 12.5px system-ui, sans-serif" : "11.5px system-ui, sans-serif";
      ctx.fillText(truncate(line, 60), x + 12, y + 14 + i * 18);
    });
  }

  // The rectangle of a pipeline's card in the current frame, or null when it is not on the stage.
  findRect(pid) {
    const rect = this.frameIndex.get(pid);
    return rect ? { rect } : null;
  }
}

// The soft disc drawn behind a lit station or a packet. Idle stations have none.
function haloFor(state, color, failColor) {
  if (state === OK) return { color, alpha: 0.28, radius: 9 };
  if (state === ACTIVE) return { color, alpha: 0.35, radius: 11 };
  if (state === FAIL) return { color: failColor, alpha: 0.3, radius: 9 };
  return null;
}

function roundRect(ctx, x, y, w, h, r) {
  const rr = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
}
