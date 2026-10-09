// Pipeline theatre: the canvas half. Plays the effects produced by theatre_model.js in real time.
//
// Every pipeline is drawn as a card with eight stations. Packets travel between stations as stages
// complete, stations light up or fail, revision loops arc back to the coder, and each finished
// pipeline bursts green (passed) or red (failed). While the run is busy the loop runs at display rate,
// so the hub and the card glows keep breathing. With prefers-reduced-motion the ambient animation stops,
// and the loop redraws only when something changes: state changes stay visible, the motion does not.
import { STAGES, IDLE, ACTIVE, OK, FAIL, TIMING, TheatreModel, truncate } from "./theatre_model.js";
import { prefersReducedMotion } from "./common.js";

const CARD_H = 146;
const GAP = 18;
const HEADER_H = 92;
const PAD = 22;
const TRACK_Y = 92; // station row, relative to the card top
const MIN_CARD_W = 252;
const MAX_CARD_W = 320;
const MIN_WIDTH = 280;
const MIN_HEIGHT = 240;
const MAX_LIVE_EVENTS = 60; // a backlog larger than this is fast-forwarded without animation
const IDLE_REDRAW_MS = 250; // with reduced motion, how often the still picture is redrawn

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

  // Feed a batch of events from the stream. Large backlogs are fast-forwarded.
  ingest(events) {
    const now = performance.now();
    const cut = Math.max(0, events.length - MAX_LIVE_EVENTS);
    for (let i = 0; i < events.length; i++) {
      const animate = i >= cut;
      const effects = this.model.apply(events[i], animate);
      for (const effect of effects) this.queue.push({ due: now + (effect.at || 0), effect });
    }
    this.dirty = true;
  }

  // ----- layout ---------------------------------------------------------------------------

  // Cards stretch to fill the width, within sensible minimum and maximum sizes.
  layout() {
    const usableW = this.width - PAD * 2;
    const cols = Math.max(1, Math.floor((usableW + GAP) / (MIN_CARD_W + GAP)));
    const cardW = Math.min(MAX_CARD_W, (usableW - (cols - 1) * GAP) / cols);
    const rows = Math.max(1, Math.floor((this.height - HEADER_H - PAD + GAP) / (CARD_H + GAP)));
    const gridW = cols * cardW + (cols - 1) * GAP;
    const left = PAD + Math.max(0, (usableW - gridW) / 2);
    return { cols, rows, left, top: HEADER_H, capacity: cols * rows, cardW };
  }

  cardRect(index, layout) {
    const col = index % layout.cols;
    const row = Math.floor(index / layout.cols);
    return {
      x: layout.left + col * (layout.cardW + GAP),
      y: layout.top + row * (CARD_H + GAP),
      w: layout.cardW,
      h: CARD_H,
    };
  }

  stationPos(rect, index) {
    const inner = rect.w - 48;
    return { x: rect.x + 24 + (index * inner) / (STAGES.length - 1), y: rect.y + TRACK_Y };
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
    if (this.reduceMotion && !busy && !this.dirty && now - this.lastDraw < IDLE_REDRAW_MS) return;
    this.lastDraw = now;
    this.dirty = false;

    const ctx = this.ctx;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, this.width, this.height);
    this.drawBackground(now);

    const layout = this.layout();
    const cards = this.model.visibleCards(layout.capacity);
    this.hover = null;
    cards.forEach((card, index) => {
      const v = this.vcards.get(card.id);
      if (!v) return;
      const rect = this.cardRect(index, layout);
      this.drawCard(card, v, rect, now);
      if (this.mouse && this.inside(this.mouse, rect)) this.hover = { ...card, rect };
    });
    const archived = this.model.order.length - cards.length;
    this.drawHub(now, archived);

    this.drawPackets(now, layout);
    this.drawArcs(now, layout);
    this.drawFloats(now, layout);
    this.drawBursts(now, layout);
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
    // Stars that twinkle slowly. With reduced motion they stay still.
    for (const s of this.stars) {
      const wave = this.reduceMotion ? 0 : Math.sin(now / 900 + s.p);
      const alpha = 0.12 + 0.12 * wave;
      ctx.fillStyle = `rgba(200,215,255,${alpha.toFixed(3)})`;
      ctx.beginPath();
      ctx.arc(s.x * this.width, s.y * this.height, s.s, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  drawHub(now, archived) {
    const ctx = this.ctx;
    const { x, y } = this.hubPos();
    const stats = this.model.stats();
    const pulse = this.reduceMotion ? 0.5 : 0.5 + 0.5 * Math.sin(now / 420);

    // Halo.
    const halo = ctx.createRadialGradient(x, y, 6, x, y, 58);
    halo.addColorStop(0, hexToRgba(this.stageColor[0], 0.28 + 0.12 * pulse));
    halo.addColorStop(1, hexToRgba(this.stageColor[0], 0));
    ctx.fillStyle = halo;
    ctx.beginPath();
    ctx.arc(x, y, 58, 0, Math.PI * 2);
    ctx.fill();

    // Orbiting dots, one per running pipeline up to twelve. With reduced motion they stay where they are.
    const orbiting = Math.min(12, stats.running);
    const turn = this.reduceMotion ? 0 : now / 1100;
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
    ctx.fillStyle = this.colors.muted;
    ctx.font = "10px system-ui, sans-serif";
    ctx.fillText("running", x, y + 40);

    // Headline text next to the hub. Every line is fitted to the space to the right of the hub.
    const avail = Math.max(40, this.width - 86 - PAD);
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
    ctx.fillStyle = this.colors.text;
    ctx.font = "600 15px system-ui, sans-serif";
    const total = stats.total ?? "?";
    ctx.fillText(fitText(ctx, `${stats.done} of ${total} pipelines`, avail), 86, 34);
    ctx.font = "12px system-ui, sans-serif";
    ctx.fillStyle = this.colors.muted;
    const tail = [`${stats.passed} passed`, `$${stats.spent.toFixed(3)}${stats.budget ? ` of $${stats.budget.toFixed(2)}` : ""}`];
    if (stats.duplicates) tail.push(`${stats.duplicates} duplicate ideas rejected`);
    if (archived > 0) tail.push(`${archived} earlier pipelines archived`);
    ctx.fillText(fitText(ctx, tail.join("  ·  "), avail), 86, 54);

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
      ctx.fillText(fitText(ctx, this.hubText.text, avail), this.width - PAD, 18);
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

  drawCard(card, v, rect, now) {
    const ctx = this.ctx;
    const age = now - v.spawnAt;
    const appear = this.reduceMotion ? 1 : Math.min(1, Math.max(0, age / 380));
    const eased = 1 - Math.pow(1 - appear, 3);
    let dx = 0;
    if (!this.reduceMotion && v.shakeAt >= 0 && now - v.shakeAt < 520) {
      const t = (now - v.shakeAt) / 520;
      dx = 7 * Math.sin(t * 42) * Math.exp(-t * 5);
    }
    const finished = v.status !== "running";
    const failedDone = finished && !PASSED(v.status);
    const alpha = (failedDone ? 0.55 : 1) * eased;
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

    // Border: breathing while running, green for passes, red for failures.
    let border = hexToRgba(this.stageColor[0], 0.28);
    let glow = null;
    if (!finished) {
      const breathe = this.reduceMotion ? 0.5 : 0.5 + 0.5 * Math.sin(now / 600 + (v.spawnAt % 97));
      border = hexToRgba(this.stageColor[0], 0.3 + 0.35 * breathe);
      glow = hexToRgba(this.stageColor[0], 0.25 * breathe);
    } else if (PASSED(v.status)) {
      border = hexToRgba(this.colors.ok, 0.75);
      glow = hexToRgba(this.colors.ok, 0.35);
    } else {
      border = hexToRgba(this.colors.fail, 0.45);
    }
    ctx.strokeStyle = border;
    ctx.lineWidth = 1.2;
    if (glow) { ctx.shadowColor = glow; ctx.shadowBlur = 16; }
    roundRect(ctx, x, y, rect.w, rect.h, 14);
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Header: short id, theme pill, attempt.
    ctx.fillStyle = this.colors.muted;
    ctx.font = "500 10px ui-monospace, Consolas, monospace";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(`#${card.short}`, x + 14, y + 17);
    if (card.theme) {
      ctx.font = "600 9.5px system-ui, sans-serif";
      const label = card.theme.toUpperCase();
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
      ctx.font = "600 9.5px system-ui, sans-serif";
      ctx.fillText(`try ${card.attempt}`, x + 14 + 56, y + 17);
    }

    // Idea name.
    ctx.fillStyle = this.colors.text;
    ctx.font = "600 12.5px system-ui, sans-serif";
    ctx.fillText(fitText(ctx, card.idea || "generating idea…", rect.w - 28), x + 14, y + 40);

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
      ctx.globalAlpha = node.stale ? 0.38 : 1;
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
      ctx.beginPath();
      ctx.arc(sx, sy, 6.5 * pop, 0, Math.PI * 2);
      if (node.state === OK) {
        ctx.fillStyle = color;
        ctx.shadowColor = hexToRgba(color, 0.7);
        ctx.shadowBlur = 10;
      } else if (node.state === ACTIVE) {
        ctx.fillStyle = color;
        ctx.shadowColor = hexToRgba(color, 0.9);
        ctx.shadowBlur = 14;
      } else if (node.state === FAIL) {
        ctx.fillStyle = this.colors.fail;
        ctx.shadowColor = hexToRgba(this.colors.fail, 0.7);
        ctx.shadowBlur = 10;
      } else {
        ctx.fillStyle = "#0e1424";
      }
      ctx.fill();
      ctx.shadowBlur = 0;
      ctx.strokeStyle = node.state === IDLE ? "rgba(120,130,160,0.5)" : "rgba(255,255,255,0.18)";
      ctx.lineWidth = 1;
      ctx.stroke();

      // Station value above, label below.
      if (node.text) {
        ctx.fillStyle = node.state === FAIL ? this.colors.fail : this.colors.muted;
        ctx.font = "500 8.5px ui-monospace, Consolas, monospace";
        ctx.textAlign = "center";
        const value = fitText(ctx, truncate(node.text, 10), spacing - 4);
        if (value) ctx.fillText(value, sx, sy - 14);
      }
      ctx.fillStyle = node.state === IDLE ? "rgba(138,148,168,0.6)" : node.state === FAIL ? this.colors.fail : color;
      ctx.font = "600 7.5px system-ui, sans-serif";
      ctx.textAlign = "center";
      if (ctx.measureText(stage.short).width > spacing - 2) ctx.font = "600 6.5px system-ui, sans-serif";
      const label = fitText(ctx, stage.short, spacing - 2);
      if (label) ctx.fillText(label, sx, sy + 20);
      ctx.globalAlpha = alpha;
    });
    ctx.textAlign = "left";

    // Result badge in the footer when the pipeline has finished.
    if (finished) {
      const label = card.status.replace(/_/g, " ");
      ctx.font = "600 9.5px system-ui, sans-serif";
      const bw = ctx.measureText(label).width + 16;
      const bx = x + 14;
      const by = y + rect.h - 22;
      ctx.fillStyle = PASSED(v.status) ? hexToRgba(this.colors.ok, 0.18) : hexToRgba(this.colors.fail, 0.16);
      roundRect(ctx, bx, by, bw, 15, 7.5);
      ctx.fill();
      ctx.fillStyle = PASSED(v.status) ? this.colors.ok : this.colors.fail;
      ctx.textBaseline = "middle";
      ctx.fillText(label, bx + 8, by + 8);
    }
    ctx.restore();
  }

  drawPackets(now, layout) {
    const ctx = this.ctx;
    for (const p of this.packets) {
      const t = easeInOut(Math.min(1, Math.max(0, (now - p.start) / p.dur)));
      const rect = this.findRect(p.pid, layout);
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
      ctx.shadowColor = hexToRgba(color, 0.9);
      ctx.shadowBlur = 14;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, y, 3.8, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;
    }
  }

  drawArcs(now, layout) {
    const ctx = this.ctx;
    for (const a of this.arcs) {
      const rect = this.findRect(a.pid, layout);
      if (!rect) continue;
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
      ctx.shadowColor = hexToRgba(this.colors.warn, 0.9);
      ctx.shadowBlur = 12;
      ctx.fillStyle = this.colors.warn;
      ctx.beginPath();
      ctx.arc(px, py, 3.6, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;
    }
  }

  drawFloats(now, layout) {
    const ctx = this.ctx;
    ctx.textAlign = "center";
    for (const f of this.floats) {
      const rect = this.findRect(f.pid, layout);
      if (!rect) continue;
      const t = (now - f.start) / 1100;
      const pos = f.index >= 0 ? this.stationPos(rect.rect, f.index) : { x: rect.rect.x + rect.rect.w / 2, y: rect.rect.y };
      ctx.globalAlpha = Math.max(0, 1 - t);
      ctx.fillStyle = this.toneColor(f.tone);
      ctx.font = "600 10px ui-monospace, Consolas, monospace";
      const text = fitText(ctx, f.text, (rect.rect.w - 48) / (STAGES.length - 1) * 2 - 4);
      if (text) ctx.fillText(text, pos.x, pos.y - 26 - t * 22);
    }
    ctx.globalAlpha = 1;
    ctx.textAlign = "left";
  }

  drawBursts(now, layout) {
    const ctx = this.ctx;
    for (const b of this.bursts) {
      const rect = this.findRect(b.pid, layout);
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

  findRect(pid, layout) {
    const cards = this.model.visibleCards(layout.capacity);
    const index = cards.findIndex((c) => c && c.id === pid);
    if (index < 0) return null;
    return { rect: this.cardRect(index, layout) };
  }
}

function PASSED(status) {
  return status === "critic_pass" || status === "validation_pass";
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
