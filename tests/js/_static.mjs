// Loads the dashboard's browser modules under Node for the tests in this folder.
//
// The modules under src/alpha_lab/dashboard/static/js are plain ES modules written for a browser.
// Node only treats them as ES modules when a package.json says so, and this folder must not gain one
// (it would be served). So each test process copies static/js into a temporary folder with a
// package.json marking it as ES modules, and imports the modules from that copy. The helper also
// installs the few browser globals the modules touch, and a fake ECharts. Nothing here uses the
// network.
import { cpSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const SOURCE_JS = join(HERE, "..", "..", "src", "alpha_lab", "dashboard", "static", "js");

// Settings the tests can change before they construct a module that reads them.
export const media = { reduceMotion: false, lightScheme: false };

// Only defines a global when Node does not already provide it.
function defineIfMissing(name, value) {
  if (globalThis[name] === undefined) globalThis[name] = value;
}

defineIfMissing("window", { devicePixelRatio: 1, addEventListener() {}, removeEventListener() {} });
defineIfMissing("document", { documentElement: {}, createElement: () => fakeElement() });
defineIfMissing("getComputedStyle", () => ({ getPropertyValue: () => "" }));
defineIfMissing("requestAnimationFrame", () => 0);
defineIfMissing("cancelAnimationFrame", () => {});
defineIfMissing("ResizeObserver", class {
  constructor(callback) { this.callback = callback; }
  observe() {}
  disconnect() {}
});
defineIfMissing("matchMedia", (query) => ({
  get matches() {
    if (query.includes("reduced-motion")) return media.reduceMotion;
    if (query.includes("light")) return media.lightScheme;
    return false;
  },
  addEventListener() {},
  removeEventListener() {},
  addListener() {},
  removeListener() {},
}));

let copy = null;

function copyStaticJs() {
  const root = mkdtempSync(join(tmpdir(), "alpha-lab-js-"));
  cpSync(SOURCE_JS, join(root, "js"), { recursive: true });
  writeFileSync(join(root, "package.json"), '{"type": "module"}\n', "utf-8");
  process.on("exit", () => rmSync(root, { recursive: true, force: true }));
  return root;
}

// Imports one module from static/js, for example loadStatic("core/api.js").
export async function loadStatic(relative) {
  copy ??= copyStaticJs();
  return import(pathToFileURL(join(copy, "js", relative)).href);
}

// The same shape the server sends from /api/statuses and /api/limits. The tests load it the way a page does,
// so no test depends on a status name the JavaScript itself has to know.
// Mirrors statuses.py: a validation failure still passed the critic.
export const FIXTURE_STATUSES = {
  passed_critic: ["critic_pass", "validation_pass", "validation_fail"],
  passed_validation: ["validation_pass"],
  validation_verdicts: ["validation_pass", "validation_fail"],
};
export const FIXTURE_LIMITS = { pricing_tier_tokens: 100000, context_limit_tokens: 90000 };

export async function useFixtureContract() {
  const { normaliseStatuses, normaliseLimits, setContract } = await loadStatic("core/contract.js");
  setContract({ ...normaliseStatuses(FIXTURE_STATUSES), ...normaliseLimits(FIXTURE_LIMITS) });
}

// A stand-in for a DOM element: enough for the modules to set text, attributes and listeners.
export function fakeElement(extra = {}) {
  const attributes = new Map();
  return {
    innerHTML: "",
    textContent: "",
    hidden: false,
    dataset: {},
    style: {},
    classList: { add() {}, remove() {}, toggle() {} },
    listeners: {},
    addEventListener(type, handler) { this.listeners[type] = handler; },
    dispatchEvent() { return true; },
    setAttribute(name, value) { attributes.set(name, String(value)); },
    getAttribute(name) { return attributes.has(name) ? attributes.get(name) : null; },
    appendChild() {},
    replaceChildren() {},
    ...extra,
  };
}

// Records what the viz modules ask ECharts to draw. Tests read the last option and call its formatters.
export class FakeChart {
  constructor(element) {
    this.element = element;
    this.options = [];
    this.handlers = {};
    this.clearCount = 0;
    this.resizeCount = 0;
  }

  setOption(option, notMerge = false) { this.options.push({ option, notMerge }); }

  get option() { return this.options.at(-1)?.option; }

  clear() { this.clearCount += 1; }

  on(name, handler) { this.handlers[name] = handler; }

  resize() { this.resizeCount += 1; }
}

// Replaces the global echarts with a recorder and returns the list of charts created.
export function installFakeEcharts() {
  const charts = [];
  globalThis.echarts = {
    init(element) {
      const chart = new FakeChart(element);
      charts.push(chart);
      return chart;
    },
  };
  return charts;
}
