// Stand-ins for a browser, so a page module (static/js/pages/*.js) can be imported and run under Node (R4-FE-10).
// The document creates each element on first use, fetch answers from a table of routes, and setInterval keeps
// its callbacks so a test can fire one by hand instead of waiting 15 seconds. Import this helper before the page.
import { fakeElement, loadStatic } from "./_static.mjs";

// A context whose drawing calls do nothing and whose measureText returns a width. Enough for the theatre.
function fakeContext() {
  const gradient = () => ({ addColorStop() {} });
  const special = {
    measureText: (text) => ({ width: String(text).length * 6 }),
    createRadialGradient: gradient,
    createLinearGradient: gradient,
  };
  return new Proxy({}, {
    get(target, prop) {
      if (prop in target) return target[prop];
      if (prop in special) return special[prop];
      return () => undefined;
    },
    set(target, prop, value) {
      target[prop] = value;
      return true;
    },
  });
}

// The stage canvas: a parent with a size, and a context for the theatre to draw on.
export function fakeStageCanvas() {
  const parent = {
    clientWidth: 1036,
    clientHeight: 666,
    getBoundingClientRect() { return { left: 0, top: 0, width: this.clientWidth, height: this.clientHeight }; },
  };
  const ctx = fakeContext();
  return {
    width: 0,
    height: 0,
    style: {},
    parentElement: parent,
    listeners: {},
    addEventListener(type, handler) { this.listeners[type] = handler; },
    getBoundingClientRect() { return parent.getBoundingClientRect(); },
    getContext() { return ctx; },
  };
}

// An element with the few DOM methods the page modules call on lists and the feed.
function pageElement(id) {
  return fakeElement({
    id,
    tabIndex: 0,
    contains: () => false,
    querySelectorAll: () => [],
    querySelector: () => null,
    focus() {},
    insertAdjacentElement() {},
  });
}

export function fakeDocument() {
  const elements = new Map();
  return {
    elements,
    body: { prepend() {} },
    documentElement: {},
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, id === "stage" ? fakeStageCanvas() : pageElement(id));
      return elements.get(id);
    },
    querySelector: () => null,
    querySelectorAll: () => [],
    createElement: (tag) => fakeElement({ tagName: tag, contains: () => false, querySelectorAll: () => [] }),
    createDocumentFragment: () => ({ appendChild() {} }),
    addEventListener() {},
  };
}

export function jsonReply(status, body) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => {
      if (body === undefined) throw new SyntaxError("not JSON");
      return body;
    },
  };
}

// A stand-in for EventSource that records each connection. The tests decide what a stream sends, if anything.
class FakeEventSource {
  constructor(url) {
    this.url = url;
    this.closed = false;
    FakeEventSource.instances.push(this);
  }

  close() { this.closed = true; }
}
FakeEventSource.instances = [];

// Installs document, location, fetch, setInterval and EventSource for one page. routes maps a URL (or a prefix
// ending in *) to { status, body }. Unknown URLs answer 404. Returns the document, the captured intervals, the
// requested URLs and the stream connections.
export function installPage({ path = "/live", search = "", routes = {} }) {
  const doc = fakeDocument();
  const intervals = [];
  const requested = [];
  FakeEventSource.instances = [];
  globalThis.document = doc;
  globalThis.location = { search, href: `http://127.0.0.1:8600${path}${search}`, pathname: path };
  globalThis.setInterval = (fn) => { intervals.push(fn); return intervals.length; };
  globalThis.EventSource = FakeEventSource;
  globalThis.fetch = async (url) => {
    requested.push(url);
    const route = Object.entries(routes).find(([pattern]) => (pattern.endsWith("*") ? url.startsWith(pattern.slice(0, -1)) : url === pattern));
    const hit = route ? route[1] : { status: 404, body: { detail: "not found" } };
    return jsonReply(hit.status, hit.body);
  };
  return { doc, intervals, requested, streams: FakeEventSource.instances };
}

// Lets pending promises (the page's first loads) finish.
export async function settle(ms = 30) {
  await new Promise((resolve) => setTimeout(resolve, ms));
}

export async function loadPage(relative) {
  return loadStatic(relative);
}
