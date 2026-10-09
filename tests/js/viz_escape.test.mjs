// Every ledger value that reaches innerHTML or an HTML tooltip is escaped (FE-02). Each view gets a
// hostile idea name, theme and status, and the test checks that no element is created.
import { describe, it, beforeEach } from "node:test";
import assert from "node:assert/strict";
import { loadStatic, installFakeEcharts, fakeElement, useFixtureContract } from "./_static.mjs";

// Charts that colour a status need the server's status sets, as a page loads them before drawing.
await useFixtureContract();

const HOSTILE = '<img src=x onerror="window.__xss=(window.__xss||0)+1">';
const NOW = "2026-10-09T00:00:00.000+00:00";

// The text of a formatter's output must contain the escaped payload and no live markup from it.
function assertEscaped(html) {
  assert.equal(typeof html, "string", "formatter returned a string");
  assert.ok(!html.includes("<img"), `no live <img> in: ${html}`);
  assert.ok(html.includes("&lt;img"), `payload is shown as text in: ${html}`);
}

let charts;
beforeEach(() => { charts = installFakeEcharts(); });

describe("HTML tooltips escape ledger text", () => {
  it("galaxy tooltip escapes the idea name and the theme", async () => {
    const galaxy = await loadStatic("viz/galaxy.js");
    const chart = galaxy.init(fakeElement());
    galaxy.update({
      points: [{ attempt_id: "a1", name: HOSTILE, theme: HOSTILE, train_sharpe: 1, validation_sharpe: 0.5,
        deflated_sharpe: 0.3, turnover: 0.1, revisions: 1, passed: true }],
      luck: { train_sharpes: [1], expected_max: 0.8, n_trials: 3 },
    });
    const point = chart.option.series[0].data[0];
    assertEscaped(chart.option.tooltip.formatter({ data: point, seriesName: HOSTILE }));
    assertEscaped(chart.option.tooltip.formatter({ data: {}, seriesName: HOSTILE }));
  });

  it("correlation tooltip escapes the names of both signals", async () => {
    const corr = await loadStatic("viz/correlation.js");
    const chart = corr.init(fakeElement());
    corr.update({ names: [HOSTILE, "beta"], matrix: [[1, 0.42], [0.42, 1]] });
    assertEscaped(chart.option.tooltip.formatter({ data: [0, 1, "0.420"] }));
  });

  it("swimlanes tooltip escapes the agent and theme shown on a segment", async () => {
    const swimlanes = await loadStatic("viz/swimlanes.js");
    const chart = swimlanes.init(fakeElement());
    swimlanes.update({ pipelines: [{ id: "abcdef12", started: NOW, finished: null, status: "running", theme: HOSTILE,
      tokens: 0, markers: [], segments: [{ t: NOW, agent: HOSTILE, latency: 10, status: "ok" }] }] });
    assertEscaped(chart.option.tooltip.formatter({ seriesName: "segments", value: [0, 0, 0, HOSTILE, "ok", HOSTILE] }));
  });

  it("swarm node tooltip escapes the theme and status", async () => {
    const swarm = await loadStatic("viz/swarm.js");
    const chart = swarm.init(fakeElement());
    swarm.update({ pipelines: [{ id: "abcdef12", theme: HOSTILE, status: HOSTILE, tokens: 5 }] });
    const node = chart.option.series[0].data.find((n) => n.pipelineId === "abcdef12");
    assertEscaped(chart.option.tooltip.formatter({ data: node }));
  });

  it("failure anatomy tooltip escapes the status name", async () => {
    const failures = await loadStatic("viz/failures.js");
    const chart = failures.init(fakeElement());
    failures.update([{ status: HOSTILE, count: 2 }]);
    assertEscaped(chart.option.tooltip.formatter({ name: HOSTILE, value: 2 }));
  });
});

describe("event ticker escapes ledger text", () => {
  it("renders hostile names, pipeline ids and agents as text", async () => {
    const ticker = await loadStatic("viz/ticker.js");
    const el = fakeElement();
    ticker.init(el);
    ticker.update([
      { seq: 1, type: "idea_created", timestamp: NOW, pipeline_id: HOSTILE, idea_id: "a", agent: HOSTILE,
        stage: "", payload: { name: HOSTILE } },
      { seq: 2, type: HOSTILE, timestamp: NOW, pipeline_id: 'x" onmouseover="alert(1)', agent: "ideas",
        stage: "", payload: {} },
    ]);
    assertEscaped(el.innerHTML);
    assert.ok(!el.innerHTML.includes('" onmouseover="alert'), "a quote in a pipeline id cannot open an attribute");
  });
});
