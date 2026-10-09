// What the charts show in edge cases: no data, unknown budget, unusual numbers, user zoom.
import { describe, it, beforeEach } from "node:test";
import assert from "node:assert/strict";
import { loadStatic, installFakeEcharts, fakeElement } from "./_static.mjs";

let charts;
beforeEach(() => { charts = installFakeEcharts(); });

const CURVE = { name: "idea-a", dates: ["2020-01-02", "2020-01-03"], equity: [1, 1.01], drawdown: [0, -0.01],
  train_end: "2019-12-31", validation_end: "2020-06-30", holdout: "locked" };

describe("research charts with no data", () => {
  for (const [file, label] of [["viz/equity.js", "equity"], ["viz/drawdown.js", "drawdown"]]) {
    it(`${label}: an empty run clears the previous curves and says why (FE-03)`, async () => {
      const mod = await loadStatic(file);
      const chart = mod.init(fakeElement());
      mod.update([CURVE]);
      assert.ok(chart.option.series.length >= 1, "the first run draws its curve");
      mod.update([]);
      assert.ok(chart.clearCount >= 1, "the chart is cleared");
      assert.match(chart.option.title.text, /no signal reached validation/);
      assert.equal(chart.option.series, undefined, "no series from the previous run remain");
    });
  }
});

describe("budget and projection", () => {
  it("the gauge says the budget is unknown instead of showing $1.00 (FE-15)", async () => {
    const gauge = await loadStatic("viz/budget_gauge.js");
    const chart = gauge.init(fakeElement());
    gauge.update({ spent: 0.123, budget: null });
    const text = chart.option.series[0].detail.formatter(0.123);
    assert.match(text, /budget unknown/);
    assert.doesNotMatch(text, /\$1\.00/);
  });

  it("the spend projection never runs past one more run length (FE-15)", async () => {
    const spend = await loadStatic("viz/budget_spend.js");
    const chart = spend.init(fakeElement());
    const runStart = "2026-10-09T00:00:00.000+00:00";
    const runEnd = "2026-10-09T00:10:00.000+00:00";
    spend.update({
      spendSeries: [{ t: runStart, spent: 0 }, { t: runEnd, spent: 0.01 }],
      budget: 2, spent: 0.01, spendRate: 0.001, runStart, runEnd,
      projectedExhaustion: "2026-10-12T00:00:00.000+00:00",
    });
    const projected = chart.option.series[1].data;
    const horizon = Date.parse(runEnd) + (Date.parse(runEnd) - Date.parse(runStart));
    assert.ok(projected.at(-1)[0] <= horizon, "the projection stops at the horizon");
  });
});

describe("luck and large inputs", () => {
  it("the noise-line label never says N=undefined", async () => {
    const luck = await loadStatic("viz/luck.js");
    const chart = luck.init(fakeElement());
    luck.update({ luck: { train_sharpes: [0.1, 0.5], expected_max: 0.8 }, points: [] });
    const label = chart.option.series[0].markLine.data[0].name;
    assert.doesNotMatch(label, /undefined/);
  });

  it("with no expected maximum: no noise line, the axis ignores zero, and the text says n/a (R4-F1 request)", async () => {
    const luck = await loadStatic("viz/luck.js");
    const element = fakeElement();
    const chart = luck.init(element);
    luck.update({ luck: { train_sharpes: [1.2, 1.5, 1.8], expected_max: null, n_trials: 3 }, points: [] });
    const series = chart.option.series[0];
    assert.deepEqual(series.markLine.data, [], "no noise line is drawn");
    assert.equal(chart.option.title.text, "noise max: n/a");
    assert.match(element.getAttribute("aria-label"), /Expected best from noise with 3 trials: n\/a/);
    assert.ok(Number(chart.option.xAxis.data[0]) > 0.5, `the first bin starts at ${chart.option.xAxis.data[0]}, not at zero`);
  });

  it("luck histogram handles a very large number of trials without a RangeError (FE-20)", async () => {
    const luck = await loadStatic("viz/luck.js");
    const chart = luck.init(fakeElement());
    const sharpes = Array.from({ length: 200000 }, (_, i) => (i % 1000) / 1000 - 0.5);
    luck.update({ luck: { train_sharpes: sharpes, expected_max: 0.8, n_trials: 200000 }, points: [] });
    assert.equal(chart.option.series[0].data.reduce((a, b) => a + b, 0), 200000);
  });

  it("cost by agent handles a very large number of calls without a RangeError (FE-20)", async () => {
    const stack = await loadStatic("viz/budget_stack.js");
    const chart = stack.init(fakeElement());
    const calls = Array.from({ length: 200000 }, (_, i) => ({
      t: new Date(Date.UTC(2026, 9, 9) + i * 1000).toISOString(), cost: 0.0001,
    }));
    stack.update({ agentSpend: { ideas: calls } });
    assert.ok(chart.option.series.length === 1);
  });
});

describe("funnel and galaxy", () => {
  it("a funnel link never carries more ideas than the stage before it (FE-05)", async () => {
    const funnel = await loadStatic("viz/funnel.js");
    const { links } = funnel.buildGraph({ generated: 5, coded: 9, static: 2 }, {});
    const intoCoded = links.find((l) => l.target === "coded");
    assert.ok(intoCoded === undefined || intoCoded.value <= 5, "coded cannot exceed generated");
  });

  it("the galaxy tooltip shows n/a, not 0.000, for a missing deflated Sharpe (FE-20)", async () => {
    const galaxy = await loadStatic("viz/galaxy.js");
    const chart = galaxy.init(fakeElement());
    galaxy.update({
      points: [{ attempt_id: "a1", name: "idea", theme: "t", train_sharpe: 1, validation_sharpe: 0.5,
        deflated_sharpe: null, turnover: 0.1, revisions: 1, passed: true }],
      luck: { train_sharpes: [1], expected_max: 0.8, n_trials: 3 },
    });
    const point = chart.option.series[0].data[0];
    const text = chart.option.tooltip.formatter({ data: point, seriesName: "t" });
    assert.match(text, /deflated n\/a/);
  });

  it("the galaxy sets a text alternative on its container (FE-08)", async () => {
    const galaxy = await loadStatic("viz/galaxy.js");
    const container = fakeElement();
    galaxy.init(container);
    galaxy.update({
      points: [{ attempt_id: "a1", name: "idea", theme: "t", train_sharpe: 1, validation_sharpe: 0.5,
        deflated_sharpe: 0.2, turnover: 0.1, revisions: 1, passed: true }],
      luck: { train_sharpes: [1], expected_max: 0.8, n_trials: 3 },
    });
    assert.equal(container.getAttribute("role"), "img");
    assert.match(container.getAttribute("aria-label"), /1 signal that reached validation/);
  });
});

describe("swimlanes zoom", () => {
  const snapshot = () => ({
    pipelines: [{ id: "abcdef12", started: "2026-10-09T00:00:00.000+00:00", finished: null, status: "running",
      theme: "t", tokens: 0, markers: [], segments: [] }],
  });

  it("a zoom the user made survives the next redraw (FE-10)", async () => {
    const swimlanes = await loadStatic("viz/swimlanes.js");
    const chart = swimlanes.init(fakeElement());
    swimlanes.setAutoscroll(false);
    swimlanes.update(snapshot());
    chart.handlers.datazoom({ batch: [{ start: 40, end: 60 }] });
    swimlanes.update(snapshot());
    assert.equal(chart.option.dataZoom[0].start, 40);
    assert.equal(chart.option.dataZoom[0].end, 60);
  });

  it("auto-scroll keeps the newest end of the time axis in view (FE-10)", async () => {
    const swimlanes = await loadStatic("viz/swimlanes.js");
    const chart = swimlanes.init(fakeElement());
    swimlanes.setAutoscroll(true);
    swimlanes.update(snapshot());
    chart.handlers.datazoom({ batch: [{ start: 40, end: 60 }] });
    swimlanes.update(snapshot());
    assert.equal(chart.option.dataZoom[0].end, 100);
    assert.equal(chart.option.dataZoom[0].start, 80);
  });
});
