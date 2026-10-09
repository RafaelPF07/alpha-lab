// HTML escaping helper (static/js/core/escape.js).
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { loadStatic } from "./_static.mjs";

const { esc } = await loadStatic("core/escape.js");

describe("esc", () => {
  it("escapes the five HTML-significant characters", () => {
    assert.equal(esc(`<img src=x onerror="a">&'`), "&lt;img src=x onerror=&quot;a&quot;&gt;&amp;&#39;");
  });

  it("renders null and undefined as empty text, and numbers as text", () => {
    assert.equal(esc(undefined), "");
    assert.equal(esc(null), "");
    assert.equal(esc(0), "0");
    assert.equal(esc(0.25), "0.25");
  });
});
