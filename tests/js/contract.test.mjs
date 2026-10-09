// The status sets and thresholds that the pages take from the API (DECISION-13). Missing values must fail,
// and the page must never fall back to a number of its own.
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { loadStatic } from "./_static.mjs";

const { normaliseStatuses, normaliseLimits, setContract, contract, isPassedCritic, isPassedValidation, isPassedOutcome } = await loadStatic("core/contract.js");
const { FIXTURE_STATUSES } = await import("./_static.mjs");

describe("dashboard contract", () => {
  it("reads the passed status sets from /api/statuses", () => {
    const sets = normaliseStatuses(FIXTURE_STATUSES);
    assert.deepEqual([...sets.passedCritic].sort(), ["critic_pass", "validation_fail", "validation_pass"]);
    assert.deepEqual([...sets.passedValidation], ["validation_pass"]);
    assert.deepEqual([...sets.validationVerdicts].sort(), ["validation_fail", "validation_pass"]);
  });

  it("refuses a status payload with a missing or malformed list", () => {
    assert.throws(() => normaliseStatuses({ passed_critic: ["critic_pass"] }), /passed_validation/);
    assert.throws(() => normaliseStatuses({ passed_critic: "critic_pass", passed_validation: [], validation_verdicts: [] }), /passed_critic/);
    assert.throws(() => normaliseStatuses({ passed_critic: [1], passed_validation: [], validation_verdicts: [] }), /passed_critic/);
    assert.throws(() => normaliseStatuses({ passed_critic: [], passed_validation: [] }), /validation_verdicts/);
  });

  it("reads the thresholds from /api/limits and refuses zero or missing values", () => {
    const limits = normaliseLimits({ pricing_tier_tokens: 100000, context_limit_tokens: 90000 });
    assert.deepEqual(limits, { pricingTierTokens: 100000, contextLimitTokens: 90000 });
    assert.throws(() => normaliseLimits({ pricing_tier_tokens: 100000 }), /context_limit_tokens/);
    assert.throws(() => normaliseLimits({ pricing_tier_tokens: 0, context_limit_tokens: 90000 }), /pricing_tier_tokens/);
  });

  it("refuses to answer status questions before the contract is loaded", () => {
    setContract(null);
    assert.throws(() => contract(), /not loaded/);
    assert.throws(() => isPassedCritic("critic_pass"), /not loaded/);
  });

  it("answers status questions from the loaded sets", () => {
    setContract({ ...normaliseStatuses(FIXTURE_STATUSES), pricingTierTokens: 1, contextLimitTokens: 2 });
    assert.equal(isPassedCritic("critic_pass"), true);
    assert.equal(isPassedCritic("validation_fail"), true, "a validation failure passed the critic");
    assert.equal(isPassedValidation("critic_pass"), false);
    setContract(null);
  });

  it("colours a finished pipeline green only when it did not fail validation", () => {
    setContract({ ...normaliseStatuses(FIXTURE_STATUSES), pricingTierTokens: 1, contextLimitTokens: 2 });
    assert.equal(isPassedOutcome("validation_pass"), true);
    assert.equal(isPassedOutcome("validation_fail"), false, "a validation failure is red");
    assert.equal(isPassedOutcome("critic_pass"), true, "a critic pass with no verdict is green");
    assert.equal(isPassedOutcome("failed_critic"), false);
    setContract(null);
  });
});
