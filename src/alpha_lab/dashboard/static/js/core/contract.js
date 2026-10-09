// Facts the server owns, loaded once per page: which pipeline statuses count as passed (DECISION-13) and
// the token thresholds of the pricing tier and the context limit. Pages await loadContract() before they draw
// anything. There are no defaults: a missing value is an error, so a page never shows a guessed number.
import { fetchJson } from "./api.js";

let current = null;

// Expects these lists from /api/statuses (statuses.py): passed_critic, passed_validation and validation_verdicts.
// passed_critic includes validation_fail, because a signal that failed validation did pass the critic.
export function normaliseStatuses(payload) {
  const list = (key) => {
    const value = payload?.[key];
    if (!Array.isArray(value) || value.some((s) => typeof s !== "string")) {
      throw new Error(`/api/statuses has no list named ${key}`);
    }
    return new Set(value);
  };
  return {
    passedCritic: list("passed_critic"),
    passedValidation: list("passed_validation"),
    validationVerdicts: list("validation_verdicts"),
  };
}

// Expects { pricing_tier_tokens: n, context_limit_tokens: n } from /api/limits.
export function normaliseLimits(payload) {
  const positive = (key) => {
    const n = Number(payload?.[key]);
    if (!Number.isFinite(n) || n <= 0) throw new Error(`/api/limits has no positive number named ${key}`);
    return n;
  };
  return { pricingTierTokens: positive("pricing_tier_tokens"), contextLimitTokens: positive("context_limit_tokens") };
}

export async function loadContract() {
  const [statuses, limits] = await Promise.all([fetchJson("/api/statuses"), fetchJson("/api/limits")]);
  current = { ...normaliseStatuses(statuses), ...normaliseLimits(limits) };
  return current;
}

// Tests and the loader set the contract directly.
export function setContract(contract) {
  current = contract;
}

export function contract() {
  if (!current) throw new Error("the dashboard contract is not loaded yet");
  return current;
}

export const isPassedCritic = (status) => contract().passedCritic.has(status);
export const isPassedValidation = (status) => contract().passedValidation.has(status);
export const isValidationVerdict = (status) => contract().validationVerdicts.has(status);

// The colour rule for a finished pipeline: green when it passed validation, or passed the critic and got no
// validation verdict. A validation failure is red, even though it passed the critic.
export const isPassedOutcome = (status) =>
  isPassedValidation(status) || (isPassedCritic(status) && !isValidationVerdict(status));
