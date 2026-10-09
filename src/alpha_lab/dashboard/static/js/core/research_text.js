// Wording for the research pages (R4-FE-08, R4-WHOLE-01). A signal that reached validation has a validation
// result. A signal passed validation only when its own passed flag is true. The pages never call a signal
// "validated" (DECISION-13).
import { isPassedValidation, isValidationVerdict } from "./contract.js";

// A point's status is the attempt's stored status (R5-F4). It is read with the server's sets (DECISION-13).
// A point with no status (an older payload) falls back to its passed flag.
const hasStatus = (p) => typeof p.status === "string";
const reachedValidation = (p) => (hasStatus(p) ? isValidationVerdict(p.status) : true);
const passedValidation = (p) => (hasStatus(p) ? isPassedValidation(p.status) : p.passed === true);

// Counts from the points: how many reached validation, and how many of those passed it.
export function validationCounts(points) {
  const reached = points.filter(reachedValidation);
  return { reached: reached.length, passed: reached.filter(passedValidation).length };
}

// The run note under the research heading.
export function runNote(points, trials, dsrReason) {
  const { reached, passed } = validationCounts(points);
  const dsr = dsrReason ? ` · deflated Sharpe not defined: ${dsrReason}` : "";
  return `${reached} reached validation, ${passed} passed · ${trials} trials in the ledger${dsr}`;
}

// The outcome of one signal, from its status (or its passed flag, for an older payload).
export function signalOutcome(point) {
  return passedValidation(point) ? "passed validation" : "failed validation";
}

// "1 signal that reached validation", "3 signals that reached validation": for chart text and headings.
export function reachedText(n) {
  return `${n} ${n === 1 ? "signal" : "signals"} that reached validation`;
}
