// The holdout note and the LLM-calls table of the signal inspector. Pure functions, so the tests can load them.
// Every ledger value that reaches innerHTML goes through esc() (R4-FE-12).
import { esc } from "./escape.js";
import { clockOf } from "./runstatus.js";

// What the holdout row says. A holdout still marked running is "stale" once the server says so (R3-SEC-11).
// Started times are shown as HH:MM only, so the note contains no ledger text.
// A failed holdout used the one holdout for its idea, so it is not "not run" and cannot be run again (R4-FE-07).
export function holdoutNote(holdout) {
  switch (holdout.status) {
    case "complete":
      return "Holdout run recorded.";
    case "running": {
      const started = Date.parse(holdout.started_at || "");
      const since = Number.isFinite(started) ? ` since ${clockOf(started)}` : "";
      return holdout.stale
        ? `<span class="stale">Holdout run stale${since}: it is still marked running, but has not finished in time. Its numbers stay hidden.</span>`
        : `Holdout running${since}. Its numbers stay hidden until it completes.`;
    }
    case "error":
      return "Holdout run failed. It used the one holdout for this idea, so it cannot be run again. Its numbers stay hidden.";
    case "locked":
      return "Holdout not run for this signal. Its numbers stay hidden until the official holdout run.";
    default:
      return "Holdout status is not known to this page. Its numbers stay hidden.";
  }
}

// The LLM calls made for an idea, as a table. fmtCost formats a cost in dollars.
export function callsTable(calls, fmtCost = (x) => String(x ?? "n/a")) {
  const rows = calls.map((c) => `<tr><td>${esc(c.agent)}</td><td class="mono">${esc(c.model)}</td>`
    + `<td class="num">${esc(c.input_tokens ?? "n/a")}</td><td class="num">${esc(c.output_tokens ?? "n/a")}</td>`
    + `<td class="num">${esc(fmtCost(c.cost_usd))}</td></tr>`).join("");
  return `<table class="data"><thead><tr><th>agent</th><th>model</th><th class="num">in</th><th class="num">out</th>`
    + `<th class="num">cost</th></tr></thead><tbody>${rows}</tbody></table>`;
}
