// HTML escaping for every ledger value that reaches innerHTML or an HTML tooltip. Ledger text (idea
// names, themes, statuses, agent names) is written by an LLM and must never be parsed as markup.
const ENTITIES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };

export function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ENTITIES[c]);
}
