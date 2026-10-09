"""The one definition of every status string, and of what "passed" means (DECISION-13).

The pipeline, the ledger report, the dashboard API and the CLI import their status sets from here.
The JavaScript pages receive these sets through /api/statuses and do not hard-code them.

Two different questions are asked of a pipeline, and they are kept apart:
  passed the critic      PASSED_CRITIC      the attempt reached and passed the critic stage
  passed validation      PASSED_VALIDATION  the attempt passed the validation gate
A signal that passed the critic and then failed validation (validation_fail) counts as passing the
critic, and not as passing validation. Never report a single "passes" count.
"""

from __future__ import annotations

# Attempt statuses (attempts.status). "running" is written when an attempt starts, and "abandoned"
# when a run that died is closed by Ledger.abandon_stale_runs.
ATTEMPT_STATUSES: frozenset[str] = frozenset(
    {
        "running",
        "rejected_duplicate_code",
        "failed_static",
        "failed_sandbox",
        "failed_lookahead",
        "failed_backtest",
        "failed_gate",
        "failed_critic",
        "failed_critic_output",
        "critic_pass",
        "validation_pass",
        "validation_fail",
        "aborted_budget",
        "aborted_error",
        "aborted_cancelled",
        "abandoned",
    }
)

# Pipeline-only statuses, written to pipeline_finished and PipelineResult.status.
PIPELINE_ONLY_STATUSES: frozenset[str] = frozenset(
    {"failed_ideas", "failed_coder", "failed_api", "failed_error"}
)

# What a finished pipeline can end with. A pipeline has no "running" or "abandoned" status: a
# pipeline that a dead process left open is closed as aborted_error.
PIPELINE_STATUSES: frozenset[str] = (
    ATTEMPT_STATUSES - {"running", "abandoned"}
) | PIPELINE_ONLY_STATUSES

# Run statuses (runs.status).
RUN_STATUSES: frozenset[str] = frozenset(
    {
        "running",
        "completed",
        "budget_stopped",
        "failed",
        "failed_data",
        "interrupted",
        "abandoned",
    }
)
# failed_data: every pipeline ended in failed_backtest, so no window could be scored (R3-STAT-02).
RUN_FAILED_DATA = "failed_data"

# Pipelines that stopped on an error before a verdict.
ERROR_PIPELINE_STATUSES: frozenset[str] = frozenset({"failed_error", "failed_api"})
# Pipelines that stopped with nothing learned: the ideas or the coder gave no usable output. A run
# where every pipeline ends in one of these, or in an error, is "failed" (R4-AGT-01).
RUN_FAILING_PIPELINE_STATUSES: frozenset[str] = ERROR_PIPELINE_STATUSES | {
    "failed_ideas",
    "failed_coder",
}

ABORTED_STATUSES: frozenset[str] = frozenset(
    {"aborted_budget", "aborted_error", "aborted_cancelled"}
)

# The training gate was passed, so the attempt reached the critic (failed_critic_output is a
# critic that gave no usable review, after the gate).
PASSED_TRAIN_GATE: frozenset[str] = frozenset(
    {"critic_pass", "failed_critic", "failed_critic_output", "validation_pass", "validation_fail"}
)
PASSED_CRITIC: frozenset[str] = frozenset({"critic_pass", "validation_pass", "validation_fail"})
PASSED_VALIDATION: frozenset[str] = frozenset({"validation_pass"})
# The attempt reached the validation gate and received a verdict.
VALIDATION_VERDICTS: frozenset[str] = frozenset({"validation_pass", "validation_fail"})

# llm_calls.status. "cancelled_before_send" is a call cancelled while it waited to retry, when no
# request was in flight, so it is charged nothing. "discarded" is a reply that was billed but that
# the harness could not use (truncated, no tool call, or invalid input). It is not a successful call.
LLM_CALL_STATUSES: frozenset[str] = frozenset(
    {"ok", "error", "cancelled", "cancelled_before_send", "refused", "discarded"}
)

# Holdout rows (holdout_runs.status). "running" is claimed before the holdout is read, and "complete"
# or "error" is written after. The dashboard shows "locked" when no holdout has completed; that is a
# view, not a stored status (R4-WHOLE-05).
HOLDOUT_STATUSES: frozenset[str] = frozenset({"running", "complete", "error"})
HOLDOUT_LOCKED_VIEW = "locked"
HOLDOUT_VIEWS: frozenset[str] = HOLDOUT_STATUSES | {HOLDOUT_LOCKED_VIEW}

__all__ = [
    "ABORTED_STATUSES",
    "ATTEMPT_STATUSES",
    "ERROR_PIPELINE_STATUSES",
    "HOLDOUT_LOCKED_VIEW",
    "HOLDOUT_STATUSES",
    "HOLDOUT_VIEWS",
    "LLM_CALL_STATUSES",
    "PASSED_CRITIC",
    "PASSED_TRAIN_GATE",
    "PASSED_VALIDATION",
    "PIPELINE_ONLY_STATUSES",
    "PIPELINE_STATUSES",
    "RUN_FAILED_DATA",
    "RUN_FAILING_PIPELINE_STATUSES",
    "RUN_STATUSES",
    "VALIDATION_VERDICTS",
]
