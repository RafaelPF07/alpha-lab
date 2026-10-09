"""Coder agent: writes the signal(prices) function for one idea, and revises it after failures."""

from __future__ import annotations

from dataclasses import dataclass

from alpha_lab.agents.llm import LLMClient, PromptSpec
from alpha_lab.agents.models import CoderOutput, IdeaOutput

TOOL_NAME = "submit_code"
TOOL_DESCRIPTION = "Submit the full Python source defining signal(prices)."


@dataclass(frozen=True)
class Revision:
    previous_code: str
    failure: str  # the exact check failure, in plain words
    train_metrics: dict | None = None  # training metrics only, never validation


def build_user_text(idea: IdeaOutput, revision: Revision | None) -> str:
    text = (
        f"Idea name: {idea.name}\n"
        f"Hypothesis: {idea.hypothesis}\n"
        f"Description: {idea.description}\n"
        f"Expected holding period: {idea.expected_holding_period_days} trading days\n"
        f"The harness takes your weights every {idea.expected_holding_period_days} trading days and holds them in between. "
        "Write the signal for that rebalance frequency, not for daily trading.\n"
    )
    if revision is None:
        return text + "\nWrite the signal. Call the submit_code tool."
    metrics = ""
    if revision.train_metrics:
        metrics = "Training metrics for this code:\n" + "\n".join(
            f"- {k}: {v}" for k, v in revision.train_metrics.items()
        )
    return (
        text
        + "\nYour previous code:\n```python\n"
        + revision.previous_code
        + "\n```\n"
        + f"It failed this check:\n{revision.failure}\n"
        + (metrics + "\n" if metrics else "")
        + "\nFix the problem and call the submit_code tool with the full corrected source."
    )


async def write_signal_code(
    llm: LLMClient,
    *,
    prompt: PromptSpec,
    model: str,
    idea: IdeaOutput,
    revision: Revision | None,
    pipeline_id: str,
    idea_id: str,
) -> CoderOutput:
    result = await llm.structured_call(
        agent="coder",
        prompt=prompt,
        model=model,
        user_text=build_user_text(idea, revision),
        output_model=CoderOutput,
        tool_name=TOOL_NAME,
        tool_description=TOOL_DESCRIPTION,
        pipeline_id=pipeline_id,
        idea_id=idea_id,
    )
    assert isinstance(result, CoderOutput)
    return result
