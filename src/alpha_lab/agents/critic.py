"""Critic agent: looks for hidden lookahead, overfitting and fragility in a signal that passed training."""

from __future__ import annotations

from alpha_lab.agents.llm import LLMClient, PromptSpec
from alpha_lab.agents.models import CriticOutput

TOOL_NAME = "submit_critique"
TOOL_DESCRIPTION = "Submit the critique of this signal."


def build_user_text(idea_name: str, hypothesis: str, code: str, metrics: dict) -> str:
    shown = "\n".join(
        f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}" for k, v in metrics.items()
    )
    return (
        f"Idea: {idea_name}\nHypothesis: {hypothesis}\n\n"
        f"Code:\n```python\n{code}\n```\n\n"
        f"Training-period metrics (2010 to train end only):\n{shown}\n\n"
        "Critique it and call the submit_critique tool."
    )


async def critique_signal(
    llm: LLMClient,
    *,
    prompt: PromptSpec,
    model: str,
    idea_name: str,
    hypothesis: str,
    code: str,
    metrics: dict,
    pipeline_id: str,
    idea_id: str,
) -> CriticOutput:
    result = await llm.structured_call(
        agent="critic",
        prompt=prompt,
        model=model,
        user_text=build_user_text(idea_name, hypothesis, code, metrics),
        output_model=CriticOutput,
        tool_name=TOOL_NAME,
        tool_description=TOOL_DESCRIPTION,
        pipeline_id=pipeline_id,
        idea_id=idea_id,
    )
    assert isinstance(result, CriticOutput)
    return result
