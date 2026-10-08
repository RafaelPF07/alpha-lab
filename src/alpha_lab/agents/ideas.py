"""Ideas agent: proposes one daily-equity signal idea per call."""

from __future__ import annotations

from alpha_lab.agents.llm import LLMClient, PromptSpec
from alpha_lab.agents.models import IdeaOutput

TOOL_NAME = "submit_idea"
TOOL_DESCRIPTION = "Submit one new trading signal idea."
AVOID_LIST_SIZE = 30


def format_avoid_list(recent: list) -> str:
    if not recent:
        return "No ideas have been tested yet."
    lines = []
    for row in recent:
        one_liner = " ".join(row["description"].split())[:120]
        lines.append(f"- {row['name']}: {one_liner}")
    return "Ideas already tested. Do not repeat or closely resemble these:\n" + "\n".join(lines)


async def generate_idea(
    llm: LLMClient,
    *,
    prompt: PromptSpec,
    model: str,
    recent_ideas: list,
    pipeline_id: str,
) -> IdeaOutput:
    user_text = (
        f"{format_avoid_list(recent_ideas[:AVOID_LIST_SIZE])}\n\n"
        "Propose one new idea. Call the submit_idea tool."
    )
    result = await llm.structured_call(
        agent="ideas",
        prompt=prompt,
        model=model,
        user_text=user_text,
        output_model=IdeaOutput,
        tool_name=TOOL_NAME,
        tool_description=TOOL_DESCRIPTION,
        pipeline_id=pipeline_id,
    )
    assert isinstance(result, IdeaOutput)
    return result
