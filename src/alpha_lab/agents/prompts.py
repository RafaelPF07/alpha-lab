"""Versioned prompt files. The version is the file name, so every ledger row says which prompt ran."""

from __future__ import annotations

from alpha_lab.agents.llm import PromptSpec
from alpha_lab.config import PROJECT_ROOT

PROMPTS_DIR = PROJECT_ROOT / "prompts"


def load_prompt(name: str) -> PromptSpec:
    path = PROMPTS_DIR / f"{name}.md"
    return PromptSpec(name=name, text=path.read_text(encoding="utf-8").strip())
