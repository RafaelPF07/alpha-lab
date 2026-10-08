"""Structured outputs the agents must return. Validated by pydantic before anything runs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class IdeaOutput(BaseModel):
    name: str = Field(min_length=3, max_length=80, description="Short unique name for the idea.")
    hypothesis: str = Field(
        min_length=10, max_length=600, description="Why the signal should predict returns."
    )
    description: str = Field(
        min_length=10,
        max_length=1200,
        description="Exactly how the signal ranks or weights stocks, in plain English.",
    )
    expected_holding_period_days: int = Field(
        ge=1, le=252, description="Typical number of trading days a position is held."
    )


class CoderOutput(BaseModel):
    code: str = Field(
        min_length=20,
        max_length=20_000,
        description="Full Python source. It must define signal(prices) and nothing else needs to run.",
    )


class CriticOutput(BaseModel):
    concerns: list[str] = Field(
        max_length=10, description="Specific concerns. Empty if none. Each one short and concrete."
    )
    suspicion_level: Literal["low", "medium", "high"] = Field(
        description="high means the result is probably an artefact of lookahead or overfitting."
    )
    suggested_change: str = Field(
        max_length=800, description="One concrete change to the code, or 'none'."
    )
