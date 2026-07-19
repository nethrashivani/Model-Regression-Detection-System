"""
Typed interface contract for the email classifier feature.

PromptConfig is what the eval pipeline consumes as "the code under test."
EmailClassification is the structured output every prompt version must produce,
regardless of how the prompt itself is written.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Category = Literal["billing", "technical", "account", "general"]


class FewShotExample(BaseModel):
    """A single input/output pair embedded in the prompt for few-shot guidance."""

    input: str
    output: "EmailClassification"


class PromptConfig(BaseModel):
    """
    A versioned, immutable snapshot of a prompt.

    Loaded from /prompts/*.yaml. Each YAML file is one version — never edit
    a version in place once it's been run against the golden dataset; add a
    new version instead, so eval history stays comparable across versions.
    """

    version: str
    created_at: datetime
    description: str = ""
    system_prompt: str
    few_shot_examples: list[FewShotExample] = Field(default_factory=list)

    # Model call parameters live here too, since they're part of "the prompt"
    # from an eval perspective — changing temperature is a behavior change
    # just as much as changing the wording.
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 200

    @field_validator("version")
    @classmethod
    def version_must_be_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("PromptConfig.version cannot be empty")
        return v


class EmailClassification(BaseModel):
    """The structured output contract every prompt version must satisfy."""

    category: Category
    summary: str = Field(..., max_length=280)

    @field_validator("summary")
    @classmethod
    def summary_must_be_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("summary cannot be empty")
        return v


# Resolve the forward reference in FewShotExample.output
FewShotExample.model_rebuild()
