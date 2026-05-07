"""Intent decomposition for Planning Mode."""

from __future__ import annotations

import asyncio
from typing import Tuple

from pydantic import BaseModel, Field, field_validator

from src.core.model_provider import ModelProvider, StructuredParams, TokenUsage
from src.core.types import SceneType


class IntentAnalysis(BaseModel):
    """WHO/WHAT/HOW/FORMAT decomposition of a vibe coding prompt."""

    who: str = Field(
        default="developer",
        description='Role definition, such as "Python developer".',
    )
    what: str = Field(
        ...,
        description="Core task or functionality to implement.",
    )
    how: str = Field(
        default="",
        description="Technical stack, framework, language, and constraints.",
    )
    format: str = Field(
        default="",
        description="Output format requirements.",
    )
    raw_prompt: str = Field(
        default="",
        description="Original prompt preserved verbatim.",
    )

    @field_validator("who", mode="before")
    @classmethod
    def _default_who(cls, value: object) -> str:
        """Default empty role values to developer."""

        if value is None or str(value).strip() == "":
            return "developer"
        return str(value).strip()

    @field_validator("what")
    @classmethod
    def _what_must_not_be_empty(cls, value: str) -> str:
        """Ensure model output includes a non-empty core task."""

        if value.strip() == "":
            raise ValueError("what must not be empty")
        return value.strip()

    @field_validator("how", "format", mode="before")
    @classmethod
    def _default_optional_text(cls, value: object) -> str:
        """Normalize absent optional dimensions to empty strings."""

        if value is None:
            return ""
        return str(value).strip()


class IntentAnalyzer:
    """Use a model provider to extract structured intent dimensions."""

    def __init__(self, model_provider: ModelProvider) -> None:
        """Create an analyzer with an instantiated model provider."""

        self.model_provider = model_provider

    async def analyze(
        self,
        raw_prompt: str,
        scene: SceneType,
    ) -> Tuple[IntentAnalysis, TokenUsage]:
        """Analyze ``raw_prompt`` into WHO/WHAT/HOW/FORMAT dimensions."""

        prompt = (
            "Analyze this vibe coding prompt and extract the intent dimensions.\n\n"
            f'Prompt: "{raw_prompt}"\n\n'
            "Extract:\n"
            '- WHO: The role/persona implied (if none, use "developer")\n'
            "- WHAT: The core task/functionality to implement (required)\n"
            "- HOW: Technical stack, framework, language, constraints mentioned\n"
            '- FORMAT: Output format requirements (e.g., "return only code", "add docstrings")\n\n'
            "Return JSON with keys: who, what, how, format"
        )
        params = StructuredParams(
            prompt=prompt,
            schema=IntentAnalysis.model_json_schema(),
            max_tokens=512,
        )
        result = await self.model_provider.generate_structured(
            params=params,
            response_type=IntentAnalysis,
        )
        analysis = result.parsed
        if not isinstance(analysis, IntentAnalysis):
            analysis = IntentAnalysis.model_validate(result.parsed)
        analysis.raw_prompt = raw_prompt
        return analysis, result.usage

    def analyze_sync(
        self,
        raw_prompt: str,
        scene: SceneType,
    ) -> Tuple[IntentAnalysis, TokenUsage]:
        """Synchronous wrapper for non-async callers."""

        return asyncio.run(self.analyze(raw_prompt, scene))
