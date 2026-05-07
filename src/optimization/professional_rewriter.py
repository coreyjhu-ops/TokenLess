"""Model-driven professional prompt rewriting for the optimization layer."""

from __future__ import annotations

import asyncio
import re

from pydantic import BaseModel, Field, field_validator

from src.core.model_provider import ModelProvider, StructuredParams, TokenUsage
from src.core.types import PlanningResult


class _ProfessionalRewrite(BaseModel):
    """Structured response returned by the professional rewrite model call."""

    rewritten_prompt: str = Field(
        default="",
        description="The rewritten professional prompt in Markdown section format.",
    )

    @field_validator("rewritten_prompt", mode="before")
    @classmethod
    def _normalize_prompt(cls, value: object) -> str:
        """Normalize absent or non-string model output to a string."""

        if value is None:
            return ""
        return str(value).strip()


class ProfessionalRewriter:
    """Rewrite raw user prompts into precise professional prompt structures."""

    def __init__(self, model_provider: ModelProvider) -> None:
        """Create the rewriter with an optimization-capable model provider."""

        self.model_provider = model_provider

    async def rewrite(
        self,
        raw_prompt: str,
        planning_result: PlanningResult,
    ) -> tuple[str, TokenUsage]:
        """Rewrite ``raw_prompt`` into a professional structured prompt.

        The rewrite intentionally optimizes for reduced ambiguity and lower
        downstream reasoning cost, not for the shortest possible wording.
        """

        prompt = self._build_rewrite_prompt(raw_prompt, planning_result)
        result = await self.model_provider.generate_structured(
            StructuredParams(
                prompt=prompt,
                schema=_ProfessionalRewrite.model_json_schema(),
                max_tokens=1200,
            ),
            _ProfessionalRewrite,
        )

        parsed = result.parsed
        if not isinstance(parsed, _ProfessionalRewrite):
            parsed = _ProfessionalRewrite.model_validate(parsed)

        rewritten_prompt = parsed.rewritten_prompt.strip()
        if not rewritten_prompt:
            rewritten_prompt = self._fallback_rewrite(raw_prompt, planning_result)
        return rewritten_prompt, result.usage

    def rewrite_sync(
        self,
        raw_prompt: str,
        planning_result: PlanningResult,
    ) -> tuple[str, TokenUsage]:
        """Synchronous wrapper around :meth:`rewrite`."""

        return asyncio.run(self.rewrite(raw_prompt, planning_result))

    def _build_rewrite_prompt(
        self,
        raw_prompt: str,
        planning_result: PlanningResult,
    ) -> str:
        """Build the structured rewrite instruction for the model."""

        missing_fields = "\n".join(
            f"- {field.field} ({field.importance})"
            for field in planning_result.missingFields
        )
        refined_requirements = "\n".join(
            f"- {requirement}" for requirement in planning_result.refinedRequirements
        )
        missing_text = missing_fields or "- None"
        requirements_text = refined_requirements or "- None"

        return (
            "Rewrite the user's prompt into a professional, unambiguous prompt "
            "for a downstream software-development LLM.\n\n"
            "Core objective:\n"
            "- Eliminate ambiguity and minimize downstream reasoning cost; do not optimize for brevity.\n"
            "- Preserve the user's intent and concrete details.\n"
            "- Do not invent missing facts.\n\n"
            "Required structure and exact section order:\n"
            "1. Role\n"
            "2. Task\n"
            "3. Tech Stack\n"
            "4. Constraints\n"
            "5. Output Format\n"
            "6. Reminder\n\n"
            "Rewrite rules:\n"
            "- Replace vague words with precise technical terminology when supported by the input.\n"
            "- Use imperative phrasing such as Implement, Return, Ensure, Use, Include.\n"
            '- Make implicit requirements explicit, e.g. "fast" -> "< 100ms p99 latency" only when the prompt implies performance.\n'
            "- For missing critical fields, use placeholders like [REQUIRED: language] instead of guessing.\n"
            "- Keep short prompts actionable; never refuse because the prompt has fewer than 30 tokens.\n"
            "- Return only the rewritten prompt as Markdown headings and body text.\n"
            "- The Output Format section must include a visible fenced ```json code block.\n"
            "- Do not hide JSON in HTML comments.\n"
            "- Use this JSON-compatible structure unless the user clearly requested a stricter schema: "
            '{"summary":"","files_to_create_or_modify":[],"implementation_steps":[],"acceptance_criteria":[],"tests":[]}.\n\n'
            f"Detected intent:\n{planning_result.detectedIntent}\n\n"
            f"Refined requirements:\n{requirements_text}\n\n"
            f"Missing fields:\n{missing_text}\n\n"
            f"Raw prompt:\n{raw_prompt}"
        )

    @staticmethod
    def _fallback_rewrite(
        raw_prompt: str,
        planning_result: PlanningResult,
    ) -> str:
        """Return a deterministic structured rewrite when model output is empty."""

        missing = {field.field for field in planning_result.missingFields}
        role = ProfessionalRewriter._extract_requirement(
            planning_result,
            "Role:",
            default="software developer",
        )
        task = ProfessionalRewriter._extract_requirement(
            planning_result,
            "Task:",
            default=planning_result.detectedIntent or raw_prompt.strip(),
        )
        stack = ProfessionalRewriter._extract_requirement(
            planning_result,
            "Stack:",
            default="[REQUIRED: language/framework]"
            if {"language", "framework"} & missing
            else "[REQUIRED: tech stack]",
        )
        constraints = (
            "[REQUIRED: constraints]"
            if "constraints" in missing
            else "Ensure the implementation is deterministic, maintainable, and testable."
        )
        output_format = (
            "[REQUIRED: output format]"
            if "output_format" in missing
            else "Return clean, well-structured implementation details."
        )

        return (
            f"## Role\nAct as a {role}.\n\n"
            f"## Task\nImplement {ProfessionalRewriter._imperative_task(task)}.\n\n"
            f"## Tech Stack\nUse {stack}.\n\n"
            f"## Constraints\n- {constraints}\n\n"
            f"## Output Format\nReturn {output_format}\n\n"
            "Return the answer using this JSON-compatible structure:\n\n"
            "```json\n"
            "{\n"
            '  "summary": "",\n'
            '  "files_to_create_or_modify": [],\n'
            '  "implementation_steps": [],\n'
            '  "acceptance_criteria": [],\n'
            '  "tests": []\n'
            "}\n"
            "```\n\n"
            f"## Reminder\n{planning_result.detectedIntent or raw_prompt.strip()}"
        )

    @staticmethod
    def _extract_requirement(
        planning_result: PlanningResult,
        prefix: str,
        default: str,
    ) -> str:
        """Extract a refined requirement by prefix."""

        for requirement in planning_result.refinedRequirements:
            if requirement.lower().startswith(prefix.lower()):
                return requirement[len(prefix) :].strip() or default
        return default

    @staticmethod
    def _imperative_task(task: str) -> str:
        """Clean task text for use after an imperative verb."""

        cleaned = re.sub(
            r"^(build|create|implement|write|make)\s+",
            "",
            task.strip(),
            flags=re.IGNORECASE,
        )
        return cleaned or "the requested functionality"


__all__ = ["ProfessionalRewriter"]
