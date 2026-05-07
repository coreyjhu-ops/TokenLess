"""Planning Mode orchestration for TokenLess."""

from __future__ import annotations

import asyncio
from typing import Tuple

from src.core.model_provider import ModelProvider, TokenUsage
from src.core.types import MissingField, PlanningResult
from src.planning.gap_analyzer import GapAnalyzer
from src.planning.intent_analyzer import IntentAnalysis, IntentAnalyzer
from src.planning.scene_detector import SceneDetector


class Planner:
    """Run scene detection, intent analysis, and gap analysis as one flow."""

    IMPORTANCE_ORDER = {"critical": 0, "recommended": 1, "optional": 2}

    def __init__(
        self,
        model_provider: ModelProvider,
        refs_dir: str = "src/planning/instruction_refs",
    ) -> None:
        """Wire Planning Mode dependencies."""

        self.model_provider = model_provider
        self.scene_detector = SceneDetector(refs_dir)
        self.intent_analyzer = IntentAnalyzer(model_provider)
        self.gap_analyzer = GapAnalyzer(self.scene_detector)

    async def plan(self, raw_prompt: str) -> PlanningResult:
        """Run the full Planning Mode flow for ``raw_prompt``."""

        planning_result, _usage = await self.plan_with_usage(raw_prompt)
        return planning_result

    async def plan_with_usage(
        self,
        raw_prompt: str,
    ) -> Tuple[PlanningResult, TokenUsage]:
        """Run Planning Mode and return the intent analyzer token usage."""

        scene = self.scene_detector.detect(raw_prompt)
        intent, usage = await self.intent_analyzer.analyze(raw_prompt, scene)
        missing = self.gap_analyzer.analyze(intent, scene)
        missing = sorted(
            missing,
            key=lambda field: self.IMPORTANCE_ORDER[field.importance],
        )

        refined_requirements = self._build_refined_requirements(intent)
        clarification_message = await self._generate_clarification_message(
            raw_prompt,
            intent,
            missing,
        )
        return (
            PlanningResult(
                detectedIntent=intent.what,
                scene=scene,
                missingFields=missing,
                refinedRequirements=refined_requirements,
                instructionRefs=["vibe_coding"],
                clarification_message=clarification_message,
            ),
            usage,
        )

    def plan_sync(self, raw_prompt: str) -> PlanningResult:
        """Synchronous wrapper for non-async callers."""

        return asyncio.run(self.plan(raw_prompt))

    async def plan_with_supplement(
        self,
        raw_prompt: str,
        supplements: dict[str, str],
    ) -> PlanningResult:
        """Merge user supplements into the raw prompt and re-run planning."""

        supplement_text = ", ".join(
            f"{key}={value}" for key, value in supplements.items()
        )
        prompt_with_supplement = f"{raw_prompt}\nAdditional info: {supplement_text}"
        return await self.plan(prompt_with_supplement)

    async def plan_with_supplement_and_usage(
        self,
        raw_prompt: str,
        supplements: dict[str, str],
    ) -> Tuple[PlanningResult, TokenUsage]:
        """Merge supplements into the prompt and return planning usage."""

        supplement_text = ", ".join(
            f"{key}={value}" for key, value in supplements.items()
        )
        prompt_with_supplement = f"{raw_prompt}\nAdditional info: {supplement_text}"
        return await self.plan_with_usage(prompt_with_supplement)

    @staticmethod
    def _build_refined_requirements(intent: IntentAnalysis) -> list[str]:
        """Convert non-empty intent dimensions into requirement strings."""

        requirements: list[str] = []
        if intent.who.strip():
            requirements.append(f"Role: {intent.who.strip()}")
        if intent.what.strip():
            requirements.append(f"Task: {intent.what.strip()}")
        if intent.how.strip():
            requirements.append(f"Stack: {intent.how.strip()}")
        if intent.format.strip():
            requirements.append(f"Format: {intent.format.strip()}")
        return requirements

    async def _generate_clarification_message(
        self,
        raw_prompt: str,
        intent: IntentAnalysis,
        missing_fields: list[MissingField],
    ) -> str | None:
        """Generate dynamic clarification text for missing planning fields."""

        if not missing_fields:
            return None
        return await self.gap_analyzer.generate_targeted_questions(
            raw_prompt=raw_prompt,
            intent=intent,
            missing_fields=missing_fields,
            model_provider=self.model_provider,
        )
