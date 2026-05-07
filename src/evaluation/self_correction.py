"""Self-correction flow triggered when an optimized prompt loses evaluation."""

from __future__ import annotations

import logging
from typing import Literal

from src.core.types import (
    EvaluationResult,
    OptimizationConstraints,
    PlanningResult,
)
from src.evaluation.pairwise_battle import PairwiseBattle
from src.optimization.optimizer import Optimizer

logger = logging.getLogger(__name__)

FailureDimension = Literal[
    "intentAlignment",
    "logicCoherence",
    "concisenessScore",
    "formatCompliance",
]

CORRECTION_PROMPTS: dict[FailureDimension, str] = {
    "intentAlignment": (
        "Focus strictly on the core task. Remove any content not directly related "
        "to the objective."
    ),
    "logicCoherence": (
        "Ensure the prompt establishes clear logical flow. Add explicit "
        "step-by-step structure if needed."
    ),
    "concisenessScore": (
        "Aggressively remove all redundant phrases, filler words, and repeated "
        "information."
    ),
    "formatCompliance": (
        "Strictly enforce the output format. Place format instructions prominently "
        "at the tail."
    ),
}


class SelfCorrector:
    """Retry prompt optimization with targeted requirements after evaluation loss."""

    def __init__(
        self,
        optimizer: Optimizer,
        battle: PairwiseBattle,
        constraints: OptimizationConstraints | None = None,
    ) -> None:
        """Create a self-corrector from the optimizer, evaluator, and guardrails."""

        self.optimizer = optimizer
        self.battle = battle
        self.constraints = constraints or OptimizationConstraints()

    async def correct(
        self,
        raw_prompt: str,
        planning_result: PlanningResult,
        initial_result: EvaluationResult,
        task_description: str,
    ) -> tuple[str, EvaluationResult]:
        """Retry optimization until the optimized prompt wins or retries are exhausted."""

        max_retries = max(0, self.constraints.maxSelfCorrectionRetries)
        last_result = initial_result
        corrected_planning = planning_result

        for retry in range(1, max_retries + 1):
            failed_dimension, correction_prompt = self._analyze_failure(last_result)
            logger.info(
                "Self-correction retry=%s failed_dimension=%s correction_prompt=%s",
                retry,
                failed_dimension,
                correction_prompt,
            )

            corrected_planning = self._append_refined_requirement(
                corrected_planning,
                correction_prompt,
            )
            optimization_result, _pipeline_usage = await self.optimizer.optimize(
                raw_prompt,
                corrected_planning,
            )
            last_result, _judge_usage, _target_usage = await self.battle.evaluate(
                raw_prompt,
                optimization_result.optimizedPrompt,
                task_description,
            )

            if last_result.winner == "optimized":
                return optimization_result.optimizedPrompt, last_result

        exhausted_feedback = f"Self-correction exhausted after {max_retries} retries"
        return raw_prompt, last_result.model_copy(update={"feedback": exhausted_feedback})

    @staticmethod
    def _analyze_failure(result: EvaluationResult) -> tuple[FailureDimension, str]:
        """Return the lowest-scoring non-overall dimension and its correction prompt."""

        score_by_dimension: dict[FailureDimension, float] = {
            "intentAlignment": result.scores.intentAlignment,
            "logicCoherence": result.scores.logicCoherence,
            "concisenessScore": result.scores.concisenessScore,
            "formatCompliance": result.scores.formatCompliance,
        }
        failed_dimension = min(
            score_by_dimension,
            key=lambda dimension: score_by_dimension[dimension],
        )
        return failed_dimension, CORRECTION_PROMPTS[failed_dimension]

    @staticmethod
    def _append_refined_requirement(
        planning_result: PlanningResult,
        correction_prompt: str,
    ) -> PlanningResult:
        """Return a planning result with the correction prompt appended immutably."""

        return planning_result.model_copy(
            update={
                "refinedRequirements": [
                    *planning_result.refinedRequirements,
                    correction_prompt,
                ],
            }
        )


__all__ = ["SelfCorrector"]
