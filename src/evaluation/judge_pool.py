"""Judge model orchestration for pairwise prompt evaluation."""

from __future__ import annotations

import asyncio
import logging
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from src.core.model_provider import ModelProvider, StructuredParams, TokenUsage
from src.core.types import EvaluationScores, JudgeVote

logger = logging.getLogger(__name__)


class EvaluationError(RuntimeError):
    """Raised when the evaluation layer cannot produce a valid result."""


class _JudgeAssessment(BaseModel):
    """Structured JSON payload expected from each judge model."""

    winner: Literal["A", "B", "tie"]
    intentAlignment: float = Field(..., ge=0, le=10)
    logicCoherence: float = Field(..., ge=0, le=10)
    concisenessScore: float = Field(..., ge=0, le=10)
    formatCompliance: float = Field(..., ge=0, le=10)
    reasoning: str

    @field_validator("winner", mode="before")
    @classmethod
    def normalize_winner(cls, value: object) -> object:
        """Accept lowercase judge outputs while keeping the public schema strict."""

        if isinstance(value, str) and value.lower() == "tie":
            return "tie"
        if isinstance(value, str):
            return value.strip().upper()
        return value


class JudgePool:
    """Manage judge providers and collect independent pairwise votes."""

    def __init__(self, judge_providers: list[ModelProvider]) -> None:
        """Create a judge pool from already-instantiated model providers."""

        self.judge_providers = judge_providers

    async def evaluate(
        self,
        original_prompt: str,
        optimized_prompt: str,
        response_a: str,
        response_b: str,
        task_description: str,
    ) -> tuple[list[JudgeVote], TokenUsage]:
        """Evaluate one original-vs-optimized response pair across all judges.

        Judges receive only Response A and Response B, so they do not know that
        A came from the original prompt and B came from the optimized prompt.
        Individual judge failures are logged and ignored; at least one valid
        judge vote is required.
        """

        del original_prompt, optimized_prompt

        tasks = [
            self._evaluate_with_judge(
                judge_provider=judge_provider,
                response_a=response_a,
                response_b=response_b,
                task_description=task_description,
            )
            for judge_provider in self.judge_providers
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        votes: list[JudgeVote] = []
        total_usage = TokenUsage(prompt_tokens=0, completion_tokens=0)
        for result in results:
            if isinstance(result, Exception):
                logger.warning(
                    "Judge evaluation failed and was ignored.",
                    exc_info=result,
                )
                continue
            if result is not None:
                vote, usage = result
                votes.append(vote)
                total_usage = TokenUsage(
                    prompt_tokens=total_usage.prompt_tokens + usage.prompt_tokens,
                    completion_tokens=(
                        total_usage.completion_tokens + usage.completion_tokens
                    ),
                )

        if not votes:
            raise EvaluationError("All judge model evaluations failed.")

        return votes, total_usage

    async def _evaluate_with_judge(
        self,
        *,
        judge_provider: ModelProvider,
        response_a: str,
        response_b: str,
        task_description: str,
    ) -> tuple[JudgeVote, TokenUsage]:
        """Run one structured judge call and convert it to the public vote type."""

        assessment = await judge_provider.generate_structured(
            StructuredParams(
                prompt=self._build_judge_prompt(
                    task_description=task_description,
                    response_a=response_a,
                    response_b=response_b,
                ),
                schema=_JudgeAssessment.model_json_schema(),
                max_tokens=2000,
            ),
            _JudgeAssessment,
        )

        parsed = assessment.parsed
        if not isinstance(parsed, _JudgeAssessment):
            parsed = _JudgeAssessment.model_validate(parsed)

        scores = self._scores_from_assessment(parsed)
        vote = JudgeVote(
            model=self._provider_label(judge_provider),
            winner=self._map_winner(parsed.winner),
            reasoning=parsed.reasoning.strip(),
        )
        object.__setattr__(vote, "_scores", scores)
        return vote, assessment.usage

    @staticmethod
    def _build_judge_prompt(
        *,
        task_description: str,
        response_a: str,
        response_b: str,
    ) -> str:
        """Build the double-blind judge prompt from the Phase 4 specification."""

        return f"""You are an expert evaluator assessing two AI-generated responses to a coding task.
Your goal is to determine which response better fulfills the task requirements.

Task: {task_description}

Response A:
{response_a}

Response B:
{response_b}

Evaluate both responses on these dimensions (0-10, higher is always better):
- intentAlignment: Does the response accurately fulfill the task intent?
- logicCoherence: Is the reasoning/code logically sound and complete?
- concisenessScore: How concise is the response? (10 = no redundancy, 0 = highly redundant)
- formatCompliance: Does the response follow expected output format?

Then pick the overall winner.

Return JSON:
{{
  "winner": "A" or "B" or "tie",
  "intentAlignment": <float 0-10>,
  "logicCoherence": <float 0-10>,
  "concisenessScore": <float 0-10>,
  "formatCompliance": <float 0-10>,
  "reasoning": "<brief explanation>"
}}"""

    @staticmethod
    def _scores_from_assessment(assessment: _JudgeAssessment) -> EvaluationScores:
        """Create weighted scores from judge dimensions, computing overall locally."""

        overall = (
            assessment.intentAlignment * 0.4
            + assessment.logicCoherence * 0.3
            + assessment.concisenessScore * 0.2
            + assessment.formatCompliance * 0.1
        )
        return EvaluationScores(
            intentAlignment=assessment.intentAlignment,
            logicCoherence=assessment.logicCoherence,
            concisenessScore=assessment.concisenessScore,
            formatCompliance=assessment.formatCompliance,
            overall=overall,
        )

    @staticmethod
    def _map_winner(
        winner: Literal["A", "B", "tie"],
    ) -> Literal["original", "optimized", "tie"]:
        """Map blind A/B labels back to prompt versions."""

        if winner == "A":
            return "original"
        if winner == "B":
            return "optimized"
        return "tie"

    @staticmethod
    def _provider_label(provider: ModelProvider) -> str:
        """Return a stable human-readable model label for a provider."""

        model = getattr(provider, "model", None)
        if model:
            return f"{provider.id}:{model}"
        return provider.id


__all__ = ["EvaluationError", "JudgePool"]
