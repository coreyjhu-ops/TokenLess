"""Pairwise battle evaluation between original and optimized prompts."""

from __future__ import annotations

import asyncio
import logging
from typing import Literal

from src.core.model_provider import GenerateParams, ModelProvider, TokenUsage
from src.core.types import (
    EvaluationResult,
    EvaluationScores,
    JudgeVote,
    OptimizationConstraints,
)
from src.evaluation.judge_pool import EvaluationError, JudgePool

logger = logging.getLogger(__name__)


class PairwiseBattle:
    """Run Phase 4 pairwise response generation, judging, voting, and quality gating."""

    def __init__(
        self,
        target_providers: list[ModelProvider],
        judge_pool: JudgePool,
        constraints: OptimizationConstraints | None = None,
    ) -> None:
        """Create a pairwise evaluator from target models, judges, and guardrails."""

        self.target_providers = target_providers
        self.judge_pool = judge_pool
        self.constraints = constraints or OptimizationConstraints()

    async def evaluate(
        self,
        original_prompt: str,
        optimized_prompt: str,
        task_description: str,
    ) -> tuple[EvaluationResult, TokenUsage, TokenUsage]:
        """Evaluate original and optimized prompts via blind target responses."""

        response_a_result, response_b_result = await asyncio.gather(
            self._generate_first_successful(original_prompt, prompt_label="original"),
            self._generate_first_successful(optimized_prompt, prompt_label="optimized"),
        )
        response_a, response_a_usage = response_a_result
        response_b, response_b_usage = response_b_result

        judge_results, judge_usage = await self.judge_pool.evaluate(
            original_prompt=original_prompt,
            optimized_prompt=optimized_prompt,
            response_a=response_a,
            response_b=response_b,
            task_description=task_description,
        )

        scores = self._average_scores(judge_results)
        winner = self._decide_winner(judge_results)
        feedback = self._feedback_for_original_winner(winner, judge_results)

        if scores.overall < self.constraints.minQualityScore:
            winner = "original"
            feedback = (
                f"Overall score {scores.overall:.1f} below threshold "
                f"{self.constraints.minQualityScore:.1f}"
            )

        target_usage = TokenUsage(
            prompt_tokens=response_a_usage.prompt_tokens + response_b_usage.prompt_tokens,
            completion_tokens=(
                response_a_usage.completion_tokens + response_b_usage.completion_tokens
            ),
        )
        return (
            EvaluationResult(
                winner=winner,
                scores=scores,
                judgeResults=judge_results,
                feedback=feedback if winner == "original" else None,
            ),
            judge_usage,
            target_usage,
        )

    async def run(
        self,
        original_prompt: str,
        optimized_prompt: str,
        task_description: str,
    ) -> tuple[EvaluationResult, TokenUsage]:
        """Compatibility wrapper returning the battle result and target usage."""

        evaluation_result, _judge_usage, target_usage = await self.evaluate(
            original_prompt=original_prompt,
            optimized_prompt=optimized_prompt,
            task_description=task_description,
        )
        return evaluation_result, target_usage

    async def _generate_first_successful(
        self,
        prompt: str,
        *,
        prompt_label: str,
    ) -> tuple[str, TokenUsage]:
        """Generate a target response by trying all target providers concurrently."""

        tasks = [
            provider.generate(
                GenerateParams(
                    prompt=prompt,
                    temperature=0.0,
                )
            )
            for provider in self.target_providers
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        failures: list[Exception] = []
        for result in results:
            if isinstance(result, Exception):
                failures.append(result)
                logger.warning(
                    "Target generation failed for %s prompt and was ignored.",
                    prompt_label,
                    exc_info=result,
                )
                continue
            return result.text, result.usage

        failure_text = "; ".join(str(failure) for failure in failures)
        raise EvaluationError(
            f"All target providers failed while generating the {prompt_label} response."
            + (f" Failures: {failure_text}" if failure_text else "")
        )

    @staticmethod
    def _decide_winner(
        judge_results: list[JudgeVote],
    ) -> Literal["original", "optimized", "tie"]:
        """Choose the final winner by majority vote, preferring optimized on ties."""

        original_votes = sum(1 for vote in judge_results if vote.winner == "original")
        optimized_votes = sum(1 for vote in judge_results if vote.winner == "optimized")
        if original_votes > optimized_votes:
            return "original"
        return "optimized"

    @staticmethod
    def _average_scores(judge_results: list[JudgeVote]) -> EvaluationScores:
        """Average per-judge score dimensions and recompute overall locally."""

        scores = [
            getattr(vote, "_scores")
            for vote in judge_results
            if hasattr(vote, "_scores")
        ]
        if not scores:
            raise EvaluationError("No judge scores were available for aggregation.")

        count = len(scores)
        intent_alignment = sum(score.intentAlignment for score in scores) / count
        logic_coherence = sum(score.logicCoherence for score in scores) / count
        conciseness_score = sum(score.concisenessScore for score in scores) / count
        format_compliance = sum(score.formatCompliance for score in scores) / count
        overall = (
            intent_alignment * 0.4
            + logic_coherence * 0.3
            + conciseness_score * 0.2
            + format_compliance * 0.1
        )

        return EvaluationScores(
            intentAlignment=intent_alignment,
            logicCoherence=logic_coherence,
            concisenessScore=conciseness_score,
            formatCompliance=format_compliance,
            overall=overall,
        )

    @staticmethod
    def _feedback_for_original_winner(
        winner: Literal["original", "optimized", "tie"],
        judge_results: list[JudgeVote],
    ) -> str | None:
        """Explain why the optimized prompt failed when original wins."""

        if winner != "original":
            return None

        original_votes = sum(1 for vote in judge_results if vote.winner == "original")
        optimized_votes = sum(1 for vote in judge_results if vote.winner == "optimized")
        tie_votes = sum(1 for vote in judge_results if vote.winner == "tie")
        return (
            "Optimized prompt lost judge vote "
            f"(original={original_votes}, optimized={optimized_votes}, tie={tie_votes})."
        )


__all__ = ["EvaluationError", "PairwiseBattle"]
