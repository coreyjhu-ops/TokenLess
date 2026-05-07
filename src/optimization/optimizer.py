"""Integrated optimization engine pipeline for TokenLess."""

from __future__ import annotations

import asyncio
import re

from src.core.model_provider import ModelProvider, TokenUsage
from src.core.token_estimator import TokenEstimator
from src.core.types import (
    OptimizationConstraints,
    OptimizationResult,
    OptimizationROIReport,
    PipelineTokenUsage,
    PlanningResult,
    PromptSection,
    TokenStats,
)
from src.optimization.formatter import PromptFormatter
from src.optimization.positional_anchor import PositionalAnchorer, PromptSegment
from src.optimization.professional_rewriter import ProfessionalRewriter
from src.optimization.semantic_pruner import SemanticPruner
from src.optimization.word_pruner import WordPruner


class Optimizer:
    """Coordinate anchoring, pruning, semantic filtering, and output formatting."""

    def __init__(
        self,
        model_provider: ModelProvider,
        token_estimator: TokenEstimator,
        constraints: OptimizationConstraints | None = None,
    ) -> None:
        """Create an optimizer with provider, token estimator, and constraints."""

        self.anchorer = PositionalAnchorer()
        self.pruner = WordPruner()
        self.semantic_pruner = SemanticPruner(model_provider)
        self.rewriter = ProfessionalRewriter(model_provider)
        self.formatter = PromptFormatter()
        self.token_estimator = token_estimator
        self.constraints = constraints or OptimizationConstraints(
            maxCompressionRate=0.50,
            minQualityScore=6.0,
            requirePositiveROI=True,
            maxSelfCorrectionRetries=2,
        )

    async def optimize(
        self,
        raw_prompt: str,
        planning_result: PlanningResult,
        intent_usage: TokenUsage | None = None,
    ) -> tuple[OptimizationResult, PipelineTokenUsage]:
        """Optimize a raw prompt and return a fully populated result object."""

        original_count = self.token_estimator.exact_count(raw_prompt)
        segments = self.anchorer.segment(raw_prompt, planning_result)
        segments = self.anchorer.reorder(segments)

        applied_techniques: list[str] = ["positional anchoring"]

        segments = self._apply_word_pruning(segments, applied_techniques)
        semantic_usage = await self._apply_semantic_pruning(
            segments,
            planning_result,
            applied_techniques,
        )
        rewritten_prompt, rewrite_usage = await self.rewriter.rewrite(
            raw_prompt,
            planning_result,
        )
        rewritten_segments = self._segments_from_rewritten_prompt(rewritten_prompt)
        if rewritten_segments:
            segments = rewritten_segments
        if "professional rewrite" not in applied_techniques:
            applied_techniques.append("professional rewrite")

        pipeline_usage = PipelineTokenUsage(
            intent_analyzer_prompt=intent_usage.prompt_tokens if intent_usage else 0,
            intent_analyzer_completion=(
                intent_usage.completion_tokens if intent_usage else 0
            ),
            semantic_pruner_prompt=semantic_usage.prompt_tokens,
            semantic_pruner_completion=semantic_usage.completion_tokens,
            rewriter_prompt=rewrite_usage.prompt_tokens,
            rewriter_completion=rewrite_usage.completion_tokens,
        )

        optimized_text = self.formatter.format(segments, planning_result)
        optimized_count = self.token_estimator.exact_count(optimized_text)
        reduction_rate = self._reduction_rate(original_count, optimized_count)

        optimization_skipped = False
        skip_reason: str | None = None

        if 0 < reduction_rate > self.constraints.maxCompressionRate:
            optimization_skipped = True
            skip_reason = (
                f"Reduction rate {reduction_rate:.1%} exceeds the limit "
                f"{self.constraints.maxCompressionRate:.1%}; returning original prompt"
            )
            optimized_text = raw_prompt
            optimized_count = original_count
            reduction_rate = 0.0

        optimization_cost_tokens = pipeline_usage.total_pipeline_tokens
        net_savings = (original_count - optimized_count) - optimization_cost_tokens
        roi_positive = net_savings > 0
        rewrite_only = (
            applied_techniques == ["positional anchoring", "professional rewrite"]
            or "professional rewrite" in applied_techniques
        )

        if (
            self.constraints.requirePositiveROI
            and not roi_positive
            and not optimization_skipped
            and not rewrite_only
        ):
            optimization_skipped = True
            skip_reason = "Net optimization savings are negative because the raw prompt is too short; returning original prompt"
            optimized_text = raw_prompt
            optimized_count = original_count
            reduction_rate = 0.0
            net_savings = -optimization_cost_tokens
            roi_positive = False

        optimization_result = OptimizationResult(
            optimizedPrompt=optimized_text,
            tokenStats=TokenStats(
                originalCount=original_count,
                optimizedCount=optimized_count,
                reductionRate=round(reduction_rate, 4),
            ),
            appliedTechniques=applied_techniques,
            structureMap=[
                PromptSection(
                    position=segment.position,
                    type=segment.section_type,
                    content=segment.content,
                    tokenCount=self.token_estimator.exact_count(segment.content),
                )
                for segment in segments
            ],
            roiReport=OptimizationROIReport(
                inputTokensSaved=original_count - optimized_count,
                optimizationCostTokens=optimization_cost_tokens,
                netTokenSavings=net_savings,
                roiPositive=roi_positive,
                pipeline_breakdown=pipeline_usage,
            ),
            optimizationSkipped=optimization_skipped,
            skipReason=skip_reason if optimization_skipped else None,
        )
        return optimization_result, pipeline_usage

    def optimize_sync(
        self,
        raw_prompt: str,
        planning_result: PlanningResult,
        intent_usage: TokenUsage | None = None,
    ) -> tuple[OptimizationResult, PipelineTokenUsage]:
        """Synchronous wrapper around :meth:`optimize`."""

        return asyncio.run(self.optimize(raw_prompt, planning_result, intent_usage))

    def _apply_word_pruning(
        self,
        segments: list[PromptSegment],
        applied_techniques: list[str],
    ) -> list[PromptSegment]:
        """Apply rule-based compression to each segment."""

        updated_segments: list[PromptSegment] = []
        for segment in segments:
            compressed_content, rules = self.pruner.prune(segment.content)
            for rule in rules:
                if rule not in applied_techniques:
                    applied_techniques.append(rule)
            updated_segments.append(
                segment.model_copy(update={"content": compressed_content})
            )
        return updated_segments

    async def _apply_semantic_pruning(
        self,
        segments: list[PromptSegment],
        planning_result: PlanningResult,
        applied_techniques: list[str],
    ) -> TokenUsage:
        """Prune long middle-context segments in place."""

        total_usage = TokenUsage(prompt_tokens=0, completion_tokens=0)
        for index, segment in enumerate(segments):
            if (
                segment.position != "middle"
                or segment.section_type != "context"
                or len(segment.content) <= 50
            ):
                continue

            pruned_content, noise_ratio, usage = await self.semantic_pruner.prune(
                segment.content,
                planning_result.detectedIntent,
            )
            total_usage = TokenUsage(
                prompt_tokens=total_usage.prompt_tokens + usage.prompt_tokens,
                completion_tokens=total_usage.completion_tokens
                + usage.completion_tokens,
            )
            segments[index] = segment.model_copy(update={"content": pruned_content})
            if noise_ratio > 0:
                applied_techniques.append(
                    f"semantic pruning ({noise_ratio:.1%} noise removed)"
                )
            else:
                applied_techniques.append("semantic pruning")
        return total_usage

    @staticmethod
    def _reduction_rate(original_count: int, optimized_count: int) -> float:
        """Return token reduction rate with zero-input protection."""

        if original_count <= 0:
            return 0.0
        return (original_count - optimized_count) / original_count

    @staticmethod
    def _segments_from_rewritten_prompt(rewritten_prompt: str) -> list[PromptSegment]:
        """Parse a professional Markdown rewrite into formatter segments."""

        section_specs: dict[str, tuple[str, str]] = {
            "role": ("role", "head"),
            "task": ("task", "head"),
            "tech stack": ("tech_stack", "middle"),
            "constraints": ("constraint", "middle"),
            "output format": ("format", "tail"),
            "reminder": ("context", "tail"),
        }
        heading_pattern = re.compile(
            r"(?im)^#{1,3}\s*(Role|Task|Tech Stack|Constraints|Output Format|Reminder)\s*$"
        )
        matches = list(heading_pattern.finditer(rewritten_prompt))
        if not matches:
            return []

        segments: list[PromptSegment] = []
        for index, match in enumerate(matches):
            title = match.group(1).strip().lower()
            start = match.end()
            end = (
                matches[index + 1].start()
                if index + 1 < len(matches)
                else len(rewritten_prompt)
            )
            content = rewritten_prompt[start:end].strip()
            if not content or title not in section_specs:
                continue
            section_type, position = section_specs[title]
            segments.append(
                PromptSegment(
                    content=content,
                    section_type=section_type,
                    position=position,
                )
            )
        return segments


__all__ = ["Optimizer"]
