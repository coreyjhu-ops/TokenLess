"""Main chatbot controller that orchestrates the four-layer TokenLess pipeline."""

from __future__ import annotations

from uuid import uuid4

from src.chat.conversation import ConversationState
from src.core.types import PlanningResult
from src.evaluation.pairwise_battle import PairwiseBattle
from src.evaluation.self_correction import SelfCorrector
from src.optimization.optimizer import Optimizer
from src.planning.planner import Planner


class TokenLessChatbot:
    """Interactive Python-facing controller for the TokenLess pipeline."""

    def __init__(
        self,
        planner: Planner,
        optimizer: Optimizer,
        battle: PairwiseBattle,
        corrector: SelfCorrector,
    ) -> None:
        """Wire the four pipeline components used by the chat interface."""

        self.planner = planner
        self.optimizer = optimizer
        self.battle = battle
        self.corrector = corrector

    async def process(
        self,
        raw_prompt: str,
        supplements: dict[str, str] | None = None,
    ) -> ConversationState:
        """Run Planning, Optimization, Pairwise Battle, and optional correction."""

        state = ConversationState(
            session_id=str(uuid4()),
            raw_prompt=raw_prompt,
            supplements=supplements or {},
            status="planning",
        )

        try:
            if supplements:
                (
                    planning_result,
                    intent_usage,
                ) = await self.planner.plan_with_supplement_and_usage(
                    raw_prompt,
                    supplements,
                )
            else:
                planning_result, intent_usage = await self.planner.plan_with_usage(
                    raw_prompt
                )
            state.planning_result = planning_result

            state.status = "optimizing"
            optimization_result, pipeline_usage = await self.optimizer.optimize(
                raw_prompt,
                planning_result,
                intent_usage=intent_usage,
            )
            state.optimization_result = optimization_result
            state.token_ledger = pipeline_usage

            if optimization_result.optimizationSkipped:
                state.final_prompt = raw_prompt
            else:
                state.status = "evaluating"
                (
                    evaluation_result,
                    judge_usage,
                    target_usage,
                ) = await self.battle.evaluate(
                    original_prompt=raw_prompt,
                    optimized_prompt=optimization_result.optimizedPrompt,
                    task_description=planning_result.detectedIntent,
                )
                pipeline_usage = pipeline_usage.model_copy(
                    update={
                        "judge_pool_prompt": judge_usage.prompt_tokens,
                        "judge_pool_completion": judge_usage.completion_tokens,
                        "target_model_prompt": target_usage.prompt_tokens,
                        "target_model_completion": target_usage.completion_tokens,
                    }
                )
                state.token_ledger = pipeline_usage
                if optimization_result.roiReport:
                    state.optimization_result = optimization_result.model_copy(
                        update={
                            "roiReport": optimization_result.roiReport.model_copy(
                                update={
                                    "optimizationCostTokens": (
                                        pipeline_usage.total_pipeline_tokens
                                    ),
                                    "netTokenSavings": (
                                        optimization_result.roiReport.inputTokensSaved
                                        - pipeline_usage.total_pipeline_tokens
                                    ),
                                    "roiPositive": (
                                        optimization_result.roiReport.inputTokensSaved
                                        - pipeline_usage.total_pipeline_tokens
                                        > 0
                                    ),
                                    "pipeline_breakdown": pipeline_usage,
                                }
                            )
                        }
                    )

                state.evaluation_result = evaluation_result

                if evaluation_result.winner == "original":
                    final_prompt, final_result = await self.corrector.correct(
                        raw_prompt=raw_prompt,
                        planning_result=planning_result,
                        initial_result=evaluation_result,
                        task_description=planning_result.detectedIntent,
                    )
                    state.final_prompt = final_prompt
                    state.evaluation_result = final_result
                else:
                    state.final_prompt = optimization_result.optimizedPrompt

            state.status = "done"
            return state
        except Exception as exc:
            state.status = "error"
            state.error_message = str(exc)
            return state

    def format_missing_fields_message(self, planning_result: PlanningResult) -> str:
        """Format missing fields into UI-friendly supplement prompts."""

        if planning_result.clarification_message:
            return planning_result.clarification_message

        if not planning_result.missingFields:
            return "No missing fields were found."

        lines: list[str] = []
        for missing_field in planning_result.missingFields:
            marker = self._importance_marker(missing_field.importance)
            line = f"{marker} {missing_field.question}"
            if missing_field.defaultValue:
                line += f" (default: {missing_field.defaultValue})"
            lines.append(line)
        return "\n".join(lines)

    def format_result_summary(self, state: ConversationState) -> str:
        """Format a conversation state into a concise final result summary."""

        if state.status == "error":
            return f"Processing failed: {state.error_message or 'unknown error'}"

        lines = [f"Status: {state.status}"]

        if state.planning_result:
            lines.append(f"Scene: {state.planning_result.scene}")
            lines.append(f"Intent: {state.planning_result.detectedIntent}")
            critical_count = sum(
                1
                for field in state.planning_result.missingFields
                if field.importance == "critical"
            )
            lines.append(f"Critical missing fields: {critical_count}")

        if state.optimization_result:
            token_stats = state.optimization_result.tokenStats
            lines.append(
                "Tokens: "
                f"{token_stats.originalCount} -> {token_stats.optimizedCount} "
                f"(reduction rate {token_stats.reductionRate:.1%})"
            )

        if state.evaluation_result:
            lines.append(f"Judge winner: {state.evaluation_result.winner}")
            lines.append(f"Overall: {state.evaluation_result.scores.overall:.2f}")
            if state.evaluation_result.feedback:
                lines.append(f"Feedback: {state.evaluation_result.feedback}")

        if state.final_prompt is not None:
            lines.append("Final prompt:")
            lines.append(state.final_prompt)

        return "\n".join(lines)

    @staticmethod
    def _importance_marker(importance: str) -> str:
        """Map missing-field importance to the GapAnalyzer display marker."""

        if importance == "critical":
            return "[Required]"
        if importance == "recommended":
            return "[Recommended]"
        return "[Optional]"


__all__ = ["TokenLessChatbot"]
