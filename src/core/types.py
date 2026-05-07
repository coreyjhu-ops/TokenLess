"""Core Pydantic data structures for TokenLess, based on tech-spec section 2.3.

V1 Domain Scope: Vibe Coding / Software Development only.
Other scenes (image_gen, email_writing, speech_writing, general) are deferred to V2.
See changelog v0.1.1 and tech-spec section 1.3 for the domain scope decision.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field

# tech-spec section 2.3 scene categories.
# V1: only 'vibe_coding' is actively supported.
# 'custom' is kept as an interface placeholder for V2 extensibility.
# Removed from V1: 'image_gen', 'email_writing', 'speech_writing', 'general'
SceneType = Literal[
    "vibe_coding",   # V1 primary scene: natural-language coding prompts (Cursor/Copilot workflows)
    "custom",        # V2 placeholder - do not implement routing logic for this in V1
]


class InputConstraints(BaseModel):
    """Optional user-specified constraints attached to the raw prompt."""

    preserveKeywords: Optional[list[str]] = Field(
        default=None,
        description="Keywords that must be preserved during prompt optimization.",
    )


class UserInput(BaseModel):
    """User-provided prompt input before planning or optimization."""

    rawPrompt: str = Field(
        ...,
        description="The user's original prompt text.",
    )
    targetModel: Optional[str] = Field(
        default=None,
        description="Optional target model identifier used to select tokenization behavior.",
    )
    scene: Optional[SceneType] = Field(
        default=None,
        description="Optional user-specified scene classification.",
    )
    constraints: Optional[InputConstraints] = Field(
        default=None,
        description="Optional optimization constraints declared by the user.",
    )


class MissingField(BaseModel):
    """A missing requirement that should be clarified with the user."""

    field: str = Field(
        ...,
        description="The name of the missing field or requirement.",
    )
    question: str = Field(
        ...,
        description="The follow-up question to ask the user.",
    )
    importance: Literal["critical", "recommended", "optional"] = Field(
        ...,
        description="The importance level of the missing requirement.",
    )
    defaultValue: Optional[str] = Field(
        default=None,
        description="Optional default value that can be used if the user does not answer.",
    )


class PlanningResult(BaseModel):
    """Structured planning output generated from the raw user request."""

    detectedIntent: str = Field(
        ...,
        description="The core intent detected from the user's prompt.",
    )
    scene: SceneType = Field(
        ...,
        description="The matched scene category for the request.",
    )
    missingFields: list[MissingField] = Field(
        default_factory=list,
        description="Missing fields that should be clarified with the user.",
    )
    refinedRequirements: list[str] = Field(
        default_factory=list,
        description="Structured requirements distilled from the original prompt.",
    )
    instructionRefs: list[str] = Field(
        default_factory=list,
        description="Matched preset instruction reference identifiers.",
    )
    clarification_message: Optional[str] = Field(
        default=None,
        description="Optional model-generated clarification questions for missing fields.",
    )


class PromptSection(BaseModel):
    """A mapped section of the optimized prompt structure."""

    position: Literal["head", "middle", "tail"] = Field(
        ...,
        description="The section position within the prompt.",
    )
    type: Literal["role", "task", "context", "tech_stack", "constraint", "format"] = Field(
        ...,
        description="The semantic type of the prompt section.",
    )
    content: str = Field(
        ...,
        description="The text content of the prompt section.",
    )
    tokenCount: int = Field(
        ...,
        description="The token count of this section.",
    )


class PipelineTokenUsage(BaseModel):
    """Real LLM token usage by pipeline stage."""

    # Planning layer: IntentAnalyzer to Gemma.
    intent_analyzer_prompt: int = 0
    intent_analyzer_completion: int = 0

    # Optimization layer: SemanticPruner to Gemma; zero when skipped.
    semantic_pruner_prompt: int = 0
    semantic_pruner_completion: int = 0

    # Optimization layer: ProfessionalRewriter to Gemma.
    rewriter_prompt: int = 0
    rewriter_completion: int = 0

    # Evaluation layer: JudgePool, accumulated across all judges.
    judge_pool_prompt: int = 0
    judge_pool_completion: int = 0

    # Evaluation layer: PairwiseBattle target model, original plus optimized calls.
    target_model_prompt: int = 0
    target_model_completion: int = 0

    @property
    def total_pipeline_tokens(self) -> int:
        """Total token usage across all pipeline stages."""

        return (
            self.intent_analyzer_prompt
            + self.intent_analyzer_completion
            + self.semantic_pruner_prompt
            + self.semantic_pruner_completion
            + self.rewriter_prompt
            + self.rewriter_completion
            + self.judge_pool_prompt
            + self.judge_pool_completion
            + self.target_model_prompt
            + self.target_model_completion
        )


class TokenStats(BaseModel):
    """Token counts before and after optimization."""

    originalCount: int = Field(
        ...,
        description="The original token count before optimization.",
    )
    optimizedCount: int = Field(
        ...,
        description="The optimized token count after optimization.",
    )
    reductionRate: float = Field(
        ...,
        description="The token reduction rate expressed as a value from 0 to 1.",
    )


class OptimizationResult(BaseModel):
    """Output produced by the prompt optimization engine.

    The pipeline must populate roiReport and check OptimizationConstraints
    before returning this object. If ROI is negative or compression exceeds the cap,
    the pipeline should return the original prompt instead (see OptimizationConstraints).
    """

    optimizedPrompt: str = Field(
        ...,
        description="The optimized prompt text.",
    )
    tokenStats: TokenStats = Field(
        ...,
        description="Token statistics comparing the original and optimized prompt.",
    )
    appliedTechniques: list[str] = Field(
        default_factory=list,
        description="The optimization techniques applied to the prompt.",
    )
    structureMap: list[PromptSection] = Field(
        default_factory=list,
        description="A structural mapping of the optimized prompt.",
    )
    roiReport: Optional["OptimizationROIReport"] = Field(
        default=None,
        description=(
            "Token cost accounting for the optimization process itself. "
            "Must be populated by the pipeline before the result is returned. "
            "If roiReport.roiPositive is False and OptimizationConstraints.requirePositiveROI "
            "is True, the pipeline must return the original prompt instead."
        ),
    )
    optimizationSkipped: bool = Field(
        default=False,
        description=(
            "True when the pipeline decided not to optimize (e.g. negative ROI, prompt already minimal). "
            "When True, optimizedPrompt equals the original prompt and tokenStats shows no reduction."
        ),
    )
    skipReason: Optional[str] = Field(
        default=None,
        description="Human-readable explanation for why optimization was skipped (populated when optimizationSkipped=True).",
    )


class EvaluationScores(BaseModel):
    """Detailed scoring dimensions used during evaluation.

    All scores are on a 0-10 scale where HIGHER IS BETTER.
    Weights per tech-spec section 4.2 and Project Plan section 4:
        intentAlignment   40%
        logicCoherence    30%
        concisenessScore  20%  (replaces old 'redundancyScore'; higher = more concise)
        formatCompliance  10%
    overall = intentAlignment*0.4 + logicCoherence*0.3 + concisenessScore*0.2 + formatCompliance*0.1

    NOTE: field was previously named 'redundancyScore' (lower=better), which caused a sign
    error in the overall formula. Renamed to 'concisenessScore' (higher=better) for
    directional consistency. Judge prompt must instruct: 10 = no redundancy, 0 = highly redundant.
    """

    intentAlignment: float = Field(
        ...,
        description="Intent alignment score 0-10 (higher is better). Weight: 40%.",
    )
    logicCoherence: float = Field(
        ...,
        description="Logic coherence score 0-10 (higher is better). Weight: 30%.",
    )
    concisenessScore: float = Field(
        ...,
        description=(
            "Conciseness score 0-10 (higher is better, 10 = no redundancy). Weight: 20%. "
            "Judges should award 10 when the response has zero unnecessary repetition, "
            "and 0 when the response is highly redundant. "
            "Renamed from 'redundancyScore' to fix directional inconsistency with overall formula."
        ),
    )
    formatCompliance: float = Field(
        ...,
        description="Format compliance score 0-10 (higher is better). Weight: 10%.",
    )
    overall: float = Field(
        ...,
        description=(
            "Overall weighted evaluation score 0-10 (higher is better). "
            "Formula: intentAlignment*0.4 + logicCoherence*0.3 + concisenessScore*0.2 + formatCompliance*0.1. "
            "Optimization is rejected when overall < 6.0 (see OptimizationConstraints)."
        ),
    )


class JudgeVote(BaseModel):
    """A single judge model's evaluation vote and reasoning."""

    model: str = Field(
        ...,
        description="The judge model identifier.",
    )
    winner: Literal["original", "optimized", "tie"] = Field(
        ...,
        description="The prompt version chosen by this judge.",
    )
    reasoning: str = Field(
        ...,
        description="The judge's reasoning for the decision.",
    )


class EvaluationResult(BaseModel):
    """Aggregated evaluation result for an original vs optimized prompt."""

    winner: Literal["original", "optimized", "tie"] = Field(
        ...,
        description="The final winning prompt version across judges.",
    )
    scores: EvaluationScores = Field(
        ...,
        description="The aggregated evaluation scores.",
    )
    judgeResults: list[JudgeVote] = Field(
        default_factory=list,
        description="Votes returned by each judge model.",
    )
    feedback: Optional[str] = Field(
        default=None,
        description="Optional feedback describing why the optimized prompt failed.",
    )


class OptimizationConstraints(BaseModel):
    """Governance guardrails applied at runtime per Project Plan section 7 and tech-spec section 10.

    These constraints enforce positive ROI and quality floors.
    The pipeline must check these before returning any OptimizationResult to the user.
    """

    maxCompressionRate: float = Field(
        default=0.50,
        description=(
            "Hard ceiling on token reduction rate (0-1). "
            "Exceeding 50% triggers a human-review prompt instead of silent output. "
            "Source: Project Plan section 7 Controls and Boundaries."
        ),
    )
    minQualityScore: float = Field(
        default=6.0,
        description=(
            "Minimum acceptable Pairwise Battle overall score (0-10). "
            "When overall < 6.0 after self-correction retries, return the original prompt. "
            "Source: Project Plan section 7."
        ),
    )
    requirePositiveROI: bool = Field(
        default=True,
        description=(
            "If True, the pipeline must verify that tokens consumed by the optimization process "
            "itself (planning + optimization model calls) are fewer than tokens saved. "
            "When ROI is negative (e.g. short prompts), skip optimization and return original. "
            "Source: Project Plan section 7 Risks - negative ROI for already-concise prompts."
        ),
    )
    maxSelfCorrectionRetries: int = Field(
        default=2,
        description=(
            "Maximum number of self-correction retry rounds when the optimized prompt loses "
            "the Pairwise Battle. After exhausting retries, return the original prompt. "
            "Source: tech-spec section 4.1 Step 4."
        ),
    )


class OptimizationROIReport(BaseModel):
    """Token cost accounting for the optimization process itself.

    Used to enforce the positive-ROI constraint in OptimizationConstraints.
    The pipeline populates this before returning OptimizationResult.
    optimizationCostTokens must come from PipelineTokenUsage.total_pipeline_tokens;
    hard-coded optimization costs are forbidden.
    """

    inputTokensSaved: int = Field(
        ...,
        description="Tokens saved in the user's prompt (originalCount - optimizedCount).",
    )
    optimizationCostTokens: int = Field(
        ...,
        description=(
            "Tokens consumed by planning + optimization model calls during the optimization process. "
            "This is the 'cost' that must be lower than inputTokensSaved for positive ROI."
        ),
    )
    netTokenSavings: int = Field(
        ...,
        description="inputTokensSaved - optimizationCostTokens. Positive means ROI is achieved.",
    )
    roiPositive: bool = Field(
        ...,
        description="True when netTokenSavings > 0.",
    )
    pipeline_breakdown: Optional[PipelineTokenUsage] = Field(
        default=None,
        description="Per-stage token usage used to compute optimizationCostTokens.",
    )


__all__ = [
    "SceneType",
    "InputConstraints",
    "UserInput",
    "MissingField",
    "PlanningResult",
    "PromptSection",
    "PipelineTokenUsage",
    "TokenStats",
    "OptimizationResult",
    "EvaluationScores",
    "JudgeVote",
    "EvaluationResult",
    "OptimizationConstraints",
    "OptimizationROIReport",
]
