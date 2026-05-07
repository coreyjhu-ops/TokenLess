"""Conversation state models for TokenLess chat orchestration."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.core.types import (
    EvaluationResult,
    OptimizationResult,
    PipelineTokenUsage,
    PlanningResult,
)


ConversationStatus = Literal[
    "planning",
    "awaiting_supplement",
    "optimizing",
    "evaluating",
    "done",
    "error",
]

ChatRole = Literal["user", "assistant", "system"]

ConversationStage = Literal[
    "welcome",
    "collecting",
    "clarifying",
    "confirming",
    "optimizing",
    "done",
    "error",
]


class ChatMessage(BaseModel):
    """One message in the mentor-style chat session."""

    role: ChatRole
    content: str


class ConversationState(BaseModel):
    """Single-turn conversation state across the full TokenLess pipeline."""

    session_id: str
    raw_prompt: str
    planning_result: PlanningResult | None = None
    optimization_result: OptimizationResult | None = None
    evaluation_result: EvaluationResult | None = None
    final_prompt: str | None = None
    supplements: dict[str, str] = Field(default_factory=dict)
    status: ConversationStatus = "planning"
    error_message: str | None = None
    token_ledger: PipelineTokenUsage | None = Field(
        default=None,
        description="Pipeline real token usage by stage.",
    )


class ChatSessionState(BaseModel):
    """Multi-turn UI state for the beginner-friendly mentor flow."""

    session_id: str
    messages: list[ChatMessage] = Field(default_factory=list)
    stage: ConversationStage = "welcome"
    raw_prompt: str = ""
    supplements: dict[str, str] = Field(default_factory=dict)
    planning_result: PlanningResult | None = None
    final_state: ConversationState | None = None
    turn_count: int = 0
    error_message: str | None = None


__all__ = [
    "ChatMessage",
    "ChatRole",
    "ChatSessionState",
    "ConversationStage",
    "ConversationState",
    "ConversationStatus",
]
