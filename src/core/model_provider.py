"""Model provider abstractions and shared request/response payloads."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from src.core.token_estimator import TokenEstimator

T = TypeVar("T", bound=BaseModel)


class TokenUsage(BaseModel):
    """Token usage metadata for a completed model generation."""

    prompt_tokens: int = Field(
        ...,
        description="The number of tokens consumed by the input prompt.",
    )
    completion_tokens: int = Field(
        ...,
        description="The number of tokens generated in the completion.",
    )


class GenerateParams(BaseModel):
    """Parameters used for plain-text generation requests."""

    prompt: str = Field(
        ...,
        description="The full prompt text sent to the model.",
    )
    max_tokens: int | None = Field(
        default=None,
        description="Optional upper bound for generated completion tokens.",
    )
    temperature: float | None = Field(
        default=0.7,
        description="Optional sampling temperature for the generation call.",
    )
    stop_sequences: list[str] | None = Field(
        default=None,
        description="Optional stop sequences that terminate generation early.",
    )


class GenerateResult(BaseModel):
    """A text generation response returned by a model provider."""

    text: str = Field(
        ...,
        description="The generated text returned by the provider.",
    )
    usage: TokenUsage = Field(
        ...,
        description="Token usage metadata for the completed generation.",
    )
    parsed: Any | None = Field(
        default=None,
        description="Optional parsed structured payload for schema-constrained calls.",
    )


class StructuredParams(BaseModel):
    """Parameters used for schema-constrained structured generation."""

    model_config = ConfigDict(populate_by_name=True)

    prompt: str = Field(
        ...,
        description="The prompt text used to request structured output.",
    )
    schema_: dict = Field(
        ...,
        alias="schema",
        serialization_alias="schema",
        description="The JSON schema that describes the expected response shape.",
    )
    max_tokens: int | None = Field(
        default=None,
        description="Optional upper bound for generated completion tokens.",
    )

    @property
    def schema(self) -> dict:
        """Return the structured output schema using the tech-spec field name."""

        return self.schema_


class HealthCheckResult(BaseModel):
    """Availability and latency information for a model provider."""

    available: bool = Field(
        ...,
        description="Whether the provider is currently reachable and operational.",
    )
    latency_ms: float = Field(
        ...,
        description="Observed round-trip latency in milliseconds.",
    )


class ModelProvider(ABC):
    """Abstract contract implemented by every model backend in TokenLess."""

    id: str
    type: Literal["local", "api", "hybrid"]

    @abstractmethod
    async def generate(self, params: GenerateParams) -> GenerateResult:
        """Generate plain text from the supplied prompt parameters."""

    @abstractmethod
    async def generate_structured(
        self,
        params: StructuredParams,
        response_type: type[T],
    ) -> GenerateResult:
        """Generate structured output and validate it against a Pydantic model."""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens for a text payload using the provider's tokenization rules."""

    @abstractmethod
    async def health_check(self) -> HealthCheckResult:
        """Check whether the provider is available and report basic latency."""


class ModelRegistry:
    """Container that wires the active providers and tokenizer for the system."""

    def __init__(
        self,
        planning_model: ModelProvider,
        optimization_model: ModelProvider,
        evaluation_models: list[ModelProvider],
        tokenizer: TokenEstimator,
    ) -> None:
        """Create a registry from already-instantiated provider dependencies."""

        self.planning_model = planning_model
        self.optimization_model = optimization_model
        self.evaluation_models = evaluation_models
        self.tokenizer = tokenizer

    @classmethod
    def from_config(cls, config_path: str) -> "ModelRegistry":
        """Build a registry from a model configuration file once providers exist."""

        raise NotImplementedError(
            f"ModelRegistry.from_config is not implemented yet for {config_path!r}."
        )


__all__ = [
    "T",
    "TokenUsage",
    "GenerateParams",
    "GenerateResult",
    "StructuredParams",
    "HealthCheckResult",
    "ModelProvider",
    "ModelRegistry",
]
