"""OpenAI API-backed model provider implementation for hosted chat models."""

from __future__ import annotations

import json
import logging
import os
import re
from importlib import import_module
from time import perf_counter
from typing import Any

from pydantic import ValidationError

from src.core.model_provider import (
    GenerateParams,
    GenerateResult,
    HealthCheckResult,
    ModelProvider,
    StructuredParams,
    T,
    TokenUsage,
)
from src.core.token_estimator import TokenEstimator

logger = logging.getLogger(__name__)


class OpenAIProviderError(RuntimeError):
    """Raised when OpenAI provider setup or requests fail."""


class OpenAIProvider(ModelProvider):
    """Model provider implementation backed by the OpenAI API."""

    id = "openai"
    type = "api"

    def __init__(self, model: str = "gpt-4o", api_key: str | None = None) -> None:
        """Create an OpenAI provider using the supplied or environment API key."""

        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise OpenAIProviderError(
                "OpenAIProvider requires an API key. Set OPENAI_API_KEY or pass api_key."
            )

        self._token_estimator = TokenEstimator()
        self._openai = self._load_openai_sdk()
        self._tiktoken = self._load_tiktoken_sdk()

        try:
            self._client = self._openai.AsyncOpenAI(api_key=self.api_key)
        except Exception as exc:  # pragma: no cover - thin construction wrapper
            logger.exception(
                "Failed to initialize OpenAI client for model %s",
                self.model,
            )
            raise OpenAIProviderError(
                f"Failed to initialize OpenAI client for model '{self.model}'."
            ) from exc

    async def generate(self, params: GenerateParams) -> GenerateResult:
        """Generate plain text via OpenAI Chat Completions."""

        try:
            completion = await self._client.chat.completions.create(
                **self._build_chat_kwargs(
                    messages=[{"role": "user", "content": params.prompt}],
                    max_tokens=params.max_tokens,
                    temperature=params.temperature,
                    stop_sequences=params.stop_sequences,
                )
            )
        except Exception as exc:
            self._raise_request_error("text generation", exc)

        text = self._extract_message_text(completion)
        return GenerateResult(
            text=text,
            usage=self._extract_usage(
                completion=completion,
                prompt_text=params.prompt,
                completion_text=text,
            ),
        )

    async def generate_structured(
        self,
        params: StructuredParams,
        response_type: type[T],
    ) -> GenerateResult:
        """Generate JSON output with OpenAI JSON mode and validate it locally."""

        schema_text = json.dumps(params.schema, ensure_ascii=False, indent=2)
        messages = [
            {
                "role": "system",
                "content": (
                    "Return valid JSON only. "
                    "Do not include markdown, explanations, or extra text. "
                    f"Match this schema:\n{schema_text}"
                ),
            },
            {"role": "user", "content": params.prompt},
        ]

        try:
            completion = await self._client.chat.completions.create(
                **self._build_chat_kwargs(
                    messages=messages,
                    max_tokens=params.max_tokens,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
            )
        except Exception as exc:
            self._raise_request_error("structured generation", exc)

        raw_text = self._extract_message_text(completion)
        try:
            parsed = response_type.model_validate_json(
                self._normalize_json_text(raw_text)
            )
        except ValidationError as exc:
            logger.exception(
                "OpenAI returned invalid structured output for model %s",
                self.model,
            )
            raise OpenAIProviderError(
                f"OpenAI returned JSON that did not match schema for model '{self.model}'."
            ) from exc
        return GenerateResult(
            text=raw_text,
            usage=self._extract_usage(
                completion=completion,
                prompt_text=params.prompt,
                completion_text=raw_text,
            ),
            parsed=parsed,
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens with tiktoken using the active model mapping when available."""

        if not text:
            return 0

        try:
            encoder = self._tiktoken.encoding_for_model(self.model)
        except Exception:
            logger.warning(
                "No exact tiktoken mapping found for model %s; using %s fallback",
                self.model,
                self._token_estimator.default_encoding,
            )
            encoder = self._tiktoken.get_encoding(self._token_estimator.default_encoding)

        try:
            return len(encoder.encode(text))
        except Exception as exc:
            logger.exception(
                "Failed to count tokens for OpenAI model %s",
                self.model,
            )
            raise OpenAIProviderError(
                f"Failed to count tokens for OpenAI model '{self.model}'."
            ) from exc

    async def health_check(self) -> HealthCheckResult:
        """Check OpenAI availability by retrieving metadata for the configured model."""

        started = perf_counter()
        try:
            await self._client.models.retrieve(self.model)
        except Exception:
            latency_ms = (perf_counter() - started) * 1000
            logger.exception(
                "OpenAI health check failed for model %s",
                self.model,
            )
            return HealthCheckResult(available=False, latency_ms=latency_ms)

        latency_ms = (perf_counter() - started) * 1000
        return HealthCheckResult(available=True, latency_ms=latency_ms)

    def _build_chat_kwargs(
        self,
        *,
        messages: list[dict[str, str]],
        max_tokens: int | None,
        temperature: float | None,
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Build a chat completion payload while omitting unset optional fields."""

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature
        if stop_sequences:
            kwargs["stop"] = stop_sequences
        if response_format is not None:
            kwargs["response_format"] = response_format
        return kwargs

    def _extract_usage(
        self,
        *,
        completion: Any,
        prompt_text: str,
        completion_text: str,
    ) -> TokenUsage:
        """Extract OpenAI usage metadata, falling back to local counting if absent."""

        usage = getattr(completion, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)

        if prompt_tokens is None:
            prompt_tokens = self.count_tokens(prompt_text)
        if completion_tokens is None:
            completion_tokens = self.count_tokens(completion_text)

        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    @staticmethod
    def _extract_message_text(completion: Any) -> str:
        """Pull the first assistant message text from a chat completion response."""

        choices = getattr(completion, "choices", None) or []
        if not choices:
            return ""

        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text_chunks = []
            for item in content:
                text = getattr(item, "text", None)
                if text:
                    text_chunks.append(text)
            return "\n".join(text_chunks).strip()
        return ""

    @staticmethod
    def _normalize_json_text(text: str) -> str:
        """Strip markdown code fences so validation can parse the JSON payload."""

        normalized = text.strip()
        fence_pattern = r"^```(?:json)?\s*(.*?)\s*```$"
        match = re.match(fence_pattern, normalized, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
        return normalized

    @staticmethod
    def _load_openai_sdk() -> Any:
        """Import the OpenAI SDK lazily so the module imports without dependencies."""

        try:
            return import_module("openai")
        except ModuleNotFoundError as exc:
            raise OpenAIProviderError(
                "OpenAIProvider requires the 'openai' package. "
                "Install project dependencies from requirements.txt before use."
            ) from exc

    @staticmethod
    def _load_tiktoken_sdk() -> Any:
        """Import tiktoken lazily for precise OpenAI token counting."""

        try:
            return import_module("tiktoken")
        except ModuleNotFoundError as exc:
            raise OpenAIProviderError(
                "OpenAIProvider requires the 'tiktoken' package for token counting."
            ) from exc

    def _raise_request_error(self, action: str, exc: Exception) -> None:
        """Log SDK failures and raise a clear provider-specific runtime error."""

        message = f"OpenAI {action} failed for model '{self.model}'."
        timeout_error = getattr(self._openai, "APITimeoutError", None)
        connection_error = getattr(self._openai, "APIConnectionError", None)
        status_error = getattr(self._openai, "APIStatusError", None)

        if timeout_error and isinstance(exc, timeout_error):
            message = f"OpenAI {action} timed out for model '{self.model}'."
        elif connection_error and isinstance(exc, connection_error):
            message = f"Unable to reach the OpenAI API for model '{self.model}'."
        elif status_error and isinstance(exc, status_error):
            status_code = getattr(exc, "status_code", "unknown")
            message = (
                f"OpenAI {action} failed for model '{self.model}' "
                f"with status {status_code}."
            )

        logger.exception(message)
        raise OpenAIProviderError(message) from exc


__all__ = ["OpenAIProvider", "OpenAIProviderError"]
