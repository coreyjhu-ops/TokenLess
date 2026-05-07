"""LM Studio-backed model provider using the local OpenAI-compatible API."""

from __future__ import annotations

import json
import logging
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


class LMStudioProviderError(RuntimeError):
    """Raised when LM Studio requests fail or return invalid payloads."""


class LMStudioProvider(ModelProvider):
    """Model provider implementation for local models served by LM Studio."""

    id = "lmstudio"
    type = "local"

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "lm-studio",
        extra_body: dict | None = None,
    ) -> None:
        """Create a provider that talks to an LM Studio OpenAI-compatible server."""

        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self._extra_body = extra_body
        self._token_estimator = TokenEstimator()
        self._openai = self._load_openai_sdk()

        try:
            self._client = self._openai.AsyncOpenAI(
                base_url=base_url,
                api_key=api_key,
            )
        except Exception as exc:  # pragma: no cover - constructor is thin wrapper
            logger.exception(
                "Failed to initialize LM Studio client for model %s at %s",
                self.model,
                self.base_url,
            )
            raise LMStudioProviderError(
                f"Failed to initialize LM Studio client for model '{self.model}'."
            ) from exc

    async def generate(self, params: GenerateParams) -> GenerateResult:
        """Generate plain text through LM Studio's Chat Completions API."""

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
        """Generate JSON text and validate it against the requested Pydantic model."""

        schema_text = json.dumps(params.schema, ensure_ascii=False, indent=2)
        messages = [
            {
                "role": "system",
                "content": (
                    "You must respond with valid JSON only. "
                    "Do not include markdown fences, explanations, or extra text. "
                    f"Follow this JSON schema exactly:\n{schema_text}"
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
                ),
                response_format=self._build_response_format(
                    schema=params.schema,
                    response_type=response_type,
                ),
            )
        except Exception as exc:
            self._raise_request_error("structured generation", exc)

        raw_text = self._extract_structured_text(completion)
        try:
            parsed = response_type.model_validate_json(
                self._normalize_json_text(raw_text)
            )
        except ValidationError as exc:
            logger.warning(
                "LM Studio returned invalid structured output for model %s. Raw text: %r",
                self.model,
                raw_text,
            )
            raise LMStudioProviderError(
                f"LM Studio returned JSON that did not match schema for model '{self.model}'."
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
        """Count tokens using the shared exact estimator for local models."""

        try:
            return self._token_estimator.exact_count(text, model=self.model)
        except Exception as exc:
            logger.exception(
                "Failed to count tokens for LM Studio model %s",
                self.model,
            )
            raise LMStudioProviderError(
                f"Failed to count tokens for LM Studio model '{self.model}'."
            ) from exc

    async def health_check(self) -> HealthCheckResult:
        """Check LM Studio availability by listing models from the local server."""

        started = perf_counter()
        try:
            await self._client.models.list()
        except Exception:
            latency_ms = (perf_counter() - started) * 1000
            logger.exception(
                "LM Studio health check failed for model %s at %s",
                self.model,
                self.base_url,
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
        if self._extra_body:
            kwargs["extra_body"] = self._extra_body
        return kwargs

    def _extract_usage(
        self,
        *,
        completion: Any,
        prompt_text: str,
        completion_text: str,
    ) -> TokenUsage:
        """Extract provider usage metadata, falling back to local token counting."""

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
        if message is None:
            return ""

        content_parts: list[str] = []
        content = getattr(message, "content", "")
        if isinstance(content, str):
            stripped_content = content.strip()
            if stripped_content:
                content_parts.append(stripped_content)
        elif isinstance(content, list):
            for item in content:
                text = getattr(item, "text", None)
                if text:
                    stripped_text = text.strip()
                    if stripped_text:
                        content_parts.append(stripped_text)

        for reasoning_attr in ("reasoning_content", "reasoning"):
            reasoning_value = getattr(message, reasoning_attr, "")
            if isinstance(reasoning_value, str):
                stripped_reasoning = reasoning_value.strip()
                if stripped_reasoning:
                    content_parts.append(stripped_reasoning)

        return "\n\n".join(content_parts).strip()

    @staticmethod
    def _extract_structured_text(completion: Any) -> str:
        """Pull structured output text, falling back to LM Studio's reasoning field when needed."""

        choices = getattr(completion, "choices", None) or []
        if not choices:
            return ""

        message = getattr(choices[0], "message", None)
        if message is None:
            return ""

        content = getattr(message, "content", "")
        if isinstance(content, str):
            stripped_content = content.strip()
            if stripped_content:
                return stripped_content
        elif isinstance(content, list):
            text_chunks = []
            for item in content:
                text = getattr(item, "text", None)
                if text:
                    stripped_text = text.strip()
                    if stripped_text:
                        text_chunks.append(stripped_text)
            if text_chunks:
                return "\n".join(text_chunks).strip()

        for reasoning_attr in ("reasoning_content", "reasoning"):
            reasoning_value = getattr(message, reasoning_attr, "")
            if isinstance(reasoning_value, str):
                stripped_reasoning = reasoning_value.strip()
                if stripped_reasoning:
                    return stripped_reasoning
        return ""

    @staticmethod
    def _build_response_format(
        *,
        schema: dict[str, Any],
        response_type: type[T],
    ) -> dict[str, Any]:
        """Build a JSON schema response format payload for LM Studio structured output."""

        return {
            "type": "json_schema",
            "json_schema": {
                "name": response_type.__name__,
                "strict": True,
                "schema": schema,
            },
        }

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
        """Import the OpenAI SDK lazily so module import stays lightweight."""

        try:
            return import_module("openai")
        except ModuleNotFoundError as exc:
            raise LMStudioProviderError(
                "LMStudioProvider requires the 'openai' package. "
                "Install project dependencies from requirements.txt before use."
            ) from exc

    def _raise_request_error(self, action: str, exc: Exception) -> None:
        """Log SDK failures and raise a clear provider-specific runtime error."""

        message = f"LM Studio {action} failed for model '{self.model}'."
        timeout_error = getattr(self._openai, "APITimeoutError", None)
        connection_error = getattr(self._openai, "APIConnectionError", None)
        status_error = getattr(self._openai, "APIStatusError", None)

        if timeout_error and isinstance(exc, timeout_error):
            message = (
                f"LM Studio {action} timed out for model '{self.model}' "
                f"at '{self.base_url}'."
            )
        elif connection_error and isinstance(exc, connection_error):
            message = (
                f"Unable to reach LM Studio for model '{self.model}' "
                f"at '{self.base_url}'."
            )
        elif status_error and isinstance(exc, status_error):
            status_code = getattr(exc, "status_code", "unknown")
            message = (
                f"LM Studio {action} failed for model '{self.model}' "
                f"with status {status_code}."
            )

        logger.exception(message)
        raise LMStudioProviderError(message) from exc


__all__ = ["LMStudioProvider", "LMStudioProviderError"]
