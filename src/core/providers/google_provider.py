"""Google Gemini model provider backed by the google-genai SDK."""

from __future__ import annotations

import logging
import os
from importlib import import_module
from time import perf_counter
from typing import Any

from pydantic import BaseModel, ValidationError

from src.core.model_provider import (
    GenerateParams,
    GenerateResult,
    HealthCheckResult,
    ModelProvider,
    StructuredParams,
    T,
    TokenUsage,
)

logger = logging.getLogger(__name__)


class GoogleProviderError(RuntimeError):
    """Raised when Gemini provider setup or requests fail."""


class GoogleProvider(ModelProvider):
    """Model provider implementation backed by Google's Gemini Developer API."""

    id = "google"
    type = "api"

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: str | None = None,
    ) -> None:
        """Create a Google Gemini provider using the supplied or environment API key."""

        self.model = model
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise GoogleProviderError(
                "GoogleProvider requires an API key. Set GOOGLE_API_KEY or pass api_key."
            )

        self._genai = self._load_genai_sdk()
        self._types = self._load_genai_types()

        try:
            self._client = self._genai.Client(api_key=self.api_key)
        except Exception as exc:  # pragma: no cover - thin construction wrapper
            logger.exception(
                "Failed to initialize Google GenAI client for model %s",
                self.model,
            )
            raise GoogleProviderError(
                f"Failed to initialize Google provider client for model '{self.model}'."
            ) from exc

    async def generate(self, params: GenerateParams) -> GenerateResult:
        """Generate plain text through Gemini's async content generation API."""

        try:
            response = await self._client.aio.models.generate_content(
                model=self.model,
                contents=params.prompt,
                config=self._build_generate_config(
                    max_tokens=params.max_tokens,
                    temperature=params.temperature,
                    stop_sequences=params.stop_sequences,
                ),
            )
        except Exception as exc:
            self._raise_request_error("text generation", exc)

        text = self._extract_response_text(response)
        return GenerateResult(
            text=text,
            usage=self._extract_usage(response),
        )

    async def generate_structured(
        self,
        params: StructuredParams,
        response_type: type[T],
    ) -> GenerateResult:
        """Generate JSON output from Gemini and validate it as a Pydantic model."""

        try:
            response = await self._client.aio.models.generate_content(
                model=self.model,
                contents=params.prompt,
                config=self._types.GenerateContentConfig(
                    max_output_tokens=params.max_tokens,
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=response_type,
                ),
            )
        except Exception as exc:
            self._raise_request_error("structured generation", exc)

        parsed = getattr(response, "parsed", None)
        try:
            if isinstance(parsed, response_type):
                validated = parsed
            elif isinstance(parsed, BaseModel):
                validated = response_type.model_validate(parsed.model_dump())
            elif parsed is not None:
                validated = response_type.model_validate(parsed)
            else:
                validated = response_type.model_validate_json(
                    self._extract_response_text(response)
                )
        except ValidationError as exc:
            logger.exception(
                "Google Gemini returned invalid structured output for model %s",
                self.model,
            )
            raise GoogleProviderError(
                f"Google Gemini returned JSON that did not match schema for model '{self.model}'."
            ) from exc
        text = self._extract_response_text(response)
        return GenerateResult(
            text=text,
            usage=self._extract_usage(response),
            parsed=validated,
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens using Gemini's token counting endpoint."""

        try:
            response = self._client.models.count_tokens(
                model=self.model,
                contents=text,
            )
        except Exception as exc:
            logger.exception(
                "Google Gemini token counting failed for model %s",
                self.model,
            )
            raise GoogleProviderError(
                f"Failed to count tokens for Google model '{self.model}'."
            ) from exc

        total_tokens = getattr(response, "total_tokens", None)
        if total_tokens is None:
            total_tokens = getattr(response, "total_token_count", None)
        if total_tokens is None:
            raise GoogleProviderError(
                f"Google token counting did not return total_tokens for model '{self.model}'."
            )
        return int(total_tokens)

    async def health_check(self) -> HealthCheckResult:
        """Check Gemini availability by fetching model metadata."""

        started = perf_counter()
        try:
            await self._client.aio.models.get(model=self.model)
        except Exception:
            latency_ms = (perf_counter() - started) * 1000
            logger.exception(
                "Google Gemini health check failed for model %s",
                self.model,
            )
            return HealthCheckResult(available=False, latency_ms=latency_ms)

        latency_ms = (perf_counter() - started) * 1000
        return HealthCheckResult(available=True, latency_ms=latency_ms)

    def _build_generate_config(
        self,
        *,
        max_tokens: int | None,
        temperature: float | None,
        stop_sequences: list[str] | None,
    ) -> Any:
        """Build a Gemini GenerateContentConfig while omitting unset fields."""

        config_kwargs: dict[str, Any] = {}
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max_tokens
        if temperature is not None:
            config_kwargs["temperature"] = temperature
        if stop_sequences:
            config_kwargs["stop_sequences"] = stop_sequences
        return self._types.GenerateContentConfig(**config_kwargs)

    def _extract_usage(self, response: Any) -> TokenUsage:
        """Extract Gemini usage metadata from a generate_content response."""

        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
        completion_tokens = getattr(usage, "candidates_token_count", 0) or 0
        return TokenUsage(
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
        )

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        """Extract the text payload from a Gemini response object."""

        text = getattr(response, "text", None)
        if isinstance(text, str):
            return text.strip()
        return ""

    @staticmethod
    def _load_genai_sdk() -> Any:
        """Import google-genai lazily so module import stays lightweight."""

        try:
            return import_module("google.genai")
        except ModuleNotFoundError as exc:
            raise GoogleProviderError(
                "GoogleProvider requires the 'google-genai' package. "
                "Install project dependencies from requirements.txt before use."
            ) from exc

    @staticmethod
    def _load_genai_types() -> Any:
        """Import google.genai.types lazily for typed config construction."""

        try:
            return import_module("google.genai.types")
        except ModuleNotFoundError as exc:
            raise GoogleProviderError(
                "GoogleProvider requires the 'google-genai' package. "
                "Install project dependencies from requirements.txt before use."
            ) from exc

    def _raise_request_error(self, action: str, exc: Exception) -> None:
        """Log SDK failures and raise a clear provider-specific runtime error."""

        message = f"Google Gemini {action} failed for model '{self.model}'."
        lower_message = str(exc).lower()
        if "timeout" in lower_message:
            message = f"Google Gemini {action} timed out for model '{self.model}'."
        elif any(
            token in lower_message
            for token in ("connection", "network", "dns", "unreachable")
        ):
            message = f"Unable to reach Google Gemini for model '{self.model}'."

        logger.exception(message)
        raise GoogleProviderError(message) from exc


__all__ = ["GoogleProvider", "GoogleProviderError"]
