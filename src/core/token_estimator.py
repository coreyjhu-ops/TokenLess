"""Token counting and estimation utilities used across TokenLess."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from pydantic import BaseModel, Field


class TokenCompareReport(BaseModel):
    """Comparison metrics between an original and optimized prompt."""

    original_tokens: int = Field(
        ...,
        description="The token count of the original prompt.",
    )
    optimized_tokens: int = Field(
        ...,
        description="The token count of the optimized prompt.",
    )
    saved_tokens: int = Field(
        ...,
        description="The number of tokens saved after optimization.",
    )
    reduction_percent: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="The percentage of tokens reduced, clamped between 0 and 100.",
    )


class TokenEstimator:
    """Utility for exact token counting, fast estimation, and comparison reports."""

    def __init__(self, default_encoding: str = "cl100k_base") -> None:
        """Initialize the estimator and cache the default tokenizer encoding."""

        self.default_encoding = default_encoding
        self._tiktoken = self._load_tiktoken()
        self._default_encoder = self._tiktoken.get_encoding(default_encoding)

    def exact_count(self, text: str, model: str | None = None) -> int:
        """Return the exact token count using a model-specific tokenizer when possible."""

        if not text:
            return 0

        encoder = self._default_encoder
        if model:
            try:
                encoder = self._tiktoken.encoding_for_model(model)
            except Exception:
                encoder = self._default_encoder

        return len(encoder.encode(text))

    def quick_estimate(self, text: str) -> int:
        """Estimate token usage quickly with simple ASCII and CJK heuristics."""

        if not text:
            return 0

        estimate = 0
        in_ascii_segment = False

        for char in text:
            if self._is_cjk(char):
                if in_ascii_segment:
                    estimate += 1
                    in_ascii_segment = False
                estimate += 2
                continue

            if self._is_ascii(char):
                if char.isspace():
                    if in_ascii_segment:
                        estimate += 1
                        in_ascii_segment = False
                else:
                    in_ascii_segment = True
                continue

            if in_ascii_segment:
                estimate += 1
                in_ascii_segment = False
            estimate += 1

        if in_ascii_segment:
            estimate += 1

        return estimate

    def compare_report(
        self,
        original: str,
        optimized: str,
        model: str | None = None,
    ) -> TokenCompareReport:
        """Compare exact token counts for original and optimized prompt variants."""

        original_tokens = self.exact_count(original, model=model)
        optimized_tokens = self.exact_count(optimized, model=model)
        saved_tokens = original_tokens - optimized_tokens

        if original_tokens == 0:
            reduction_percent = 0.0
        else:
            reduction_percent = (saved_tokens / original_tokens) * 100
            reduction_percent = min(max(reduction_percent, 0.0), 100.0)

        return TokenCompareReport(
            original_tokens=original_tokens,
            optimized_tokens=optimized_tokens,
            saved_tokens=saved_tokens,
            reduction_percent=reduction_percent,
        )

    @staticmethod
    def _is_ascii(char: str) -> bool:
        """Return whether a character belongs to the ASCII range."""

        return ord(char) < 128

    @staticmethod
    def _is_cjk(char: str) -> bool:
        """Return whether a character belongs to a common CJK Unicode block."""

        code_point = ord(char)
        return (
            0x3400 <= code_point <= 0x4DBF
            or 0x4E00 <= code_point <= 0x9FFF
            or 0xF900 <= code_point <= 0xFAFF
            or 0x20000 <= code_point <= 0x2A6DF
            or 0x2A700 <= code_point <= 0x2B73F
            or 0x2B740 <= code_point <= 0x2B81F
            or 0x2B820 <= code_point <= 0x2CEAF
            or 0x2CEB0 <= code_point <= 0x2EBEF
            or 0x30000 <= code_point <= 0x3134F
        )

    @staticmethod
    def _load_tiktoken() -> Any:
        """Import tiktoken lazily and raise a clear runtime error if unavailable."""

        try:
            return import_module("tiktoken")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "TokenEstimator requires the 'tiktoken' package. "
                "Install project dependencies from requirements.txt before use."
            ) from exc


__all__ = [
    "TokenCompareReport",
    "TokenEstimator",
]
