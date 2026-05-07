"""Model-assisted semantic pruning for middle-context prompt content."""

from __future__ import annotations

import asyncio
import re

from pydantic import BaseModel, Field

from src.core.model_provider import ModelProvider, StructuredParams, TokenUsage


class _RelevanceScores(BaseModel):
    """Structured relevance scores returned by the optimization model."""

    scores: list[float] = Field(default_factory=list)


class SemanticPruner:
    """Filter low-relevance middle context using a structured model call."""

    def __init__(
        self,
        model_provider: ModelProvider,
        relevance_threshold: float = 0.3,
    ) -> None:
        """Create a semantic pruner with the supplied provider and threshold."""

        self.model_provider = model_provider
        self.relevance_threshold = relevance_threshold

    async def prune(
        self,
        context_text: str,
        core_objective: str,
    ) -> tuple[str, float, TokenUsage]:
        """Prune irrelevant context sentences and return noise ratio.

        Failures in model scoring are intentionally non-fatal: if scoring fails
        or returns invalid data, every sentence receives score 1.0 and the input
        context is preserved apart from whitespace normalization.
        """

        sentences = self._split_sentences(context_text)
        if not sentences:
            return "", 0.0, TokenUsage(prompt_tokens=0, completion_tokens=0)

        scores, usage = await self._score_sentences(sentences, core_objective)
        if len(scores) != len(sentences):
            scores = [1.0] * len(sentences)

        kept_sentences = [
            sentence
            for sentence, score in zip(sentences, scores, strict=True)
            if score >= self.relevance_threshold
        ]
        removed_count = len(sentences) - len(kept_sentences)
        noise_ratio = removed_count / len(sentences)

        deduped_sentences = self._deduplicate_adjacent(kept_sentences)
        return self._clean_whitespace(" ".join(deduped_sentences)), noise_ratio, usage

    def prune_sync(
        self,
        context_text: str,
        core_objective: str,
    ) -> tuple[str, float, TokenUsage]:
        """Synchronous wrapper around :meth:`prune`."""

        return asyncio.run(self.prune(context_text, core_objective))

    def _split_sentences(self, context_text: str) -> list[str]:
        """Split text into sentences while preserving terminal punctuation."""

        parts = re.split(r"(?<=[.！？.!?])\s*", context_text)
        return [part.strip() for part in parts if part.strip()]

    async def _score_sentences(
        self,
        sentences: list[str],
        core_objective: str,
    ) -> tuple[list[float], TokenUsage]:
        """Ask the provider to score sentence relevance, falling back on failure."""

        numbered_sentences = "\n".join(
            f"{index}. {sentence}"
            for index, sentence in enumerate(sentences, start=1)
        )
        prompt = (
            "Rate each sentence's relevance to the core objective on a scale of "
            "0.0 to 1.0.\n\n"
            f'Core objective: "{core_objective}"\n\n'
            "Sentences:\n"
            f"{numbered_sentences}\n\n"
            'Return JSON: {"scores": [0.8, 0.2, ...]}  '
            "(same order as sentences)"
        )

        try:
            response = await self.model_provider.generate_structured(
                StructuredParams(
                    prompt=prompt,
                    schema=_RelevanceScores.model_json_schema(),
                    max_tokens=256,
                ),
                _RelevanceScores,
            )
        except Exception:
            return [1.0] * len(sentences), TokenUsage(
                prompt_tokens=0,
                completion_tokens=0,
            )

        parsed = response.parsed
        if not isinstance(parsed, _RelevanceScores):
            parsed = _RelevanceScores.model_validate(parsed)

        scores = parsed.scores
        if len(scores) != len(sentences):
            return [1.0] * len(sentences), response.usage
        return [min(max(float(score), 0.0), 1.0) for score in scores], response.usage

    def _deduplicate_adjacent(self, sentences: list[str]) -> list[str]:
        """Remove adjacent sentences with high token Jaccard similarity."""

        if not sentences:
            return []

        deduped = [sentences[0]]
        for sentence in sentences[1:]:
            if self._jaccard_similarity(deduped[-1], sentence) > 0.7:
                continue
            deduped.append(sentence)
        return deduped

    @staticmethod
    def _jaccard_similarity(left: str, right: str) -> float:
        """Return simple word-set Jaccard similarity for two sentences."""

        left_words = set(re.findall(r"\w+", left.lower()))
        right_words = set(re.findall(r"\w+", right.lower()))
        if not left_words and not right_words:
            return 1.0
        if not left_words or not right_words:
            return 0.0
        return len(left_words & right_words) / len(left_words | right_words)

    @staticmethod
    def _clean_whitespace(text: str) -> str:
        """Collapse repeated whitespace."""

        return re.sub(r"\s+", " ", text).strip()


__all__ = ["SemanticPruner"]
