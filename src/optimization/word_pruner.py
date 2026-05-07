"""Rule-based word and phrase compression for prompts."""

from __future__ import annotations

import re


REDUNDANT_PHRASES = [
    (r"\bin order to\b", "to"),
    (r"\bit is important to note(?: that)?\b", ""),
    (r"\bplease make sure(?: that)?\b", "ensure"),
    (r"\bdue to the fact that\b", "because"),
    (r"\bin the event that\b", "if"),
    (r"\bfor the purpose of\b", "to"),
    (r"\bat this point in time\b", "now"),
    (r"\bin order for\b", "for"),
    (r"\bthe fact that\b", "that"),
    (r"\ba large number of\b", "many"),
    (r"\bprior to\b", "before"),
]

PASSIVE_PATTERNS = [
    (r"\bshould be processed by the system\b", "process"),
    (r"\bshould be (\w+)ed by\b", r"should \1"),
    (r"\bneeds to be (\w+)ed\b", r"\1"),
    (r"\bhas to be (\w+)ed\b", r"\1"),
    (r"\bmust be (\w+)ed\b", r"\1"),
    (r"\bis (\w+)ed\b", r"\1"),
]

NESTED_PATTERNS = [
    (r"\bgenerate a comprehensive and detailed\b", "generate a"),
    (r"\bcomprehensive and detailed\b", "detailed"),
    (r"\bthat includes? the following\b", "including"),
    (r"\bwhich (includes?|contains?)\b", r"\1"),
    (r"\bvery (important|critical|crucial|essential)\b", r"\1"),
]


class WordPruner:
    """Apply deterministic word-level prompt compression rules."""

    def prune(self, text: str) -> tuple[str, list[str]]:
        """Compress text and return the names of rules that changed it."""

        compressed = text
        applied_rules: list[str] = []

        compressed, changed = self._apply_patterns(compressed, REDUNDANT_PHRASES)
        if changed:
            applied_rules.append("Rule 1 - redundant phrase elimination")

        compressed, changed = self._apply_patterns(compressed, PASSIVE_PATTERNS)
        if changed:
            applied_rules.append("Rule 2 - passive voice simplification")

        compressed, changed = self._apply_patterns(compressed, NESTED_PATTERNS)
        if changed:
            applied_rules.append("Rule 3 - nested modifier simplification")

        compressed, changed = self._remove_repeated_references(compressed)
        if changed:
            applied_rules.append("Rule 4 - reference deduplication")

        compressed, changed = self._mark_format_instruction(compressed)
        if changed:
            applied_rules.append("Rule 5 - format instruction marker")

        return self._clean_whitespace(compressed), applied_rules

    def estimate_reduction(self, original: str, compressed: str) -> float:
        """Return character-level reduction rate, clamped to a minimum of 0."""

        if not original:
            return 0.0
        reduction = (len(original) - len(compressed)) / len(original)
        return max(reduction, 0.0)

    @staticmethod
    def _apply_patterns(
        text: str,
        patterns: list[tuple[str, str]],
    ) -> tuple[str, bool]:
        """Apply regex replacements and report whether text changed."""

        updated = text
        for pattern, replacement in patterns:
            updated = re.sub(pattern, replacement, updated, flags=re.IGNORECASE)
        return updated, updated != text

    def _remove_repeated_references(self, text: str) -> tuple[str, bool]:
        """Replace repeated long noun phrases after their first occurrence."""

        pattern = re.compile(r"\bThe\s+\w+\s+\w+\s+\w+(?:\s+\w+)?", re.IGNORECASE)
        matches = list(pattern.finditer(text))
        phrases: dict[str, list[re.Match[str]]] = {}
        for match in matches:
            phrase = match.group(0)
            phrases.setdefault(phrase.lower(), []).append(match)

        repeated = {
            key: occurrences
            for key, occurrences in phrases.items()
            if len(occurrences) >= 2
        }
        if not repeated:
            return text, False

        updated = text
        for key, occurrences in sorted(
            repeated.items(),
            key=lambda item: item[1][0].start(),
            reverse=True,
        ):
            phrase = occurrences[0].group(0)
            replacement = self._reference_replacement(phrase)
            spans = [(match.start(), match.end()) for match in occurrences[1:]]
            for start, end in sorted(spans, reverse=True):
                updated = updated[:start] + replacement + updated[end:]

        return updated, updated != text

    @staticmethod
    def _reference_replacement(phrase: str) -> str:
        """Choose a short generic reference for a repeated noun phrase."""

        lower_phrase = phrase.lower()
        if "system" in lower_phrase:
            return "the system"
        if "function" in lower_phrase:
            return "the function"
        return "the model"

    def _mark_format_instruction(self, text: str) -> tuple[str, bool]:
        """Append a JSON schema hint when natural-language JSON output is requested."""

        if "[Use JSON Schema if needed]" in text:
            return text, False
        if re.search(r"\boutput a?\s+(?:json|dict|list|array)\b", text, re.IGNORECASE):
            return f"{text} [Use JSON Schema if needed]", True
        return text, False

    @staticmethod
    def _clean_whitespace(text: str) -> str:
        """Collapse excess spaces and blank lines."""

        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


__all__ = ["WordPruner"]
