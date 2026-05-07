"""Lightweight extraction of natural-language mentor answers."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field


class ExtractedAnswers(BaseModel):
    """Fields inferred from one user reply."""

    answers: dict[str, str] = Field(default_factory=dict)
    confidence: float = 0.0


class AnswerExtractor:
    """Rule-based answer extraction for beginner chat replies."""

    LANGUAGE_HINTS = {
        "python": "Python",
        "typescript": "TypeScript",
        "ts": "TypeScript",
        "javascript": "JavaScript",
        "js": "JavaScript",
        "go": "Go",
        "golang": "Go",
        "rust": "Rust",
        "java": "Java",
    }
    FRAMEWORK_HINTS = {
        "react": "React",
        "next": "Next.js",
        "next.js": "Next.js",
        "vue": "Vue",
        "fastapi": "FastAPI",
        "flask": "Flask",
        "django": "Django",
        "express": "Express",
        "tailwind": "Tailwind CSS",
    }

    def extract(
        self,
        user_message: str,
        raw_prompt: str = "",
        existing: dict[str, str] | None = None,
    ) -> ExtractedAnswers:
        """Extract known supplement fields from a free-form reply."""

        existing = existing or {}
        text = user_message.strip()
        lower = text.lower()
        answers: dict[str, str] = {}

        if self._asks_for_default(lower):
            answers.update(self._practical_defaults(raw_prompt, existing))

        language = self._first_hint(lower, self.LANGUAGE_HINTS)
        if language:
            answers["language"] = language

        frameworks = self._all_hints(lower, self.FRAMEWORK_HINTS)
        if frameworks:
            answers["framework"] = ", ".join(frameworks)

        if re.search(r"mobile|responsive", lower):
            answers["constraints"] = self._append(
                answers.get("constraints") or existing.get("constraints", ""),
                "Must work well on mobile and desktop.",
            )
        if re.search(r"animation|transition", lower):
            answers["constraints"] = self._append(
                answers.get("constraints") or existing.get("constraints", ""),
                "Include polished but lightweight animations.",
            )
        if re.search(r"login|auth|jwt|account", lower):
            answers["functionality"] = self._append(
                answers.get("functionality") or existing.get("functionality", ""),
                "User authentication and account flow.",
            )
        if re.search(r"test|pytest|jest|unit", lower):
            answers["output_format"] = self._append(
                answers.get("output_format") or existing.get("output_format", ""),
                "Include implementation steps, file structure, and tests.",
            )
        if re.search(r"explain|beginner|new developer", lower):
            answers["output_format"] = self._append(
                answers.get("output_format") or existing.get("output_format", ""),
                "Explain the code in beginner-friendly language.",
            )
        if re.search(r"pure code|code only", lower):
            answers["output_format"] = "Return only code and file paths."
        if re.search(r"input|output|user.*enter|display|generate|return|upload", lower):
            answers["input_output"] = text.strip()
        if re.search(r"api|database|db|mock|csv|excel|file|local", lower):
            answers["data_requirements"] = text.strip()

        feature_phrases = self._extract_feature_phrase(text)
        if feature_phrases:
            answers["functionality"] = self._append(
                answers.get("functionality") or existing.get("functionality", ""),
                feature_phrases,
            )

        confidence = min(1.0, 0.25 + 0.18 * len(answers)) if answers else 0.0
        return ExtractedAnswers(answers=answers, confidence=confidence)

    def _practical_defaults(
        self,
        raw_prompt: str,
        existing: dict[str, str],
    ) -> dict[str, str]:
        """Choose practical defaults when the user is unsure."""

        combined = f"{raw_prompt} {' '.join(existing.values())}".lower()
        if re.search(r"api|backend|server", combined):
            return {
                "language": existing.get("language", "Python"),
                "framework": existing.get("framework", "FastAPI"),
                "output_format": existing.get(
                    "output_format",
                    "Return file structure, implementation steps, and pytest tests.",
                ),
            }
        if re.search(r"script|automation", combined):
            return {
                "language": existing.get("language", "Python"),
                "framework": existing.get("framework", "standard library where possible"),
                "output_format": existing.get(
                    "output_format",
                    "Return a runnable script with usage instructions and basic tests.",
                ),
            }
        return {
            "language": existing.get("language", "TypeScript"),
            "framework": existing.get("framework", "React"),
            "constraints": existing.get(
                "constraints",
                "Use beginner-friendly defaults and keep the implementation maintainable.",
            ),
            "output_format": existing.get(
                "output_format",
                "Return file structure, implementation steps, and tests.",
            ),
        }

    @staticmethod
    def _asks_for_default(text: str) -> bool:
        return bool(
            re.search(
                r"not sure|default|choose for me|you choose|recommend",
                text,
            )
        )

    @staticmethod
    def _first_hint(text: str, hints: dict[str, str]) -> str | None:
        for key, value in hints.items():
            if re.search(rf"\b{re.escape(key)}\b", text) or key in text:
                return value
        return None

    @staticmethod
    def _all_hints(text: str, hints: dict[str, str]) -> list[str]:
        found: list[str] = []
        for key, value in hints.items():
            if (re.search(rf"\b{re.escape(key)}\b", text) or key in text) and value not in found:
                found.append(value)
        return found

    @staticmethod
    def _append(current: str, addition: str) -> str:
        if not current:
            return addition
        if addition in current:
            return current
        return f"{current}; {addition}"

    @staticmethod
    def _extract_feature_phrase(text: str) -> str:
        if len(text) < 8:
            return ""
        if re.search(r"include|support|feature|need|must|should", text, re.IGNORECASE):
            return text.strip()
        return ""


__all__ = ["AnswerExtractor", "ExtractedAnswers"]
