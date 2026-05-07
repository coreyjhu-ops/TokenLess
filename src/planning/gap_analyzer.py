"""Missing-field analysis against scene-specific instruction references."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator

from src.core.model_provider import ModelProvider, StructuredParams
from src.core.types import MissingField, SceneType
from src.planning.intent_analyzer import IntentAnalysis
from src.planning.scene_detector import SceneDetector


class _ClarificationQuestions(BaseModel):
    """Structured model output for targeted clarification questions."""

    questions: str = Field(
        default="",
        description="A numbered list of targeted clarification questions.",
    )

    @field_validator("questions", mode="before")
    @classmethod
    def _normalize_questions(cls, value: object) -> str:
        """Normalize missing model output to an empty string."""

        if value is None:
            return ""
        return str(value).strip()


class GapAnalyzer:
    """Pure rule engine that identifies missing vibe coding requirements."""

    LANGUAGE_KEYWORDS = (
        "python",
        "javascript",
        "typescript",
        "java",
        "go",
        "golang",
        "rust",
        "sql",
        "c++",
        "cpp",
        "c#",
        "csharp",
        "php",
        "ruby",
        "swift",
        "kotlin",
        "scala",
        "r",
        "matlab",
        "bash",
        "shell",
        "html",
        "css",
        "js",
        "ts",
    )
    FRAMEWORK_KEYWORDS = (
        "react",
        "vue",
        "angular",
        "svelte",
        "next",
        "next.js",
        "nuxt",
        "fastapi",
        "django",
        "flask",
        "express",
        "nestjs",
        "spring",
        "rails",
        "laravel",
        "pytest",
        "jest",
        "pytorch",
        "tensorflow",
        "pandas",
        "numpy",
    )
    CODE_STYLE_KEYWORDS = (
        "lint",
        "style",
        "type hint",
        "type hints",
        "typing",
        "docstring",
        "docstrings",
        "pep8",
        "prettier",
        "eslint",
        "black",
    )
    IMPORTANCE_ORDER = {"critical": 0, "recommended": 1, "optional": 2}

    def __init__(self, scene_detector: SceneDetector) -> None:
        """Create the analyzer with access to loaded instruction refs."""

        self.scene_detector = scene_detector

    def analyze(self, intent: IntentAnalysis, scene: SceneType) -> list[MissingField]:
        """Compare extracted intent against the scene's expected fields."""

        self.scene_detector.get_ref(scene)
        how = intent.how.lower()
        what = intent.what.strip()
        output_format = intent.format.strip()

        missing: list[MissingField] = []
        if not self._contains_any(how, self.LANGUAGE_KEYWORDS):
            missing.append(
                MissingField(
                    field="language",
                    question=self.beginner_question_for("language", raw_prompt=what),
                    importance="critical",
                    defaultValue=None,
                )
            )
        if not self._contains_any(how, self.FRAMEWORK_KEYWORDS):
            missing.append(
                MissingField(
                    field="framework",
                    question=self.beginner_question_for("framework", raw_prompt=what),
                    importance="critical",
                    defaultValue=None,
                )
            )
        if len(what) < 10:
            missing.append(
                MissingField(
                    field="functionality",
                    question=self.beginner_question_for("functionality", raw_prompt=what),
                    importance="critical",
                    defaultValue=None,
                )
            )
        if not self._has_constraints(how):
            missing.append(
                MissingField(
                    field="constraints",
                    question=self.beginner_question_for("constraints", raw_prompt=what),
                    importance="recommended",
                    defaultValue="Use practical beginner-friendly defaults",
                )
            )
        if output_format == "":
            missing.append(
                MissingField(
                    field="output_format",
                    question=self.beginner_question_for("output_format", raw_prompt=what),
                    importance="recommended",
                    defaultValue="Return code, file structure, explanation, and tests",
                )
            )
        if not self._contains_any(how, self.CODE_STYLE_KEYWORDS):
            missing.append(
                MissingField(
                    field="code_style",
                    question=self.beginner_question_for("code_style", raw_prompt=what),
                    importance="optional",
                    defaultValue="Follow the standard style for the chosen stack",
                )
            )
        return sorted(
            missing,
            key=lambda field: self.IMPORTANCE_ORDER[field.importance],
        )

    def generate_questions(self, missing_fields: list[MissingField]) -> str:
        """Format missing fields as a chatbot-friendly clarification message."""

        if not missing_fields:
            return "No missing information was detected. You can continue to optimization."

        lines = ["The following information is missing. Please add details:"]
        for field in missing_fields:
            marker = self._importance_marker(field.importance)
            line = f"{marker} {field.question}"
            if field.defaultValue:
                line = f"{line} (default: {field.defaultValue})"
            lines.append(line)
        return "\n".join(lines)

    async def generate_targeted_questions(
        self,
        raw_prompt: str,
        intent: IntentAnalysis,
        missing_fields: list[MissingField],
        model_provider: ModelProvider,
    ) -> str:
        """Generate prompt-specific clarification questions with a model.

        The original rule-based :meth:`generate_questions` remains the fallback
        path when no critical/recommended fields are missing or the model cannot
        produce a usable numbered list.
        """

        target_fields = [
            field
            for field in missing_fields
            if field.importance in {"critical", "recommended"}
        ]
        if not target_fields:
            return self.generate_questions(missing_fields)

        missing_text = "\n".join(
            (
                f"- field: {field.field}; importance: {field.importance}; "
                f"fallback_question: {field.question}"
            )
            for field in target_fields
        )
        prompt = (
            "Generate targeted clarification questions for a software-development prompt.\n\n"
            "Requirements:\n"
            "- Base every question on the user's concrete raw prompt content.\n"
            "- Detect whether the user wrote mainly Chinese or English; write the questions in the same language.\n"
            "- Ask only about critical and recommended missing fields listed below.\n"
            "- Produce 2 to 4 questions in numbered-list format.\n"
            "- Do not use generic templates; make each question specific to the user's scenario.\n"
            "- Do not answer the questions or invent missing facts.\n\n"
            f"Raw prompt:\n{raw_prompt}\n\n"
            "Intent analysis:\n"
            f"- who: {intent.who}\n"
            f"- what: {intent.what}\n"
            f"- how: {intent.how}\n"
            f"- format: {intent.format}\n\n"
            f"Missing fields to ask about:\n{missing_text}\n\n"
            'Return JSON with key "questions" containing only the numbered list.'
        )

        try:
            response = await model_provider.generate_structured(
                StructuredParams(
                    prompt=prompt,
                    schema=_ClarificationQuestions.model_json_schema(),
                    max_tokens=512,
                ),
                _ClarificationQuestions,
            )
        except Exception:
            return self.generate_questions(target_fields)

        try:
            parsed = response.parsed
            if not isinstance(parsed, _ClarificationQuestions):
                parsed = _ClarificationQuestions.model_validate(parsed)
        except Exception:
            return self.generate_questions(target_fields)

        questions = parsed.questions.strip()
        if not questions:
            return self.generate_questions(target_fields)
        return questions

    @staticmethod
    def beginner_question_for(field: str, raw_prompt: str = "") -> str:
        """Return beginner-friendly clarification wording for a missing field."""

        questions = {
            "language": "Which programming language should this project use? If you are unsure, say 'choose for me'.",
            "framework": (
                "Which framework do you prefer? For example, a web app could use React "
                "and a backend API could use FastAPI. If you are unsure, I can recommend one."
            ),
            "functionality": "What are the 2-3 most important features?",
            "constraints": "Any specific requirements, such as mobile support, login, saved data, speed, or visual polish?",
            "output_format": "Should the AI return only code, or code plus explanation, file structure, and tests?",
            "code_style": "Should it follow any code style, such as type hints, docstrings, linting, or detailed comments?",
        }
        return questions.get(field, "What other important details should be included?")

    @staticmethod
    def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
        """Return whether any keyword appears in text."""

        for keyword in keywords:
            if GapAnalyzer._keyword_matches(text, keyword):
                return True
        return False

    @staticmethod
    def _keyword_matches(text: str, keyword: str) -> bool:
        """Match keywords without letting short terms hit inside longer words."""

        escaped = re.escape(keyword)
        if re.fullmatch(r"[a-z0-9]+", keyword):
            return bool(re.search(rf"\b{escaped}\b", text))
        return keyword in text

    @staticmethod
    def _has_constraints(text: str) -> bool:
        """Detect version or constraint hints in the HOW dimension."""

        return bool(re.search(r"\d", text)) or "latest" in text or "version" in text

    @staticmethod
    def _importance_marker(importance: str) -> str:
        """Map importance to the display label expected by the chatbot."""

        if importance == "critical":
            return "[Required]"
        if importance == "recommended":
            return "[Recommended]"
        return "[Optional]"
