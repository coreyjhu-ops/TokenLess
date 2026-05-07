"""Mentor-style multi-turn orchestration for TokenLess."""

from __future__ import annotations

import re
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from src.chat.answer_extractor import AnswerExtractor
from src.chat.chatbot import TokenLessChatbot
from src.chat.conversation import ChatMessage, ChatSessionState
from src.core.model_provider import StructuredParams
from src.core.token_estimator import TokenEstimator
from src.core.types import EvaluationResult, OptimizationResult, PlanningResult


class _MentorQuestion(BaseModel):
    """Structured model output for one reflective follow-up question."""

    question: str = Field(default="")

    @field_validator("question", mode="before")
    @classmethod
    def _normalize_question(cls, value: object) -> str:
        if value is None:
            return ""
        return str(value).strip()


class TokenLessMentor:
    """Beginner-friendly chat layer that wraps the existing pipeline."""

    CRITICAL_FIELDS = {"language", "framework", "functionality"}

    def __init__(
        self,
        chatbot: TokenLessChatbot,
        token_estimator: TokenEstimator | None = None,
        min_turns: int = 5,
        max_turns: int = 10,
    ) -> None:
        self.chatbot = chatbot
        self.extractor = AnswerExtractor()
        self.token_estimator = token_estimator or TokenEstimator()
        self.min_turns = min_turns
        self.max_turns = max_turns

    async def start(self) -> ChatSessionState:
        """Create a new chat session with a welcoming assistant message."""

        return ChatSessionState(
            session_id=str(uuid4()),
            stage="welcome",
            messages=[
                ChatMessage(
                    role="assistant",
                    content=(
                        "Hi, I am TokenLess. Tell me what project you want an AI coding "
                        "model to help you build. You do not need to explain everything "
                        "at once; I will ask focused questions like a prompt mentor."
                    ),
                )
            ],
        )

    async def receive(
        self,
        state: ChatSessionState,
        user_message: str,
    ) -> ChatSessionState:
        """Handle one user turn and return the updated chat session."""

        message = user_message.strip()
        if not message:
            return state

        state.messages.append(ChatMessage(role="user", content=message))
        state.turn_count += 1

        try:
            if not state.raw_prompt:
                state.raw_prompt = message
            else:
                extracted = self.extractor.extract(
                    message,
                    raw_prompt=state.raw_prompt,
                    existing=state.supplements,
                )
                state.supplements.update(extracted.answers)

            if state.raw_prompt and state.planning_result is None:
                state.planning_result, _usage = await self.chatbot.planner.plan_with_usage(
                    state.raw_prompt
                )

            if self.should_optimize(state, latest_user_message=message):
                return await self._optimize(state)

            state.stage = self._next_stage(state, message)
            question = await self.next_question(state)
            state.messages.append(
                ChatMessage(role="assistant", content=question)
            )
            return state
        except Exception as exc:
            state.stage = "error"
            state.error_message = str(exc)
            state.messages.append(
                ChatMessage(
                    role="assistant",
                    content=f"I hit a problem while processing this: {state.error_message}",
                )
            )
            return state

    def should_optimize(
        self,
        state: ChatSessionState,
        latest_user_message: str = "",
    ) -> bool:
        """Return whether the mentor should run the full pipeline now."""

        latest = latest_user_message.lower()
        explicit_finish = any(
            phrase in latest
            for phrase in (
                "finish now",
                "run now",
                "optimize now",
                "generate now",
                "skip questions",
                "start optimization",
            )
        )
        enough_turns = state.turn_count >= self.min_turns
        confirmed = enough_turns and self._is_confirmation(latest_user_message)
        if explicit_finish:
            return bool(state.raw_prompt)
        return bool(state.raw_prompt and enough_turns and confirmed)

    async def next_question(self, state: ChatSessionState) -> str:
        """Choose the next focused mentor question."""

        if state.turn_count == 1 and state.planning_result:
            summary = state.planning_result.detectedIntent or state.raw_prompt
            first_focus = await self._reflective_question(state)
            return (
                f"I understand the project as: {summary}\n\n"
                f"{first_focus}"
            )

        if state.stage == "confirming" and state.turn_count >= self.min_turns:
            state.stage = "confirming"
            return (
                "I have enough information to generate the final prompt. Reply "
                "'confirm' and I will start optimization; if you want to add details, "
                "send them directly and I will keep asking around your project."
            )
        return await self._reflective_question(state)

    def user_metrics(self, state: ChatSessionState) -> dict[str, str]:
        """Return simple user-facing metrics for the final result."""

        final_state = state.final_state
        if not final_state or not final_state.optimization_result:
            original = self.token_estimator.exact_count(state.raw_prompt)
            return {
                "original_prompt_tokens": str(original),
                "optimized_prompt_tokens": "-",
                "target_reasoning_tokens": "-",
                "target_answer_tokens": "-",
                "target_total_tokens": "-",
                "model_score": "-",
                "original_tokens": str(original),
                "optimized_tokens": "-",
                "token_savings": "-",
            }

        stats = final_state.optimization_result.tokenStats
        final_prompt = self.display_prompt(state)
        target_usage = self.estimate_target_model_usage(final_prompt, state)
        score = (
            f"{final_state.evaluation_result.scores.overall:.2f}"
            if final_state.evaluation_result
            else "-"
        )
        metrics = {
            "original_prompt_tokens": str(stats.originalCount),
            "optimized_prompt_tokens": str(stats.optimizedCount),
            "target_reasoning_tokens": str(target_usage["reasoning_tokens"]),
            "target_answer_tokens": str(target_usage["answer_tokens"]),
            "target_total_tokens": str(target_usage["total_tokens"]),
            "model_score": score,
        }
        metrics.update(
            {
                "original_tokens": metrics["original_prompt_tokens"],
                "optimized_tokens": metrics["optimized_prompt_tokens"],
                "token_savings": str(stats.originalCount - stats.optimizedCount),
            }
        )
        return metrics

    def display_prompt(self, state: ChatSessionState) -> str:
        """Choose the best prompt for the user-facing final prompt module."""

        final_state = state.final_state
        if not final_state:
            return state.raw_prompt
        optimization = final_state.optimization_result
        optimized = optimization.optimizedPrompt if optimization else ""
        pipeline_final = final_state.final_prompt or ""

        if optimized and self._looks_like_structured_prompt(optimized):
            if not self._looks_like_structured_prompt(pipeline_final):
                return optimized
            if self.token_estimator.exact_count(optimized) > self.token_estimator.exact_count(pipeline_final) * 1.4:
                return optimized
        return pipeline_final or optimized or state.raw_prompt

    def estimate_target_model_usage(
        self,
        final_prompt: str,
        state: ChatSessionState,
    ) -> dict[str, int]:
        """Estimate downstream model input, reasoning, and answer token usage."""

        prompt_tokens = self.token_estimator.exact_count(final_prompt)
        text = f"{state.raw_prompt} {' '.join(state.supplements.values())}".lower()
        complexity = 1.0
        if re.search(r"auth|login|database|api|backend", text):
            complexity += 0.35
        if re.search(r"test|responsive|mobile|animation|dashboard", text):
            complexity += 0.20
        if re.search(r"full[- ]?stack|multi-page|multi", text):
            complexity += 0.30

        reasoning_tokens = max(220, int(prompt_tokens * (1.2 + complexity * 0.45)))
        answer_tokens = max(650, int(prompt_tokens * (2.8 + complexity * 0.65)))
        return {
            "prompt_tokens": prompt_tokens,
            "reasoning_tokens": reasoning_tokens,
            "answer_tokens": answer_tokens,
            "total_tokens": prompt_tokens + reasoning_tokens + answer_tokens,
        }

    def advanced_details(self, state: ChatSessionState) -> dict[str, object]:
        """Return a compact user-readable quality and cost summary."""

        final_state = state.final_state
        if not final_state:
            return {}
        optimization: OptimizationResult | None = final_state.optimization_result
        evaluation: EvaluationResult | None = final_state.evaluation_result
        planning: PlanningResult | None = final_state.planning_result
        estimated_usage = (
            self.estimate_target_model_usage(self.display_prompt(state), state)
            if self.display_prompt(state)
            else {}
        )
        return {
            "project_summary": planning.detectedIntent if planning else state.raw_prompt,
            "collected_details": dict(state.supplements),
            "quality_score": evaluation.scores.overall if evaluation else None,
            "quality_breakdown": evaluation.scores.model_dump() if evaluation else None,
            "applied_improvements": optimization.appliedTechniques if optimization else [],
            "estimated_target_model_usage": estimated_usage,
        }

    async def _optimize(self, state: ChatSessionState) -> ChatSessionState:
        state.stage = "optimizing"
        state.messages.append(
            ChatMessage(role="assistant", content="Great. I am turning these details into the final optimized prompt now.")
        )
        final_state = await self.chatbot.process(
            state.raw_prompt,
            supplements=state.supplements,
        )
        state.final_state = final_state
        state.planning_result = final_state.planning_result or state.planning_result
        if final_state.status == "error":
            state.stage = "error"
            state.error_message = final_state.error_message
            state.messages.append(
                ChatMessage(
                    role="assistant",
                    content=f"Optimization failed: {final_state.error_message or 'unknown error'}",
                )
            )
            return state

        state.stage = "done"
        metrics = self.user_metrics(state)
        state.messages.append(
            ChatMessage(
                role="assistant",
                content=(
                    "Done. The final optimized prompt is ready.\n\n"
                    f"- Original prompt tokens: {metrics['original_prompt_tokens']}\n"
                    f"- Optimized prompt tokens: {metrics['optimized_prompt_tokens']}\n"
                    f"- Estimated target-model total: {metrics['target_total_tokens']}\n"
                    f"- Model score: {metrics['model_score']}"
                ),
            )
        )
        return state

    def _next_stage(self, state: ChatSessionState, latest_user_message: str) -> str:
        if state.stage == "confirming" and self._is_substantive_detail(latest_user_message):
            return "clarifying"
        if state.turn_count >= self.min_turns:
            return "confirming"
        if state.turn_count <= 1:
            return "collecting"
        return "clarifying"

    def _next_missing_field(self, state: ChatSessionState):
        if not state.planning_result:
            return None
        for field in state.planning_result.missingFields:
            if field.field not in state.supplements and field.importance != "optional":
                return field
        return None

    async def _reflective_question(self, state: ChatSessionState) -> str:
        """Ask the local planning model for one specific follow-up, with fallback."""

        fallback = self._contextual_question(state)
        prompt = self._build_question_prompt(state, fallback)
        try:
            response = await self.chatbot.planner.model_provider.generate_structured(
                StructuredParams(
                    prompt=prompt,
                    schema=_MentorQuestion.model_json_schema(),
                    max_tokens=220,
                ),
                _MentorQuestion,
            )
            parsed = response.parsed
            if not isinstance(parsed, _MentorQuestion):
                parsed = _MentorQuestion.model_validate(parsed)
            question = parsed.question.strip()
        except Exception:
            return fallback

        if not question or self._is_generic_or_repeated(question, state):
            return fallback
        return question

    def _build_question_prompt(self, state: ChatSessionState, fallback: str) -> str:
        """Build a short reflection prompt for targeted mentor questions."""

        history = "\n".join(
            f"{message.role}: {message.content}"
            for message in state.messages[-8:]
        )
        supplements = "\n".join(
            f"- {key}: {value}" for key, value in state.supplements.items()
        ) or "- None yet"
        return (
            "You are TokenLess, a patient prompt mentor for beginner programmers.\n"
            "Ask exactly ONE concise follow-up question in the same language the user mainly uses.\n"
            "The question must refer to the user's concrete project, not a generic checklist.\n"
            "Do not ask for confirmation unless the user clearly says they are ready.\n"
            "Do not repeat questions that already appear in the chat history.\n"
            "If the user just provided useful details, ask the next missing practical detail.\n"
            "Good topics include data source, UI behavior, edge cases, deployment/runtime, testing, or output format.\n\n"
            f"Raw project idea:\n{state.raw_prompt}\n\n"
            f"Known details:\n{supplements}\n\n"
            f"Recent chat:\n{history}\n\n"
            f"Fallback question if uncertain:\n{fallback}\n\n"
            'Return JSON: {"question": "..."}'
        )

    @staticmethod
    def _is_generic_or_repeated(question: str, state: ChatSessionState) -> bool:
        normalized = re.sub(r"\s+", "", question.lower())
        repeated_confirm = (
            "enough information" in question.lower()
            or "reply 'confirm'" in question.lower()
        )
        if repeated_confirm:
            return True
        previous_assistant = [
            re.sub(r"\s+", "", message.content.lower())
            for message in state.messages
            if message.role == "assistant"
        ]
        return any(normalized and normalized in previous for previous in previous_assistant)

    @staticmethod
    def _is_substantive_detail(message: str) -> bool:
        stripped = message.strip().lower()
        if not stripped:
            return False
        confirm_phrases = {"confirm", "yes", "y", "run", "optimize", "go"}
        if stripped in confirm_phrases:
            return False
        return len(stripped) >= 6

    @staticmethod
    def _is_confirmation(message: str) -> bool:
        """Accept only short, intentional confirmation replies."""

        stripped = message.strip().lower()
        normalized = re.sub(r"[\s..!！,,;;\"'""]+", "", stripped)
        exact = {
            "confirm",
            "yes",
            "y",
            "ok",
            "run",
            "optimize",
            "go",
            "start",
            "generate",
        }
        if normalized in exact:
            return True
        return bool(re.fullmatch(r"(confirm|start|generate|optimize)(now)?", normalized))

    @staticmethod
    def _looks_like_structured_prompt(prompt: str) -> bool:
        return bool(
            prompt
            and "## " in prompt
            and ("```json" in prompt or "Output Format" in prompt)
        )

    def _contextual_question(self, state: ChatSessionState) -> str:
        """Ask a follow-up that reacts to the user's actual project idea."""

        raw = state.raw_prompt.strip()
        message_history = " ".join(
            message.content for message in state.messages if message.role == "user"
        )
        context = f"{raw} {message_history} {' '.join(state.supplements.values())}".lower()
        project_label = self._project_label(raw)

        missing_field = self._next_missing_field(state)
        if missing_field and missing_field.field in {"language", "framework"}:
            default = self._default_stack_for(context)
            return (
                f"The tech stack for this {project_label} is still open. Do you want "
                f"to choose the language and framework yourself, or should I use "
                f"`{default}`? If you are unsure, reply 'choose for me'."
            )

        if not state.supplements.get("functionality"):
            return (
                f"For this {project_label}, what are the 2-3 actions the end user "
                "will do most often, such as create, search, filter, upload, export, "
                "sign in, or view charts?"
            )

        if not self._has_io_details(context):
            return (
                f"To make the prompt more executable, what are the inputs and outputs "
                f"for this {project_label}? In other words, what does the user provide, "
                "and what should the system display or generate?"
            )

        if self._looks_like_ui_project(context) and not self._has_ui_details(context):
            return (
                "What should the interface feel like: mobile-first, admin dashboard, "
                "simple form, card list, dark mode, or lightly animated?"
            )

        if not self._has_data_details(context):
            return (
                f"Does this {project_label} need real data, such as an API, local file, "
                "or database, or is mock data enough for the first version?"
            )

        if not self._has_testing_details(context):
            return (
                "Should the AI include tests so the final prompt reads like an "
                "engineering task? You can choose unit tests, manual acceptance steps, "
                "or no tests for now."
            )

        return (
            "Last delivery choice: should the AI return the full file structure and "
            "code, or first provide implementation steps and then generate files one by one?"
        )

    @staticmethod
    def _beginner_question_for(field, raw_prompt: str) -> str:
        questions = {
            "language": "Which programming language should this project use? If you are unsure, say 'choose for me'.",
            "framework": (
                "Which framework do you prefer? A web app could use React and a backend "
                "API could use FastAPI. I can recommend one if you are unsure."
            ),
            "functionality": "What are the 2-3 most important features? A rough answer is fine.",
            "constraints": "Any specific requirements, such as mobile support, login, saved data, speed, or visual polish?",
            "output_format": "Should the AI return only code, or code plus explanation, file structure, and tests?",
        }
        return questions.get(field.field, field.question)

    @staticmethod
    def _project_label(raw_prompt: str) -> str:
        text = raw_prompt.lower()
        if re.search(r"dashboard|admin", text):
            return "dashboard"
        if re.search(r"api|backend|server", text):
            return "backend API"
        if re.search(r"script|automation", text):
            return "automation script"
        if re.search(r"app|web|site", text):
            return "web app"
        if re.search(r"data|analytics|csv|excel", text):
            return "data tool"
        return "project"

    @staticmethod
    def _default_stack_for(context: str) -> str:
        if re.search(r"api|backend|server", context):
            return "Python + FastAPI"
        if re.search(r"script|automation|csv|excel", context):
            return "Python"
        return "React + TypeScript"

    @staticmethod
    def _looks_like_ui_project(context: str) -> bool:
        return bool(re.search(r"web|app|dashboard|ui|page|site|screen", context))

    @staticmethod
    def _has_io_details(context: str) -> bool:
        return bool(re.search(r"input|output|user.*enter|display|generate|return|upload", context))

    @staticmethod
    def _has_ui_details(context: str) -> bool:
        return bool(re.search(r"mobile|responsive|dashboard|dark|animation|card|form|simple", context))

    @staticmethod
    def _has_data_details(context: str) -> bool:
        return bool(re.search(r"api|database|db|mock|csv|excel|file|local", context))

    @staticmethod
    def _has_testing_details(context: str) -> bool:
        return bool(re.search(r"test|pytest|jest|unit|e2e|acceptance", context))


__all__ = ["TokenLessMentor"]
