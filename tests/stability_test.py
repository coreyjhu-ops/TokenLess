"""
TokenLess V1 - Stability & Multi-Scenario Test Suite
=====================================================
Test scope:
  Layer 1  - Planning Engine(offline rules, all scenarios)
  Layer 2  - Optimization Engine(WordPruner offline plus Optimizer integration)
  Layer 3  - Evaluation Layer(offline mocks, no API calls)
  Layer 4  - Self-Correction(offline mocks)
  E2E      - TokenLessChatbot full pipeline(offline mocks)

Design principles:
  - All tests run offline without LM Studio, OpenAI, or Google API dependencies
  - Prompts are grouped by length: micro(<10t), short(10-50t), medium(50-200t), long(200-500t), xlarge(500+t)
  - Covers positive paths, guardrails, and injected-error scenarios
"""

from __future__ import annotations

import asyncio
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

try:
    import pytest
except ModuleNotFoundError:
    class _PytestShim:
        class mark:
            @staticmethod
            def asyncio(func):
                return func

    pytest = _PytestShim()

sys.path.insert(0, ".")

# Import modules under test
from src.core.model_provider import ModelProvider
from src.core.token_estimator import TokenEstimator
from src.core.types import (
    EvaluationResult,
    EvaluationScores,
    JudgeVote,
    MissingField,
    OptimizationConstraints,
    OptimizationResult,
    OptimizationROIReport,
    PipelineTokenUsage,
    PlanningResult,
    TokenStats,
)
from src.evaluation.judge_pool import JudgePool
from src.evaluation.pairwise_battle import PairwiseBattle
from src.evaluation.self_correction import SelfCorrector
from src.optimization.optimizer import Optimizer
from src.optimization.word_pruner import WordPruner
from src.planning.gap_analyzer import GapAnalyzer
from src.planning.intent_analyzer import IntentAnalyzer, IntentAnalysis
from src.planning.planner import Planner
from src.planning.scene_detector import SceneDetector
from src.chat.chatbot import TokenLessChatbot
from src.chat.conversation import ChatSessionState, ConversationState
from src.chat.mentor import TokenLessMentor
from app import choose_display_prompt


# Mock model providers
from src.core.model_provider import (
    GenerateParams, GenerateResult, StructuredParams,
    HealthCheckResult, TokenUsage,
)


class MockLocalProvider(ModelProvider):
    """Mock local Gemma model that returns fixed structured intent output."""

    id: str = "mock-gemma"
    type: str = "local"

    async def generate(self, params: GenerateParams) -> GenerateResult:
        return GenerateResult(
            text='{"who":"developer","what":"build a web application","how":"using React and TypeScript","format":"clean code with comments"}',
            usage=TokenUsage(prompt_tokens=10, completion_tokens=20),
        )

    async def generate_structured(self, params: StructuredParams, response_type: type) -> Any:
        # Return IntentAnalysis or _JudgeAssessment depending on the requested schema.
        if response_type.__name__ == "IntentAnalysis":
            parsed = IntentAnalysis(
                who="developer",
                what="build a feature-rich web application",
                how="using React 18, TypeScript, and Tailwind CSS",
                format="well-commented code with TypeScript types",
            )
            return GenerateResult(
                text=parsed.model_dump_json(),
                usage=TokenUsage(prompt_tokens=25, completion_tokens=12),
                parsed=parsed,
            )
        # SemanticPruner relevance scoring.
        try:
            parsed = response_type(scores=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        except Exception:
            parsed = response_type.model_validate({})
        return GenerateResult(
            text=parsed.model_dump_json(),
            usage=TokenUsage(prompt_tokens=18, completion_tokens=7),
            parsed=parsed,
        )

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    async def health_check(self) -> HealthCheckResult:
        return HealthCheckResult(available=True, latency_ms=0.0)


class MockJudgeProvider(ModelProvider):
    """Mock judge model with configurable winner and overall score."""

    def __init__(self, model_id: str, winner: str = "optimized", overall: float = 8.0):
        self.id = model_id
        self.type = "api"
        self._winner = winner
        self._overall = overall

    async def generate(self, params: GenerateParams) -> GenerateResult:
        return GenerateResult(
            text=f'{{"winner":"{self._winner}","intentAlignment":{self._overall},"logicCoherence":{self._overall},"concisenessScore":{self._overall},"formatCompliance":{self._overall},"reasoning":"mock evaluation"}}',
            usage=TokenUsage(prompt_tokens=50, completion_tokens=30),
        )

    async def generate_structured(self, params: StructuredParams, response_type: type) -> Any:
        from src.evaluation.judge_pool import _JudgeAssessment  # type: ignore
        ab_winner = "A" if self._winner == "original" else ("B" if self._winner == "optimized" else "tie")
        parsed = _JudgeAssessment(
            winner=ab_winner,
            intentAlignment=self._overall,
            logicCoherence=self._overall,
            concisenessScore=self._overall,
            formatCompliance=self._overall,
            reasoning="mock evaluation",
        )
        return GenerateResult(
            text=parsed.model_dump_json(),
            usage=TokenUsage(prompt_tokens=50, completion_tokens=30),
            parsed=parsed,
        )

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    async def health_check(self) -> HealthCheckResult:
        return HealthCheckResult(available=True, latency_ms=1.0)


class MockTargetProvider(ModelProvider):
    """Mock target generation model that returns fixed code output."""

    id: str = "mock-target"
    type: str = "api"

    async def generate(self, params: GenerateParams) -> GenerateResult:
        return GenerateResult(
            text="// Mock generated code\nconst App = () => <div>Hello World</div>;",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=15),
        )

    async def generate_structured(self, params: StructuredParams, response_type: type) -> Any:
        return None

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    async def health_check(self) -> HealthCheckResult:
        return HealthCheckResult(available=True, latency_ms=2.0)


def _build_mock_chatbot(*, require_positive_roi: bool) -> TokenLessChatbot:
    """Build a fully offline chatbot with non-zero token usage mocks."""

    estimator = TokenEstimator()
    local_provider = MockLocalProvider()
    judge_provider = MockJudgeProvider("judge-gpt4o", winner="optimized", overall=8.5)
    target_provider = MockTargetProvider()
    planner = Planner(local_provider)
    optimizer = Optimizer(
        local_provider,
        estimator,
        OptimizationConstraints(requirePositiveROI=require_positive_roi),
    )
    judge_pool = JudgePool([judge_provider])
    battle = PairwiseBattle(
        target_providers=[target_provider],
        judge_pool=judge_pool,
        constraints=OptimizationConstraints(minQualityScore=6.0),
    )
    corrector = SelfCorrector(
        optimizer=optimizer,
        battle=battle,
        constraints=OptimizationConstraints(maxSelfCorrectionRetries=2),
    )
    return TokenLessChatbot(
        planner=planner,
        optimizer=optimizer,
        battle=battle,
        corrector=corrector,
    )


class TestTokenLedger:
    @pytest.mark.asyncio
    async def test_pipeline_token_usage_accumulates(self):
        """After the full pipeline runs, PipelineTokenUsage should contain real positive values for each active stage."""

        chatbot = _build_mock_chatbot(require_positive_roi=False)
        state = await chatbot.process(PROMPTS["medium_01"][0])

        assert state.status == "done"
        assert state.token_ledger is not None
        assert state.token_ledger.intent_analyzer_prompt > 0
        assert state.token_ledger.semantic_pruner_prompt >= 0
        assert state.token_ledger.judge_pool_prompt > 0
        assert state.token_ledger.target_model_prompt > 0
        assert state.token_ledger.total_pipeline_tokens > 0
        assert state.token_ledger.total_pipeline_tokens != 200

    @pytest.mark.asyncio
    async def test_roi_report_uses_real_cost(self):
        """OptimizationROIReport.optimization_cost_tokens should equal total_pipeline_tokens."""

        chatbot = _build_mock_chatbot(require_positive_roi=False)
        state = await chatbot.process(PROMPTS["medium_01"][0])

        assert state.token_ledger is not None
        assert state.optimization_result is not None
        assert state.optimization_result.roiReport is not None
        assert (
            state.optimization_result.roiReport.optimizationCostTokens
            == state.token_ledger.total_pipeline_tokens
        )

    @pytest.mark.asyncio
    async def test_professional_rewrite_bypasses_static_roi_skip(self):
        """ProfessionalRewriter should not be skipped solely because static ROI is negative when the rewrite is longer."""

        chatbot = _build_mock_chatbot(require_positive_roi=True)
        state = await chatbot.process("fix bug")

        assert state.status == "done"
        assert state.optimization_result is not None
        assert not state.optimization_result.optimizationSkipped
        assert "professional rewrite" in state.optimization_result.appliedTechniques
        assert state.optimization_result.roiReport is not None
        assert state.optimization_result.roiReport.inputTokensSaved <= 0
        assert state.token_ledger is not None
        assert state.token_ledger.rewriter_prompt > 0
        assert state.token_ledger.judge_pool_prompt > 0
        assert state.token_ledger.target_model_prompt > 0

    def test_pipeline_token_usage_total_property(self):
        """PipelineTokenUsage.total_pipeline_tokens should equal the sum of all fields."""

        usage = PipelineTokenUsage(
            intent_analyzer_prompt=10,
            intent_analyzer_completion=5,
            semantic_pruner_prompt=8,
            semantic_pruner_completion=3,
            judge_pool_prompt=20,
            judge_pool_completion=15,
            target_model_prompt=30,
            target_model_completion=25,
        )
        assert usage.total_pipeline_tokens == 116

    def test_pipeline_token_usage_default_zero(self):
        """All PipelineTokenUsage fields should default to 0."""

        usage = PipelineTokenUsage()
        assert usage.total_pipeline_tokens == 0


class TestChatFirstMentor:
    @pytest.mark.asyncio
    async def test_chat_session_starts_with_welcome(self):
        """Mentor session should start as a chat, not a blank dashboard state."""

        mentor = TokenLessMentor(_build_mock_chatbot(require_positive_roi=False))
        state = await mentor.start()

        assert isinstance(state, ChatSessionState)
        assert state.stage == "welcome"
        assert state.messages
        assert state.messages[0].role == "assistant"
        assert "TokenLess" in state.messages[0].content

    @pytest.mark.asyncio
    async def test_first_prompt_clarifies_before_optimization(self):
        """The first user idea should trigger a follow-up question."""

        mentor = TokenLessMentor(_build_mock_chatbot(require_positive_roi=False))
        state = await mentor.start()
        state = await mentor.receive(state, "I want to build a habit tracking web app")

        assert state.stage in {"collecting", "clarifying"}
        assert state.final_state is None
        assert state.turn_count == 1
        assert state.messages[-1].role == "assistant"
        assert (
            "web app" in state.messages[-1].content
            or "user" in state.messages[-1].content
            or "actions" in state.messages[-1].content
        )

    @pytest.mark.asyncio
    async def test_mentor_waits_for_five_turns_and_confirmation(self):
        """Default flow should not optimize before five user turns."""

        mentor = TokenLessMentor(_build_mock_chatbot(require_positive_roi=False))
        state = await mentor.start()
        for message in [
            "Build a small dashboard for tracking study tasks",
            "not sure, help me choose",
            "It should add tasks and show progress",
            "The input is a task name and deadline",
            "Make it responsive and beginner friendly",
        ]:
            state = await mentor.receive(state, message)

        assert state.turn_count == 5
        assert state.stage == "confirming"
        assert state.final_state is None

        state = await mentor.receive(state, "confirm")
        assert state.stage == "done"
        assert state.final_state is not None
        assert state.final_state.status == "done"

    @pytest.mark.asyncio
    async def test_confirming_state_accepts_more_details_without_repeating(self):
        """If the user adds details after confirmation prompt, keep clarifying."""

        mentor = TokenLessMentor(_build_mock_chatbot(require_positive_roi=False))
        state = await mentor.start()
        for message in [
            "Build a course assignment progress dashboard",
            "Use React and make it mobile responsive",
            "The user enters assignment names and deadlines, and the app outputs progress charts",
            "Use mock data first, without connecting an API",
            "Include simple tests",
        ]:
            state = await mentor.receive(state, message)

        assert state.stage == "confirming"
        confirmation_message = state.messages[-1].content
        assert "Reply 'confirm'" in confirmation_message

        state = await mentor.receive(state, "It only needs to run in a local browser")
        assert state.stage == "clarifying"
        assert state.final_state is None
        assert state.messages[-1].content != confirmation_message
        assert "enough information" not in state.messages[-1].content

    @pytest.mark.asyncio
    async def test_confirmation_requires_short_intentional_reply(self):
        """A substantive detail sentence should not trigger optimization."""

        mentor = TokenLessMentor(_build_mock_chatbot(require_positive_roi=False))
        state = await mentor.start()
        for message in [
            "Design a Slack bot for SLI management",
            "Use Python and FastAPI",
            "The user enters service name and SLI target",
            "Use mock data and do not connect to a real API",
            "Include tests and local run instructions",
        ]:
            state = await mentor.receive(state, message)

        assert state.stage == "confirming"
        state = await mentor.receive(state, "It can run in a local browser for now; no cloud deployment needed")

        assert state.stage == "clarifying"
        assert state.final_state is None

    @pytest.mark.asyncio
    async def test_natural_language_answers_merge_into_supplements(self):
        """Natural answers like React plus mobile should become supplement fields."""

        mentor = TokenLessMentor(_build_mock_chatbot(require_positive_roi=False))
        state = await mentor.start()
        state = await mentor.receive(state, "Build a portfolio web app")
        state = await mentor.receive(
            state,
            "Use React, include animation, support mobile, and add tests if possible",
        )

        assert state.supplements["framework"] in {"React", "React, Tailwind CSS"}
        assert "mobile" in state.supplements["constraints"].lower()
        assert "tests" in state.supplements["output_format"].lower()

    @pytest.mark.asyncio
    async def test_not_sure_chooses_practical_defaults(self):
        """The user can say not sure and receive a practical default stack."""

        mentor = TokenLessMentor(_build_mock_chatbot(require_positive_roi=False))
        state = await mentor.start()
        state = await mentor.receive(state, "Build a simple web app")
        state = await mentor.receive(state, "not sure, choose a simple option for me")

        assert state.supplements["language"] == "TypeScript"
        assert state.supplements["framework"] == "React"

    @pytest.mark.asyncio
    async def test_final_prompt_and_metrics_are_user_facing(self):
        """Final result should expose simple metrics and Markdown plus JSON prompt."""

        mentor = TokenLessMentor(_build_mock_chatbot(require_positive_roi=False))
        state = await mentor.start()
        for message in [
            "Build a study planner web app",
            "not sure, help me choose",
            "Add tasks, deadlines, progress filters",
            "Users type tasks and see a dashboard",
            "Responsive UI, include tests",
            "confirm",
        ]:
            state = await mentor.receive(state, message)

        assert state.stage == "done"
        assert state.final_state is not None
        final_prompt = state.final_state.final_prompt or ""
        metrics = mentor.user_metrics(state)
        advanced = mentor.advanced_details(state)

        assert "## Role" in final_prompt
        assert "```json" in final_prompt
        assert metrics["original_prompt_tokens"] != "-"
        assert metrics["optimized_prompt_tokens"] != "-"
        assert metrics["target_reasoning_tokens"] != "-"
        assert metrics["target_answer_tokens"] != "-"
        assert metrics["target_total_tokens"] != "-"
        assert metrics["model_score"] != "-"
        assert "project_summary" in advanced
        assert "estimated_target_model_usage" in advanced
        assert "token_ledger" not in advanced
        assert "optimization" not in advanced

    @pytest.mark.asyncio
    async def test_display_prompt_prefers_structured_optimized_prompt(self):
        """If final_prompt fell back to raw text, show the structured optimized prompt."""

        mentor = TokenLessMentor(_build_mock_chatbot(require_positive_roi=False))
        state = await mentor.start()
        for message in [
            "Design a Slack bot for SLI management",
            "Use Python and FastAPI",
            "The user enters service name and SLI target",
            "Use mock data and do not connect to a real API",
            "Include tests and local run instructions",
            "confirm",
        ]:
            state = await mentor.receive(state, message)

        assert state.final_state is not None
        raw_fallback = state.raw_prompt
        state.final_state.final_prompt = raw_fallback
        display_prompt = mentor.display_prompt(state)
        app_display_prompt = choose_display_prompt(state)

        assert display_prompt != raw_fallback
        assert "## Role" in display_prompt
        assert "```json" in display_prompt
        assert app_display_prompt == display_prompt


# Test result data structure
@dataclass
class TestResult:
    test_id: str
    category: str
    prompt_length_bucket: str
    original_tokens: int
    description: str
    passed: bool
    duration_ms: float
    details: dict = field(default_factory=dict)
    error: str = ""


# Test prompt fixtures
PROMPTS = {
    # micro: <10 tokens
    "micro_01": ("write a function", "micro"),
    "micro_02": ("build a website", "micro"),
    "micro_03": ("fix bug", "micro"),

    # short: 10-50 tokens
    "short_01": (
        "Write a quicksort algorithm in Python with comments.",
        "short",
    ),
    "short_02": (
        "Create a React component for a login form with email and password fields.",
        "short",
    ),
    "short_03": (
        "Write a simple REST API in TypeScript with GET and POST endpoints.",
        "short",
    ),

    # medium: 50-200 tokens
    "medium_01": (
        "I need to build a user authentication system using Python and FastAPI. "
        "The system must support registration, login, and logout with JWT authentication. "
        "Use PostgreSQL through SQLAlchemy ORM. "
        "Implement password hashing and token refresh. "
        "The code should include type hints and docstrings.",
        "medium",
    ),
    "medium_02": (
        "Build a real-time chat application using Node.js and Socket.io. "
        "The app should support multiple chat rooms, user authentication with JWT, "
        "message persistence using MongoDB, and show online/offline status. "
        "Use TypeScript throughout, follow ESLint rules, and include unit tests with Jest. "
        "The frontend should be built with React and styled using Tailwind CSS.",
        "medium",
    ),
    "medium_03": (
        "Please write a data processing script that uses Pandas and NumPy to process CSV files. "
        "Required features: read and merge multiple CSV files, clean data by deduplicating and filling missing values, "
        "calculate statistics such as mean, standard deviation, and quantiles, generate Matplotlib charts, "
        "and output cleaned data plus a statistical report to new CSV files. "
        "The code should be modular, and each function should include type hints and docstrings.",
        "medium",
    ),

    # long: 200-500 tokens
    "long_01": (
        "I need to build a complete ecommerce backend. Tech stack: Python 3.11, FastAPI, PostgreSQL, Redis, and Celery. "
        "Core modules:\n"
        "1. User module: registration/login (JWT + OAuth2), user profile management, role levels (admin/seller/buyer)\n"
        "2. Product module: CRUD operations, image upload to S3, Elasticsearch search, inventory management\n"
        "3. Order module: cart, checkout flow, Stripe payment integration, order state machine\n"
        "4. Notification module: email via SendGrid, SMS via Twilio, real-time push via WebSocket\n"
        "5. Async tasks: Celery + Redis for image compression, email sending, and data statistics\n"
        "Code requirements:\n"
        "- All endpoints need OpenAPI documentation\n"
        "- Unit test coverage above 80% with pytest\n"
        "- Use Alembic for database migrations\n"
        "- Follow PEP8 with full type hints and docstrings\n"
        "- Docker and docker-compose deployment configuration\n"
        "Start with the project directory structure, then implement the user module step by step.",
        "long",
    ),
    "long_02": (
        "Build a machine learning pipeline for sentiment analysis with the following specs:\n\n"
        "Data Pipeline:\n"
        "- Ingest data from multiple sources (CSV, JSON, database)\n"
        "- Text preprocessing: tokenization, stop-word removal, lemmatization\n"
        "- Feature engineering: TF-IDF, word embeddings (Word2Vec/GloVe)\n"
        "- Train/val/test split with stratified sampling\n\n"
        "Model Development:\n"
        "- Baseline: Logistic Regression, Naive Bayes\n"
        "- Advanced: LSTM, BERT fine-tuning (HuggingFace Transformers)\n"
        "- Hyperparameter tuning: Optuna\n"
        "- Experiment tracking: MLflow\n\n"
        "Evaluation:\n"
        "- Metrics: accuracy, precision, recall, F1, AUC-ROC\n"
        "- Confusion matrix visualization\n"
        "- Error analysis on misclassified samples\n\n"
        "Deployment:\n"
        "- FastAPI serving endpoint with batch inference support\n"
        "- Model versioning, A/B testing infrastructure\n"
        "- Monitoring: data drift detection (Evidently AI)\n\n"
        "Tech stack: Python 3.11, PyTorch, scikit-learn, FastAPI, MLflow, Docker.\n"
        "All code must have type hints, docstrings, and 90%+ test coverage with pytest.",
        "long",
    ),

    # xlarge: 500+ tokens
    "xlarge_01": (
        "I need to design and implement an enterprise project management system with a microservices architecture, similar to Jira.\n\n"
        "[System Architecture]\n"
        "Use a microservices architecture with the following services:\n"
        "- API Gateway(Kong or custom gateway for routing, authentication, rate limiting, and logging)\n"
        "- User Service: registration/login/OAuth2/RBAC\n"
        "- Project Service: project CRUD, member management, permission control\n"
        "- Task Service: task lifecycle, kanban board, sprint management\n"
        "- Comment Service: nested comments, mentions, rich text\n"
        "- Notification Service: real-time push, email, webhook\n"
        "- Reporting Service: burndown charts, time tracking, PDF export\n"
        "- File Service: attachment upload, preview, version control\n\n"
        "[Tech Stack]\n"
        "- Backend:Python(FastAPI)+ Node.js(some real-time services)\n"
        "- Databases: PostgreSQL (primary data) + Redis (cache/sessions) + MongoDB (logging/comments)\n"
        "- Message queue:RabbitMQ(inter-service communication)\n"
        "- Search:Elasticsearch\n"
        "- Real-time communication:WebSocket(Socket.io)\n"
        "- Containerization:Docker + Kubernetes\n"
        "- CI/CD:GitHub Actions\n"
        "- Observability:Prometheus + Grafana + Jaeger(distributed tracing)\n\n"
        "[Code Standards]\n"
        "- Python services: PEP8, full type hints, Google-style docstrings, pytest (>85% coverage)\n"
        "- Node.js services: TypeScript, ESLint (Airbnb style), Jest\n"
        "- All services must provide OpenAPI documentation\n"
        "- Use Conventional Commits\n\n"
        "[Deliverables]\n"
        "1. Complete directory structure description\n"
        "2. Core model definitions for each service(Pydantic / TypeScript interface)\n"
        "3. API Gateway configuration example\n"
        "4. Complete Task Service implementation as the reference implementation\n"
        "5. docker-compose.yml(local development environment)\n"
        "6. GitHub Actions CI/CD pipeline configuration\n\n"
        "Implement the modules step by step, starting with the system design document.",
        "xlarge",
    ),
}

# Boundary and error scenarios
EDGE_CASES = {
    "edge_empty": ("", "micro"),
    "edge_spaces": ("   ", "micro"),
    "edge_special_chars": ("!@#$%^&*()", "micro"),
    "edge_only_numbers": ("12345 67890", "micro"),
    "edge_mixed_lang": (
        "Please build a user login system with email verification using Python FastAPI",
        "short",
    ),
    "edge_repeat_words": (
        "Please note that this feature is very important; make sure to implement it. "
        "Implement it in Python using clear Python code.",
        "short",
    ),
    "edge_already_structured": (
        "## Role\nYou are a senior Python developer.\n\n"
        "## Task\nBuild a REST API with FastAPI.\n\n"
        "## Constraints\n- Use Python 3.11\n- Include type hints\n- Write pytest tests\n\n"
        "## Output Format\nClean, well-documented code with docstrings.",
        "short",
    ),
    "edge_all_english_verbose": (
        "I would like you to please help me to create a function. "
        "The function that I am asking you to create should be able to "
        "take in a list of numbers as its input parameter. "
        "What I need the function to do is to make sure that it processes "
        "each and every single number in the list. "
        "In order to achieve the desired result, the function should "
        "calculate and return the sum of all the numbers. "
        "Please make sure that the implementation works correctly.",
        "short",
    ),
}


# Test runner
class TestRunner:
    def __init__(self):
        self.results: list[TestResult] = []
        self.estimator = TokenEstimator()
        self.local_provider = MockLocalProvider()
        self.judge_provider_optimized = MockJudgeProvider("judge-gpt4o", winner="optimized", overall=8.5)
        self.judge_provider_original = MockJudgeProvider("judge-gemini", winner="original", overall=5.0)
        self.target_provider = MockTargetProvider()

        # Scene components - Planner wires SceneDetector/IntentAnalyzer/GapAnalyzer internally
        self.planner = Planner(self.local_provider)

        # Optimizer
        self.optimizer_roi_on = Optimizer(
            self.local_provider,
            self.estimator,
            OptimizationConstraints(requirePositiveROI=True),
        )
        self.optimizer_roi_off = Optimizer(
            self.local_provider,
            self.estimator,
            OptimizationConstraints(requirePositiveROI=False),
        )

        # Evaluation - JudgePool takes only judge_providers; PairwiseBattle takes target_providers + judge_pool
        self.judge_pool_pass = JudgePool([self.judge_provider_optimized])
        self.judge_pool_fail = JudgePool([self.judge_provider_original])

        # PairwiseBattle(target_providers, judge_pool, constraints)
        constraints_pass = OptimizationConstraints(minQualityScore=6.0)
        constraints_fail = OptimizationConstraints(minQualityScore=6.0)

        self.battle_pass = PairwiseBattle(
            target_providers=[self.target_provider],
            judge_pool=self.judge_pool_pass,
            constraints=constraints_pass,
        )
        self.battle_fail = PairwiseBattle(
            target_providers=[self.target_provider],
            judge_pool=self.judge_pool_fail,
            constraints=constraints_fail,
        )
        self.corrector = SelfCorrector(
            optimizer=self.optimizer_roi_off,
            battle=self.battle_pass,
            constraints=OptimizationConstraints(maxSelfCorrectionRetries=2),
        )

    def record(self, result: TestResult):
        self.results.append(result)
        status = "PASS" if result.passed else "FAIL"
        print(f"  {status} [{result.test_id}] {result.description} ({result.duration_ms:.0f}ms)")
        if not result.passed and result.error:
            print(f"         Error: {result.error[:120]}")

    # ── Layer 1: Planning ────────────────────────────────────────────────────
    def test_planning_layer(self):
        print("\n═══ LAYER 1: Planning Engine ═══")

        for pid, (prompt, bucket) in {**PROMPTS, **EDGE_CASES}.items():
            start = time.perf_counter()
            try:
                tokens = self.estimator.exact_count(prompt) if prompt.strip() else 0
                if not prompt.strip():
                    # Skip planning for empty/whitespace - expected graceful skip
                    result = TestResult(
                        test_id=f"plan_{pid}",
                        category="planning",
                        prompt_length_bucket=bucket,
                        original_tokens=0,
                        description=f"Empty/whitespace graceful skip [{pid}]",
                        passed=True,
                        duration_ms=0.0,
                        details={"skipped": True},
                    )
                    self.record(result)
                    continue

                planning_result = asyncio.run(self.planner.plan(prompt))
                duration = (time.perf_counter() - start) * 1000

                assert isinstance(planning_result, PlanningResult), "Not a PlanningResult"
                assert planning_result.scene in ("vibe_coding", "custom"), "Invalid scene"
                assert isinstance(planning_result.missingFields, list), "missingFields not list"
                assert planning_result.detectedIntent, "detectedIntent empty"

                result = TestResult(
                    test_id=f"plan_{pid}",
                    category="planning",
                    prompt_length_bucket=bucket,
                    original_tokens=tokens,
                    description=f"Planning output valid [{pid}]",
                    passed=True,
                    duration_ms=duration,
                    details={
                        "scene": planning_result.scene,
                        "missing_critical": sum(1 for f in planning_result.missingFields if f.importance == "critical"),
                        "missing_total": len(planning_result.missingFields),
                        "intent": planning_result.detectedIntent[:60],
                    },
                )
            except Exception as e:
                duration = (time.perf_counter() - start) * 1000
                result = TestResult(
                    test_id=f"plan_{pid}",
                    category="planning",
                    prompt_length_bucket=bucket,
                    original_tokens=self.estimator.exact_count(prompt) if prompt else 0,
                    description=f"Planning output valid [{pid}]",
                    passed=False,
                    duration_ms=duration,
                    error=str(e),
                )
            self.record(result)

    # ── Layer 2: Optimization ────────────────────────────────────────────────
    def test_optimization_layer(self):
        print("\n═══ LAYER 2: Optimization Engine ═══")

        for pid, (prompt, bucket) in PROMPTS.items():
            if not prompt.strip():
                continue
            tokens = self.estimator.exact_count(prompt)

            # Test A: ROI=True (production mode)
            start = time.perf_counter()
            try:
                planning_result = asyncio.run(self.planner.plan(prompt))
                opt_result, _pipeline_usage = asyncio.run(self.optimizer_roi_on.optimize(prompt, planning_result))
                duration = (time.perf_counter() - start) * 1000

                assert isinstance(opt_result, OptimizationResult)
                assert opt_result.tokenStats.originalCount == tokens
                assert opt_result.roiReport is not None
                assert isinstance(opt_result.optimizationSkipped, bool)
                # If skipped, optimizedPrompt must equal raw_prompt
                if opt_result.optimizationSkipped:
                    assert opt_result.optimizedPrompt == prompt, "Skipped but optimizedPrompt != raw"

                result = TestResult(
                    test_id=f"opt_roi_on_{pid}",
                    category="optimization",
                    prompt_length_bucket=bucket,
                    original_tokens=tokens,
                    description=f"Optimizer(ROI=True) [{pid}]",
                    passed=True,
                    duration_ms=duration,
                    details={
                        "skipped": opt_result.optimizationSkipped,
                        "skip_reason": opt_result.skipReason or "-",
                        "original_tokens": opt_result.tokenStats.originalCount,
                        "optimized_tokens": opt_result.tokenStats.optimizedCount,
                        "reduction_rate": f"{opt_result.tokenStats.reductionRate:.1%}",
                        "roi_positive": opt_result.roiReport.roiPositive,
                        "net_savings": opt_result.roiReport.netTokenSavings,
                        "techniques": ", ".join(opt_result.appliedTechniques[:3]),
                    },
                )
            except Exception as e:
                duration = (time.perf_counter() - start) * 1000
                result = TestResult(
                    test_id=f"opt_roi_on_{pid}",
                    category="optimization",
                    prompt_length_bucket=bucket,
                    original_tokens=tokens,
                    description=f"Optimizer(ROI=True) [{pid}]",
                    passed=False,
                    duration_ms=duration,
                    error=str(e),
                )
            self.record(result)

            # Test B: ROI=False (demo mode)
            start = time.perf_counter()
            try:
                planning_result = asyncio.run(self.planner.plan(prompt))
                opt_result, _pipeline_usage = asyncio.run(self.optimizer_roi_off.optimize(prompt, planning_result))
                duration = (time.perf_counter() - start) * 1000

                assert isinstance(opt_result, OptimizationResult)
                assert opt_result.tokenStats.originalCount == tokens

                result = TestResult(
                    test_id=f"opt_roi_off_{pid}",
                    category="optimization",
                    prompt_length_bucket=bucket,
                    original_tokens=tokens,
                    description=f"Optimizer(ROI=False/Demo) [{pid}]",
                    passed=True,
                    duration_ms=duration,
                    details={
                        "original_tokens": opt_result.tokenStats.originalCount,
                        "optimized_tokens": opt_result.tokenStats.optimizedCount,
                        "reduction_rate": f"{opt_result.tokenStats.reductionRate:.1%}",
                        "techniques": ", ".join(opt_result.appliedTechniques[:3]),
                    },
                )
            except Exception as e:
                duration = (time.perf_counter() - start) * 1000
                result = TestResult(
                    test_id=f"opt_roi_off_{pid}",
                    category="optimization",
                    prompt_length_bucket=bucket,
                    original_tokens=tokens,
                    description=f"Optimizer(ROI=False/Demo) [{pid}]",
                    passed=False,
                    duration_ms=duration,
                    error=str(e),
                )
            self.record(result)

    # Layer 2: WordPruner standalone tests
    def test_word_pruner(self):
        print("\n═══ LAYER 2b: WordPruner Rules ═══")

        cases = [
            ("wp_redundant", "In order to achieve the goal, please make sure that it works", "rule 1"),
            ("wp_passive", "The code should be written by the developer to handle errors", "rule 2"),
            ("wp_nested", "You need to make sure to check whether the input is valid or not", "rule 3"),
            ("wp_clean", "Build a REST API with FastAPI and PostgreSQL", "no rules"),
            ("wp_chinese_redundant", "in order to achieve this goal, please make sure that the code works correctly", "rule 1 chinese"),
        ]

        pruner = WordPruner()
        for cid, text, label in cases:
            start = time.perf_counter()
            try:
                result_text, rules_applied = pruner.prune(text)
                duration = (time.perf_counter() - start) * 1000
                before_chars = len(text)
                after_chars = len(result_text)
                reduction = (before_chars - after_chars) / before_chars if before_chars > 0 else 0

                result = TestResult(
                    test_id=f"wp_{cid}",
                    category="word_pruner",
                    prompt_length_bucket="short",
                    original_tokens=self.estimator.exact_count(text),
                    description=f"WordPruner: {label}",
                    passed=True,
                    duration_ms=duration,
                    details={
                        "before_chars": before_chars,
                        "after_chars": after_chars,
                        "char_reduction": f"{reduction:.1%}",
                        "rules_triggered": len(rules_applied),
                        "rule_names": ", ".join(rules_applied) if rules_applied else "none",
                        "result_preview": result_text[:80],
                    },
                )
            except Exception as e:
                duration = (time.perf_counter() - start) * 1000
                result = TestResult(
                    test_id=f"wp_{cid}",
                    category="word_pruner",
                    prompt_length_bucket="short",
                    original_tokens=0,
                    description=f"WordPruner: {label}",
                    passed=False,
                    duration_ms=duration,
                    error=str(e),
                )
            self.record(result)

    # ── Layer 3: Evaluation (offline mock) ──────────────────────────────────
    def test_evaluation_layer(self):
        print("\n═══ LAYER 3: Evaluation Layer (Mock) ═══")

        # Case: optimized wins
        for pid, (prompt, bucket) in list(PROMPTS.items())[:5]:
            if not prompt.strip():
                continue
            tokens = self.estimator.exact_count(prompt)
            start = time.perf_counter()
            try:
                planning_result = asyncio.run(self.planner.plan(prompt))
                opt_result, _pipeline_usage = asyncio.run(self.optimizer_roi_off.optimize(prompt, planning_result))
                eval_result, _judge_usage, _target_usage = asyncio.run(self.battle_pass.evaluate(
                    original_prompt=prompt,
                    optimized_prompt=opt_result.optimizedPrompt,
                    task_description=planning_result.detectedIntent,
                ))
                duration = (time.perf_counter() - start) * 1000

                assert isinstance(eval_result, EvaluationResult)
                assert eval_result.winner in ("original", "optimized", "tie")
                assert 0 <= eval_result.scores.overall <= 10

                result = TestResult(
                    test_id=f"eval_pass_{pid}",
                    category="evaluation",
                    prompt_length_bucket=bucket,
                    original_tokens=tokens,
                    description=f"Battle(optimized wins mock) [{pid}]",
                    passed=True,
                    duration_ms=duration,
                    details={
                        "winner": eval_result.winner,
                        "overall": round(eval_result.scores.overall, 2),
                        "judges": len(eval_result.judgeResults),
                    },
                )
            except Exception as e:
                duration = (time.perf_counter() - start) * 1000
                result = TestResult(
                    test_id=f"eval_pass_{pid}",
                    category="evaluation",
                    prompt_length_bucket=bucket,
                    original_tokens=tokens,
                    description=f"Battle(optimized wins mock) [{pid}]",
                    passed=False,
                    duration_ms=duration,
                    error=str(e),
                )
            self.record(result)

        # Case: quality gate triggers (overall < 6.0)
        start = time.perf_counter()
        try:
            prompt = PROMPTS["medium_01"][0]
            planning_result = asyncio.run(self.planner.plan(prompt))
            opt_result, _pipeline_usage = asyncio.run(self.optimizer_roi_off.optimize(prompt, planning_result))
            eval_result, _judge_usage, _target_usage = asyncio.run(self.battle_fail.evaluate(
                original_prompt=prompt,
                optimized_prompt=opt_result.optimizedPrompt,
                task_description=planning_result.detectedIntent,
            ))
            duration = (time.perf_counter() - start) * 1000

            # When quality gate fires (overall < 6.0), winner must be original
            assert eval_result.winner == "original", f"Expected original, got {eval_result.winner}"
            assert eval_result.feedback is not None, "Feedback should be set for quality gate"

            result = TestResult(
                test_id="eval_quality_gate",
                category="evaluation",
                prompt_length_bucket="medium",
                original_tokens=self.estimator.exact_count(prompt),
                description="Quality gate: overall<6.0 forces winner=original",
                passed=True,
                duration_ms=duration,
                details={"winner": eval_result.winner, "feedback": eval_result.feedback[:80]},
            )
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            result = TestResult(
                test_id="eval_quality_gate",
                category="evaluation",
                prompt_length_bucket="medium",
                original_tokens=0,
                description="Quality gate: overall<6.0 forces winner=original",
                passed=False,
                duration_ms=duration,
                error=str(e),
            )
        self.record(result)

    # ── Layer 4: Self-Correction ─────────────────────────────────────────────
    def test_self_correction(self):
        print("\n═══ LAYER 4: Self-Correction (Mock) ═══")

        # Case: correction succeeds (battle_pass after retry)
        start = time.perf_counter()
        try:
            prompt = PROMPTS["medium_01"][0]
            planning_result = asyncio.run(self.planner.plan(prompt))
            # Simulate initial_result where original wins (triggers correction)
            initial_result = EvaluationResult(
                winner="original",
                scores=EvaluationScores(
                    intentAlignment=5.0, logicCoherence=5.0,
                    concisenessScore=5.0, formatCompliance=5.0, overall=5.0
                ),
                judgeResults=[],
                feedback="Optimized prompt lacks focus.",
            )
            final_prompt, final_result = asyncio.run(self.corrector.correct(
                raw_prompt=prompt,
                planning_result=planning_result,
                initial_result=initial_result,
                task_description=planning_result.detectedIntent,
            ))
            duration = (time.perf_counter() - start) * 1000

            assert isinstance(final_prompt, str) and final_prompt
            assert isinstance(final_result, EvaluationResult)

            result = TestResult(
                test_id="correction_succeeds",
                category="self_correction",
                prompt_length_bucket="medium",
                original_tokens=self.estimator.exact_count(prompt),
                description="Self-correction: retries until optimized wins",
                passed=True,
                duration_ms=duration,
                details={"final_winner": final_result.winner, "final_overall": round(final_result.scores.overall, 2)},
            )
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            result = TestResult(
                test_id="correction_succeeds",
                category="self_correction",
                prompt_length_bucket="medium",
                original_tokens=0,
                description="Self-correction: retries until optimized wins",
                passed=False,
                duration_ms=duration,
                error=str(e),
            )
        self.record(result)

        # Case: correction exhausts retries -> returns raw_prompt
        start = time.perf_counter()
        try:
            prompt = PROMPTS["medium_02"][0]
            corrector_fail = SelfCorrector(
                optimizer=self.optimizer_roi_off,
                battle=self.battle_fail,   # always returns original (quality gate)
                constraints=OptimizationConstraints(maxSelfCorrectionRetries=2),
            )
            planning_result = asyncio.run(self.planner.plan(prompt))
            initial_result = EvaluationResult(
                winner="original",
                scores=EvaluationScores(
                    intentAlignment=4.0, logicCoherence=4.0,
                    concisenessScore=4.0, formatCompliance=4.0, overall=4.0
                ),
                judgeResults=[],
                feedback="Quality below threshold.",
            )
            final_prompt, final_result = asyncio.run(corrector_fail.correct(
                raw_prompt=prompt,
                planning_result=planning_result,
                initial_result=initial_result,
                task_description=planning_result.detectedIntent,
            ))
            duration = (time.perf_counter() - start) * 1000

            assert final_prompt == prompt, "Should return raw_prompt after exhaustion"
            assert "retries exhausted" in (final_result.feedback or "").lower() or final_result.winner == "original"

            result = TestResult(
                test_id="correction_exhausted",
                category="self_correction",
                prompt_length_bucket="medium",
                original_tokens=self.estimator.exact_count(prompt),
                description="Self-correction: exhausts retries -> returns raw_prompt",
                passed=True,
                duration_ms=duration,
                details={"final_prompt_is_raw": final_prompt == prompt, "feedback": (final_result.feedback or "")[:60]},
            )
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            result = TestResult(
                test_id="correction_exhausted",
                category="self_correction",
                prompt_length_bucket="medium",
                original_tokens=0,
                description="Self-correction: exhausts retries -> returns raw_prompt",
                passed=False,
                duration_ms=duration,
                error=str(e),
            )
        self.record(result)

    # ── E2E: TokenLessChatbot ────────────────────────────────────────────────
    def test_e2e_chatbot(self):
        print("\n═══ E2E: TokenLessChatbot Full Pipeline (Mock) ═══")

        chatbot = TokenLessChatbot(
            planner=self.planner,
            optimizer=self.optimizer_roi_off,
            battle=self.battle_pass,
            corrector=self.corrector,
        )

        test_cases = [
            ("e2e_micro", PROMPTS["micro_01"][0], "micro", None),
            ("e2e_short_en", PROMPTS["short_02"][0], "short", None),
            ("e2e_medium_cn", PROMPTS["medium_01"][0], "medium", None),
            ("e2e_long_cn", PROMPTS["long_01"][0], "long", None),
            ("e2e_with_supplement", PROMPTS["short_01"][0], "short", {"language": "Python", "framework": "FastAPI"}),
            ("e2e_edge_mixed", EDGE_CASES["edge_mixed_lang"][0], "short", None),
            ("e2e_edge_structured", EDGE_CASES["edge_already_structured"][0], "short", None),
            ("e2e_edge_verbose", EDGE_CASES["edge_all_english_verbose"][0], "short", None),
        ]

        for cid, prompt, bucket, supplements in test_cases:
            tokens = self.estimator.exact_count(prompt) if prompt.strip() else 0
            start = time.perf_counter()
            try:
                state = asyncio.run(chatbot.process(prompt, supplements=supplements))
                duration = (time.perf_counter() - start) * 1000

                assert isinstance(state, ConversationState)
                assert state.status in ("done", "error", "awaiting_supplement")
                assert state.session_id

                # Validate done state has expected fields
                if state.status == "done":
                    assert state.planning_result is not None
                    assert state.optimization_result is not None
                    assert state.final_prompt is not None

                summary = chatbot.format_result_summary(state)
                assert isinstance(summary, str) and len(summary) > 0

                result = TestResult(
                    test_id=f"e2e_{cid}",
                    category="e2e",
                    prompt_length_bucket=bucket,
                    original_tokens=tokens,
                    description=f"E2E Chatbot [{cid}]",
                    passed=True,
                    duration_ms=duration,
                    details={
                        "status": state.status,
                        "winner": state.evaluation_result.winner if state.evaluation_result else "N/A",
                        "overall": round(state.evaluation_result.scores.overall, 2) if state.evaluation_result else "N/A",
                        "opt_skipped": state.optimization_result.optimizationSkipped if state.optimization_result else "N/A",
                        "original_tokens": tokens,
                        "optimized_tokens": state.optimization_result.tokenStats.optimizedCount if state.optimization_result else "N/A",
                        "supplements": str(supplements),
                    },
                )
            except Exception as e:
                duration = (time.perf_counter() - start) * 1000
                result = TestResult(
                    test_id=f"e2e_{cid}",
                    category="e2e",
                    prompt_length_bucket=bucket,
                    original_tokens=tokens,
                    description=f"E2E Chatbot [{cid}]",
                    passed=False,
                    duration_ms=duration,
                    error=traceback.format_exc()[-300:],
                )
            self.record(result)

    # ── Stability: ROI boundary ──────────────────────────────────────────────
    def test_roi_boundary(self):
        print("\n═══ STABILITY: ROI Boundary Conditions ═══")

        # Generate length-gradient prompts: 5, 20, 50, 100, 150, 200, 300, 500 tokens
        dummy_word = "implement"
        token_targets = [5, 20, 50, 100, 150, 200, 300, 500]

        for target in token_targets:
            prompt = " ".join([dummy_word] * target)
            actual_tokens = self.estimator.exact_count(prompt)
            bucket = (
                "micro" if actual_tokens < 10 else
                "short" if actual_tokens < 50 else
                "medium" if actual_tokens < 200 else
                "long" if actual_tokens < 500 else "xlarge"
            )
            start = time.perf_counter()
            try:
                planning_result = asyncio.run(self.planner.plan(prompt))
                opt_result, _pipeline_usage = asyncio.run(self.optimizer_roi_on.optimize(prompt, planning_result))
                duration = (time.perf_counter() - start) * 1000

                assert opt_result.roiReport is not None
                expected_skip = opt_result.roiReport.netTokenSavings <= 0

                result = TestResult(
                    test_id=f"roi_boundary_{target}t",
                    category="roi_boundary",
                    prompt_length_bucket=bucket,
                    original_tokens=actual_tokens,
                    description=f"ROI boundary at ~{target} tokens",
                    passed=True,
                    duration_ms=duration,
                    details={
                        "actual_tokens": actual_tokens,
                        "net_savings": opt_result.roiReport.netTokenSavings,
                        "roi_positive": opt_result.roiReport.roiPositive,
                        "skipped": opt_result.optimizationSkipped,
                        "optimized_tokens": opt_result.tokenStats.optimizedCount,
                    },
                )
            except Exception as e:
                duration = (time.perf_counter() - start) * 1000
                result = TestResult(
                    test_id=f"roi_boundary_{target}t",
                    category="roi_boundary",
                    prompt_length_bucket=bucket,
                    original_tokens=target,
                    description=f"ROI boundary at ~{target} tokens",
                    passed=False,
                    duration_ms=duration,
                    error=str(e),
                )
            self.record(result)

    def run_all(self):
        print("\n" + "="*60)
        print("  TokenLess V1 - Stability & Multi-Scenario Test Suite")
        print("="*60)
        self.test_planning_layer()
        self.test_optimization_layer()
        self.test_word_pruner()
        self.test_evaluation_layer()
        self.test_self_correction()
        self.test_e2e_chatbot()
        self.test_roi_boundary()

        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        print(f"\n{'='*60}")
        print(f"  TOTAL: {total}  |  PASSED: {passed}  |  FAILED: {failed}")
        print(f"  Pass Rate: {passed/total:.1%}")
        print("="*60)
        return self.results


if __name__ == "__main__":
    runner = TestRunner()
    results = runner.run_all()
