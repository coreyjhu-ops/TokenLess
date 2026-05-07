# TokenLess

**Vibe Coding Prompt Mentor** - A chat-first prompt optimization system for software development AI workflows.

TokenLess turns vague natural-language coding requests into precise, structured prompts through a 5-10 turn mentor conversation, then automatically optimizes them using a four-layer pipeline: Planning -> Optimization -> Pairwise Evaluation -> Self-Correction.

> **JHU ISAI - Generative AI Course, Week 8 Final Submission**
> Developed by Jiayi Zhuo (Corey)

---

## Context, User, and Problem

**User:** Software developers using vibe coding tools (Cursor, GitHub Copilot, Claude) who describe what they want to build in plain language and let AI generate the code.

**Workflow being improved:** The moment a developer types a natural-language request into a coding AI. This prompt is the single highest-leverage input in the entire workflow - its quality directly determines whether the AI writes correct, idiomatic code on the first try or requires multiple costly correction rounds.

**Why this problem matters:** Vague, redundant, or poorly structured prompts force the target LLM to do more inference work: resolving ambiguity consumes output tokens, misunderstood requirements cause re-runs, and "Lost-in-the-Middle" attention decay buries critical constraints in long prompts. The cost compounds - not just the tokens wasted on a bad first attempt, but every follow-up clarification exchange that follows.

**Why GenAI is necessary here:** A pure rule-engine can remove filler phrases but cannot understand *intent*. Deciding which background context is semantically relevant, what technical constraints are implied by a framework choice, or whether a sentence is redundant given the overall goal requires natural language understanding. TokenLess uses a local LLM (Gemma-4-E4B-IT) for intent analysis and semantic pruning, and cloud LLMs (GPT-4o + Gemini) as independent quality judges - tasks that are fundamentally beyond deterministic rules.

**Where a human should stay involved:** Domain validation (is the optimized prompt still asking for the right thing?), edge-case judgment when the system flags low quality scores (overall < 6.0), and any decision to expand beyond Vibe Coding into new prompt categories. The system surfaces these moments explicitly: it shows quality scores and refuses to output prompts that fail the quality gate, rather than silently degrading.

---

## What It Does

You describe a coding project in plain language. TokenLess asks clarifying questions one at a time, then produces a professional structured prompt with measurable token savings and a quality score from real LLM judges (GPT-4o + Gemini).

```
User: "I want to build a weather app"
          ↓  5-10 clarifying turns
TokenLess: "What language? React or Vue? Do you need an API? Tests?"
          ↓  confirmed
[Planning -> Optimization -> Pairwise Battle -> Self-Correction]
          ↓
Final Prompt (Markdown + JSON schema) + Token savings + Quality score
```

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11+ | Tested on macOS M4 |
| LM Studio | at least  0.3 | Must be running locally |
| Local Model | Gemma-4-E4B-IT | Download in LM Studio |
| OpenAI API Key | N/A | For GPT-4o judge |
| Google API Key | N/A | For Gemini 2.5 Flash judge |

---

## Setup

### 1. Clone the repo

```bash
git clone <repo-url>
cd TokenLess
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> On macOS with system Python, add `--break-system-packages` if needed.

### 3. Configure API keys

Create a `.env` file in the project root:

```bash
cp .env.example .env   # if .env.example exists, otherwise create manually
```

Edit `.env`:

```env
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...
```

> Both keys are required for the Pairwise Battle evaluation layer. Without them the pipeline will raise an `EvaluationError` and fall back to returning the original prompt.

### 4. Set up LM Studio

1. Download and install [LM Studio](https://lmstudio.ai)
2. Search for and download **Gemma-4-E4B-IT** (about 4B params, fits 8 GB RAM)
3. Load the model and start the local server on **port 1234** (default)
4. Verify it's running: `curl http://localhost:1234/v1/models`

The sidebar in the app will show a green "LM Studio connected" badge when ready.

> **Alternative local models** (update `config/models.yaml` if you use these):
> - Qwen3.5-4B - stable, roughly 3x slower latency
> - GPT-OSS-Nano - high throughput, less consistent on multilingual input

---

## Running the App

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

### Quick Demo (no LM Studio)

If you don't have a local model running, enable **Demo mode** in the sidebar. This bypasses the ROI guardrail and lets you see the full optimization pipeline with shorter prompts. The evaluation layer still requires API keys.

---

## Usage

1. **Describe your project** in the chat input - any natural language works
2. **Answer the follow-up questions** - TokenLess asks about language, framework, core functionality, I/O, etc., one question at a time
3. **Confirm** when asked - type "confirm", "yes", "ok", or "optimize"
4. **View results** - token savings, quality score, and the final structured prompt appear below the chat
5. **Copy the prompt** - use the one-click copy button and paste it into Cursor, Copilot, or Claude

---

## Project Structure

```
TokenLess/
├── app.py                        # Streamlit entry point
├── requirements.txt
├── config/
│   └── models.yaml               # Model configuration (hot-swappable)
├── src/
│   ├── core/                     # Types, ModelProvider ABC, TokenEstimator
│   │   └── providers/            # LMStudio / OpenAI / Google providers
│   ├── planning/                 # Scene detection, intent analysis, gap analysis
│   ├── optimization/             # Word pruner, positional anchor, semantic pruner,
│   │                             #   professional rewriter, optimizer pipeline
│   ├── evaluation/               # Pairwise battle, judge pool, self-correction
│   └── chat/                     # Chatbot controller, mentor orchestration,
│                                 #   conversation state, answer extractor
├── notebooks/                    # Phase validation notebooks (01-06)
├── tests/
│   └── stability_test.py         # Offline full-pipeline stability tests
└── docs/                         # Tech spec, task plan, changelogs
```

---

## Running Tests

All tests run fully offline - no API keys or local model required:

```bash
pytest tests/stability_test.py -v
```

Expected result: **14 passed, 2 warnings**

---

## Architecture Overview

TokenLess uses a four-layer pipeline architecture:

| Layer | Module | Model |
|---|---|---|
| Planning | `SceneDetector` -> `IntentAnalyzer` -> `GapAnalyzer` | Gemma-4-E4B-IT (local) |
| Optimization | `WordPruner` -> `PositionalAnchorer` -> `SemanticPruner` -> `ProfessionalRewriter` | Gemma-4-E4B-IT (local) |
| Evaluation | `JudgePool` (dual-blind) -> `PairwiseBattle` (majority vote) | GPT-4o + Gemini 2.5 Flash (API) |
| Self-Correction | `SelfCorrector` (up to 2 retries) | Gemma-4-E4B-IT (local) |

The Mentor layer (`TokenLessMentor`) wraps the full pipeline with a 5-10 turn chat interface that collects structured requirements before triggering optimization.

**Token tracking**: Every LLM call reports real `prompt_tokens` and `completion_tokens` via `PipelineTokenUsage`. The ROI report compares the pipeline's total token cost against the savings delivered to the target model.

---

## Configuration

Edit `config/models.yaml` to change models without touching code:

```yaml
planning:
  provider: "lmstudio"
  model: "gemma-4-e4b-it"
  base_url: "http://localhost:1234/v1"

evaluation:
  judges:
    - provider: "openai"
      model: "gpt-4o"
    - provider: "google"
      model: "gemini-2.5-flash"
```

---

## Demo Tips for Evaluators

For the best demonstration of the full optimization pipeline:

- **Use a prompt with 50+ words** - short prompts (< 20 tokens) trigger ROI protection by default; enable Demo mode in the sidebar to bypass this
- **Try a vague prompt** like "build me a todo app" - the mentor will ask about language, framework, data persistence, auth, and deployment
- **Try a detailed prompt** - the system will skip questions where information is already present and go straight to optimization

Example prompts to try:

```
I want to make a React dashboard that shows real-time stock prices
with charts and alerts when a stock moves more than 5%.
```

```
Build a Python FastAPI todo API with authentication, CRUD operations, and PostgreSQL storage.
```

---

## Evaluation and Results

### Baseline

The baseline for all evaluations is **sending the raw, unmodified user prompt directly to the target model** (Gemini 2.5 Flash) and measuring the response against the same rubric as the optimized prompt. This represents the status quo - what a developer gets when they paste a vague request into a coding AI with no preparation.

A secondary, weaker baseline is a **prompt-only rewrite**: asking GPT-4o to "clean up" the prompt in a single pass, without structured planning, positional anchoring, or pairwise evaluation.

### Test Cases

Five representative prompts were used to evaluate the full pipeline (run via `notebooks/05_pairwise_battle.ipynb` and `notebooks/06_end_to_end.ipynb` with live API calls):

| Case | Raw Prompt | Raw Tokens | Optimized Tokens | Quality Score | Winner |
|------|-----------|------------|-----------------|---------------|--------|
| EB-01 | "Build a React weather app with animation" | 24 | 187 | 9.55 / 10 | optimized |
| EB-02 | "Write a function" | 3 | N/A | 2.20 / 10 | original (quality gate blocked) |
| EB-03 | "Build a todo API with user auth" (vague) | 18 | 203 | 7.20 / 10 | optimized (after 2 self-corrections) |
| WP-01-04 | Redundant English prompts (unit tests) | 30-65 | N/A | N/A | 32-50% character compression |
| TC-01-05 | Planning gap analysis (multilingual) | 8-45 | N/A | N/A | All critical gaps resolved |

**Quality scoring rubric** (all dimensions higher = better):
- Intent Alignment (40%) - does the output address what the user actually asked?
- Logic Coherence (30%) - is the reasoning chain complete and unambiguous?
- Conciseness Score (20%) - how free of redundancy is the prompt?
- Format Compliance (10%) - does the output follow the specified structure?

### What the Evaluation Showed

The system performs well on prompts with 20+ tokens that contain recoverable gaps (missing language, framework, or constraints). In EB-01, the optimized prompt scored 9.55/10 against the baseline and was chosen by both GPT-4o and Gemini judges as the clear winner. In EB-03, the first optimization attempt scored below threshold; the self-correction loop retried twice and ultimately produced a 7.20/10 result - demonstrating that the failure feedback loop works.

### Where It Fails

**Short prompts trigger ROI protection.** When the raw prompt is under ~20 tokens, the structured Markdown overhead added by `ProfessionalRewriter` produces a longer output than the original, and the pipeline's token cost exceeds the savings. In production mode the system correctly refuses to output the "optimized" version and returns the original. In Demo mode (sidebar toggle), this protection is bypassed so evaluators can see the full pipeline output even on short prompts.

**Vague three-word requests hit the quality gate.** EB-02 ("Write a function") scored 2.20/10 - even after two self-correction retries, the prompt was too underspecified for the optimizer to produce a high-quality structured output. The system correctly rejected the output and returned the original, but it also means the Mentor's clarification phase is critical: if a user skips all questions and forces optimization, they may get no output improvement.

**The structured output is longer, not shorter.** A common misconception about this system: the *optimized prompt sent to the developer* is typically longer than the original (because it adds structural clarity), while the *token savings* are realized at the target model's inference step - clearer prompts produce fewer output tokens per code generation run. This is a known limitation documented in the Known Limitations section below.

---

## Artifact Snapshot

This section satisfies the final-project artifact snapshot requirement through a concise sample input/output walkthrough. A standalone version is also available at [`artifacts/artifact_snapshot.md`](artifacts/artifact_snapshot.md), with the longer transcript and metrics in [`artifacts/sample_output.md`](artifacts/sample_output.md).

### Input/Output Screenshot

![TokenLess input and output artifact screenshot](artifacts/input-output-screenshot.png)

This screenshot shows the raw user input, the mentor clarification questions, the optimized output prompt, evaluation metrics, and baseline comparison in one view. A plain runnable-app screenshot is also included at [`artifacts/app-screenshot.png`](artifacts/app-screenshot.png).

### Sample Input -> Output

**User input (raw):**
```
I want to build a React dashboard that shows real-time stock prices, includes charts, and sends an alert when a stock moves more than 5%.
```

**After 5-turn Mentor conversation** (clarified: TypeScript, Recharts, WebSocket API, no auth needed, tests optional):

**TokenLess output (optimized prompt):**
```markdown
## Role
You are a senior React/TypeScript developer. Your expertise: real-time data dashboards, financial UI components.

## Task
Build a real-time stock price dashboard with chart visualization and price-movement alerts.

## Context
- Framework: React 18 + TypeScript
- Charting: Recharts
- Data source: WebSocket connection to a stock price API (endpoint TBD)
- Alert threshold: plus or minus 5% price movement triggers an in-app notification

## Constraints
- No user authentication required
- Use functional components and React hooks only
- Handle WebSocket reconnection gracefully

## Output Format
Return the answer using this JSON-compatible structure:

​```json
{
  "files_to_create_or_modify": [],
  "implementation_steps": [],
  "acceptance_criteria": [],
  "tests": []
}
​```

## Reminder
Implement the WebSocket connection and alert logic first; the chart component is secondary.
```

**Result:** Token estimate before sending to target model - raw: 47 tokens -> structured: 183 tokens. Quality score: 9.55/10. Pipeline token cost: ~2,600 tokens (planning + optimization + dual-judge evaluation).

### App Interface

The Streamlit app (`streamlit run app.py`) opens to a chat interface. The user types a description, answers 5-10 follow-up questions from the Mentor, confirms, and the optimization pipeline runs automatically. The main result panel shows token estimates, quality score, and the final structured prompt with a one-click copy button. Advanced details (judge votes, token ledger, structure map) are hidden in an expandable section.

For the standalone artifact snapshot, see [`artifacts/artifact_snapshot.md`](artifacts/artifact_snapshot.md). For a complete session transcript with metrics, see [`artifacts/sample_output.md`](artifacts/sample_output.md). To see the live app, follow the setup instructions above and run `streamlit run app.py`.

---

## Known Limitations (V1)

- Domain is locked to **Vibe Coding / Software Development** - general prompts, image generation prompts, and email writing are out of scope (V2)
- The local model (Gemma-4-E4B-IT) requires LM Studio running on `localhost:1234`; there is no cloud fallback for the planning and optimization layers
- `ProfessionalRewriter` output is typically longer than the original prompt (structured Markdown overhead); ROI savings are realized at the target model's inference step, not in raw character count
- Evaluation layer requires both `OPENAI_API_KEY` and `GOOGLE_API_KEY`; missing keys will cause the pipeline to skip evaluation and return the optimized prompt directly

---

## License

Academic project - JHU ISAI Generative AI course, Spring 2026.
