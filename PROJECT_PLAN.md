# TokenLess - Project Plan

## 1. Project Title

**TokenLess: A Prompt Optimization Chatbot for Token Efficiency**

---

## 2. Target User, Workflow, and Business Value

TokenLess targets **developers and technical professionals** who frequently interact with LLM APIs. These users write prompts daily for tasks such as coding assistance, content drafting, and data analysis. Most lack a systematic approach to prompt optimization. This leads to **excessive token consumption** and **high inference costs**.

The recurring task this system addresses is the translation of natural language requirements into effective prompts. The core decision at each interaction is how to convey intent precisely with the **fewest possible tokens**. Currently, users rely on intuition when writing prompts. This often results in redundant phrasing, ambiguous intent, and disorganized information structure.

The workflow begins when a user inputs a raw prompt or a natural language description of their need. It ends when the system outputs a **structurally reorganized, word-level compressed, and positionally anchored prompt**. The output includes a **token savings report** and a **quality evaluation score**.

Better performance on this workflow matters for both economic and technical reasons. LLM inference cost scales directly with token consumption. The self-attention mechanism in the Transformer architecture scales at **O(n²) complexity**. Doubling the token count quadruples the computational cost. For enterprise users, even a **10% reduction** in tokens across millions of daily API calls represents significant cost savings. Research also shows that excessively long prompts trigger the **"Lost-in-the-Middle" attention degradation phenomenon**. This causes output quality to decline. Therefore, prompt optimization **reduces cost and improves output quality simultaneously**.

---

## 3. Problem Statement and GenAI Fit

TokenLess receives a user's raw prompt. It applies intent analysis, missing-field completion, positional anchoring, word-level compression, and semantic pruning. It then outputs an optimized prompt that uses fewer tokens while preserving or improving semantic equivalence. A **Pairwise Battle** evaluation framework verifies the optimization quality.

**Three parts of this workflow require GenAI capabilities.**

**Part 1 - Planning Mode.** It uses an LLM's semantic understanding to extract core intent from raw prompts. It decomposes intent into four dimensions: `WHO`, `WHAT`, `HOW`, and `FORMAT`. It matches the input against predefined scene templates and identifies missing fields. This task requires language comprehension and reasoning that rule engines cannot provide.

**Part 2 - Semantic Pruning.** It uses an LLM to score the relevance of background material in the prompt's middle section. It removes information unrelated to the core task. This judgment requires deep language understanding at the sentence level.

**Part 3 - Pairwise Battle Evaluation.** It uses an **LLM-as-a-Judge** paradigm. Multiple judge models perform blind comparisons between outputs generated from the original and optimized prompts.

> A simpler non-GenAI tool would not be sufficient. Rule-based engines such as regex substitution or template matching can only handle surface-level word replacements. They cannot understand semantic context, judge information relevance, or intelligently restructure prompt architecture. Intent analysis requires deep natural language understanding. Semantic pruning requires cross-sentence relevance reasoning. Quality evaluation requires human-like judgment. **These are tasks where GenAI is irreplaceable.**

---

## 4. Planned System Design and Baseline

TokenLess uses a **Layered Pipeline** architecture with **four layers**.

**Layer 1 - Input Layer.** It handles token estimation and input parsing. It is a rule-based engine using `tiktoken`. No LLM is required at this stage.

**Layer 2 - Planning Mode.** It performs scene detection, intent decomposition into `WHO` / `WHAT` / `HOW` / `FORMAT` dimensions, gap analysis against scene templates, and user-guided completion of missing fields. This layer requires a lightweight local model that supports structured output.

**Layer 3 - Optimization.** It performs positional anchoring using a **HEAD / MIDDLE / TAIL** three-zone restructuring strategy. It applies **five word-level compression rules**:

1. Redundant phrase elimination
2. Passive-to-active conversion
3. Nested simplification
4. Reference resolution
5. Format instruction compression

It also performs semantic pruning on the middle zone. The final output uses a **Markdown and JSON hybrid template**.

**Layer 4 - Evaluation.** It implements a **Pairwise Battle** framework. It sends both the original and optimized prompts to the same target model. It collects responses and submits them to at least **two independent judge models** for blind scoring. The judges use four weighted dimensions:

| Dimension | Weight |
| --- | --- |
| Intent Alignment | **40%** |
| Logic Coherence | **30%** |
| Conciseness Score | **20%** |
| Format Compliance | **10%** |

A **majority vote** determines the winner. In the case of a tie, the system keeps the optimized version because it uses fewer tokens. A **failure feedback loop** records failure patterns and triggers self-correction with a maximum of **two retries**.

### Local Inference

The Planning Mode and Optimization layers will run a small open-source model through **LM Studio**. LM Studio provides an OpenAI-compatible API at `localhost:1234/v1`. The parameter constraint is **at most  5B** to ensure compatibility with consumer-grade hardware. Candidate models:

| Model | Provider | Parameters | Notes |
| --- | --- | --- | --- |
| `Gemma-4-E4B-IT` | Google | **4B** | Compact, multilingual |
| `Qwen3.5-4B` | Alibaba | **5B** | Strong structured output |
| `GPT-OSS-Nano` | Community | **9B total / 3B active** | MoE architecture |

The final selection will be determined through a **benchmark evaluation** measuring latency, throughput, stability, structured output quality, and multilingual compatibility. For cloud-based evaluation, the judge models will use **GPT-4o** and **Gemini 2.5 Flash** APIs.

### Course Concept Integration

**Course Concept 1 - Anatomy of an LLM Call (Weeks 2-3):** Planning Mode uses a low temperature (~0.3) and enforces JSON Schema output constraints to produce a deterministic `PlanningResult` structure containing detected intent, scene type, and missing fields. The optimization layer similarly uses structured generation at temperature ~0.2 to ensure stable, reproducible compression output.

**Course Concept 2 - Evaluation Design:** The evaluation layer implements a Pairwise Battle framework where two independent judge models (GPT-4o and Gemini 2.5 Flash) perform blind scoring of outputs from the original versus optimized prompts across a four-dimensional weighted rubric (Intent Alignment 40%, Logic Coherence 30%, Conciseness Score 20%, Format Compliance 10%), with majority vote determining the winner.

### Baseline

The baseline is the **direct use of the original unoptimized prompt**. The user sends their raw input to the target LLM without any modification. Comparison dimensions: token count and compression rate, output quality via Pairwise Battle scores, inference latency, and total cost.

### App Description

The final deliverable is a browser-based mentor chatbot. When the user opens the app, they see a clean chat interface with a single text input field. The user describes a coding idea in natural language, and TokenLess asks beginner-friendly follow-up questions over a default 5-10 turn clarification flow. It collects target platform, language, framework, core features, input/output behavior, UI expectations, data/API needs, testing expectations, and output format. If the user is unsure, TokenLess chooses practical defaults such as React + TypeScript for a web app or FastAPI for a simple backend. Once the user confirms, the optimization pipeline runs automatically. The main result shows only original token estimate, optimized token estimate, token savings, model score, and the final Markdown + visible JSON prompt. Technical evaluation details remain available in an advanced expander. The UI framework is Streamlit (or equivalent), to be deployed as the Week 8 final deliverable.

### Development Plan

**V1 / Phases 1-4 - Core Algorithm Validation (Jupyter Notebooks).** The goal is to confirm model selection, tune optimization parameters, and validate the end-to-end pipeline from input through planning, optimization, and evaluation.

**Phase 5 - User-Facing Web Interface (Week 7-8).** The core pipeline validated in V1 notebooks will be wrapped in the interactive web application described above. This runnable app is a required deliverable for the Week 8 final presentation and live demo. *(Previously labeled "V2" in this document; aligned with docs/task-plan.md Phase 5 and docs/tech-spec.md section 8.)*

---

## 5. Evaluation Plan

> **Success means:** the optimized prompt achieves a **5% to 15% token reduction rate**, and the optimized version must win at least **60% of Pairwise Battle evaluations**. This means the output quality from the optimized prompt is at least as good as the output from the original prompt.

The system will measure multiple dimensions:

| Metric | Definition | Scale / Unit |
| --- | --- | --- |
| **Token Reduction Rate** | (original - optimized) / original | Percentage |
| **Intent Alignment** | Whether the optimized output fulfills the original intent | 0-10 |
| **Logic Coherence** | Whether the reasoning chain is complete and gap-free | 0-10 |
| **Conciseness Score** | Absence of unnecessary repetition (*higher is better*; 10 = no redundancy, 0 = highly redundant) | 0-10 |
| **Format Compliance** | Adherence to specified output formats | 0-10 |
| **Latency** | Time cost of the optimization process itself | Seconds |
| **Cost ROI** | Whether tokens consumed by optimization < tokens saved | **Hard requirement: positive ROI** |

### Test Set Design

The test set will contain **20 to 30 test cases** focused on the **vibe coding / software development** scene (V1 domain scope). Each test case will vary along two axes: prompt complexity (*short*, *medium*, *long*) and sub-type (*feature request*, *bug fix*, *refactor*, *data pipeline*, *API integration*). Test prompts will come from publicly shared Cursor and Copilot usage examples, manually constructed prompts with typical redundancy patterns, and extreme edge cases (e.g. prompts already near-optimal, or prompts with critical technical terms that must not be compressed).

### Baseline Comparison

Each test prompt will be processed through both the original and optimized paths. The system will record token counts, Pairwise Battle scores, and target model output quality for both versions. Paired comparison statistics will measure the **win rate**, **average compression rate**, and **average quality improvement** of the optimized version. Cases where the original version wins will be recorded and analyzed for **failure patterns**.

---

## 6. Example Inputs and Failure Cases

### Example Inputs

**Example 1 - Vibe Coding.** The user asks for a Python weather application with a React frontend, animation effects, a 7-day forecast display, and dynamic background colors based on weather conditions. *Expected optimization:* identifies missing fields such as the backend framework and API source. After user confirmation, the system applies positional anchoring and compression.

**Example 2 - Vibe Coding (Bug Fix).** The user asks to "fix the login bug where users get logged out randomly." *Expected optimization:* identifies missing fields (language, framework, reproduction steps, suspected cause) via Planning Mode. After user confirmation, applies positional anchoring - moves the reproduction condition to HEAD, moves stack trace context to MIDDLE, adds output constraint (return only the changed function) to TAIL.

**Example 3 - Vibe Coding (Refactor).** The user asks to "refactor this messy Python file to be cleaner." *Expected optimization:* detects vague intent ("cleaner"), asks critical missing field (what does clean mean: PEP8 compliance? extract functions? reduce nesting?). After clarification, applies word-level compression - removes filler like "can you help me" and restructures into `Role -> Tech Stack -> Requirements -> Constraints`.

**Example 4 - Vibe Coding (Data Pipeline).** The user asks for a "Python script that reads from a database and generates a report." *Expected optimization:* identifies missing fields (DB type, ORM or raw SQL, report format, scheduling requirements). Applies semantic pruning to any background context provided, retaining only schema details relevant to the task.

**Example 5 - Vibe Coding (Long Codebase Context, 500+ tokens).** The user pastes a large block of existing code, error logs, and database schema as context, then asks to "add a caching layer." Only the schema and the error log are relevant; the rest is noise. *Expected optimization:* applies semantic pruning to retain only the relevant schema fields and error patterns, removing over **60%** of the background noise, while keeping all critical technical identifiers (table names, column types, error codes) intact.

### Anticipated Failure Cases

**Failure 1 - Over-Compression Leading to Semantic Loss.** For highly specialized technical prompts containing specific API parameter names or database schema definitions, the system may incorrectly identify critical technical terms as redundant. Removing these terms would cause the optimized prompt to **lose key constraint information** and degrade output quality.

**Failure 2 - Chinese Tokenization Bias.** Chinese characters tokenize differently from English. Chinese requires approximately **2-3 tokens per character** versus approximately **1 token per English word**. This may cause inaccurate compression rate calculations. Word-level substitution strategies may also perform poorly in Chinese contexts.

**Failure 3 - Over-Compression of Critical Technical Identifiers.** For vibe coding prompts that include specific variable names, function signatures, or database column names as requirements, the word-level pruner may incorrectly flag repeated technical terms as redundant. Removing them would cause the generated code to use **wrong identifiers**, producing output that does not integrate with the existing codebase.

---

## 7. Risks and Governance

### Potential System Failures

The optimization process itself consumes tokens. For short prompts that are already concise, the tokens consumed by optimization may **exceed** the tokens saved. This would result in a **negative ROI**. A small local model with at most  5B parameters has limited reasoning capacity. It may fail to adequately analyze prompts with complex multi-layer nesting. The Pairwise Battle evaluation depends on cloud APIs. Network latency or API rate limiting could reduce evaluation efficiency.

### Trust Boundaries

> The system should **not** be trusted for prompts in **high-risk domains** such as medicine, law, or finance. Any semantic deviation in these fields could lead to serious consequences.

Higher compression rates are not always better. The system must enforce a **compression ceiling** to prevent overly aggressive pruning. The judge models in the evaluation layer have their own biases. Pairwise Battle results should not be treated as absolute truth.

### Controls and Boundaries

| Control | Threshold / Behavior |
| --- | --- |
| **Minimum quality threshold** | Reject optimization when Pairwise Battle overall score < **6.0**; return the original prompt with explanation |
| **Maximum compression rate** | **50%** cap; exceeding triggers a human review prompt |
| **Sensitive scene detection** | Structural reorganization only; **no content deletion** for prompts involving personal privacy or sensitive topics |
| **User confirmation step** | Required for critical missing fields identified during Planning Mode before optimization proceeds |

### Data Privacy

Data privacy is addressed through the system architecture. The Planning Mode and Optimization layers run **entirely on a local model**. No user data leaves the local machine during these stages. Only the Evaluation layer sends data to cloud APIs. API cost is controlled by limiting evaluation frequency. Each evaluation call to GPT-4o and Gemini consumes approximately **1,000-2,000 tokens**. All test data consists of **synthetic prompts** and publicly shared community examples. **No private user prompts will be used.**

---

## 8. Final Delivery - Week 8 Presentation

The Week 6 check-in has been cancelled. The project targets a **complete, runnable agent** for the Week 8 final presentation.

**Deliverable**: A fully functional, browser-based Prompt Optimization Chatbot for vibe coding prompts, deployed via Streamlit (or equivalent). The grader must be able to clone the repo, run `pip install -r requirements.txt`, and launch the app in one command.

**What the app must demonstrate:**

The complete four-layer pipeline running end-to-end behind a chat-first mentor: Input Layer (token estimation) -> Planning Mode (scene detection, intent decomposition, gap analysis with multi-turn clarification) -> Optimization Engine (positional anchoring, word-level compression, semantic pruning, professional rewrite) -> Pairwise Battle Evaluation (blind scoring with GPT-4o and Gemini 2.5 Flash, Conciseness Score replacing old Redundancy Score). The main result shows simple token metrics, model score, and the final Markdown + JSON prompt, with technical evaluation details hidden by default.

**Success criteria:**

5%-15% token reduction rate on vibe coding prompts, with the optimized version winning at least 60% of Pairwise Battle evaluations. ROI must be positive (tokens saved > tokens consumed by optimization). Compression rate must not exceed 50%. Quality floor: overall score at least  6.0 before the result is returned.

**Current status (Week 7):**

Phase 1 (infrastructure) is complete. Phase 2 (Planning Engine) is next - starting with Task 2.1 (vibe_coding.json). Phases 3-5 (Optimization, Evaluation, Web UI) must be completed this week.
