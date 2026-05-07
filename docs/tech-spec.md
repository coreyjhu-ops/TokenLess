# TokenLess V1 Technical Specification

## Product Scope

TokenLess is a vibe-coding prompt mentor for software-development AI workflows. It helps a developer turn a vague natural-language coding request into a structured prompt that can be sent to a target AI coding model.

V1 supports one business use case: improving the prompt-writing workflow for developers who use tools such as Cursor, GitHub Copilot, Claude, or ChatGPT for code generation.

## Architecture

TokenLess uses a four-layer pipeline.

| Layer | Responsibility | Main Modules | Model Use |
|---|---|---|---|
| Planning | Detect the scene, extract intent, and identify missing fields | `src/planning/` | Local LM Studio model for structured intent analysis |
| Optimization | Reorder, prune, and rewrite the prompt into Markdown plus JSON structure | `src/optimization/` | Local LM Studio model for semantic pruning and professional rewrite |
| Evaluation | Compare original and optimized outputs with independent judges | `src/evaluation/` | GPT-4o and Gemini 2.5 Flash as judges |
| Self-Correction | Retry optimization when evaluation fails | `src/evaluation/self_correction.py` | Local model plus evaluation feedback |

The chat layer in `src/chat/` wraps this pipeline in a 5-10 turn mentor flow. The Streamlit interface in `app.py` exposes the workflow as a usable app.

## Core Data Contracts

The shared Pydantic models live in `src/core/types.py`. The most important contracts are `PlanningResult`, `OptimizationResult`, `EvaluationResult`, `OptimizationROIReport`, and `PipelineTokenUsage`.

Evaluation scores use four weighted dimensions: intent alignment at 40%, logic coherence at 30%, conciseness score at 20%, and format compliance at 10%.

## Configuration

`config/models.yaml` controls model providers and model names. Planning and optimization default to LM Studio at `http://localhost:1234/v1` with `gemma-4-e4b-it`. Evaluation defaults to GPT-4o and Gemini 2.5 Flash.

## Failure Handling

- If required API keys are missing, evaluation can fail and the app returns the best available prompt state instead of exposing secrets.
- If the quality gate rejects an optimized prompt, self-correction can retry up to two times.
- If ROI protection blocks a very short prompt in production mode, the app returns the original prompt. Demo mode can bypass this to show the full workflow.

## Validation

Offline validation is provided by `tests/stability_test.py`. Notebook evidence is retained under `notebooks/` for phase-level validation and reproducibility.
