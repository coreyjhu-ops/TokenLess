# Model Selection Report

## Selected Local Model

TokenLess uses `gemma-4-e4b-it` through LM Studio for planning and optimization. It was selected because it fits consumer hardware, supports structured output well enough for the project, and had the best practical latency among the tested local options.

## Alternatives Considered

| Model | Result | Decision |
|---|---|---|
| Gemma-4-E4B-IT | Low latency and stable enough for structured calls | Primary local model |
| Qwen3.5-4B | Stable but slower | Backup option |
| GPT-OSS-Nano | High throughput but less consistent | Conditional option only |

## Judge Models

The evaluation layer uses GPT-4o and Gemini 2.5 Flash as independent LLM judges. This separation keeps the local model focused on generation and uses stronger cloud models for quality comparison.
