# TokenLess Changelog

## v1.0.0 - 2026-05-07

TokenLess V1 is ready for the BU.330.760 final project submission.

### Completed
- Built a runnable Streamlit app in `app.py` for a chat-first prompt mentor workflow.
- Implemented the four-layer pipeline: Planning, Optimization, Pairwise Evaluation, and Self-Correction.
- Added offline stability tests in `tests/stability_test.py`.
- Added reproducible setup and usage instructions in `README.md`.
- Added sample output, benchmark artifacts, and technical documentation for grader review.

### Known V1 Boundaries
- Scope is limited to vibe-coding and software-development prompts.
- Planning and optimization require LM Studio on `localhost:1234`.
- Pairwise evaluation requires `OPENAI_API_KEY` and `GOOGLE_API_KEY`.
- The optimized prompt is often longer than the raw prompt because it adds structure; the business value is better downstream code generation quality and fewer correction rounds.
