# Chat-First Refactor Summary

TokenLess was refactored from a pipeline dashboard into a chat-first prompt mentor. The final interface asks focused questions, collects project details over several turns, then produces a structured final prompt with user-facing metrics.

## Final UX Requirements
- Start with a natural-language project idea.
- Ask one focused question at a time.
- Continue for at least five user turns unless the user explicitly asks to optimize now.
- Confirm before running the full pipeline.
- Show final prompt, token metrics, and model score.
- Keep advanced evaluation details in an expandable section.

## Implemented Files
- `src/chat/conversation.py`
- `src/chat/mentor.py`
- `src/chat/chatbot.py`
- `app.py`
- `tests/stability_test.py`
