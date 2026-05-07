"""Markdown and JSON-schema formatting for optimized prompts."""

from __future__ import annotations

import re
from typing import Any

from src.core.types import PlanningResult
from src.optimization.positional_anchor import PromptSegment


class PromptFormatter:
    """Render optimized prompt segments into the final Markdown format."""

    def format(
        self,
        segments: list[PromptSegment],
        planning_result: PlanningResult,
    ) -> str:
        """Format segments with section headings and a tail reminder anchor."""

        sections: list[str] = []
        for section_type, title in [
            ("role", "Role"),
            ("task", "Task"),
            ("tech_stack", "Tech Stack"),
            ("context", "Context"),
            ("constraint", "Constraints"),
            ("format", "Output Format"),
        ]:
            content = self._content_for(segments, section_type)
            if section_type == "format" and not content:
                content = "Return a clear implementation plan with file paths, code, and tests."
            if not content:
                continue
            if section_type == "constraint":
                content = self._format_constraints(content)
            elif section_type == "format":
                content = self.inject_json_schema(content)
            sections.append(f"## {title}\n{content}")

        if planning_result.detectedIntent.strip():
            sections.append(f"## Reminder\n{planning_result.detectedIntent.strip()}")

        return "\n\n".join(sections)

    def inject_json_schema(self, format_section: str) -> str:
        """Append a visible JSON-compatible response template."""

        if "```json" in format_section:
            return format_section

        schema_placeholder = """Return the answer using this JSON-compatible structure:

```json
{
  "summary": "",
  "files_to_create_or_modify": [],
  "implementation_steps": [],
  "acceptance_criteria": [],
  "tests": []
}
```"""
        return f"{format_section.rstrip()}\n\n{schema_placeholder}"

    def to_token_stats(
        self,
        original: str,
        optimized: str,
        token_estimator: Any,
    ) -> dict:
        """Return canonical token stats from a token estimator compare report."""

        report = token_estimator.compare_report(original, optimized)
        return {
            "original_count": report.original_tokens,
            "optimized_count": report.optimized_tokens,
            "reduction_rate": report.reduction_percent / 100,
        }

    @staticmethod
    def _content_for(segments: list[PromptSegment], section_type: str) -> str:
        """Collect all non-empty segment content for a section type."""

        contents = [
            segment.content.strip()
            for segment in segments
            if segment.section_type == section_type and segment.content.strip()
        ]
        return "\n".join(contents).strip()

    @staticmethod
    def _format_constraints(content: str) -> str:
        """Render constraints as one Markdown bullet per line."""

        lines = [line.strip() for line in content.splitlines() if line.strip()]
        return "\n".join(
            line if line.startswith("- ") else f"- {line}"
            for line in lines
        )


__all__ = ["PromptFormatter"]
