"""Position-based prompt anchoring for the optimization engine.

This module implements the U-shaped attention layout from tech-spec §3.2 by
placing role and task instructions at the head, context and constraints in the
middle, and output-format instructions at the tail.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from src.core.types import PlanningResult


class PromptSegment(BaseModel):
    """A semantic prompt section assigned to an attention-aware position."""

    content: str
    section_type: Literal["role", "task", "context", "tech_stack", "constraint", "format"]
    position: Literal["head", "middle", "tail"]
    token_count: int = Field(default=0)


class PositionalAnchorer:
    """Split and reorder prompts into HEAD, MIDDLE, and TAIL sections."""

    _SECTION_PREFIXES = {
        "role": "Role:",
        "task": "Task:",
        "constraint": "Stack:",
        "format": "Format:",
    }

    def segment(
        self,
        raw_prompt: str,
        planning_result: PlanningResult,
    ) -> list[PromptSegment]:
        """Create prompt segments from refined planning requirements and context.

        Role and task requirements are anchored at the head, stack constraints and
        remaining raw context are placed in the middle, and format instructions
        are anchored at the tail. If no format instruction is present, a concise
        default output-format section is generated.
        """

        role_items: list[str] = []
        task_items: list[str] = []
        constraint_items: list[str] = []
        format_items: list[str] = []

        for requirement in planning_result.refinedRequirements:
            label, value = self._split_requirement(requirement)
            if label == "role":
                role_items.append(value)
            elif label == "task":
                task_items.append(value)
            elif label == "constraint":
                constraint_items.append(value)
            elif label == "format":
                format_items.append(value)

        context = self._remove_known_fragments(
            raw_prompt,
            planning_result.refinedRequirements,
        )

        segments: list[PromptSegment] = []
        if role_items:
            segments.append(
                PromptSegment(
                    content=self._join_items(role_items),
                    section_type="role",
                    position="head",
                )
            )
        if task_items:
            segments.append(
                PromptSegment(
                    content=self._join_items(task_items),
                    section_type="task",
                    position="head",
                )
            )
        if context:
            segments.append(
                PromptSegment(
                    content=context,
                    section_type="context",
                    position="middle",
                )
            )
        if constraint_items:
            segments.append(
                PromptSegment(
                    content=self._join_items(constraint_items),
                    section_type="constraint",
                    position="middle",
                )
            )

        format_content = (
            self._join_items(format_items)
            if format_items
            else "Return clean, well-structured output."
        )
        segments.append(
            PromptSegment(
                content=format_content,
                section_type="format",
                position="tail",
            )
        )

        return segments

    def reorder(self, segments: list[PromptSegment]) -> list[PromptSegment]:
        """Return segments ordered as HEAD, MIDDLE, then TAIL."""

        position_order = {"head": 0, "middle": 1, "tail": 2}
        section_order = {
            "role": 0,
            "task": 1,
            "context": 2,
            "tech_stack": 3,
            "constraint": 4,
            "format": 5,
        }
        return sorted(
            segments,
            key=lambda segment: (
                position_order[segment.position],
                section_order[segment.section_type],
            ),
        )

    def to_structured_prompt(self, segments: list[PromptSegment]) -> str:
        """Render ordered segments as Markdown sections."""

        section_titles = {
            "role": "Role",
            "task": "Task",
            "context": "Context",
            "tech_stack": "Tech Stack",
            "constraint": "Constraints",
            "format": "Output Format",
        }
        sections: list[str] = []
        for segment in self.reorder(segments):
            content = segment.content.strip()
            if not content:
                continue
            title = section_titles[segment.section_type]
            sections.append(f"## {title}\n{content}")

        return "\n\n".join(sections)

    def _split_requirement(self, requirement: str) -> tuple[str | None, str]:
        """Return the recognized section label and cleaned content."""

        text = requirement.strip()
        for section_type, prefix in self._SECTION_PREFIXES.items():
            if text.lower().startswith(prefix.lower()):
                return section_type, text[len(prefix) :].strip()
        return None, text

    def _remove_known_fragments(
        self,
        raw_prompt: str,
        refined_requirements: list[str],
    ) -> str:
        """Remove already-extracted requirement fragments from raw context."""

        context = raw_prompt.strip()
        for requirement in refined_requirements:
            _, value = self._split_requirement(requirement)
            candidates = {requirement.strip(), value.strip()}
            for candidate in sorted(candidates, key=len, reverse=True):
                if not candidate:
                    continue
                context = re.sub(
                    re.escape(candidate),
                    " ",
                    context,
                    flags=re.IGNORECASE,
                )

        return self._clean_whitespace(context)

    @staticmethod
    def _join_items(items: list[str]) -> str:
        """Join requirement values while removing accidental blank entries."""

        return "\n".join(item.strip() for item in items if item.strip()).strip()

    @staticmethod
    def _clean_whitespace(text: str) -> str:
        """Normalize spacing while preserving paragraph boundaries lightly."""

        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" *\n *", "\n", text)
        return text.strip()


__all__ = ["PromptSegment", "PositionalAnchorer"]
