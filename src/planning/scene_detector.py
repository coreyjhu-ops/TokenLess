"""Scene detection for Planning Mode.

V1 only supports the ``vibe_coding`` scene. The detector still loads scene
configuration from JSON files so V2 can add new scenes by dropping additional
instruction reference files into the refs directory.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.core.types import SceneType


class SceneDetector:
    """Map raw user prompts to a supported planning scene."""

    def __init__(self, refs_dir: str = "src/planning/instruction_refs") -> None:
        """Load all instruction reference JSON files from ``refs_dir``."""

        self.refs_dir = Path(refs_dir)
        self.refs: dict[str, dict] = {}
        self._load_refs()

    def detect(self, raw_prompt: str) -> SceneType:
        """Detect the planning scene for ``raw_prompt``.

        V1 is intentionally locked to vibe coding: keyword hits return
        ``"vibe_coding"``, and prompts without hits also fall back to
        ``"vibe_coding"``.
        """

        ref = self.get_ref("vibe_coding")
        prompt = raw_prompt.lower()
        for keyword in ref.get("scene_keywords", []):
            if keyword.lower() in prompt:
                return "vibe_coding"
        return "vibe_coding"

    def get_ref(self, scene: SceneType) -> dict:
        """Return the full instruction reference config for ``scene``."""

        if scene not in self.refs:
            raise KeyError(f"Instruction reference not found for scene: {scene}")
        return self.refs[scene]

    def _load_refs(self) -> None:
        """Load scene configs keyed by their top-level JSON scene name."""

        if not self.refs_dir.exists():
            raise FileNotFoundError(
                f"Instruction references directory not found: {self.refs_dir}"
            )

        for ref_path in sorted(self.refs_dir.glob("*.json")):
            with ref_path.open("r", encoding="utf-8") as ref_file:
                payload = json.load(ref_file)
            for scene, config in payload.items():
                self.refs[scene] = config
