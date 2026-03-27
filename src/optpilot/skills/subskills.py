"""SubSkill storage — persist successful repair experiences for reuse.

Each sub-skill captures a successful repair pattern: what was diagnosed,
what changes were made, and the measured effect.  Future evolve steps
can query relevant sub-skills and try them first before inventing new fixes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from optpilot.config import LIBRARY_DIR


@dataclass
class SubSkill:
    """A single successful repair recipe."""

    name: str
    fm_group: str
    trigger: str                       # root cause pattern that this fixes
    changes: list[dict[str, str]]      # [{old_str, new_str, source}, ...]
    effect_fm_rate: tuple[float, float] = (0.0, 0.0)   # before → after
    effect_pass_rate: tuple[float, float] = (0.0, 0.0)  # before → after
    summary: str = ""
    created_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


_SUBSKILLS_DIR = LIBRARY_DIR / "subskills"


class SubSkillStore:
    """Persist and load sub-skills per FM group."""

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or _SUBSKILLS_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _group_dir(self, fm_group: str) -> Path:
        d = self.base_dir / fm_group
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save(self, subskill: SubSkill) -> Path:
        """Save a sub-skill to disk. Returns the file path."""
        group_dir = self._group_dir(subskill.fm_group)

        if not subskill.created_at:
            subskill.created_at = datetime.now().isoformat()

        # Auto-number
        existing = sorted(group_dir.glob("subskill_*.json"))
        idx = len(existing) + 1
        path = group_dir / f"subskill_{idx:03d}.json"

        with open(path, "w") as f:
            json.dump(asdict(subskill), f, indent=2, ensure_ascii=False)

        return path

    def load(self, fm_group: str) -> list[SubSkill]:
        """Load all sub-skills for an FM group."""
        group_dir = self._group_dir(fm_group)
        subskills: list[SubSkill] = []

        for path in sorted(group_dir.glob("subskill_*.json")):
            try:
                with open(path) as f:
                    data = json.load(f)
                subskills.append(SubSkill(
                    name=data.get("name", ""),
                    fm_group=data.get("fm_group", fm_group),
                    trigger=data.get("trigger", ""),
                    changes=data.get("changes", []),
                    effect_fm_rate=tuple(data.get("effect_fm_rate", (0, 0))),
                    effect_pass_rate=tuple(data.get("effect_pass_rate", (0, 0))),
                    summary=data.get("summary", ""),
                    created_at=data.get("created_at", ""),
                    metadata=data.get("metadata", {}),
                ))
            except Exception as e:
                print(f"  Warning: failed to load sub-skill {path}: {e}")

        return subskills

    def format_for_prompt(self, fm_group: str, max_subskills: int = 5) -> str:
        """Format sub-skills as text for LLM prompt injection."""
        subskills = self.load(fm_group)
        if not subskills:
            return "No prior successful repairs recorded for this failure group."

        lines: list[str] = []
        for i, ss in enumerate(subskills[-max_subskills:], 1):
            fm_before, fm_after = ss.effect_fm_rate
            pass_before, pass_after = ss.effect_pass_rate
            lines.append(
                f"SubSkill {i}: \"{ss.name}\"\n"
                f"  Trigger: {ss.trigger}\n"
                f"  Summary: {ss.summary}\n"
                f"  Effect: FM {fm_before:.2f}→{fm_after:.2f}, pass {pass_before:.3f}→{pass_after:.3f}\n"
                f"  Changes: {len(ss.changes)} modifications"
            )

        return "\n".join(lines)

    @staticmethod
    def from_evolve_result(
        fm_group: str,
        change_records: list,
        root_causes: list[str],
        before_fm: float,
        after_fm: float,
        before_pass: float,
        after_pass: float,
        summary: str,
    ) -> SubSkill:
        """Build a SubSkill from an evolve result."""
        changes = []
        for cr in change_records:
            changes.append({
                "old_str": cr.old_str[:500],
                "new_str": cr.new_str[:500],
                "source": cr.source,
            })

        return SubSkill(
            name=summary[:80],
            fm_group=fm_group,
            trigger="; ".join(root_causes[:3]),
            changes=changes,
            effect_fm_rate=(before_fm, after_fm),
            effect_pass_rate=(before_pass, after_pass),
            summary=summary,
        )
