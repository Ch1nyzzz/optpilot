"""Repair Recipe Library — actionable repair principles distilled from offline evolution.

Each RepairRecipe captures a reusable repair principle with three key fields:
- **precondition**: when to use this recipe (what symptoms indicate it's the right fix)
- **action**: concrete repair action (what to do)
- **root_cause**: what fundamental problem this solves

Recipes are more abstract than raw diffs (reusable across contexts) but more
actionable than change types (tell the LLM *what* to do, not just *which category*).

Storage: ``library_store/recipes/`` with one JSON file per (fm_group, change_type) pair.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from optpilot.config import LIBRARY_DIR
from optpilot.skills.repair_patterns import CHANGE_TYPES

_RECIPE_DIR = LIBRARY_DIR / "recipes"
SHARED_RECIPE_FM_GROUP = "_shared"


@dataclass
class RepairRecipe:
    """A reusable repair principle distilled from effective mutations."""

    recipe_id: str
    fm_group: str                # A-F
    change_type: str             # one of CHANGE_TYPES
    precondition: str            # when to use
    action: str                  # what to do
    root_cause: str              # what it fundamentally solves
    n_effective: int = 0         # how many effective diffs support this
    avg_test_delta: float = 0.0  # average test accuracy improvement
    example_diffs: list[str] = field(default_factory=list)  # 1-2 representative snippets


class RecipeLibrary:
    """Persistent recipe store indexed by (fm_group, change_type)."""

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or _RECIPE_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._recipes: dict[str, list[RepairRecipe]] = {}  # key = fm_group
        self._load()

    def get(self, fm_group: str, change_type: str | None = None) -> list[RepairRecipe]:
        """Get recipes for a FM group, optionally filtered by change type.

        Exact FM-group recipes are combined with shared cross-group recipes.
        """
        recipes = list(self._recipes.get(fm_group, []))
        recipes.extend(self._recipes.get(SHARED_RECIPE_FM_GROUP, []))
        if change_type:
            recipes = [r for r in recipes if r.change_type == change_type]
        return sorted(recipes, key=lambda r: r.n_effective, reverse=True)

    def get_top(
        self,
        fm_group: str,
        change_type: str | None = None,
        top_k: int = 3,
    ) -> list[RepairRecipe]:
        """Get top-k recipes for a FM group, sorted by evidence strength."""
        return self.get(fm_group, change_type=change_type)[:top_k]

    def add(self, recipe: RepairRecipe) -> None:
        """Add a recipe. Overwrites if same recipe_id exists, then deduplicates."""
        if recipe.fm_group not in self._recipes:
            self._recipes[recipe.fm_group] = []
        # Remove existing with same id
        self._recipes[recipe.fm_group] = [
            r for r in self._recipes[recipe.fm_group]
            if r.recipe_id != recipe.recipe_id
        ]
        self._recipes[recipe.fm_group].append(recipe)
        self._recipes[recipe.fm_group] = self._deduplicate_group(self._recipes[recipe.fm_group])

    def add_batch(self, recipes: list[RepairRecipe]) -> None:
        for r in recipes:
            self.add(r)

    def save(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        for fm_group, recipes in self._recipes.items():
            recipes = self._deduplicate_group(recipes)
            self._recipes[fm_group] = recipes
            path = self.base_dir / f"{fm_group}.json"
            data = [asdict(r) for r in recipes]
            path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def _load(self) -> None:
        for path in self.base_dir.glob("*.json"):
            fm_group = path.stem
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self._recipes[fm_group] = self._deduplicate_group([RepairRecipe(**d) for d in data])
            except Exception as e:
                print(f"  Warning: failed to load recipes for {fm_group}: {e}")

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = re.sub(r"\s+", " ", (text or "").strip().lower())
        text = re.sub(r"[^\w\s]", "", text)
        return text

    @classmethod
    def _semantic_key(cls, recipe: RepairRecipe) -> tuple[str, str, str, str, str]:
        return (
            recipe.fm_group,
            recipe.change_type,
            cls._normalize_text(recipe.precondition),
            cls._normalize_text(recipe.action),
            cls._normalize_text(recipe.root_cause),
        )

    @classmethod
    def _deduplicate_group(cls, recipes: list[RepairRecipe]) -> list[RepairRecipe]:
        merged: dict[tuple[str, str, str, str, str], RepairRecipe] = {}
        order: list[tuple[str, str, str, str, str]] = []
        for recipe in recipes:
            key = cls._semantic_key(recipe)
            if key not in merged:
                merged[key] = RepairRecipe(**asdict(recipe))
                order.append(key)
                continue
            existing = merged[key]
            total_effective = max(0, existing.n_effective) + max(0, recipe.n_effective)
            if total_effective > 0:
                weighted_delta = (
                    existing.avg_test_delta * max(0, existing.n_effective)
                    + recipe.avg_test_delta * max(0, recipe.n_effective)
                ) / total_effective
                existing.avg_test_delta = weighted_delta
            existing.n_effective = total_effective
            existing.example_diffs = list(dict.fromkeys(existing.example_diffs + recipe.example_diffs))
        return [merged[key] for key in order]

    def format_for_prompt(
        self,
        fm_group: str,
        change_type: str | None = None,
        top_k: int = 3,
    ) -> str:
        """Format top recipes for injection into LLM evolve prompt."""
        recipes = self.get_top(fm_group, change_type=change_type, top_k=top_k)
        if not recipes:
            return ""

        lines = ["## Repair Experience (from offline analysis)"]
        if change_type:
            lines.append(f"Target family: {change_type}")
        for i, r in enumerate(recipes, 1):
            lines.append(f"\nRecipe {i} ({r.n_effective}x verified, avg +{r.avg_test_delta:.1%}):")
            lines.append(f"  Precondition: {r.precondition}")
            lines.append(f"  Action: {r.action}")
            lines.append(f"  Root cause: {r.root_cause}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return sum(len(v) for v in self._recipes.values())
