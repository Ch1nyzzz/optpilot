"""SkillEvolver — LLM-powered evolution of Skill source code.

When a Skill repeatedly fails, the evolver reads its Python source,
accumulated negatives, and generates a modified version.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from optpilot.llm import call_llm
from optpilot.models import ReflectInsight
from optpilot.repair_utils import extract_fenced_block
from optpilot.skills.base import BaseSkill

EVOLVE_SKILL_PROMPT = """\
You are a meta-optimizer that improves MAS repair skills.

The following Python Skill class has repeatedly failed to fix FM group {fm_group} \
problems. Analyze the accumulated failure records and modify the Skill's strategy.

## Current Skill Source Code
```python
{source_code}
```

## Accumulated Failures ({n_failures} total)
{failures_text}

## Task
Modify the Skill class to improve its repair strategy based on the failure patterns.
You may change:
- The ANALYZE_HINT to guide better diagnosis
- Override analyze(), evolve(), or reflect() with custom implementations
- Adjust convergence parameters (MAX_INNER_ITERS, CONVERGENCE_THRESHOLD, etc.)
- Change the prompt templates used in evolve()

Output the COMPLETE modified Python file wrapped in ```python ... ```.
The file must:
- Import from optpilot.skills.base (GenericSkill or BaseSkill)
- Import from optpilot.skills.registry (register_skill)
- Use @register_skill decorator
- Keep the same FM_GROUP = "{fm_group}"
- Be valid Python that can be compiled and executed"""


class SkillEvolver:
    """Evolve Skill source code when it repeatedly fails."""

    def __init__(self, evolved_dir: Path):
        self.evolved_dir = evolved_dir
        self.evolved_dir.mkdir(parents=True, exist_ok=True)
        self._failure_counts: dict[str, int] = {}

    def record_failure(self, fm_group: str) -> None:
        self._failure_counts[fm_group] = self._failure_counts.get(fm_group, 0) + 1

    def reset_failures(self, fm_group: str) -> None:
        self._failure_counts[fm_group] = 0

    def should_evolve(self, fm_group: str, threshold: int = 3) -> bool:
        return self._failure_counts.get(fm_group, 0) >= threshold

    def evolve_skill(
        self,
        fm_group: str,
        skill: BaseSkill,
        negatives: list[ReflectInsight],
    ) -> Path | None:
        """Generate an evolved version of the skill and save it."""
        try:
            source = inspect.getsource(skill.__class__)
        except (OSError, TypeError):
            return None

        failures_text = "\n".join(
            f"Round {i+1}: changes=[{', '.join(n.changes_attempted[:2])}] "
            f"FM {n.before_fm_rate:.2f}→{n.after_fm_rate:.2f}, "
            f"pass {n.before_pass_rate:.3f}→{n.after_pass_rate:.3f}. "
            f"Reason: {n.failure_reason}. Lesson: {n.lesson}"
            for i, n in enumerate(negatives[-10:])  # last 10
        ) or "No specific failures recorded."

        prompt = EVOLVE_SKILL_PROMPT.format(
            fm_group=fm_group,
            source_code=source,
            n_failures=len(negatives),
            failures_text=failures_text,
        )

        response = call_llm(
            [{"role": "user", "content": prompt}],
            max_tokens=16384,
        )

        try:
            new_source = extract_fenced_block(response, "python")
        except ValueError:
            return None

        # Syntax check
        try:
            compile(new_source, f"evolved_skill_{fm_group}.py", "exec")
        except SyntaxError:
            return None

        # Determine version number
        existing = sorted(self.evolved_dir.glob(f"skill_{fm_group}_v*.py"))
        version = len(existing) + 1
        out_path = self.evolved_dir / f"skill_{fm_group}_v{version}.py"
        out_path.write_text(new_source, encoding="utf-8")

        self._failure_counts[fm_group] = 0
        return out_path
