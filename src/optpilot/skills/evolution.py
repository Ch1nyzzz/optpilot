"""SkillEvolver — LLM-powered evolution of Skill source code.

When a Skill repeatedly fails, the evolver reads its Python source,
accumulated negatives, and uses tool-calling to modify the source code.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import subprocess
import tempfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from optpilot.config import (
    LIBRARY_DIR,
    META_EVOLVE_FAILURE_THRESHOLD,
    META_EVOLVE_MAX_TOKENS,
    META_EVOLVE_MAX_TURNS,
    NEGATIVES_DIR,
    PROJECT_ROOT,
)
from optpilot.llm import acall_llm_with_tools
from optpilot.models import ReflectInsight
from optpilot.skills.base import BaseSkill

_EVOLVED_DIR = LIBRARY_DIR / "evolved_skills"
_SKILL_AGENT_TRACE_DIR = LIBRARY_DIR / "skill_agent_traces"
_META_EVOLVE_TRACE_DIR = LIBRARY_DIR / "meta_evolve_traces"
_SKILLS_BASE_PATH = PROJECT_ROOT / "src/optpilot/skills/base.py"
_SKILLS_TOOLS_PATH = PROJECT_ROOT / "src/optpilot/skills/tools.py"
_SKILLS_REGISTRY_PATH = PROJECT_ROOT / "src/optpilot/skills/registry.py"
_SUBSKILLS_PATH = PROJECT_ROOT / "src/optpilot/skills/subskills.py"
_DAG_EXECUTOR_PATH = PROJECT_ROOT / "src/optpilot/dag/executor.py"
_DAG_CORE_PATH = PROJECT_ROOT / "src/optpilot/dag/core.py"
_MODELS_PATH = PROJECT_ROOT / "src/optpilot/models.py"
_ARCHITECTURE_PATH = PROJECT_ROOT / "memory_bank/architecture.md"
_PROJECT_GOAL_PATH = PROJECT_ROOT / "memory_bank/project_goal.md"
_PROGRESS_PATH = PROJECT_ROOT / "memory_bank/progress.md"

_EVOLVE_SYSTEM_PROMPT = """\
You are a meta-optimizer that improves MAS repair skill source code.

You have two tools:
- **search_and_replace**: modify the Python source file by replacing exact text segments.
- **bash**: run shell commands. The current skill source is at $SKILL_FILE. \
Use `cat $SKILL_FILE` to read it. You have a generous budget: up to \
{max_turns} tool-calling turns and {max_tokens} completion tokens, so inspect \
the relevant files before editing.

## Prepared Context Files
- `$SKILL_FILE` — the current skill source you are modifying
- `meta_context.md` — prepared index of repository paths, project memory, and persistent experience
- `failure_summary.md` — concise summary of recent failed repair attempts
- `failures.json` — structured ReflectInsight records for the current FM group

## Workflow
1. Read `cat meta_context.md`, `cat failure_summary.md`, and `cat $SKILL_FILE`.
2. Use `bash` to inspect only the most relevant repository files referenced in `meta_context.md`.
3. Analyze the failure patterns and identify what needs to change.
4. Apply modifications with `search_and_replace` — targeted changes only.
5. Validate syntax with `bash`: \
`python3 -c "compile(open('$SKILL_FILE').read(), 'skill.py', 'exec')" && echo OK`
6. When done, respond with a summary of what you changed and why.

## What you can modify
- ANALYZE_HINT to guide better diagnosis
- Override analyze(), evolve(), or reflect() with custom implementations
- Adjust MAX_INNER_ITERS, CONVERGENCE_THRESHOLD, NO_IMPROVE_PATIENCE
- Change prompt templates

## Rules
- The file must import from optpilot.skills.base (GenericSkill or BaseSkill)
- Must use @register_skill decorator
- Must keep the same FM_GROUP value
- Must be valid, compilable Python"""

_EVOLVE_USER_PROMPT = """\
The Skill for FM group {fm_group} has repeatedly failed to fix problems.

## Accumulated Failures ({n_failures} total)
{failures_text}

## Prepared Context
- Read `meta_context.md` first for the key repository file addresses and one-line summaries.
- Read `failure_summary.md` for the condensed failure digest, then inspect `failures.json` if needed.

Analyze the failure patterns and modify the skill source code to improve its \
repair strategy. Use `bash` aggressively to inspect the files referenced in `meta_context.md` \
instead of relying only on the prompt summary."""


# Tool schemas for skill evolution (same structure, different target)
_SKILL_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_and_replace",
            "description": "Replace text in the skill Python source file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "old_str": {"type": "string", "description": "Exact text to find."},
                    "new_str": {"type": "string", "description": "Replacement text."},
                },
                "required": ["old_str", "new_str"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash command. $SKILL_FILE points to the source.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command to run."},
                },
                "required": ["command"],
            },
        },
    },
]


class _SkillFileContext:
    """Manages the skill source file for tool execution."""

    def __init__(self, source: str):
        self.source = source
        self._tmpdir = tempfile.mkdtemp(prefix="optpilot_meta_")
        self.skill_file = os.path.join(self._tmpdir, "skill.py")
        with open(self.skill_file, "w") as f:
            f.write(source)

    def execute_tool(self, name: str, args: dict) -> str:
        if name == "search_and_replace":
            return self._search_and_replace(args)
        elif name == "bash":
            return self._bash(args)
        return f"Error: unknown tool '{name}'"

    def _search_and_replace(self, args: dict) -> str:
        old_str = args.get("old_str", "")
        new_str = args.get("new_str", "")
        if not old_str:
            return "Error: old_str is empty."

        count = self.source.count(old_str)
        if count == 0:
            return "Error: old_str not found. Use `cat $SKILL_FILE` to see the current source."
        if count > 1:
            return f"Error: old_str matches {count} locations. Add more context."

        self.source = self.source.replace(old_str, new_str, 1)
        with open(self.skill_file, "w") as f:
            f.write(self.source)
        return f"OK. Replaced 1 occurrence."

    def _bash(self, args: dict) -> str:
        command = args.get("command", "")
        if not command:
            return "Error: command is empty."
        env = {**os.environ, "SKILL_FILE": self.skill_file}
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True,
                timeout=30, cwd=self._tmpdir, env=env,
            )
            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}"
            if result.returncode != 0:
                output += f"\n(exit code: {result.returncode})"
            if len(output) > 4000:
                output = output[:4000] + "\n... (truncated)"

            # Sync back if source was modified via bash
            if os.path.exists(self.skill_file):
                with open(self.skill_file) as f:
                    new_source = f.read()
                if new_source != self.source:
                    try:
                        compile(new_source, "skill.py", "exec")
                        self.source = new_source
                    except SyntaxError:
                        with open(self.skill_file, "w") as f:
                            f.write(self.source)

            return output or "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: command timed out."
        except Exception as e:
            return f"Error: {e}"

    def write_file(self, filename: str, content: str) -> str:
        path = os.path.join(self._tmpdir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path


def _build_failure_summary(negatives: list[ReflectInsight], max_items: int = 20) -> str:
    if not negatives:
        return "No specific failures recorded yet."

    lines = ["# Failure Summary", ""]
    for i, neg in enumerate(negatives[-max_items:], 1):
        lines.extend([
            f"## Failure {i}",
            f"- Round index: {neg.round_index}",
            f"- Changes attempted: {', '.join(neg.changes_attempted[:3]) or 'none recorded'}",
            f"- FM rate: {neg.before_fm_rate:.2f} -> {neg.after_fm_rate:.2f}",
            f"- Pass rate: {neg.before_pass_rate:.3f} -> {neg.after_pass_rate:.3f}",
            f"- Reason: {neg.failure_reason}",
            f"- Lesson: {neg.lesson}",
            "",
        ])
    return "\n".join(lines).rstrip()


def _build_meta_context(
    fm_group: str,
    source_path: Path | None,
    evolved_dir: Path,
) -> str:
    negatives_path = NEGATIVES_DIR / f"negatives_{fm_group}.json"
    subskills_dir = LIBRARY_DIR / "subskills" / fm_group
    skill_agent_trace_dir = _SKILL_AGENT_TRACE_DIR / fm_group
    meta_trace_dir = _META_EVOLVE_TRACE_DIR / fm_group
    latest_skill_agent_traces = sorted(skill_agent_trace_dir.glob("*.json"))[-5:]
    latest_meta_traces = sorted(meta_trace_dir.glob("*.json"))[-5:]

    lines = [
        "# Meta Skill Context Index",
        "",
        "Read this file first, then use bash to inspect only the files you need.",
        "",
        "## Current Target",
        f"- FM group: {fm_group}",
        f"- Working skill file: {source_path if source_path else '$SKILL_FILE (temporary copy)'}",
        f"- Output directory for evolved skills: {evolved_dir}",
        "",
        "## Prepared Local Files",
        "- $SKILL_FILE: editable temporary copy of the current skill source",
        "- meta_context.md: this index file",
        "- failure_summary.md: condensed recent failure digest",
        "- failures.json: structured ReflectInsight records copied into the temp workspace",
        "",
        "## Repository Files To Inspect With Bash",
        f"- {_SKILLS_BASE_PATH}: shared BaseSkill/GenericSkill workflow, prompts, and meta-evolution trigger behavior",
        f"- {_SKILLS_TOOLS_PATH}: bash/search_and_replace tool semantics for normal skill evolution",
        f"- {_SKILLS_REGISTRY_PATH}: registration and loading rules for evolved skill classes",
        f"- {_SUBSKILLS_PATH}: successful sub-skill persistence and prompt formatting",
        f"- {_DAG_EXECUTOR_PATH}: runtime semantics of the repaired MAS DAG",
        f"- {_DAG_CORE_PATH}: MASDAG schema and parsing behavior",
        f"- {_MODELS_PATH}: ReflectInsight and SkillBudget data models",
        f"- {_ARCHITECTURE_PATH}: current system architecture and workflow notes",
        f"- {_PROJECT_GOAL_PATH}: project mission and scope",
        f"- {_PROGRESS_PATH}: current milestones and status",
        "",
        "## Persistent Experience",
        f"- {negatives_path}: persisted negative lessons for this FM group",
        f"- {subskills_dir}: successful reusable sub-skills for this FM group",
        f"- {skill_agent_trace_dir}: persisted skill-agent tool traces for this FM group",
        *[
            f"- {path}: recent skill-agent tool trace transcript"
            for path in latest_skill_agent_traces
        ],
        f"- {evolved_dir}: previously evolved skill versions for this FM group",
        f"- {meta_trace_dir}: persisted meta-evolve tool traces for this FM group",
        *[
            f"- {path}: recent meta-evolve tool trace transcript"
            for path in latest_meta_traces
        ],
        "",
        "Use bash (`cat`, `sed -n`, `rg`) to inspect these files before making edits.",
    ]
    return "\n".join(lines)


def _persist_meta_tool_trace(
    fm_group: str,
    source_path: Path | None,
    ctx: _SkillFileContext,
    final_msgs: list[dict],
    negatives: list[ReflectInsight],
) -> str:
    trace_dir = _META_EVOLVE_TRACE_DIR / fm_group
    trace_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    trace_path = trace_dir / f"tool_trace_{timestamp}.json"
    payload = {
        "created_at": datetime.now().isoformat(),
        "fm_group": fm_group,
        "source_path": str(source_path) if source_path else "",
        "skill_file": ctx.skill_file,
        "messages": final_msgs,
        "negatives": [asdict(neg) for neg in negatives],
        "final_source": ctx.source,
    }
    trace_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return str(trace_path)


class SkillEvolver:
    """Evolve Skill source code when it repeatedly fails."""

    def __init__(self, evolved_dir: Path | None = None):
        self.evolved_dir = evolved_dir or _EVOLVED_DIR
        self.evolved_dir.mkdir(parents=True, exist_ok=True)
        self._failure_counts: dict[str, int] = {}

    def record_failure(self, fm_group: str) -> None:
        self._failure_counts[fm_group] = self._failure_counts.get(fm_group, 0) + 1

    def reset_failures(self, fm_group: str) -> None:
        self._failure_counts[fm_group] = 0

    def should_evolve(
        self,
        fm_group: str,
        threshold: int = META_EVOLVE_FAILURE_THRESHOLD,
    ) -> bool:
        return self._failure_counts.get(fm_group, 0) >= threshold

    def evolve_skill(
        self,
        fm_group: str,
        skill: BaseSkill,
        negatives: list[ReflectInsight],
    ) -> Path | None:
        """Generate an evolved version of the skill via tool-calling."""
        try:
            source = inspect.getsource(skill.__class__)
            source_path_str = inspect.getsourcefile(skill.__class__)
        except (OSError, TypeError):
            return None
        source_path = Path(source_path_str) if source_path_str else None

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're inside an async context — schedule in a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    asyncio.run,
                    self._aevolve_skill(fm_group, source, negatives, source_path),
                )
                return future.result()
        else:
            return asyncio.run(self._aevolve_skill(fm_group, source, negatives, source_path))

    async def _aevolve_skill(
        self,
        fm_group: str,
        source: str,
        negatives: list[ReflectInsight],
        source_path: Path | None = None,
    ) -> Path | None:
        failures_text = "\n".join(
            f"Round {i+1}: changes=[{', '.join(n.changes_attempted[:2])}] "
            f"FM {n.before_fm_rate:.2f}→{n.after_fm_rate:.2f}, "
            f"pass {n.before_pass_rate:.3f}→{n.after_pass_rate:.3f}. "
            f"Reason: {n.failure_reason}. Lesson: {n.lesson}"
            for i, n in enumerate(negatives[-20:])
        ) or "No specific failures recorded."

        user_prompt = _EVOLVE_USER_PROMPT.format(
            fm_group=fm_group,
            n_failures=len(negatives),
            failures_text=failures_text,
        )

        ctx = _SkillFileContext(source)
        ctx.write_file("meta_context.md", _build_meta_context(
            fm_group=fm_group,
            source_path=source_path,
            evolved_dir=self.evolved_dir,
        ))
        ctx.write_file("failure_summary.md", _build_failure_summary(negatives))
        ctx.write_file(
            "failures.json",
            json.dumps([asdict(neg) for neg in negatives], ensure_ascii=False, indent=2),
        )

        final_msgs = await acall_llm_with_tools(
            messages=[
                {"role": "system", "content": _EVOLVE_SYSTEM_PROMPT.format(
                    max_turns=META_EVOLVE_MAX_TURNS,
                    max_tokens=META_EVOLVE_MAX_TOKENS,
                )},
                {"role": "user", "content": user_prompt},
            ],
            tools=_SKILL_TOOL_SCHEMAS,
            tool_executor=ctx.execute_tool,
            max_tokens=META_EVOLVE_MAX_TOKENS,
            max_turns=META_EVOLVE_MAX_TURNS,
        )
        _persist_meta_tool_trace(
            fm_group=fm_group,
            source_path=source_path,
            ctx=ctx,
            final_msgs=final_msgs,
            negatives=negatives,
        )

        # Validate final source
        try:
            compile(ctx.source, f"evolved_skill_{fm_group}.py", "exec")
        except SyntaxError:
            return None

        # Save
        existing = sorted(self.evolved_dir.glob(f"skill_{fm_group}_v*.py"))
        version = len(existing) + 1
        out_path = self.evolved_dir / f"skill_{fm_group}_v{version}.py"
        out_path.write_text(ctx.source, encoding="utf-8")

        self._failure_counts[fm_group] = 0
        return out_path
