"""CatalogEvolver — LLM-powered evolution of the repair pattern catalog.

When repair patterns repeatedly fail for an FM group, the evolver uses
LLM tool-calling to add new patterns, refine descriptions, or mark
ineffective ones in the PatternCatalog.
"""

from __future__ import annotations

import asyncio
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
from optpilot.skills.jacobian import RepairJacobian
from optpilot.skills.repair_patterns import PatternCatalog, RepairPattern

_META_EVOLVE_TRACE_DIR = LIBRARY_DIR / "meta_evolve_traces"
_ARCHITECTURE_PATH = PROJECT_ROOT / "memory_bank/architecture.md"
_PROJECT_GOAL_PATH = PROJECT_ROOT / "memory_bank/project_goal.md"
_PROGRESS_PATH = PROJECT_ROOT / "memory_bank/progress.md"
_REPAIR_PATTERNS_PATH = PROJECT_ROOT / "src/optpilot/skills/repair_patterns.py"
_DAG_EXECUTOR_PATH = PROJECT_ROOT / "src/optpilot/dag/executor.py"
_DAG_CORE_PATH = PROJECT_ROOT / "src/optpilot/dag/core.py"


_EVOLVE_SYSTEM_PROMPT = """\
You are a meta-optimizer that improves the repair pattern catalog for a MAS \
repair system.

You have three tools:
- **add_pattern**: add a new repair pattern to the catalog.
- **update_pattern**: modify an existing pattern's description, name, \
target_components, or mark it as ineffective.
- **bash**: run shell commands to inspect project files for context.

## Prepared Context Files
- `catalog.json` — the current pattern catalog (JSON)
- `jacobian_report.md` — empirical pattern success rates and recommendation \
divergence data (which patterns work, which get ignored by the repair LLM)
- `meta_context.md` — index of repository paths and persistent experience
- `failure_summary.md` — condensed summary of recent failed repair attempts
- `failures.json` — structured failure records for this FM group

## Workflow
1. Read `cat jacobian_report.md` — this is your primary signal. It shows \
which patterns actually work, which consistently fail, and where the repair \
LLM diverges from recommendations (does something different than suggested).
2. Read `cat catalog.json` and `cat failure_summary.md`.
3. Use `bash` to inspect repository files referenced in `meta_context.md` \
if you need to understand the repair semantics.
4. Decide what to change. You have a spectrum of actions from conservative \
to aggressive — choose based on how badly things are failing:

### Conservative (some patterns still work)
   - **Refine** pattern descriptions that get ignored — the repair LLM may \
not follow them because the description is too vague or mismatched.
   - **Update target_components** if the Jacobian shows a pattern works for \
different DAG components than originally specified.

### Moderate (most patterns underperform)
   - **Disable** patterns with consistently low success rates (effective=false).
   - **Promote** observed divergences: if the repair LLM repeatedly ignores \
pattern A and does something else that works, add that "something else" as \
a first-class pattern.

### Aggressive — 不破不立 (nothing in the catalog works)
   - If existing patterns have been exhausted and keep failing, **invent \
entirely new repair strategies**. Do not just rephrase old patterns.
   - Study the failure_summary and trace files to understand *what the actual \
failure mechanism is*, then design a novel pattern that directly targets it.
   - Think beyond the current categories. Consider strategies like: \
restructuring the entire agent communication topology, introducing new \
meta-cognitive prompts, changing the information flow architecture, \
combining multiple atomic fixes into a single compound pattern, etc.
   - A creative new pattern that might work is more valuable than a refined \
old pattern that has already proven ineffective.

5. Summarize what you changed and why.

## Rules
- Each pattern must have a unique pattern_id (lowercase_with_underscores).
- Descriptions should be actionable and specific — they get injected directly \
into the repair LLM's prompt as repair directions. Vague descriptions like \
"improve the system" are useless; concrete ones like "add a carry_data=true \
edge from ProblemSolver to Verifier so the original problem statement is \
preserved across loop iterations" actually guide the repair.
- target_components must be from: agent_prompt, edge_carry_data, \
edge_condition, edge_missing, loop_config, node_config, other.
- Do not remove all patterns — keep at least 3 effective patterns.
- Focus on FM group {fm_group}."""

_EVOLVE_USER_PROMPT = """\
The repair patterns for FM group {fm_group} have repeatedly failed \
({n_failures} consecutive failures).

## Accumulated Failures
{failures_text}

## Your Task
Read these files in order:
1. `jacobian_report.md` — empirical success rates and divergence data.
2. `catalog.json` — current patterns.
3. `failure_summary.md` — what went wrong.

Then decide how aggressively to change the catalog:
- If some patterns still have decent success rates, refine and promote.
- If *everything* is failing, be bold: disable the dead weight and **invent \
new approaches** based on what the failure traces actually show. The current \
catalog clearly isn't solving this failure mode — incremental refinement of \
broken patterns won't help. Think from first principles about what repair \
strategy could actually fix this class of MAS failure."""


# Tool schemas for catalog evolution
_CATALOG_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "add_pattern",
            "description": "Add a new repair pattern to the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern_id": {
                        "type": "string",
                        "description": "Unique ID (lowercase_with_underscores).",
                    },
                    "name": {
                        "type": "string",
                        "description": "Human-readable name for the pattern.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Actionable description injected into LLM repair prompts.",
                    },
                    "target_components": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "DAG components this pattern targets.",
                    },
                },
                "required": ["pattern_id", "name", "description", "target_components"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_pattern",
            "description": "Update an existing pattern in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern_id": {
                        "type": "string",
                        "description": "ID of the pattern to update.",
                    },
                    "name": {"type": "string", "description": "New name (optional)."},
                    "description": {"type": "string", "description": "New description (optional)."},
                    "target_components": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "New target components (optional).",
                    },
                    "effective": {
                        "type": "boolean",
                        "description": "Set to false to disable this pattern.",
                    },
                },
                "required": ["pattern_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash command to inspect files.",
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


class _CatalogContext:
    """Manages the catalog and temp workspace for tool execution."""

    def __init__(self, catalog: PatternCatalog):
        self.catalog = catalog
        self._tmpdir = tempfile.mkdtemp(prefix="optpilot_catalog_")
        # Write catalog snapshot
        catalog_data = {pid: asdict(p) for pid, p in catalog.items()}
        self._write_file("catalog.json", json.dumps(catalog_data, indent=2, ensure_ascii=False))

    def execute_tool(self, name: str, args: dict) -> str:
        if name == "add_pattern":
            return self._add_pattern(args)
        elif name == "update_pattern":
            return self._update_pattern(args)
        elif name == "bash":
            return self._bash(args)
        return f"Error: unknown tool '{name}'"

    def _add_pattern(self, args: dict) -> str:
        pattern_id = args.get("pattern_id", "")
        if not pattern_id:
            return "Error: pattern_id is required."
        if pattern_id in self.catalog:
            return f"Error: pattern '{pattern_id}' already exists. Use update_pattern instead."

        pattern = RepairPattern(
            pattern_id=pattern_id,
            name=args.get("name", pattern_id),
            description=args.get("description", ""),
            target_components=args.get("target_components", []),
            effective=True,
        )
        self.catalog.add_pattern(pattern)
        return f"OK. Added pattern '{pattern_id}'."

    def _update_pattern(self, args: dict) -> str:
        pattern_id = args.get("pattern_id", "")
        if not pattern_id:
            return "Error: pattern_id is required."

        success = self.catalog.update_pattern(
            pattern_id,
            name=args.get("name"),
            description=args.get("description"),
            target_components=args.get("target_components"),
            effective=args.get("effective"),
        )
        if not success:
            return f"Error: pattern '{pattern_id}' not found."
        return f"OK. Updated pattern '{pattern_id}'."

    def _bash(self, args: dict) -> str:
        command = args.get("command", "")
        if not command:
            return "Error: command is empty."
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True,
                timeout=30, cwd=self._tmpdir,
            )
            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}"
            if result.returncode != 0:
                output += f"\n(exit code: {result.returncode})"
            if len(output) > 4000:
                output = output[:4000] + "\n... (truncated)"
            return output or "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: command timed out."
        except Exception as e:
            return f"Error: {e}"

    def _write_file(self, filename: str, content: str) -> str:
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


def _build_meta_context(fm_group: str) -> str:
    negatives_path = NEGATIVES_DIR / f"negatives_{fm_group}.json"
    meta_trace_dir = _META_EVOLVE_TRACE_DIR / fm_group

    lines = [
        "# Catalog Evolution Context Index",
        "",
        "Read this file first, then use bash to inspect only the files you need.",
        "",
        f"## Target FM Group: {fm_group}",
        "",
        "## Repository Files To Inspect With Bash",
        f"- {_REPAIR_PATTERNS_PATH}: repair pattern definitions and PatternCatalog class",
        f"- {_DAG_EXECUTOR_PATH}: runtime semantics of the repaired MAS DAG",
        f"- {_DAG_CORE_PATH}: MASDAG schema and parsing behavior",
        f"- {_ARCHITECTURE_PATH}: current system architecture",
        f"- {_PROJECT_GOAL_PATH}: project mission and scope",
        f"- {_PROGRESS_PATH}: current milestones and status",
        "",
        "## Persistent Experience",
        f"- {negatives_path}: persisted negative lessons for this FM group",
        f"- {meta_trace_dir}: previous catalog evolution traces",
        "",
        "Use bash to inspect these files before making changes.",
    ]
    return "\n".join(lines)


def _persist_trace(
    fm_group: str,
    ctx: _CatalogContext,
    final_msgs: list[dict],
    negatives: list[ReflectInsight],
) -> str:
    trace_dir = _META_EVOLVE_TRACE_DIR / fm_group
    trace_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    trace_path = trace_dir / f"catalog_evolve_{timestamp}.json"
    payload = {
        "created_at": datetime.now().isoformat(),
        "fm_group": fm_group,
        "messages": final_msgs,
        "negatives": [asdict(neg) for neg in negatives],
        "final_catalog": {pid: asdict(p) for pid, p in ctx.catalog.items()},
    }
    trace_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return str(trace_path)


class CatalogEvolver:
    """Evolve the pattern catalog when repair patterns repeatedly fail."""

    def __init__(
        self,
        catalog: PatternCatalog | None = None,
        jacobian: RepairJacobian | None = None,
    ):
        self.catalog = catalog or PatternCatalog()
        self.jacobian = jacobian
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

    def evolve_catalog(
        self,
        fm_group: str,
        negatives: list[ReflectInsight],
    ) -> bool:
        """Evolve the catalog via LLM tool-calling.  Returns True on success."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    asyncio.run,
                    self._aevolve_catalog(fm_group, negatives),
                )
                return future.result()
        else:
            return asyncio.run(self._aevolve_catalog(fm_group, negatives))

    async def _aevolve_catalog(
        self,
        fm_group: str,
        negatives: list[ReflectInsight],
    ) -> bool:
        failures_text = "\n".join(
            f"Round {i+1}: changes=[{', '.join(n.changes_attempted[:2])}] "
            f"FM {n.before_fm_rate:.2f}→{n.after_fm_rate:.2f}, "
            f"pass {n.before_pass_rate:.3f}→{n.after_pass_rate:.3f}. "
            f"Reason: {n.failure_reason}. Lesson: {n.lesson}"
            for i, n in enumerate(negatives[-20:])
        ) or "No specific failures recorded."

        ctx = _CatalogContext(self.catalog)
        ctx._write_file("meta_context.md", _build_meta_context(fm_group))
        ctx._write_file("failure_summary.md", _build_failure_summary(negatives))
        ctx._write_file(
            "failures.json",
            json.dumps([asdict(neg) for neg in negatives], ensure_ascii=False, indent=2),
        )
        # Jacobian experience report — pattern success rates + divergence data
        if self.jacobian:
            ctx._write_file("jacobian_report.md", self.jacobian.format_evolution_report(fm_group))

        final_msgs = await acall_llm_with_tools(
            messages=[
                {"role": "system", "content": _EVOLVE_SYSTEM_PROMPT.format(fm_group=fm_group)},
                {"role": "user", "content": _EVOLVE_USER_PROMPT.format(
                    fm_group=fm_group,
                    n_failures=len(negatives),
                    failures_text=failures_text,
                )},
            ],
            tools=_CATALOG_TOOL_SCHEMAS,
            tool_executor=ctx.execute_tool,
            max_tokens=META_EVOLVE_MAX_TOKENS,
            max_turns=META_EVOLVE_MAX_TURNS,
        )

        _persist_trace(fm_group, ctx, final_msgs, negatives)

        # Save updated catalog
        self.catalog.save()
        self._failure_counts[fm_group] = 0
        print(f"    Catalog evolved and saved ({len(self.catalog)} patterns).")
        return True
