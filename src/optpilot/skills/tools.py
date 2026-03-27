"""Tool definitions and execution for Skill repair workflows.

Tools available to the LLM during the evolve step:
  - search_and_replace: modify the MASDAG YAML via targeted text replacement
  - bash: execute shell commands (read files, run Python, validate syntax)
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any

import yaml

from optpilot.dag.core import MASDAG


# ── Tool Schemas (OpenAI function-calling format) ────────────────────

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_and_replace",
            "description": (
                "Replace a text segment in the MASDAG YAML configuration. "
                "The old_str must match exactly (including whitespace/indentation). "
                "Use this to modify agent prompts, roles, parameters, edge conditions, "
                "loop limits, or any other part of the DAG config."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "old_str": {
                        "type": "string",
                        "description": "The exact text to find in the current YAML. Must be unique.",
                    },
                    "new_str": {
                        "type": "string",
                        "description": "The replacement text.",
                    },
                },
                "required": ["old_str", "new_str"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Execute a bash command and return stdout+stderr. "
                "Use this to: read the current YAML (cat), run Python to test logic, "
                "validate YAML syntax, or perform any computation needed for the repair."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute.",
                    },
                },
                "required": ["command"],
            },
        },
    },
]


# ── Change record ────────────────────────────────────────────────────

@dataclass
class ChangeRecord:
    """A single search-and-replace operation, replayable by the Forger."""

    old_str: str
    new_str: str
    source: str = "search_and_replace"  # or "bash"

    def preview(self) -> str:
        return f"-{self.old_str[:120]}\n+{self.new_str[:120]}"


# ── Tool Execution Context ───────────────────────────────────────────

@dataclass
class ToolContext:
    """Mutable state shared across tool calls within one evolve step."""

    yaml_content: str
    changes: list[ChangeRecord] = field(default_factory=list)
    _tmpdir: str | None = None

    @classmethod
    def from_dag(cls, dag: MASDAG) -> "ToolContext":
        content = yaml.dump(dag.to_dict(), allow_unicode=True, sort_keys=False)
        return cls(yaml_content=content)

    @property
    def tmpdir(self) -> str:
        if self._tmpdir is None:
            self._tmpdir = tempfile.mkdtemp(prefix="optpilot_skill_")
            self._write_yaml_file()
        return self._tmpdir

    def _write_yaml_file(self) -> None:
        """Write current YAML to tmpdir for bash access."""
        import os
        path = os.path.join(self.tmpdir, "dag.yaml")
        with open(path, "w") as f:
            f.write(self.yaml_content)

    def to_dag(self) -> MASDAG:
        """Parse current YAML content into a MASDAG."""
        parsed = yaml.safe_load(self.yaml_content)
        return MASDAG.from_dict(parsed)

    def change_previews(self) -> list[str]:
        """Short diff previews for logging."""
        return [c.preview() for c in self.changes]


def execute_tool(name: str, arguments: dict[str, Any], ctx: ToolContext) -> str:
    """Execute a tool call and return the result string."""
    if name == "search_and_replace":
        return _exec_search_and_replace(arguments, ctx)
    elif name == "bash":
        return _exec_bash(arguments, ctx)
    else:
        return f"Error: unknown tool '{name}'"


def _exec_search_and_replace(args: dict[str, Any], ctx: ToolContext) -> str:
    old_str = args.get("old_str", "")
    new_str = args.get("new_str", "")

    if not old_str:
        return "Error: old_str is empty."

    count = ctx.yaml_content.count(old_str)
    if count == 0:
        return (
            f"Error: old_str not found in YAML. "
            f"Make sure whitespace and indentation match exactly. "
            f"Use the bash tool to run 'cat $DAG_FILE' to see the current YAML."
        )
    if count > 1:
        return (
            f"Error: old_str matches {count} locations. "
            f"Provide more surrounding context to make the match unique."
        )

    ctx.yaml_content = ctx.yaml_content.replace(old_str, new_str, 1)
    ctx._write_yaml_file()

    # Validate the YAML is still parseable
    try:
        yaml.safe_load(ctx.yaml_content)
    except yaml.YAMLError as e:
        ctx.yaml_content = ctx.yaml_content.replace(new_str, old_str, 1)
        ctx._write_yaml_file()
        return f"Error: replacement produced invalid YAML: {e}. Change was reverted."

    ctx.changes.append(ChangeRecord(old_str=old_str, new_str=new_str, source="search_and_replace"))
    return f"OK. Replaced 1 occurrence. Current YAML is {len(ctx.yaml_content)} chars."


def _exec_bash(args: dict[str, Any], ctx: ToolContext) -> str:
    command = args.get("command", "")
    if not command:
        return "Error: command is empty."

    import os
    dag_file = os.path.join(ctx.tmpdir, "dag.yaml")
    env = {**os.environ, "DAG_FILE": dag_file}

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=ctx.tmpdir,
            env=env,
        )
        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        if result.returncode != 0:
            output += f"\n(exit code: {result.returncode})"
        if len(output) > 4000:
            output = output[:4000] + "\n... (truncated)"

        # Sync back: if bash modified the YAML file, update ctx
        if os.path.exists(dag_file):
            with open(dag_file) as f:
                new_content = f.read()
            if new_content != ctx.yaml_content:
                try:
                    yaml.safe_load(new_content)
                    old_content = ctx.yaml_content
                    ctx.yaml_content = new_content
                    ctx.changes.append(ChangeRecord(
                        old_str=old_content, new_str=new_content, source="bash",
                    ))
                except yaml.YAMLError:
                    ctx._write_yaml_file()

        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: command timed out (30s limit)."
    except Exception as e:
        return f"Error: {e}"
