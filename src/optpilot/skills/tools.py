"""Tool definitions and execution for Skill repair workflows.

Tools available to the LLM during the evolve step:
  - search_and_replace: modify the build_dag() Python source via targeted text replacement
  - bash: execute shell commands (read files, run Python, validate syntax)
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass, field
from typing import Any

from optpilot.dag.core import MASDAG


# ── Tool Schemas (OpenAI function-calling format) ────────────────────

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_and_replace",
            "description": (
                "Replace a text segment in the build_dag() Python source code. "
                "The old_str must match exactly (including whitespace/indentation). "
                "Use this to modify agent prompts, roles, parameters, edge conditions, "
                "loop limits, or any other part of the DAG configuration."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "old_str": {
                        "type": "string",
                        "description": "The exact text to find in the current Python source. Must be unique.",
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
                "Use this to: read the current Python source (cat $DAG_FILE), "
                "run Python to test logic, validate the DAG, or perform any "
                "computation needed for the repair."
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


# ── DAG ↔ Python source conversion ──────────────────────────────────

def _sanitize_var_name(node_id: str) -> str:
    """Convert a node ID like 'Agent_Problem_Solver' to a valid Python variable name."""
    name = re.sub(r"[^a-zA-Z0-9_]", "_", node_id)
    name = name.strip("_").lower()
    if not name or name[0].isdigit():
        name = "node_" + name
    return name


def _format_python_string(s: str, indent: int = 4) -> str:
    """Format a string value as a Python multi-line parenthesized expression.

    Short strings (≤80 chars, single line) are returned as a simple repr().
    Longer strings are split into a parenthesized concatenation for readability,
    with each source line ending with a literal \\n inside the string.
    """
    if "\n" not in s and len(s) <= 80:
        return repr(s)

    # Split into lines at newlines. Each segment (except the last) gets \n appended
    # back so the reconstructed string is identical to the original.
    prefix = " " * indent
    segments = s.split("\n")
    parts = []
    for i, seg in enumerate(segments):
        if i < len(segments) - 1:
            # This segment originally ended with \n — include it in the string literal
            parts.append(f"{prefix}    {repr(seg + chr(10))}")
        else:
            parts.append(f"{prefix}    {repr(seg)}")

    return "(\n" + "\n".join(parts) + "\n" + prefix + ")"


def _repr_value(value: Any, indent: int = 4) -> str:
    """Repr a value with proper formatting for nested dicts/lists."""
    if isinstance(value, str):
        if len(value) > 80 or "\n" in value:
            return _format_python_string(value, indent)
        return repr(value)
    if isinstance(value, bool):
        return repr(value)
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, dict):
        if not value:
            return "{}"
        prefix = " " * indent
        inner = " " * (indent + 4)
        items = []
        for k, v in value.items():
            items.append(f"{inner}{repr(k)}: {_repr_value(v, indent + 4)}")
        return "{\n" + ",\n".join(items) + ",\n" + prefix + "}"
    if isinstance(value, list):
        if not value:
            return "[]"
        # Short simple lists on one line
        simple = all(isinstance(v, (str, int, float, bool)) for v in value)
        if simple and len(repr(value)) <= 80:
            return repr(value)
        prefix = " " * indent
        inner = " " * (indent + 4)
        items = []
        for v in value:
            items.append(f"{inner}{_repr_value(v, indent + 4)}")
        return "[\n" + ",\n".join(items) + ",\n" + prefix + "]"
    return repr(value)


def dag_to_python(dag: MASDAG) -> str:
    """Serialize a MASDAG to a build_dag() Python function.

    The output follows the same structure as experiments/openevolve_initial_dag.py:
    - Agent prompts and literal contents extracted into named variables
    - Nodes list with variable references
    - Edges list
    - Return dict with dag_id, nodes, edges, metadata
    """
    d = dag.to_dict()
    lines: list[str] = []

    lines.append("def build_dag():")
    lines.append('    """Build the MASDAG configuration."""')
    lines.append("")

    # ── Extract long strings into named variables ──
    prompt_vars: dict[str, str] = {}  # node_id -> var_name (for role)
    content_vars: dict[str, str] = {}  # node_id -> var_name (for config.content)

    for node in d["nodes"]:
        node_id = node["id"]
        base = _sanitize_var_name(node_id)

        # Extract role (agent prompt) into variable
        if node.get("role") and len(node["role"]) > 40:
            var_name = f"{base}_prompt"
            prompt_vars[node_id] = var_name
            lines.append(f"    {var_name} = {_format_python_string(node['role'], 4)}")
            lines.append("")

        # Extract config.content (literal content) into variable
        content = node.get("config", {}).get("content")
        if content and len(content) > 40:
            var_name = f"{base}_content"
            content_vars[node_id] = var_name
            lines.append(f"    {var_name} = {_format_python_string(content, 4)}")
            lines.append("")

    # ── Nodes list ──
    lines.append("    nodes = [")
    for node in d["nodes"]:
        node_id = node["id"]
        # Build the node dict, replacing inline strings with variable references
        parts = []
        parts.append(f'"id": {repr(node_id)}')
        parts.append(f'"type": {repr(node["type"])}')

        if node.get("role"):
            if node_id in prompt_vars:
                parts.append(f'"role": {prompt_vars[node_id]}')
            else:
                parts.append(f'"role": {repr(node["role"])}')

        if node.get("prompt"):
            parts.append(f'"prompt": {repr(node["prompt"])}')

        if node.get("config"):
            config = dict(node["config"])
            # If content was extracted to a variable, use the variable reference
            if node_id in content_vars and "content" in config:
                config_without_content = {k: v for k, v in config.items() if k != "content"}
                if config_without_content:
                    parts.append(f'"config": {{**{_repr_value(config_without_content, 12)}, "content": {content_vars[node_id]}}}')
                else:
                    parts.append(f'"config": {{"content": {content_vars[node_id]}}}')
            else:
                parts.append(f'"config": {_repr_value(config, 12)}')

        inner = ", ".join(parts)
        # Decide single-line or multi-line based on length
        candidate = f"        {{{inner}}},"
        if len(candidate) <= 100:
            lines.append(candidate)
        else:
            lines.append("        {")
            for p in parts:
                lines.append(f"            {p},")
            lines.append("        },")

    lines.append("    ]")
    lines.append("")

    # ── Edges list ──
    lines.append("    edges = [")
    for edge in d["edges"]:
        parts = []
        parts.append(f'"from": {repr(edge["from"])}')
        parts.append(f'"to": {repr(edge["to"])}')
        # Only include non-default fields
        for key in ("trigger", "condition", "carry_data"):
            if key in edge:
                parts.append(f'{repr(key)}: {_repr_value(edge[key], 12)}')
        # Extra fields (loop, etc.)
        for key, val in edge.items():
            if key not in ("from", "to", "trigger", "condition", "carry_data"):
                parts.append(f'{repr(key)}: {_repr_value(val, 12)}')

        inner = ", ".join(parts)
        candidate = f"        {{{inner}}},"
        if len(candidate) <= 100:
            lines.append(candidate)
        else:
            lines.append("        {")
            for p in parts:
                lines.append(f"            {p},")
            lines.append("        },")

    lines.append("    ]")
    lines.append("")

    # ── Return statement ──
    lines.append("    return {")
    lines.append(f'        "dag_id": {repr(d["dag_id"])},')
    lines.append('        "nodes": nodes,')
    lines.append('        "edges": edges,')
    if d.get("metadata"):
        lines.append(f'        "metadata": {_repr_value(d["metadata"], 8)},')
    lines.append("    }")

    return "\n".join(lines) + "\n"


def python_source_to_dag(source: str) -> MASDAG:
    """Execute Python source containing build_dag() and return the MASDAG.

    Raises ValueError on syntax errors, missing build_dag(), or invalid DAG dict.
    """
    namespace: dict[str, Any] = {}
    try:
        exec(source, namespace)
    except SyntaxError as e:
        raise ValueError(f"Python syntax error: {e}") from e
    except Exception as e:
        raise ValueError(f"Python execution error: {e}") from e

    if "build_dag" not in namespace:
        raise ValueError("Python source must define a build_dag() function")

    try:
        dag_dict = namespace["build_dag"]()
    except Exception as e:
        raise ValueError(f"build_dag() raised an error: {e}") from e

    if not isinstance(dag_dict, dict):
        raise ValueError(f"build_dag() must return a dict, got {type(dag_dict).__name__}")

    return MASDAG.from_dict(dag_dict)


# ── Tool Execution Context ───────────────────────────────────────────

@dataclass
class ToolContext:
    """Mutable state shared across tool calls within one evolve step."""

    python_source: str
    changes: list[ChangeRecord] = field(default_factory=list)
    _tmpdir: str | None = None

    @classmethod
    def from_dag(cls, dag: MASDAG) -> "ToolContext":
        source = dag_to_python(dag)
        return cls(python_source=source)

    @property
    def tmpdir(self) -> str:
        if self._tmpdir is None:
            self._tmpdir = tempfile.mkdtemp(prefix="optpilot_skill_")
            self._write_python_file()
        return self._tmpdir

    def _write_python_file(self) -> None:
        """Write current Python source to tmpdir for bash access."""
        import os
        path = os.path.join(self.tmpdir, "dag.py")
        with open(path, "w") as f:
            f.write(self.python_source)

    def to_dag(self) -> MASDAG:
        """Parse current Python source into a MASDAG."""
        return python_source_to_dag(self.python_source)

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

    count = ctx.python_source.count(old_str)
    if count == 0:
        return (
            f"Error: old_str not found in Python source. "
            f"Make sure whitespace and indentation match exactly. "
            f"Use the bash tool to run 'cat $DAG_FILE' to see the current source."
        )
    if count > 1:
        return (
            f"Error: old_str matches {count} locations. "
            f"Provide more surrounding context to make the match unique."
        )

    ctx.python_source = ctx.python_source.replace(old_str, new_str, 1)
    ctx._write_python_file()

    # Validate the Python source is still executable and produces a valid DAG
    try:
        python_source_to_dag(ctx.python_source)
    except (ValueError, Exception) as e:
        ctx.python_source = ctx.python_source.replace(new_str, old_str, 1)
        ctx._write_python_file()
        return f"Error: replacement produced invalid Python: {e}. Change was reverted."

    ctx.changes.append(ChangeRecord(old_str=old_str, new_str=new_str, source="search_and_replace"))
    return f"OK. Replaced 1 occurrence. Current source is {len(ctx.python_source)} chars."


def _exec_bash(args: dict[str, Any], ctx: ToolContext) -> str:
    command = args.get("command", "")
    if not command:
        return "Error: command is empty."

    import os
    dag_file = os.path.join(ctx.tmpdir, "dag.py")
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

        # Sync back: if bash modified the Python file, update ctx
        if os.path.exists(dag_file):
            with open(dag_file) as f:
                new_content = f.read()
            if new_content != ctx.python_source:
                try:
                    python_source_to_dag(new_content)
                    old_content = ctx.python_source
                    ctx.python_source = new_content
                    ctx.changes.append(ChangeRecord(
                        old_str=old_content, new_str=new_content, source="bash",
                    ))
                except (ValueError, Exception):
                    ctx._write_python_file()

        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: command timed out (30s limit)."
    except Exception as e:
        return f"Error: {e}"
