"""AgentCoder tools — Python code execution for coding agents.

Provides a python_exec tool that agents use to run code snippets,
test their implementations, and debug issues.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any

from optpilot.tools.registry import AsyncToolFn, ToolEntry, openai_tool_schema


class CodeExecutionEnvironment:
    """Sandboxed Python code execution environment."""

    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self.tool_log: list[dict] = []

    async def python_exec(self, args: dict[str, Any]) -> str:
        """Execute Python code and return stdout + stderr."""
        code = args.get("code", "")
        self.tool_log.append({"tool": "python_exec", "code": code[:300]})

        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            stdout = result.stdout[-3000:] if result.stdout else ""
            stderr = result.stderr[-1000:] if result.stderr else ""
            if result.returncode != 0:
                return f"Error (exit {result.returncode}):\n{stderr}\n{stdout}".strip()
            return stdout.strip() if stdout.strip() else "(no output)"
        except subprocess.TimeoutExpired:
            return f"Execution timed out ({self.timeout}s limit)"
        except Exception as e:
            return f"Execution error: {e}"


def build_tools(env: CodeExecutionEnvironment) -> dict[str, ToolEntry]:
    """Build AgentCoder tool registry."""
    return {
        "python_exec": (
            openai_tool_schema(
                name="python_exec",
                description=(
                    "Execute a Python code snippet and return stdout/stderr. "
                    "Use this to test your code implementation, run test cases, "
                    "or debug issues. Timeout: 15 seconds."
                ),
                parameters={
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute",
                        },
                    },
                    "required": ["code"],
                },
            ),
            env.python_exec,
        ),
    }
