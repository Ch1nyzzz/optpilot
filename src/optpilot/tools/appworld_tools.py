"""AppWorld tools — wraps the official AppWorld environment.

Each task gets its own AppWorld instance with real API server and database.
Agents interact via Python code execution through the `execute_code` tool.
"""

from __future__ import annotations

import json
from typing import Any

from optpilot.tools.registry import AsyncToolFn, ToolEntry, openai_tool_schema


class AppWorldWrapper:
    """Wraps the official AppWorld environment for one task."""

    def __init__(self, task_id: str, experiment_name: str = "optpilot"):
        from appworld.environment import AppWorld
        self.aw = AppWorld(
            task_id=task_id,
            experiment_name=experiment_name,
            load_ground_truth=False,
            raise_on_failure=False,
        )
        self.task_instruction = self.aw.task.instruction
        self.call_log: list[dict] = []

    async def execute_code(self, args: dict[str, Any]) -> str:
        """Execute Python code in the AppWorld environment.

        The code has access to `apis` object with services:
        admin, amazon, file_system, gmail, phone, simple_note,
        splitwise, spotify, supervisor.
        """
        code = args.get("code", "")
        self.call_log.append({"tool": "execute_code", "code": code[:200]})
        try:
            result = self.aw.execute(code)
            return str(result)[:3000]
        except Exception as e:
            return f"Execution error: {e}"

    async def list_apis(self, args: dict[str, Any]) -> str:
        """List available API services and their methods."""
        service = args.get("service", "")
        if service:
            try:
                result = self.aw.execute(f"print(dir(apis.{service}))")
                return str(result)
            except Exception as e:
                return f"Error: {e}"
        # List all services
        return json.dumps([
            "admin", "amazon", "file_system", "gmail", "phone",
            "simple_note", "splitwise", "spotify", "supervisor",
        ])

    async def submit_answer(self, args: dict[str, Any]) -> str:
        """Submit the final answer."""
        answer = args.get("answer", "")
        self.call_log.append({"tool": "submit_answer", "answer": answer})
        return json.dumps({"status": "submitted", "answer": answer})

    def close(self):
        try:
            self.aw.close()
        except Exception:
            pass


def build_tools(env: AppWorldWrapper) -> dict[str, ToolEntry]:
    """Build AppWorld tool registry."""
    return {
        "execute_code": (
            openai_tool_schema(
                name="execute_code",
                description=(
                    "Execute Python code in the AppWorld environment. "
                    "You have access to `apis` object with services: "
                    "admin, amazon, file_system, gmail, phone, simple_note, "
                    "splitwise, spotify, supervisor. "
                    "Example: apis.spotify.search_song(query='hello') "
                    "Use print() to see results."
                ),
                parameters={
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute (has access to `apis`)",
                        },
                    },
                    "required": ["code"],
                },
            ),
            env.execute_code,
        ),
        "list_apis": (
            openai_tool_schema(
                name="list_apis",
                description="List available API services or methods of a specific service.",
                parameters={
                    "properties": {
                        "service": {
                            "type": "string",
                            "description": "Service name (e.g., 'spotify', 'gmail'). Empty to list all services.",
                        },
                    },
                },
            ),
            env.list_apis,
        ),
        "submit_answer": (
            openai_tool_schema(
                name="submit_answer",
                description="Submit your final answer to the task.",
                parameters={
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "The final answer",
                        },
                    },
                    "required": ["answer"],
                },
            ),
            env.submit_answer,
        ),
    }
