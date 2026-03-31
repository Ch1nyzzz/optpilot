"""Tool registry builder — maps tool names to (schema, executor) pairs.

Each topology module exports a ``build_tools()`` function that returns a
dict[str, tuple[schema_dict, async_executor_fn]].  The DAGExecutor receives
this dict as ``async_tool_registry``.
"""

from __future__ import annotations

from typing import Any, Callable, Coroutine

AsyncToolFn = Callable[[dict[str, Any]], Coroutine[Any, Any, str]]
ToolEntry = tuple[dict[str, Any], AsyncToolFn]


def openai_tool_schema(
    name: str,
    description: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Build an OpenAI-compatible tool schema."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                **parameters,
            },
        },
    }
