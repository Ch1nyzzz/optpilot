"""Magentic-One-style general-purpose tools.

Provides web search, document reading, and calculation tools for
general agent tasks (GAIA-style). Web search uses LLM knowledge
as a proxy since we don't have real web access during evolution.
"""

from __future__ import annotations

import json
import math
import re
import sys
from typing import Any

from optpilot.tools.registry import AsyncToolFn, ToolEntry, openai_tool_schema


class GeneralEnvironment:
    """Environment for general-purpose agent tasks."""

    def __init__(self, context_docs: dict[str, str] | None = None):
        self.context_docs = context_docs or {}
        self.tool_log: list[dict] = []

    async def web_search(self, args: dict[str, Any]) -> str:
        """Real web search via DuckDuckGo."""
        query = args.get("query", "")
        self.tool_log.append({"tool": "web_search", "query": query})

        try:
            from ddgs import DDGS
            results = DDGS().text(query, max_results=5)
            formatted = [
                {"title": r.get("title", ""), "snippet": r.get("body", "")[:300]}
                for r in results
            ]
            return json.dumps({"results": formatted}, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": f"Web search failed: {e}"})

    async def read_document(self, args: dict[str, Any]) -> str:
        """Read a web page by URL, or a pre-loaded document by title."""
        title = args.get("title", "")
        url = args.get("url", "")
        self.tool_log.append({"tool": "read_document", "title": title, "url": url})

        # If URL provided, fetch the page
        if url:
            try:
                import httpx
                headers = {"User-Agent": "OptPilot/1.0 (research agent)"}
                async with httpx.AsyncClient(timeout=15, follow_redirects=True, headers=headers) as client:
                    resp = await client.get(url)
                    text = resp.text[:5000]
                    # Basic HTML stripping
                    import re as _re
                    text = _re.sub(r'<[^>]+>', ' ', text)
                    text = _re.sub(r'\s+', ' ', text).strip()
                    return text[:3000]
            except Exception as e:
                return json.dumps({"error": f"Failed to fetch URL: {e}"})

        # Pre-loaded doc lookup
        if title in self.context_docs:
            return self.context_docs[title]
        for doc_title, content in self.context_docs.items():
            if title.lower() in doc_title.lower() or doc_title.lower() in title.lower():
                return content

        return json.dumps({"error": f"Document '{title}' not found. Provide a URL to fetch."})

    async def calculator(self, args: dict[str, Any]) -> str:
        """Evaluate a mathematical expression safely."""
        expression = args.get("expression", "")
        self.tool_log.append({"tool": "calculator", "expression": expression})

        # Safe eval with math module and functions
        allowed_names = {
            k: getattr(math, k) for k in dir(math) if not k.startswith("_")
        }
        allowed_names.update({"abs": abs, "round": round, "int": int, "float": float, "math": math})

        try:
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return json.dumps({"result": result})
        except Exception as e:
            return json.dumps({"error": f"Calculation failed: {e}"})

    async def python_exec(self, args: dict[str, Any]) -> str:
        """Execute a Python snippet and return stdout."""
        code = args.get("code", "")
        self.tool_log.append({"tool": "python_exec", "code": code[:200]})

        import subprocess
        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True, text=True, timeout=10,
            )
            output = result.stdout[-2000:] if result.stdout else ""
            stderr = result.stderr[-500:] if result.stderr else ""
            if result.returncode != 0:
                return f"Error (exit {result.returncode}):\n{stderr}"
            return output.strip() if output.strip() else "(no output)"
        except subprocess.TimeoutExpired:
            return "Execution timed out (10s limit)"
        except Exception as e:
            return f"Execution error: {e}"


def build_tools(env: GeneralEnvironment) -> dict[str, ToolEntry]:
    """Build Magentic-One tool registry."""
    return {
        "web_search": (
            openai_tool_schema(
                name="web_search",
                description="Search the web for information. Returns relevant snippets.",
                parameters={
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                    },
                    "required": ["query"],
                },
            ),
            env.web_search,
        ),
        "read_document": (
            openai_tool_schema(
                name="read_document",
                description="Read a web page by URL or a document by title.",
                parameters={
                    "properties": {
                        "url": {"type": "string", "description": "URL to fetch (preferred)"},
                        "title": {"type": "string", "description": "Document title (fallback)"},
                    },
                },
            ),
            env.read_document,
        ),
        "calculator": (
            openai_tool_schema(
                name="calculator",
                description="Evaluate a mathematical expression (supports math.* functions).",
                parameters={
                    "properties": {
                        "expression": {"type": "string", "description": "Math expression (e.g., 'math.sqrt(2) * 3')"},
                    },
                    "required": ["expression"],
                },
            ),
            env.calculator,
        ),
        "python_exec": (
            openai_tool_schema(
                name="python_exec",
                description="Execute a Python code snippet and return stdout. 10s timeout.",
                parameters={
                    "properties": {
                        "code": {"type": "string", "description": "Python code to execute"},
                    },
                    "required": ["code"],
                },
            ),
            env.python_exec,
        ),
    }
