"""HyperAgent-style code operation tools for SWE-bench tasks.

Each task gets a real git checkout of the target repo at the specified
base_commit. Agents use file read/write/search and command execution
tools to navigate, understand, and fix code.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from optpilot.tools.registry import AsyncToolFn, ToolEntry, openai_tool_schema

_REPO_CACHE_DIR = Path("/tmp/swebench_repos")


class CodeEnvironment:
    """Real code environment for one SWE-bench task.

    Clones the repo and checks out the base_commit so agents work on
    real code.
    """

    def __init__(
        self,
        workdir: str | None = None,
        repo: str = "",
        base_commit: str = "",
    ):
        self.edit_log: list[dict] = []

        if workdir:
            self.workdir = Path(workdir)
        elif repo and base_commit:
            self.workdir = self._setup_repo(repo, base_commit)
        else:
            import tempfile
            self.workdir = Path(tempfile.mkdtemp(prefix="swebench_"))

    def _setup_repo(self, repo: str, base_commit: str) -> Path:
        """Clone repo and checkout the base commit."""
        _REPO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        repo_name = repo.replace("/", "__")
        cache_dir = _REPO_CACHE_DIR / repo_name

        # Clone if not cached (full clone, not shallow)
        if not cache_dir.exists():
            print(f"    Cloning {repo}...")
            subprocess.run(
                ["git", "clone", "--quiet", f"https://github.com/{repo}.git", str(cache_dir)],
                capture_output=True, timeout=300,
            )

        # Fetch the specific commit if not available
        check = subprocess.run(
            ["git", "cat-file", "-t", base_commit],
            cwd=str(cache_dir), capture_output=True, text=True,
        )
        if check.returncode != 0:
            subprocess.run(
                ["git", "fetch", "--quiet", "origin", base_commit],
                cwd=str(cache_dir), capture_output=True, timeout=120,
            )

        # Create a worktree for this specific commit
        import tempfile
        work_dir = Path(tempfile.mkdtemp(prefix=f"swe_{repo_name}_"))
        result = subprocess.run(
            ["git", "worktree", "add", "--detach", str(work_dir), base_commit],
            cwd=str(cache_dir), capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0 or not any(work_dir.iterdir()):
            # Fallback: direct clone at specific commit
            import shutil
            if work_dir.exists():
                shutil.rmtree(work_dir)
            shutil.copytree(cache_dir, work_dir, symlinks=True)
            subprocess.run(
                ["git", "checkout", "--quiet", "--force", base_commit],
                cwd=str(work_dir), capture_output=True, timeout=30,
            )

        return work_dir

    async def read_file(self, args: dict[str, Any]) -> str:
        """Read a file from the working directory."""
        path = args.get("path", "")
        start_line = args.get("start_line", 1)
        end_line = args.get("end_line", 0)
        full_path = self.workdir / path
        if not full_path.exists():
            return json.dumps({"error": f"File not found: {path}"})
        try:
            lines = full_path.read_text(encoding="utf-8", errors="replace").splitlines()
            if end_line > 0:
                lines = lines[max(0, start_line - 1):end_line]
            else:
                lines = lines[max(0, start_line - 1):start_line + 99]
            numbered = [f"{start_line + i}: {line}" for i, line in enumerate(lines)]
            return "\n".join(numbered)
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def edit_file(self, args: dict[str, Any]) -> str:
        """Edit a file: find old_str and replace with new_str."""
        path = args.get("path", "")
        old_str = args.get("old_str", "")
        new_str = args.get("new_str", "")
        full_path = self.workdir / path
        if not full_path.exists():
            return json.dumps({"error": f"File not found: {path}"})
        try:
            content = full_path.read_text(encoding="utf-8")
            if old_str not in content:
                return json.dumps({"error": "old_str not found in file"})
            new_content = content.replace(old_str, new_str, 1)
            full_path.write_text(new_content, encoding="utf-8")
            self.edit_log.append({"path": path, "old_str": old_str, "new_str": new_str})
            return json.dumps({"status": "ok", "path": path})
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def search_code(self, args: dict[str, Any]) -> str:
        """Search for a pattern in the codebase."""
        pattern = args.get("pattern", "")
        path = args.get("path", ".")
        search_dir = self.workdir / path
        try:
            result = subprocess.run(
                ["grep", "-rn", "--include=*.py", pattern, str(search_dir)],
                capture_output=True, text=True, timeout=10,
            )
            lines = result.stdout.strip().splitlines()[:20]
            lines = [line.replace(str(self.workdir) + "/", "") for line in lines]
            return "\n".join(lines) if lines else "No matches found."
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def list_files(self, args: dict[str, Any]) -> str:
        """List files in a directory."""
        path = args.get("path", ".")
        target = self.workdir / path
        if not target.exists():
            return json.dumps({"error": f"Path not found: {path}"})
        try:
            if target.is_file():
                return str(path)
            entries = sorted(target.iterdir())[:50]
            lines = []
            for e in entries:
                rel = e.relative_to(self.workdir)
                suffix = "/" if e.is_dir() else ""
                lines.append(f"{rel}{suffix}")
            return "\n".join(lines)
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def run_command(self, args: dict[str, Any]) -> str:
        """Run a shell command in the working directory."""
        command = args.get("command", "")
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True,
                timeout=30, cwd=str(self.workdir),
            )
            output = result.stdout[-2000:] if result.stdout else ""
            stderr = result.stderr[-1000:] if result.stderr else ""
            return f"exit_code: {result.returncode}\nstdout:\n{output}\nstderr:\n{stderr}".strip()
        except subprocess.TimeoutExpired:
            return "Command timed out (30s limit)"
        except Exception as e:
            return json.dumps({"error": str(e)})


def build_tools(env: CodeEnvironment) -> dict[str, ToolEntry]:
    """Build HyperAgent tool registry."""
    return {
        "read_file": (
            openai_tool_schema(
                name="read_file",
                description="Read lines from a file in the repository.",
                parameters={
                    "properties": {
                        "path": {"type": "string", "description": "Relative file path"},
                        "start_line": {"type": "integer", "description": "Start line (1-indexed, default 1)"},
                        "end_line": {"type": "integer", "description": "End line (0 = auto, default 100 lines)"},
                    },
                    "required": ["path"],
                },
            ),
            env.read_file,
        ),
        "edit_file": (
            openai_tool_schema(
                name="edit_file",
                description="Edit a file by replacing old_str with new_str.",
                parameters={
                    "properties": {
                        "path": {"type": "string", "description": "Relative file path"},
                        "old_str": {"type": "string", "description": "Exact text to find"},
                        "new_str": {"type": "string", "description": "Replacement text"},
                    },
                    "required": ["path", "old_str", "new_str"],
                },
            ),
            env.edit_file,
        ),
        "search_code": (
            openai_tool_schema(
                name="search_code",
                description="Search for a pattern in Python files.",
                parameters={
                    "properties": {
                        "pattern": {"type": "string", "description": "Search pattern (grep regex)"},
                        "path": {"type": "string", "description": "Directory to search (default: .)"},
                    },
                    "required": ["pattern"],
                },
            ),
            env.search_code,
        ),
        "list_files": (
            openai_tool_schema(
                name="list_files",
                description="List files and directories.",
                parameters={
                    "properties": {
                        "path": {"type": "string", "description": "Directory path (default: .)"},
                    },
                },
            ),
            env.list_files,
        ),
        "run_command": (
            openai_tool_schema(
                name="run_command",
                description="Run a shell command (e.g., python tests, git diff). 30s timeout.",
                parameters={
                    "properties": {
                        "command": {"type": "string", "description": "Shell command to execute"},
                    },
                    "required": ["command"],
                },
            ),
            env.run_command,
        ),
    }
