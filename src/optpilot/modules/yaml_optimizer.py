"""YAMLOptimizer - YAML-level MAS optimizer.

Bypasses the DAG abstraction; the Optimizer sees the full YAML config
and outputs a modified YAML based on diagnosis results.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from optpilot.data.fm_taxonomy_6group import GROUP_DEFINITIONS as FM_DEFINITIONS
from optpilot.llm import acall_llm, call_llm
from optpilot.models import FMProfile, MASTrace
from optpilot.repair_utils import extract_fenced_block, extract_preface


OPTIMIZE_PROMPT = """\
You are a multi-agent system (MAS) architecture optimizer.

Modify the YAML configuration below to fix the diagnosed faults. Make only targeted changes -- do not alter unrelated parts.

## Current YAML Configuration
```yaml
{yaml_content}
```

## Fault Diagnosis
{diagnosis_text}

## Fault Trace (reference)
{trace_excerpt}
{skills_section}
## Allowed Modifications
1. Agent prompt (nodes[].config.role) -- change agent behavior
2. Tooling (nodes[].config.tooling) -- add/remove available tools
3. Model parameters (nodes[].config.params) -- temperature, max_tokens, etc.
4. Add agent -- new node + corresponding edges
5. Remove agent -- delete node + clean up edges
6. Message flow (edges) -- trigger, condition, carry_data, clear_context
7. Termination conditions (edges[].condition) -- keyword matching rules
8. Loop limits (loop_counter.config.max_iterations)
9. Literal prompts (literal nodes' config.content)

## Output Format
1. Analysis: explain your reasoning and proposed changes (max 200 words)
2. Modified YAML: output the COMPLETE modified YAML wrapped in ```yaml ... ```
   - Must be valid, engine-executable YAML
   - Only make changes that address the diagnosed faults
   - Preserve vars, version, and other metadata unchanged"""

SKILLS_SECTION_TEMPLATE = """
## Reference Skills (from experience library, for guidance only)
{skills_text}
"""


def _build_diagnosis_text(profile: FMProfile) -> str:
    """Format all active FM diagnosis info from an FMProfile."""
    lines = []
    for fm_id in profile.active_fm_ids():
        fm_info = FM_DEFINITIONS.get(fm_id, {})
        loc = profile.localization.get(fm_id)
        lines.append(f"### FM-{fm_id}: {fm_info.get('name', '?')}")
        lines.append(f"- Definition: {fm_info.get('description', '?')}")
        if loc:
            lines.append(f"- Agent: {loc.agent}")
            lines.append(f"- Step: {loc.step}")
            lines.append(f"- Root Cause: {loc.root_cause}")
            lines.append(f"- Context: {loc.context}")
        lines.append("")
    return "\n".join(lines) if lines else "No active FMs diagnosed."


class YAMLOptimizer:
    """YAML-level MAS optimizer."""

    def __init__(self, library=None):
        self.library = library

    def optimize(
        self,
        yaml_path: str | Path,
        profile: FMProfile,
        trace: MASTrace,
    ) -> dict:
        """Optimize YAML config based on diagnosis results.

        Args:
            yaml_path: Path to the current YAML config file.
            profile: Diagnosis results (all active FM localizations).
            trace: Fault trace.

        Returns:
            {
                "original_yaml": str,
                "modified_yaml": str,
                "analysis": str,
                "fm_ids": list[str],
                "yaml_valid": bool,
            }
        """
        yaml_path = Path(yaml_path)
        original_yaml = yaml_path.read_text(encoding="utf-8")
        diagnosis_text = _build_diagnosis_text(profile)
        trace_excerpt = trace.trajectory[:3000]

        # Build optional skills section
        skills_text = self._get_relevant_skills(profile)
        skills_section = (
            SKILLS_SECTION_TEMPLATE.format(skills_text=skills_text)
            if skills_text else ""
        )

        prompt = OPTIMIZE_PROMPT.format(
            yaml_content=original_yaml,
            diagnosis_text=diagnosis_text,
            trace_excerpt=trace_excerpt,
            skills_section=skills_section,
        )

        response = call_llm(
            [{"role": "user", "content": prompt}],
            max_tokens=16384,
        )

        # Parse response
        analysis = extract_preface(response, "yaml")
        modified_yaml = extract_fenced_block(response, "yaml")

        # Validate YAML
        yaml_valid = False
        try:
            parsed = yaml.safe_load(modified_yaml)
            graph = parsed.get("graph", parsed)
            has_nodes = "nodes" in graph and len(graph["nodes"]) > 0
            has_edges = "edges" in graph
            yaml_valid = has_nodes and has_edges
        except Exception:
            pass

        return {
            "original_yaml": original_yaml,
            "modified_yaml": modified_yaml,
            "analysis": analysis,
            "fm_ids": profile.active_fm_ids(),
            "yaml_valid": yaml_valid,
        }

    async def aoptimize(
        self,
        yaml_path: str | Path,
        profile: FMProfile,
        trace: MASTrace,
    ) -> dict:
        """Async optimize variant for high-concurrency pipelines."""
        yaml_path = Path(yaml_path)
        original_yaml = yaml_path.read_text(encoding="utf-8")
        diagnosis_text = _build_diagnosis_text(profile)
        trace_excerpt = trace.trajectory[:3000]

        skills_text = self._get_relevant_skills(profile)
        skills_section = (
            SKILLS_SECTION_TEMPLATE.format(skills_text=skills_text)
            if skills_text else ""
        )

        prompt = OPTIMIZE_PROMPT.format(
            yaml_content=original_yaml,
            diagnosis_text=diagnosis_text,
            trace_excerpt=trace_excerpt,
            skills_section=skills_section,
        )

        response = await acall_llm(
            [{"role": "user", "content": prompt}],
            max_tokens=16384,
        )

        analysis = extract_preface(response, "yaml")
        modified_yaml = extract_fenced_block(response, "yaml")

        yaml_valid = False
        try:
            parsed = yaml.safe_load(modified_yaml)
            graph = parsed.get("graph", parsed)
            has_nodes = "nodes" in graph and len(graph["nodes"]) > 0
            has_edges = "edges" in graph
            yaml_valid = has_nodes and has_edges
        except Exception:
            pass

        return {
            "original_yaml": original_yaml,
            "modified_yaml": modified_yaml,
            "analysis": analysis,
            "fm_ids": profile.active_fm_ids(),
            "yaml_valid": yaml_valid,
        }

    def optimize_and_save(
        self,
        yaml_path: str | Path,
        profile: FMProfile,
        trace: MASTrace,
        output_path: str | Path | None = None,
    ) -> dict:
        """Optimize and save the modified YAML."""
        result = self.optimize(yaml_path, profile, trace)

        if result["yaml_valid"]:
            out = Path(output_path) if output_path else Path(yaml_path).with_suffix(".optimized.yaml")
            out.write_text(result["modified_yaml"], encoding="utf-8")
            result["output_path"] = str(out)
        else:
            result["output_path"] = None
            print("  WARNING: Generated YAML is invalid, not saving.")

        return result

    def _get_relevant_skills(self, profile: FMProfile) -> str:
        """Retrieve skills relevant to current FMs from the library."""
        if not self.library:
            return ""

        fm_ids = profile.active_fm_ids()
        skills = []
        for fm_id in fm_ids:
            entries = self.library.search(fm_id)
            for e in entries[:2]:  # max 2 per FM
                skills.append(
                    f"- FM-{e.fm_id} [{e.status}, success={e.success_rate:.0%}]: "
                    f"{e.root_cause_pattern}"
                )

        return "\n".join(skills) if skills else ""
