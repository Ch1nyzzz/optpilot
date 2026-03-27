"""Optimizer - generates repair candidates by retrieving skills from library or generating new ones.

Retrieval: fm_id coarse filter → LLM matches best skill by root_cause → adapt or generate new plan.
"""

from __future__ import annotations

from optpilot.dag.core import MASDAG
from optpilot.data.fm_taxonomy_6group import GROUP_DEFINITIONS
from optpilot.library.repair_library import RepairLibrary
from optpilot.llm import call_llm_json
from optpilot.models import (
    FMProfile, MASTrace, RepairAction, RepairCandidate, RepairEntry, RepairType,
)
from optpilot.repair_utils import summarize_faults

SKILL_MATCH_PROMPT = """\
You are a MAS repair expert. Match the current fault's root cause to the best skill from the library.

## Current Fault
- FM-{fm_id}: {fm_name}
{fault_summary}

## Available Skills
{skills_text}

Select the best matching skill. Respond with ONLY a JSON object:

{{
    "matched_skill_id": "<skill number (e.g. '1'), or 'none' if no good match>",
    "match_reason": "<one sentence explaining the match>",
    "confidence": <float 0.0-1.0>
}}"""

REPAIR_GENERATION_PROMPT = """\
You are a multi-agent system (MAS) architecture optimizer.

Generate a targeted repair plan based on the diagnosis below.

## Fault
- FM-{fm_id}: {fm_name} -- {fm_description}
{fault_summary}

## MAS Architecture
{mas_context}
{skills_section}
## Editable Parts
- node_mutation: update an existing node's role, prompt, or model config
- node_add: add a new node when the workflow is missing a role such as verifier, planner, or reviewer
- node_delete: remove a redundant or harmful node
- edge_mutation: change edge conditions, trigger behavior, or data-carrying behavior
- edge_rewire: change which node sends to or receives from which other node
- config_change: adjust loop limits, timeouts, or node-level parameters

Choose the smallest repair that directly addresses the diagnosed root cause.
If you add a node, explain its responsibility and include how it connects to the graph.
If you change a loop, explain both the continue condition and the exit condition.

Respond with ONLY a JSON object:

{{
    "description": "<one-sentence summary of the repair>",
    "used_skill": <true if referencing a skill above, false otherwise>,
    "actions": [
        {{
            "repair_type": "<one of: node_mutation, node_add, node_delete, edge_mutation, edge_rewire, config_change>",
            "target": "<target agent or config key>",
            "description": "<what to change>",
            "details": {{}},
            "rationale": "<why this fixes the fault>"
        }}
    ],
    "confidence": <float 0.0-1.0>
}}"""

SKILLS_SECTION_TEMPLATE = """
## Reference Skill (for guidance only -- adapt, partially use, or ignore as appropriate)
- Apply when: {skill_pattern}
- Avoid when: {when_not_to_use}
- Preferred actions: {skill_description}
- Representative actions: {skill_actions}
- Avoid actions: {avoid_actions}
"""


class Optimizer:
    """Generate repair candidates: match skill from library or create new."""

    def __init__(self, library: RepairLibrary):
        self.library = library

    def generate_repair(
        self,
        fm_id: str,
        profile: FMProfile | list[FMProfile],
        trace: MASTrace | list[MASTrace],
        dag: MASDAG | None = None,
    ) -> RepairCandidate:
        """Generate a repair candidate for the given FM.

        1. Coarse filter: find library skills for the same FM
        2. LLM matches best skill by root_cause
        3. Match found → adapt the skill
        4. No match → generate from scratch
        """
        traces, profiles = self._normalize_inputs(trace, profile)
        primary_trace = traces[0]

        # 1. Coarse filter
        candidates = self.library.search(fm_id, mas_name=primary_trace.mas_name)
        if not candidates:
            print(f"  No library skills for FM-{fm_id}, generating new repair...")
            return self._generate_new(fm_id, profiles, traces, dag)

        # 2. LLM match
        matched = self._match_skill(fm_id, profiles, traces, candidates)
        if matched:
            print(f"  Referencing skill [{matched.entry_id}]: {matched.root_cause_pattern[:60]}")
            return self._generate_with_skill(matched, fm_id, profiles, traces, dag)

        # 3. No match
        print(f"  Library has {len(candidates)} skills but none matched, generating new repair...")
        return self._generate_new(fm_id, profiles, traces, dag)

    def _match_skill(
        self,
        fm_id: str,
        profiles: list[FMProfile],
        traces: list[MASTrace],
        candidates: list[RepairEntry],
    ) -> RepairEntry | None:
        """Use LLM to match current root cause against library skills."""
        fm_info = GROUP_DEFINITIONS[fm_id]

        skills_text = "\n".join(
            f"Skill {i+1} (id={e.entry_id}): {e.root_cause_pattern} "
            f"[success_rate={e.success_rate:.0%}, applied={e.n_applied}x]"
            for i, e in enumerate(candidates)
        )

        prompt = SKILL_MATCH_PROMPT.format(
            fm_id=fm_id,
            fm_name=fm_info["name"],
            fault_summary=summarize_faults(fm_id, profiles, traces),
            skills_text=skills_text,
        )

        try:
            result = call_llm_json(
                [{"role": "user", "content": prompt}],
                max_tokens=4096,
            )
            skill_id = result.get("matched_skill_id", "none")
            confidence = result.get("confidence", 0)

            if skill_id == "none" or confidence < 0.5:
                return None

            # Map skill number to entry
            idx = int(skill_id) - 1
            if 0 <= idx < len(candidates):
                return candidates[idx]
        except Exception as e:
            print(f"  Skill matching failed: {e}")

        return None

    def _generate_with_skill(
        self,
        skill: RepairEntry,
        fm_id: str,
        profiles: list[FMProfile],
        traces: list[MASTrace],
        dag: MASDAG | None,
    ) -> RepairCandidate:
        """Generate repair with a matched skill as reference (not forced)."""
        fm_info = GROUP_DEFINITIONS[fm_id]

        skill_actions = "; ".join(
            a.description for a in (skill.candidate.actions if skill.candidate else [])
        )

        skills_section = SKILLS_SECTION_TEMPLATE.format(
            skill_pattern=skill.root_cause_pattern,
            when_not_to_use=skill.when_not_to_use or "none recorded",
            skill_description=skill.candidate.description if skill.candidate else "",
            skill_actions=skill_actions,
            avoid_actions=", ".join(skill.avoid_actions) if skill.avoid_actions else "none recorded",
        )

        prompt = REPAIR_GENERATION_PROMPT.format(
            fm_id=fm_id,
            fm_name=fm_info["name"],
            fm_description=fm_info["description"],
            fault_summary=summarize_faults(fm_id, profiles, traces),
            mas_context=self._get_mas_context(traces[0], dag),
            skills_section=skills_section,
        )

        result = call_llm_json(
            [{"role": "user", "content": prompt}],
            max_tokens=8192,
        )
        candidate = self._parse_result(result, fm_id)
        candidate.source = "library" if result.get("used_skill") else "generated"
        return candidate

    def _generate_new(
        self,
        fm_id: str,
        profiles: list[FMProfile],
        traces: list[MASTrace],
        dag: MASDAG | None,
    ) -> RepairCandidate:
        fm_info = GROUP_DEFINITIONS[fm_id]

        prompt = REPAIR_GENERATION_PROMPT.format(
            fm_id=fm_id,
            fm_name=fm_info["name"],
            fm_description=fm_info["description"],
            fault_summary=summarize_faults(fm_id, profiles, traces),
            mas_context=self._get_mas_context(traces[0], dag),
            skills_section="",
        )

        result = call_llm_json(
            [{"role": "user", "content": prompt}],
            max_tokens=8192,
        )
        return self._parse_result(result, fm_id)

    def _get_mas_context(self, trace: MASTrace, dag: MASDAG | None) -> str:
        if dag:
            return dag.summary()
        return (
            f"MAS: {trace.mas_name}, LLM: {trace.llm_name}, "
            f"Benchmark: {trace.benchmark_name}\n"
            f"Simple 2-agent dialogue system (Student + Assistant), multi-turn problem solving."
        )

    def _normalize_inputs(
        self,
        traces: MASTrace | list[MASTrace],
        profiles: FMProfile | list[FMProfile],
    ) -> tuple[list[MASTrace], list[FMProfile]]:
        norm_traces = traces if isinstance(traces, list) else [traces]
        norm_profiles = profiles if isinstance(profiles, list) else [profiles]
        if len(norm_traces) != len(norm_profiles):
            raise ValueError("Number of traces and profiles must match.")
        return norm_traces, norm_profiles

    def _parse_result(self, result: dict, fm_id: str) -> RepairCandidate:
        actions = []
        for a in result.get("actions", []):
            try:
                rt = a.get("repair_type", "node_mutation")
                if rt not in [e.value for e in RepairType]:
                    rt = "node_mutation"
                actions.append(RepairAction(
                    repair_type=RepairType(rt),
                    target=a.get("target", ""),
                    description=a.get("description", ""),
                    details=a.get("details", {}),
                    rationale=a.get("rationale", ""),
                ))
            except (ValueError, KeyError):
                continue

        return RepairCandidate(
            fm_id=fm_id,
            actions=actions,
            description=result.get("description", ""),
            source="generated",
            confidence=result.get("confidence", 0.5),
        )
