"""Wrap-up stage for consolidating raw hints into canonical skills."""

from __future__ import annotations

from statistics import mean

from optpilot.data.fm_taxonomy import FM_NAMES
from optpilot.library.repair_library import RepairLibrary
from optpilot.llm import call_llm_json
from optpilot.models import RepairCandidate, RepairEntry

WRAP_UP_PROMPT = """\
You are consolidating MAS repair hints into a compact canonical skill library.

Fault: FM-{fm_id} ({fm_name})

Validated hints:
{positive_hints}

Failed hints / counterexamples:
{negative_hints}

Produce up to {max_skills} canonical skills. Each skill must summarize:
- when_to_use: the recurring situation where the repair tends to help
- when_not_to_use: the recurring failure boundary from negative examples
- recommended_actions: short action phrases that are repeatedly useful
- avoid_actions: short action phrases that often fail or regress pass rate

Respond with ONLY JSON:
{{
  "skills": [
    {{
      "when_to_use": "<string>",
      "when_not_to_use": "<string>",
      "recommended_actions": ["<string>", "<string>"],
      "avoid_actions": ["<string>", "<string>"]
    }}
  ]
}}
"""


class WrapUp:
    """Aggregate raw hints into canonical wrapped skills."""

    def __init__(self, library: RepairLibrary, output_library: RepairLibrary | None = None):
        self.library = library
        self.output_library = output_library or library

    def wrap_fm(
        self,
        fm_id: str,
        *,
        source_mas: str | None = None,
        max_skills: int = 3,
    ) -> list[RepairEntry]:
        raw_entries = self.library.get_entries(fm_id, entry_kind="hint")
        if not raw_entries:
            return []

        positives = [
            entry for entry in raw_entries
            if entry.status == "validated" or (entry.status == "unvalidated" and entry.success_rate > 0)
        ]
        negatives = [
            entry for entry in raw_entries
            if entry.status == "failed" or (entry.status == "unvalidated" and entry.success_rate <= 0)
        ]

        wrapped_specs = self._summarize(fm_id, positives, negatives, max_skills=max_skills)
        if not wrapped_specs:
            wrapped_specs = [self._fallback_spec(positives, negatives)]

        representative = self._pick_representative(positives or negatives)
        if representative is None:
            return []

        wrapped_entries: list[RepairEntry] = []
        base_candidate = representative.candidate or RepairCandidate(
            fm_id=fm_id,
            description="Canonical wrapped skill",
        )
        mean_success = mean([entry.success_rate for entry in positives]) if positives else 0.0

        for spec in wrapped_specs[:max_skills]:
            candidate = RepairCandidate(
                fm_id=fm_id,
                actions=[],
                description=(
                    "; ".join(spec["recommended_actions"])
                    if spec["recommended_actions"]
                    else base_candidate.description
                ),
                source="wrapped",
                confidence=base_candidate.confidence,
            )
            wrapped_entries.append(RepairEntry(
                entry_kind="wrapped",
                fm_id=fm_id,
                fm_name=FM_NAMES.get(fm_id, ""),
                source_mas=source_mas or representative.source_mas,
                root_cause_pattern=spec["when_to_use"],
                when_not_to_use=spec["when_not_to_use"],
                candidate=candidate,
                status="validated" if positives else "failed",
                success_rate=mean_success,
                n_applied=sum(entry.n_applied for entry in positives),
                n_success=sum(entry.n_success for entry in positives),
                side_effects=list(dict.fromkeys(
                    side_effect
                    for entry in positives
                    for side_effect in entry.side_effects
                )),
                avoid_actions=spec["avoid_actions"],
                supporting_entry_ids=[entry.entry_id for entry in positives],
                counter_entry_ids=[entry.entry_id for entry in negatives],
                validation_metrics={
                    "recommended_actions": spec["recommended_actions"],
                    "raw_hint_count": len(raw_entries),
                    "validated_hint_count": len(positives),
                    "failed_hint_count": len(negatives),
                    "source_entry_id": representative.entry_id,
                },
            ))

        self.output_library.replace_wrapped_entries(fm_id, wrapped_entries)
        return wrapped_entries

    def _summarize(
        self,
        fm_id: str,
        positives: list[RepairEntry],
        negatives: list[RepairEntry],
        *,
        max_skills: int,
    ) -> list[dict[str, list[str] | str]]:
        prompt = WRAP_UP_PROMPT.format(
            fm_id=fm_id,
            fm_name=FM_NAMES.get(fm_id, ""),
            positive_hints=self._format_hints(positives, "none"),
            negative_hints=self._format_hints(negatives, "none"),
            max_skills=max_skills,
        )

        try:
            result = call_llm_json(
                [{"role": "user", "content": prompt}],
                max_tokens=8192,
            )
        except Exception:
            return []

        skills = result.get("skills", [])
        normalized: list[dict[str, list[str] | str]] = []
        for skill in skills:
            when_to_use = str(skill.get("when_to_use", "")).strip()
            if not when_to_use:
                continue
            normalized.append({
                "when_to_use": when_to_use,
                "when_not_to_use": str(skill.get("when_not_to_use", "")).strip(),
                "recommended_actions": [
                    str(action).strip()
                    for action in skill.get("recommended_actions", [])
                    if str(action).strip()
                ],
                "avoid_actions": [
                    str(action).strip()
                    for action in skill.get("avoid_actions", [])
                    if str(action).strip()
                ],
            })
        return normalized

    def _format_hints(self, entries: list[RepairEntry], empty_label: str) -> str:
        if not entries:
            return empty_label

        lines = []
        for entry in entries:
            actions = []
            if entry.candidate:
                actions = [action.description for action in entry.candidate.actions[:3]]
            lines.append(
                f"- id={entry.entry_id}, status={entry.status}, success={entry.success_rate:.2f}, "
                f"pattern={entry.root_cause_pattern}, "
                f"repair={entry.candidate.description if entry.candidate else ''}, "
                f"actions={actions}, "
                f"pass={entry.validation_metrics.get('after_pass_rate', 'n/a')}, "
                f"runtime_delta={entry.validation_metrics.get('runtime_delta_s', 'n/a')}"
            )
        return "\n".join(lines)

    def _fallback_spec(
        self,
        positives: list[RepairEntry],
        negatives: list[RepairEntry],
    ) -> dict[str, list[str] | str]:
        best_positive = self._pick_representative(positives)
        best_negative = self._pick_representative(negatives)
        return {
            "when_to_use": (
                best_positive.root_cause_pattern
                if best_positive and best_positive.root_cause_pattern
                else "Use when repeated evidence matches the strongest validated hint."
            ),
            "when_not_to_use": (
                best_negative.root_cause_pattern
                if best_negative and best_negative.root_cause_pattern
                else ""
            ),
            "recommended_actions": (
                [best_positive.candidate.description]
                if best_positive and best_positive.candidate and best_positive.candidate.description
                else []
            ),
            "avoid_actions": (
                [best_negative.candidate.description]
                if best_negative and best_negative.candidate and best_negative.candidate.description
                else []
            ),
        }

    def _pick_representative(self, entries: list[RepairEntry]) -> RepairEntry | None:
        if not entries:
            return None
        return max(entries, key=lambda entry: (entry.success_rate, entry.n_success, entry.n_applied))
