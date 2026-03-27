"""Forger — merge changes from multiple successful Skills into one DAG.

Each Skill works on an independent copy of the original DAG.  The Forger
replays all their search-and-replace operations onto the original YAML.
Non-conflicting changes apply directly; conflicts are resolved by LLM.
"""

from __future__ import annotations

import yaml

from optpilot.dag.core import MASDAG
from optpilot.llm import acall_llm
from optpilot.models import SkillResult
from optpilot.skills.tools import ChangeRecord


_MERGE_CONFLICT_PROMPT = """\
You are merging two independent YAML modifications that conflict.

## Original YAML segment
```
{original_segment}
```

## Modification A (from Skill-{skill_a}, fixing {fm_a})
```
{mod_a}
```

## Modification B (from Skill-{skill_b}, fixing {fm_b})
```
{mod_b}
```

Merge both modifications into a single coherent YAML segment that preserves
the intent of both repairs.  Output ONLY the merged text (no fences, no explanation)."""


async def forge(
    original_dag: MASDAG,
    skill_results: list[SkillResult],
) -> MASDAG:
    """Merge all successful skill changes into one DAG.

    Strategy:
    1. Collect all ChangeRecords from successful skills (search_and_replace only;
       bash-sourced full-file changes are applied as-is if they don't conflict).
    2. Replay them sequentially on the original YAML.
    3. If a change's old_str can't be found (already modified by another skill),
       use LLM to merge.

    Returns the merged DAG.
    """
    successful = [r for r in skill_results if r.success and r.dag is not None]
    if not successful:
        return original_dag
    if len(successful) == 1:
        return successful[0].dag

    original_yaml = yaml.dump(original_dag.to_dict(), allow_unicode=True, sort_keys=False)
    merged_yaml = original_yaml

    # Collect all change records with their source skill
    all_changes: list[tuple[str, ChangeRecord]] = []
    for result in successful:
        # Get change_records from the last evolve result
        records = _extract_change_records(result)
        for cr in records:
            if cr.source == "search_and_replace":
                all_changes.append((result.fm_id, cr))

    # For bash-sourced changes (full file replacements), we take the diff
    # against original and try to extract the meaningful delta
    for result in successful:
        records = _extract_change_records(result)
        for cr in records:
            if cr.source == "bash" and cr.old_str != cr.new_str:
                # Full file change — apply as final override if no s&r changes
                # exist for this skill (it only used bash)
                sr_records = [r for r in records if r.source == "search_and_replace"]
                if not sr_records:
                    all_changes.append((result.fm_id, cr))

    if not all_changes:
        # No replayable changes; fall back to the best single result
        best = max(successful, key=lambda r: r.final_pass_rate)
        return best.dag

    # Replay changes
    conflicts: list[tuple[str, ChangeRecord, str]] = []
    for fm_id, cr in all_changes:
        if cr.source == "bash":
            # Full file replacement — just apply it
            merged_yaml = cr.new_str
            continue

        if cr.old_str in merged_yaml:
            merged_yaml = merged_yaml.replace(cr.old_str, cr.new_str, 1)
        else:
            conflicts.append((fm_id, cr, merged_yaml))

    # Resolve conflicts via LLM
    for fm_id, cr, _snapshot in conflicts:
        merged_yaml = await _resolve_conflict(
            merged_yaml, fm_id, cr, successful,
        )

    # Parse and validate
    try:
        parsed = yaml.safe_load(merged_yaml)
        return MASDAG.from_dict(parsed)
    except Exception:
        # Fallback: return the skill with highest pass_rate
        best = max(successful, key=lambda r: r.final_pass_rate)
        return best.dag


def _extract_change_records(result: SkillResult) -> list[ChangeRecord]:
    """Extract ChangeRecords from a SkillResult's metadata or budget."""
    # change_records are stored in the EvolveResult, which is not directly
    # on SkillResult. They should be forwarded via metadata.
    records = result.metadata.get("change_records", [])
    if records and isinstance(records[0], ChangeRecord):
        return records
    # If stored as dicts (from serialization)
    if records and isinstance(records[0], dict):
        return [
            ChangeRecord(
                old_str=r.get("old_str", ""),
                new_str=r.get("new_str", ""),
                source=r.get("source", "search_and_replace"),
            )
            for r in records
        ]
    return []


async def _resolve_conflict(
    current_yaml: str,
    fm_id: str,
    cr: ChangeRecord,
    all_results: list[SkillResult],
) -> str:
    """Attempt to apply a conflicting change via LLM merge."""
    # Find a nearby segment in current_yaml that resembles old_str
    # Use the first 60 chars of old_str as search anchor
    anchor = cr.old_str[:60]
    idx = current_yaml.find(anchor[:30])
    if idx == -1:
        # Can't even find a partial match — skip this change
        print(f"  Forger: skipping Skill-{fm_id} change (no anchor found)")
        return current_yaml

    # Extract context around the anchor
    ctx_start = max(0, idx - 100)
    ctx_end = min(len(current_yaml), idx + len(cr.old_str) + 100)
    original_segment = current_yaml[ctx_start:ctx_end]

    prompt = _MERGE_CONFLICT_PROMPT.format(
        original_segment=original_segment,
        skill_a=fm_id,
        fm_a=fm_id,
        mod_a=cr.new_str[:500],
        skill_b="(already applied)",
        fm_b="(prior changes)",
        mod_b=original_segment,
    )

    try:
        merged_segment = await acall_llm(
            [{"role": "user", "content": prompt}],
            max_tokens=4096,
        )
        merged_segment = merged_segment.strip()
        if merged_segment:
            return current_yaml[:ctx_start] + merged_segment + current_yaml[ctx_end:]
    except Exception as e:
        print(f"  Forger: LLM merge failed for Skill-{fm_id}: {e}")

    return current_yaml
