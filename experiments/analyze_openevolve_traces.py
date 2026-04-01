"""Offline trace analysis for OpenEvolve cold-start → prior extraction + Jacobian warmup.

Pipeline:
1. Load all programs from OpenEvolve output (programs/*.json)
2. Build parent→child mutation tree
3. Filter train-improving mutations (child.combined_score > parent.combined_score)
4. Re-evaluate promising candidates on held-out test set
5. Filter test-improving mutations (真正有效的 mutation)
6. Classify each effective mutation (code diff → pattern category)
7. Output: PatternCatalog (data-driven priors) + Jacobian warmup data

Usage:
    python -m experiments.analyze_openevolve_traces \\
        --openevolve-dir results/.../openevolve_output \\
        --model openai/gpt-oss-120b \\
        --total-tasks 200 --n-train 100 \\
        --concurrency 512 --timeout 600
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optpilot.config import LIBRARY_DIR, RESULTS_DIR
from optpilot.dag.core import MASDAG
from optpilot.data.benchmarks import (
    OfficialBenchmarkSuite,
    load_online_benchmark_suite,
)
from optpilot.data.fm_taxonomy_6group import GROUP_IDS
from optpilot.modules.diagnoser import Diagnoser
from optpilot.modules.runner import OptPilotRunner
from optpilot.skills.jacobian import JacobianEntry, RepairJacobian, RepairOutcome
from optpilot.skills.recipes import RecipeLibrary, RepairRecipe
from optpilot.llm import acall_llm
from optpilot.skills.repair_patterns import (
    CHANGE_TYPES,
    PatternCatalog,
    RepairPattern,
    build_mutation_analysis_prompt,
    infer_all_change_types_from_dags,
)


# ------------------------------------------------------------------ #
#  Data structures                                                     #
# ------------------------------------------------------------------ #

@dataclass
class ProgramRecord:
    """Parsed record from programs/*.json."""
    id: str
    solution: str
    parent_id: str | None
    iteration_found: int
    metrics: dict  # combined_score, accuracy, fm_A-F, etc.
    metadata: dict


@dataclass
class Mutation:
    """A parent→child mutation pair."""
    parent: ProgramRecord
    child: ProgramRecord
    train_score_delta: float  # child - parent combined_score
    train_accuracy_delta: float


@dataclass
class EffectiveMutation(Mutation):
    """A mutation validated on test set."""
    parent_test_accuracy: float = 0.0
    child_test_accuracy: float = 0.0
    test_accuracy_delta: float = 0.0
    parent_dag: MASDAG | None = None
    child_dag: MASDAG | None = None
    observed_pattern: str = ""
    fm_deltas: dict = None  # per-group FM rate changes on test
    llm_analysis: dict = None  # full LLM analysis result

    def __post_init__(self):
        if self.fm_deltas is None:
            self.fm_deltas = {}
        if self.llm_analysis is None:
            self.llm_analysis = {}


# ------------------------------------------------------------------ #
#  Step 1: Load programs                                               #
# ------------------------------------------------------------------ #

def load_programs(openevolve_dir: Path) -> dict[str, ProgramRecord]:
    """Load all program records from programs/*.json.

    Searches in order: best/programs, programs/, then the highest-numbered
    checkpoint directory (which contains the most complete program archive).
    """
    programs_dir = openevolve_dir / "best" / "programs"
    if not programs_dir.exists():
        programs_dir = openevolve_dir / "programs"
    if not programs_dir.exists():
        # Find highest checkpoint
        cp_dir = openevolve_dir / "checkpoints"
        if cp_dir.exists():
            checkpoints = sorted(
                [d for d in cp_dir.iterdir() if d.is_dir() and (d / "programs").exists()],
                key=lambda d: int(d.name.split("_")[-1]) if d.name.split("_")[-1].isdigit() else 0,
                reverse=True,
            )
            if checkpoints:
                programs_dir = checkpoints[0] / "programs"
                print(f"  Using checkpoint: {checkpoints[0].name}")
    if not programs_dir.exists():
        # Last resort: any programs dir
        for p in openevolve_dir.rglob("programs"):
            if p.is_dir():
                programs_dir = p
                break

    if not programs_dir.exists():
        raise FileNotFoundError(f"No programs/ directory found under {openevolve_dir}")

    records: dict[str, ProgramRecord] = {}
    for json_file in sorted(programs_dir.glob("*.json")):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            records[data["id"]] = ProgramRecord(
                id=data["id"],
                solution=data.get("solution", ""),
                parent_id=data.get("parent_id"),
                iteration_found=data.get("iteration_found", 0),
                metrics=data.get("metrics", {}),
                metadata=data.get("metadata", {}),
            )
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Warning: skipping {json_file.name}: {e}")

    print(f"  Loaded {len(records)} programs")
    return records


# ------------------------------------------------------------------ #
#  Step 2: Build mutation pairs, filter train-improving                #
# ------------------------------------------------------------------ #

def extract_train_improving_mutations(
    programs: dict[str, ProgramRecord],
    min_score_delta: float = 0.0,
) -> list[Mutation]:
    """Find parent→child pairs where child improved on train set."""
    mutations: list[Mutation] = []
    for prog in programs.values():
        if prog.parent_id is None or prog.parent_id not in programs:
            continue
        parent = programs[prog.parent_id]

        parent_score = parent.metrics.get("combined_score", 0.0)
        child_score = prog.metrics.get("combined_score", 0.0)
        parent_acc = parent.metrics.get("accuracy", 0.0)
        child_acc = prog.metrics.get("accuracy", 0.0)

        score_delta = child_score - parent_score
        if score_delta > min_score_delta:
            mutations.append(Mutation(
                parent=parent,
                child=prog,
                train_score_delta=score_delta,
                train_accuracy_delta=child_acc - parent_acc,
            ))

    mutations.sort(key=lambda m: m.train_score_delta, reverse=True)
    print(f"  Found {len(mutations)} train-improving mutations")
    return mutations


# ------------------------------------------------------------------ #
#  Step 3: Code → MASDAG conversion                                    #
# ------------------------------------------------------------------ #

def code_to_dag(code: str) -> MASDAG | None:
    """Execute build_dag() from code string, return MASDAG or None."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(code)
            tmp_path = f.name
        spec = importlib.util.spec_from_file_location("_tmp_dag", tmp_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        os.unlink(tmp_path)
        return MASDAG.from_dict(mod.build_dag())
    except Exception as e:
        print(f"  Warning: code_to_dag failed: {e}")
        return None


# ------------------------------------------------------------------ #
#  Step 4: Test set evaluation                                         #
# ------------------------------------------------------------------ #

async def evaluate_on_test(
    runner: OptPilotRunner,
    diagnoser: Diagnoser,
    dag: MASDAG,
    test_suite: OfficialBenchmarkSuite,
    concurrency: int,
) -> dict:
    """Evaluate a DAG on test set, return accuracy + per-group FM rates."""
    tasks = test_suite.tasks()
    traces = await runner.arun_batch(tasks, dag=dag, max_concurrency=concurrency)

    n = len(traces)
    correct = sum(1 for t in traces if t.task_score and t.task_score > 0)
    accuracy = correct / n if n > 0 else 0.0

    # Classify FM groups
    fm_rates: dict[str, float] = {g: 0.0 for g in GROUP_IDS}
    try:
        from optpilot.data.fm_taxonomy_6group import GROUP_DEFINITIONS
        from optpilot.models import FMLabel, FMProfile
        import asyncio as _aio

        sem = _aio.Semaphore(64)

        async def _classify_one(trace):
            async with sem:
                return await diagnoser._aclassify(trace)

        label_dicts = await asyncio.gather(*[_classify_one(t) for t in traces])
        fm_counts: dict[str, int] = {g: 0 for g in GROUP_IDS}
        for labels_dict in label_dicts:
            for gid in GROUP_IDS:
                if labels_dict.get(gid, False):
                    fm_counts[gid] += 1
        fm_rates = {g: fm_counts[g] / n for g in GROUP_IDS} if n > 0 else fm_rates
    except Exception as e:
        print(f"  Warning: FM classification failed: {e}")

    return {"accuracy": accuracy, "correct": correct, "n": n, "fm_rates": fm_rates}


async def validate_mutations_on_test(
    mutations: list[Mutation],
    runner: OptPilotRunner,
    diagnoser: Diagnoser,
    test_suite: OfficialBenchmarkSuite,
    concurrency: int,
) -> list[EffectiveMutation]:
    """Re-evaluate train-improving mutations on test set, filter truly effective ones."""

    # Deduplicate DAGs: same code → same DAG, only evaluate once
    code_to_result: dict[str, dict] = {}
    unique_codes: dict[str, MASDAG] = {}

    # Collect all unique codes (parents + children)
    for m in mutations:
        for code in (m.parent.solution, m.child.solution):
            if code not in unique_codes:
                dag = code_to_dag(code)
                if dag is not None:
                    unique_codes[code] = dag

    print(f"  {len(unique_codes)} unique DAGs to evaluate on test set")

    # Evaluate each unique DAG
    for i, (code, dag) in enumerate(unique_codes.items()):
        if code in code_to_result:
            continue
        print(f"  Evaluating DAG {i+1}/{len(unique_codes)}...")
        try:
            result = await evaluate_on_test(runner, diagnoser, dag, test_suite, concurrency)
            code_to_result[code] = result
        except Exception as e:
            print(f"    Failed: {e}")
            code_to_result[code] = {"accuracy": 0.0, "correct": 0, "n": 0, "fm_rates": {}}

    # Filter: test accuracy must also improve
    effective: list[EffectiveMutation] = []
    for m in mutations:
        parent_result = code_to_result.get(m.parent.solution)
        child_result = code_to_result.get(m.child.solution)
        if parent_result is None or child_result is None:
            continue

        parent_dag = unique_codes.get(m.parent.solution)
        child_dag = unique_codes.get(m.child.solution)

        test_delta = child_result["accuracy"] - parent_result["accuracy"]
        if test_delta <= 0:
            continue  # Not effective on test set

        # Compute per-group FM deltas
        fm_deltas = {}
        for gid in GROUP_IDS:
            parent_rate = parent_result["fm_rates"].get(gid, 0.0)
            child_rate = child_result["fm_rates"].get(gid, 0.0)
            fm_deltas[gid] = child_rate - parent_rate  # negative = improvement

        effective.append(EffectiveMutation(
            parent=m.parent,
            child=m.child,
            train_score_delta=m.train_score_delta,
            train_accuracy_delta=m.train_accuracy_delta,
            parent_test_accuracy=parent_result["accuracy"],
            child_test_accuracy=child_result["accuracy"],
            test_accuracy_delta=test_delta,
            parent_dag=parent_dag,
            child_dag=child_dag,
            fm_deltas=fm_deltas,
        ))

    effective.sort(key=lambda e: e.test_accuracy_delta, reverse=True)
    print(f"  {len(effective)} mutations effective on test set (out of {len(mutations)} train-improving)")
    return effective


# ------------------------------------------------------------------ #
#  Step 5: Classify mutations via LLM analysis                         #
# ------------------------------------------------------------------ #

async def classify_mutations_llm(
    mutations: list[EffectiveMutation],
    model: str,
) -> list[EffectiveMutation]:
    """For each effective mutation, use LLM to analyze the code diff and classify changes.

    Falls back to deterministic DAG diff when LLM fails.
    """
    sem = asyncio.Semaphore(32)

    async def _analyze_one(m: EffectiveMutation) -> None:
        # Build prompt
        prompt = build_mutation_analysis_prompt(
            parent_code=m.parent.solution,
            child_code=m.child.solution,
            fm_deltas=m.fm_deltas,
        )
        async with sem:
            try:
                response = await acall_llm(
                    messages=[{"role": "user", "content": prompt}],
                    model=model,
                    temperature=0.0,
                    max_tokens=2048,
                )
                # Parse JSON from response
                text = response.strip()
                # Strip markdown fences if present
                if text.startswith("```"):
                    text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                    if text.endswith("```"):
                        text = text[:-3]
                    text = text.strip()
                result = json.loads(text)

                # Extract primary change type
                primary = result.get("primary_change_type", "")
                if primary in CHANGE_TYPES:
                    m.observed_pattern = primary
                # Store full analysis
                m.llm_analysis = result
            except Exception as e:
                print(f"    LLM analysis failed for {m.child.id[:8]}: {e}")
                # Fallback: deterministic DAG diff
                if m.parent_dag is not None and m.child_dag is not None:
                    types = infer_all_change_types_from_dags(m.parent_dag, m.child_dag)
                    if types:
                        m.observed_pattern = types[0]
                        m.llm_analysis = {
                            "changes": [{"type": t, "target": "", "summary": "inferred from DAG diff"} for t in types],
                            "primary_change_type": types[0],
                            "likely_fixed_fm": [],
                            "reasoning": "Deterministic fallback — LLM analysis failed.",
                        }

    await asyncio.gather(*[_analyze_one(m) for m in mutations])

    classified = sum(1 for m in mutations if m.observed_pattern)
    print(f"  Classified {classified}/{len(mutations)} mutations via LLM analysis")
    return mutations


# ------------------------------------------------------------------ #
#  Step 6: Summarize → PatternCatalog + Jacobian warmup                #
# ------------------------------------------------------------------ #

def summarize_priors(
    mutations: list[EffectiveMutation],
    output_dir: Path,
) -> dict:
    """Aggregate effective mutations into:
    1. Pattern frequency stats (which patterns actually work)
    2. Jacobian warmup data (RepairOutcome records per FM group × pattern)
    3. Updated PatternCatalog
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Aggregate stats ---
    # (fm_group, pattern_id) → list of outcomes
    pair_outcomes: dict[tuple[str, str], list[dict]] = defaultdict(list)
    pattern_counts: Counter = Counter()
    fm_group_counts: Counter = Counter()

    for m in mutations:
        if not m.observed_pattern:
            continue
        pattern_counts[m.observed_pattern] += 1

        # For each FM group that improved (delta < 0), record an outcome
        for gid in GROUP_IDS:
            delta = m.fm_deltas.get(gid, 0.0)
            if delta < 0:  # FM rate decreased = improvement
                fm_group_counts[gid] += 1
                pair_outcomes[(gid, m.observed_pattern)].append({
                    "success": True,
                    "fm_delta": delta,
                    "pass_delta": m.test_accuracy_delta,
                })

        # Also record for FM groups that didn't improve (neutral/worsened)
        for gid in GROUP_IDS:
            delta = m.fm_deltas.get(gid, 0.0)
            if delta >= 0 and m.observed_pattern:
                pair_outcomes[(gid, m.observed_pattern)].append({
                    "success": False,
                    "fm_delta": delta,
                    "pass_delta": m.test_accuracy_delta,
                })

    # --- Build Jacobian warmup records ---
    warmup_outcomes: list[dict] = []
    for (gid, pattern_id), outcomes in pair_outcomes.items():
        for o in outcomes:
            warmup_outcomes.append({
                "fm_group": gid,
                "dag_component": "other",
                "agent": "",
                "assigned_pattern_id": "",  # offline: no assignment
                "observed_pattern_id": pattern_id,
                "success": o["success"],
                "fm_delta": o["fm_delta"],
                "pass_delta": o["pass_delta"],
                "timestamp": datetime.now().isoformat(),
            })

    # --- Pattern effectiveness summary ---
    pattern_summary: dict[str, dict] = {}
    for pattern_id, count in pattern_counts.most_common():
        successes_by_fm: dict[str, int] = {}
        totals_by_fm: dict[str, int] = {}
        for gid in GROUP_IDS:
            outcomes = pair_outcomes.get((gid, pattern_id), [])
            totals_by_fm[gid] = len(outcomes)
            successes_by_fm[gid] = sum(1 for o in outcomes if o["success"])

        pattern_summary[pattern_id] = {
            "total_effective_mutations": count,
            "per_fm_group": {
                gid: {
                    "n": totals_by_fm[gid],
                    "successes": successes_by_fm[gid],
                    "success_rate": successes_by_fm[gid] / totals_by_fm[gid] if totals_by_fm[gid] > 0 else 0.0,
                }
                for gid in GROUP_IDS
            },
        }

    # --- Save outputs ---
    # 1. Full analysis report
    report = {
        "timestamp": datetime.now().isoformat(),
        "n_effective_mutations": len(mutations),
        "n_classified": sum(1 for m in mutations if m.observed_pattern),
        "pattern_summary": pattern_summary,
        "fm_group_improvement_counts": dict(fm_group_counts.most_common()),
        "mutations": [
            {
                "child_id": m.child.id,
                "parent_id": m.parent.id,
                "train_score_delta": m.train_score_delta,
                "train_accuracy_delta": m.train_accuracy_delta,
                "test_accuracy_delta": m.test_accuracy_delta,
                "observed_pattern": m.observed_pattern,
                "fm_deltas": m.fm_deltas,
                "llm_analysis": m.llm_analysis,
            }
            for m in mutations
        ],
    }
    report_path = output_dir / "offline_analysis.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Analysis report saved to {report_path}")

    # 2. Jacobian warmup outcomes
    warmup_path = output_dir / "jacobian_warmup.jsonl"
    with open(warmup_path, "w", encoding="utf-8") as f:
        for record in warmup_outcomes:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Jacobian warmup data saved to {warmup_path} ({len(warmup_outcomes)} records)")

    # 3. Data-driven cold-start priors (替换 _cold_start_score 的手写 priors)
    data_priors: dict[str, dict[str, float]] = {}
    for gid in GROUP_IDS:
        gid_priors: dict[str, float] = {}
        for pattern_id in pattern_counts:
            outcomes = pair_outcomes.get((gid, pattern_id), [])
            if outcomes:
                rate = sum(1 for o in outcomes if o["success"]) / len(outcomes)
                gid_priors[pattern_id] = round(rate, 3)
        if gid_priors:
            data_priors[gid] = dict(sorted(gid_priors.items(), key=lambda x: x[1], reverse=True))
    priors_path = output_dir / "data_driven_priors.json"
    priors_path.write_text(json.dumps(data_priors, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Data-driven priors saved to {priors_path}")

    return report


# ------------------------------------------------------------------ #
#  Step 7: Distill Repair Recipes via LLM clustering                   #
# ------------------------------------------------------------------ #

RECIPE_CLUSTER_PROMPT = """\
You are analyzing a group of effective mutations that all belong to the same
failure mode group and change type. Your job is to distill them into 1-3
reusable **repair recipes** (repair principles).

## FM Group: {fm_group} — {fm_description}
## Change Type: {change_type} — {change_type_description}

## Effective mutations in this group ({n_mutations} total):
{mutation_summaries}

Distill these into 1-3 repair recipes. Each recipe should be:
- **Abstract enough** to be reusable across different specific mutations
- **Concrete enough** that an LLM knows exactly what to do
- **Grounded** in the evidence above (not speculative)

If all mutations are essentially the same strategy, output just 1 recipe.
If there are clearly distinct strategies, output up to 3.

Respond ONLY with valid JSON:
{{
  "recipes": [
    {{
      "precondition": "<when to use: what symptoms/failure patterns indicate this is the right fix>",
      "action": "<what to do: concrete repair action, not vague>",
      "root_cause": "<what fundamental problem this solves>"
    }}
  ]
}}
"""


async def distill_recipes(
    mutations: list[EffectiveMutation],
    model: str,
) -> list[RepairRecipe]:
    """Cluster effective mutations by (fm_group, change_type) and distill recipes via LLM."""
    from optpilot.data.fm_taxonomy_6group import GROUP_DEFINITIONS

    # Group mutations by (fm_group, change_type)
    # Use LLM analysis to determine FM group association
    groups: dict[tuple[str, str], list[EffectiveMutation]] = defaultdict(list)
    for m in mutations:
        if not m.observed_pattern:
            continue
        # Determine which FM groups this mutation helped
        likely_fms = m.llm_analysis.get("likely_fixed_fm", []) if m.llm_analysis else []
        if not likely_fms:
            # Fallback: use FM groups where rate decreased
            likely_fms = [gid for gid, delta in m.fm_deltas.items() if delta < 0]
        if not likely_fms:
            likely_fms = ["unknown"]
        for fm in likely_fms:
            groups[(fm, m.observed_pattern)].append(m)

    # Filter out groups with too few mutations
    groups = {k: v for k, v in groups.items() if len(v) >= 1}
    print(f"  {len(groups)} (fm_group, change_type) clusters to distill")

    all_recipes: list[RepairRecipe] = []
    sem = asyncio.Semaphore(8)

    async def _distill_one(
        fm_group: str,
        change_type: str,
        cluster: list[EffectiveMutation],
    ) -> list[RepairRecipe]:
        # Build mutation summaries
        summaries = []
        for i, m in enumerate(cluster[:10], 1):  # cap at 10 for prompt length
            analysis = m.llm_analysis or {}
            changes = analysis.get("changes", [])
            reasoning = analysis.get("reasoning", "")
            changes_text = "; ".join(
                f"{c.get('type', '?')}: {c.get('summary', '?')}"
                for c in changes
            ) if changes else "unknown"
            summaries.append(
                f"  {i}. test_delta=+{m.test_accuracy_delta:.1%}, "
                f"changes=[{changes_text}], reasoning: {reasoning}"
            )

        fm_info = GROUP_DEFINITIONS.get(fm_group, {})
        fm_desc = fm_info.get("name", fm_group) if isinstance(fm_info, dict) else str(fm_info)
        change_desc = CHANGE_TYPES.get(change_type, change_type)

        prompt = RECIPE_CLUSTER_PROMPT.format(
            fm_group=fm_group,
            fm_description=fm_desc,
            change_type=change_type,
            change_type_description=change_desc,
            n_mutations=len(cluster),
            mutation_summaries="\n".join(summaries),
        )

        async with sem:
            try:
                response = await acall_llm(
                    messages=[{"role": "user", "content": prompt}],
                    model=model,
                    temperature=0.0,
                    max_tokens=2048,
                )
                text = response.strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                    if text.endswith("```"):
                        text = text[:-3]
                    text = text.strip()
                result = json.loads(text)

                avg_delta = sum(m.test_accuracy_delta for m in cluster) / len(cluster)
                recipes = []
                for j, raw in enumerate(result.get("recipes", [])):
                    recipe = RepairRecipe(
                        recipe_id=f"{fm_group}_{change_type}_{j}",
                        fm_group=fm_group,
                        change_type=change_type,
                        precondition=raw.get("precondition", ""),
                        action=raw.get("action", ""),
                        root_cause=raw.get("root_cause", ""),
                        n_effective=len(cluster),
                        avg_test_delta=avg_delta,
                    )
                    recipes.append(recipe)
                return recipes
            except Exception as e:
                print(f"    Recipe distillation failed for ({fm_group}, {change_type}): {e}")
                return []

    tasks = [
        _distill_one(fm_group, change_type, cluster)
        for (fm_group, change_type), cluster in groups.items()
    ]
    results = await asyncio.gather(*tasks)
    for recipe_list in results:
        all_recipes.extend(recipe_list)

    print(f"  Distilled {len(all_recipes)} recipes from {len(groups)} clusters")
    return all_recipes


# ------------------------------------------------------------------ #
#  Step 8: Apply warmup to Jacobian                                    #
# ------------------------------------------------------------------ #

def apply_jacobian_warmup(
    warmup_path: Path,
    priors_path: Path | None = None,
    jacobian_dir: Path | None = None,
) -> None:
    """Load warmup records and inject into the Jacobian matrix.

    Also copies data_driven_priors.json into the Jacobian directory so that
    ``_cold_start_score`` uses data-driven priors instead of hand-coded ones.
    """
    jacobian = RepairJacobian(base_dir=jacobian_dir)

    records = []
    for line in warmup_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))

    for rec in records:
        outcome = RepairOutcome(
            fm_group=rec["fm_group"],
            dag_component=rec.get("dag_component", "other"),
            agent=rec.get("agent", ""),
            assigned_pattern_id=rec.get("assigned_pattern_id", ""),
            observed_pattern_id=rec["observed_pattern_id"],
            success=rec["success"],
            fm_delta=rec["fm_delta"],
            pass_delta=rec["pass_delta"],
            timestamp=rec.get("timestamp", ""),
        )
        jacobian.update(outcome)

    jacobian.save()

    # Copy data-driven priors into Jacobian directory
    if priors_path and priors_path.exists():
        dest = jacobian.base_dir / "data_driven_priors.json"
        dest.write_text(priors_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"  Data-driven priors installed to {dest}")

    print(f"  Jacobian warmed up with {len(records)} offline records")
    print(f"  {jacobian.format_matrix_summary()}")


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def _load_benchmark_for_topology(
    topology: str,
    total_tasks: int,
    n_train: int,
) -> tuple[OfficialBenchmarkSuite, object, object, object]:
    """Load benchmark suite, score_fn, benchmark_name_resolver, tool_setup_fn for a topology.

    Returns (test_suite, score_fn, bench_resolver, tool_setup_fn).
    For AG2, score_fn and tool_setup_fn are None (uses test_suite.score_task).
    """
    if topology == "ag2":
        full_suite = load_online_benchmark_suite(total_tasks)
        by_bench: dict[str, list] = defaultdict(list)
        for ex in full_suite.examples:
            by_bench[ex.benchmark_name].append(ex)
        test_examples = []
        for bname, examples in sorted(by_bench.items()):
            bench_train = round(len(examples) * n_train / total_tasks)
            bench_train = min(bench_train, len(examples))
            test_examples.extend(examples[bench_train:])
        test_suite = OfficialBenchmarkSuite(test_examples)
        return test_suite, None, None, None

    elif topology == "appworld":
        from optpilot.data.benchmarks_appworld import load_appworld_examples, score_appworld
        from optpilot.tools.appworld_tools import AppWorldWrapper, build_tools
        all_examples = load_appworld_examples(
            total_tasks, splits=("train", "dev", "test_normal", "test_challenge"),
        )
        test_examples = all_examples[n_train:]
        test_suite = OfficialBenchmarkSuite(test_examples)
        lookup = {ex.prompt: ex for ex in all_examples}

        def tool_setup(tp):
            ex = lookup.get(tp)
            if ex is None:
                return None
            return build_tools(AppWorldWrapper(task_id=ex.task_id, experiment_name="offline_analysis"))

        def score_fn(tp, _dag, exec_trace):
            ex = lookup.get(tp)
            if ex is None:
                return 0.0
            pred = exec_trace.steps[-1].output_text if exec_trace.steps else ""
            return score_appworld(pred, ex.gold_answers[0])

        return test_suite, score_fn, lambda _: "AppWorld", tool_setup

    elif topology == "hyperagent":
        from optpilot.data.benchmarks_swebench import load_swebench_examples, score_swebench
        all_examples = load_swebench_examples(total_tasks)
        test_examples = all_examples[n_train:]
        test_suite = OfficialBenchmarkSuite(test_examples)
        lookup = {ex.prompt: ex for ex in all_examples}

        def tool_setup(tp):
            from optpilot.tools.hyperagent_tools import CodeEnvironment, build_tools
            ex = lookup.get(tp)
            repo = ex.metadata.get("repo", "") if ex else ""
            base_commit = ex.metadata.get("base_commit", "") if ex else ""
            return build_tools(CodeEnvironment(repo=repo, base_commit=base_commit))

        def score_fn(tp, _dag, exec_trace):
            ex = lookup.get(tp)
            if ex is None:
                return 0.0
            pred = exec_trace.steps[-1].output_text if exec_trace.steps else ""
            return score_swebench(pred, ex.gold_answers[0])

        return test_suite, score_fn, lambda _: "SWE-bench-Lite", tool_setup

    elif topology == "magentic":
        from optpilot.data.benchmarks_gaia import load_gaia_examples, score_gaia
        from optpilot.tools.magentic_tools import GeneralEnvironment, build_tools
        all_examples = load_gaia_examples(total_tasks)
        test_examples = all_examples[n_train:]
        test_suite = OfficialBenchmarkSuite(test_examples)
        lookup = {ex.prompt: ex for ex in all_examples}

        def tool_setup(tp):
            ex = lookup.get(tp)
            docs = ex.metadata.get("context_docs", {}) if ex else {}
            return build_tools(GeneralEnvironment(docs))

        def score_fn(tp, _dag, exec_trace):
            ex = lookup.get(tp)
            if ex is None:
                return 0.0
            pred = exec_trace.steps[-1].output_text if exec_trace.steps else ""
            return score_gaia(pred, ex.gold_answers[0])

        return test_suite, score_fn, lambda _: "GAIA", tool_setup

    elif topology == "simple_star":
        # Reuses GAIA benchmark + magentic tools
        return _load_benchmark_for_topology("magentic", total_tasks, n_train)

    elif topology == "simple_hier":
        # Reuses SWE-bench benchmark + hyperagent tools
        return _load_benchmark_for_topology("hyperagent", total_tasks, n_train)

    else:
        raise ValueError(f"Unknown topology: {topology}")


def run(
    openevolve_dir: str,
    model: str = "openai/gpt-oss-120b",
    total_tasks: int = 200,
    n_train: int = 100,
    concurrency: int = 512,
    timeout: int = 600,
    skip_test_eval: bool = False,
    apply_warmup: bool = True,
    topology: str = "ag2",
):
    openevolve_path = Path(openevolve_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = RESULTS_DIR / f"offline_analysis_{topology}_{timestamp}"

    print("=" * 65)
    print(f"  Offline Trace Analysis — OpenEvolve → Prior Extraction ({topology})")
    print("=" * 65)

    # --- Step 1: Load programs ---
    print("\n[1/6] Loading programs...")
    programs = load_programs(openevolve_path)

    # --- Step 2: Find train-improving mutations ---
    print("\n[2/6] Extracting train-improving mutations...")
    mutations = extract_train_improving_mutations(programs)
    if not mutations:
        print("  No train-improving mutations found. Exiting.")
        return

    # --- Step 3: Convert to DAGs ---
    print("\n[3/6] Converting code to DAGs...")
    for m in mutations:
        # Pre-convert so we can diff later
        if not hasattr(m, "parent_dag"):
            pass  # Will be done in validate step

    if skip_test_eval:
        # Skip test evaluation, use train-improving mutations directly
        print("\n  [skip_test_eval] Using train-improving mutations as-is")
        effective = []
        for m in mutations:
            parent_dag = code_to_dag(m.parent.solution)
            child_dag = code_to_dag(m.child.solution)
            effective.append(EffectiveMutation(
                parent=m.parent,
                child=m.child,
                train_score_delta=m.train_score_delta,
                train_accuracy_delta=m.train_accuracy_delta,
                parent_dag=parent_dag,
                child_dag=child_dag,
                fm_deltas={},
            ))
    else:
        # --- Step 4: Test set validation ---
        print(f"\n[4/6] Validating on held-out test set (topology={topology})...")
        test_suite, custom_score_fn, bench_resolver, tool_setup_fn = (
            _load_benchmark_for_topology(topology, total_tasks, n_train)
        )
        print(f"  Test set: {len(test_suite.tasks())} tasks")

        _BENCH_LABELS = {
            "ag2": "AG2_MathChat",
            "appworld": "AppWorld",
            "hyperagent": "SWE-bench-Lite",
            "magentic": "GAIA",
        }
        runner = OptPilotRunner(
            dag=None,
            model=model,
            benchmark_name=_BENCH_LABELS.get(topology, topology.upper()),
            score_fn=custom_score_fn or test_suite.score_task,
            benchmark_name_resolver=bench_resolver or test_suite.benchmark_name_for_task,
            timeout=timeout,
            tool_setup_fn=tool_setup_fn,
        )
        diagnoser = Diagnoser()

        effective = asyncio.run(
            validate_mutations_on_test(mutations, runner, diagnoser, test_suite, concurrency)
        )

    if not effective:
        print("\n  No mutations effective on test set. Exiting.")
        return

    # --- Step 5: Classify via LLM ---
    print("\n[5/7] Classifying effective mutations via LLM...")
    asyncio.run(classify_mutations_llm(effective, model=model))

    # --- Step 6: Summarize priors ---
    print("\n[6/7] Summarizing priors and Jacobian warmup data...")
    report = summarize_priors(effective, output_dir)

    # --- Step 7: Distill recipes ---
    print("\n[7/7] Distilling repair recipes...")
    recipes = asyncio.run(distill_recipes(effective, model=model))
    if recipes:
        # Save to output dir
        recipes_out = output_dir / "recipes.json"
        recipes_out.write_text(
            json.dumps([asdict(r) for r in recipes], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  Recipes saved to {recipes_out}")

    # --- Optional: Apply warmup ---
    if apply_warmup:
        from optpilot.config import topology_jacobian_dir, topology_recipes_dir
        warmup_path = output_dir / "jacobian_warmup.jsonl"
        priors_path = output_dir / "data_driven_priors.json"
        jac_dir = topology_jacobian_dir(topology)
        if warmup_path.exists():
            print(f"\n[Warmup] Applying offline experience to Jacobian ({topology})...")
            apply_jacobian_warmup(warmup_path, priors_path=priors_path, jacobian_dir=jac_dir)
        if recipes:
            print(f"[Warmup] Installing recipes to library ({topology})...")
            library = RecipeLibrary(base_dir=topology_recipes_dir(topology))
            library.add_batch(recipes)
            library.save()
            print(f"  {len(recipes)} recipes installed to {library.base_dir}")

    # --- Summary ---
    print(f"\n{'=' * 65}")
    print(f"  Offline Analysis Complete")
    print(f"{'=' * 65}")
    print(f"  Programs loaded:           {len(programs)}")
    print(f"  Train-improving mutations: {len(mutations)}")
    print(f"  Test-effective mutations:  {len(effective)}")
    classified = sum(1 for m in effective if m.observed_pattern)
    print(f"  Classified into patterns:  {classified}")
    if report.get("pattern_summary"):
        print(f"  Top patterns:")
        for pid, info in sorted(
            report["pattern_summary"].items(),
            key=lambda x: x[1]["total_effective_mutations"],
            reverse=True,
        )[:5]:
            print(f"    {pid}: {info['total_effective_mutations']} effective mutations")
    if recipes:
        print(f"  Recipes distilled:         {len(recipes)}")
        for r in recipes[:5]:
            print(f"    [{r.fm_group}] {r.recipe_id}: {r.action[:80]}")
    print(f"  Output: {output_dir}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    _TOPOLOGIES = ("ag2", "appworld", "hyperagent", "magentic", "simple_star", "simple_hier")
    parser = argparse.ArgumentParser(
        description="Analyze OpenEvolve traces → data-driven priors + Jacobian warmup",
    )
    parser.add_argument("--openevolve-dir", required=True, help="Path to OpenEvolve output directory")
    parser.add_argument("--topology", default="ag2", choices=_TOPOLOGIES,
                        help="MAS topology (determines benchmark for test validation)")
    parser.add_argument("--model", default="openai/gpt-oss-120b", help="Model for test evaluation")
    parser.add_argument("--total-tasks", type=int, default=200, help="Total benchmark tasks (train+test)")
    parser.add_argument("--n-train", type=int, default=100, help="Number of train tasks")
    parser.add_argument("--concurrency", type=int, default=512, help="Max concurrent tasks")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout per task")
    parser.add_argument("--skip-test-eval", action="store_true", help="Skip test set eval (use train-improving only)")
    parser.add_argument("--no-warmup", action="store_true", help="Don't apply warmup to Jacobian")
    args = parser.parse_args()

    run(
        openevolve_dir=args.openevolve_dir,
        topology=args.topology,
        model=args.model,
        total_tasks=args.total_tasks,
        n_train=args.n_train,
        concurrency=args.concurrency,
        timeout=args.timeout,
        skip_test_eval=args.skip_test_eval,
        apply_warmup=not args.no_warmup,
    )
