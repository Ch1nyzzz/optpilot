"""Orchestrator — controls the online optimization loop.

Coordinates: Runner → Diagnoser → dispatch to Skill Workflows (parallel).
"""

from __future__ import annotations

import asyncio
import copy
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TypeVar

from optpilot.config import LIBRARY_DIR
from optpilot.dag.core import MASDAG
from optpilot.data.fm_taxonomy_6group import GROUP_NAMES
from optpilot.models import FMLabel, FMProfile, FMLocalization, MASTrace, SkillBudget, SkillResult
from optpilot.modules.base_runner import MASRunner
from optpilot.modules.diagnoser import Diagnoser
from optpilot.skills.evolution import SkillEvolver
from optpilot.skills.forger import forge
from optpilot.skills.negatives import NegativesStore
from optpilot.skills.registry import get_skill, load_evolved_skill
from optpilot.tracking import Tracker


def rank_fm_groups(
    profiles: list[FMProfile],
    target_fm: str | None = None,
    min_support: int = 2,
) -> list[tuple[str, int]]:
    """Rank FM groups by frequency (descending). Returns [(fm_id, count), ...]."""
    fm_counts: Counter[str] = Counter()
    for p in profiles:
        for fm_id in p.active_fm_ids():
            fm_counts[fm_id] += 1

    if target_fm:
        count = fm_counts.get(target_fm, 0)
        return [(target_fm, count)] if count >= min_support else []

    eligible = [(fm_id, count) for fm_id, count in fm_counts.items() if count >= min_support]
    eligible.sort(key=lambda item: item[1], reverse=True)
    return eligible


T = TypeVar("T")


def _take_by_index(items: list[T], indices: list[int]) -> list[T]:
    return [items[i] for i in indices]


def _save_dag_if_possible(dag: object, path: str | Path) -> str:
    path_obj = Path(path)
    if hasattr(dag, "save"):
        dag.save(path_obj)
        return str(path_obj)
    return ""


def _task_dir_index(path: Path) -> int:
    return int(path.name.split("_", 1)[1])


def _trace_to_dict(trace) -> dict[str, object]:
    return {
        "trace_id": trace.trace_id,
        "mas_name": trace.mas_name,
        "llm_name": trace.llm_name,
        "benchmark_name": trace.benchmark_name,
        "trace_path": trace.trace_path,
        "task_key": trace.task_key,
        "task_success": trace.task_success,
        "task_score": trace.task_score,
        "latency_s": trace.latency_s,
    }


def _profile_to_dict(profile: FMProfile) -> dict[str, object]:
    return {
        "trace_id": profile.trace_id,
        "active_fm_ids": profile.active_fm_ids(),
        "labels": {
            fm_id: {
                "fm_name": label.fm_name,
                "category": label.category,
                "present": label.present,
                "confidence": label.confidence,
            }
            for fm_id, label in profile.labels.items()
        },
        "localization": {
            fm_id: {
                "agent": loc.agent,
                "step": loc.step,
                "context": loc.context,
                "root_cause": loc.root_cause,
                "dag_component": loc.dag_component,
            }
            for fm_id, loc in profile.localization.items()
        },
    }


def _profile_from_dict(data: dict[str, object]) -> FMProfile:
    profile = FMProfile(trace_id=int(data["trace_id"]))

    labels = data.get("labels", {})
    if isinstance(labels, dict):
        for fm_id, raw in labels.items():
            if not isinstance(raw, dict):
                continue
            profile.labels[str(fm_id)] = FMLabel(
                fm_id=str(fm_id),
                fm_name=str(raw.get("fm_name", fm_id)),
                category=str(raw.get("category", "")),
                present=bool(raw.get("present", False)),
                confidence=float(raw.get("confidence", 1.0)),
            )

    localization = data.get("localization", {})
    if isinstance(localization, dict):
        for fm_id, raw in localization.items():
            if not isinstance(raw, dict):
                continue
            profile.localization[str(fm_id)] = FMLocalization(
                agent=str(raw.get("agent", "")),
                step=str(raw.get("step", "")),
                context=str(raw.get("context", "")),
                root_cause=str(raw.get("root_cause", "")),
                dag_component=str(raw.get("dag_component", "other")),
            )

    return profile


def _write_json(path: str | Path, data: object) -> str:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path_obj)


def split_proposal_validation_indices(
    target_fm: str,
    profiles: list[FMProfile],
    batch_size: int = 5,
) -> tuple[list[int], list[int]] | None:
    """Split traces with target FM into proposal (5) and validation (5).

    Both sets only contain traces that exhibit the target FM.
    """
    matched_indices = [i for i, profile in enumerate(profiles) if target_fm in profile.active_fm_ids()]
    if len(matched_indices) < 2 * batch_size:
        # Not enough traces — use what we have, at least 1 per side
        if len(matched_indices) < 2:
            return None
        half = len(matched_indices) // 2
        return matched_indices[:half], matched_indices[half:]

    proposal_indices = matched_indices[:batch_size]
    validation_indices = matched_indices[batch_size: 2 * batch_size]
    return proposal_indices, validation_indices


class Orchestrator:
    """Online optimization loop controller — dispatches to Skill Workflows."""

    def __init__(
        self,
        runner: MASRunner,
        dag: MASDAG,
        use_wandb: bool = False,
        negatives_dir: str | Path | None = None,
        evolved_dir: str | Path | None = None,
        parallel: bool = True,
    ):
        self.runner = runner
        self.dag = dag
        self.diagnoser = Diagnoser()
        self.tracker = Tracker("skill_optimization", use_wandb=use_wandb)
        self.negatives_store = NegativesStore(Path(negatives_dir or LIBRARY_DIR / "negatives"))
        self.evolver = SkillEvolver(Path(evolved_dir or LIBRARY_DIR / "evolved_skills"))
        self.parallel = parallel

    def _load_persisted_traces(
        self,
        trace_base: str | Path,
        tasks: list[str],
    ) -> list:
        trace_root = Path(trace_base)
        if not trace_root.exists():
            raise FileNotFoundError(f"Persisted trace directory does not exist: {trace_root}")

        loaded: list[MASTrace | None] = [None] * len(tasks)
        for task_dir in sorted(trace_root.glob("task_*"), key=_task_dir_index):
            idx = _task_dir_index(task_dir)
            if idx >= len(tasks):
                continue
            trace_file = task_dir / "trace.txt"
            if not trace_file.exists():
                continue
            metadata_path = task_dir / "trace.json"
            metadata: dict[str, object] = {}
            if metadata_path.exists():
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                except Exception:
                    metadata = {}

            task_prompt = str(metadata.get("task_prompt", tasks[idx]))
            benchmark_name = str(
                metadata.get(
                    "benchmark_name",
                    self.runner._resolve_benchmark_name(task_prompt)
                    if hasattr(self.runner, "_resolve_benchmark_name")
                    else "",
                )
            )
            loaded[idx] = MASTrace(
                trace_id=idx,
                mas_name=str(metadata.get("mas_name", getattr(self.dag, "dag_id", "OptPilot"))),
                llm_name=str(metadata.get("llm_name", getattr(self.runner, "model", ""))),
                benchmark_name=benchmark_name,
                trajectory=trace_file.read_text(encoding="utf-8"),
                trace_path=str(trace_file),
                task_key=str(metadata.get("task_key", task_prompt[:50])),
                task_success=metadata.get("task_success"),
                task_score=metadata.get("task_score"),
                latency_s=metadata.get("latency_s"),
            )

        missing = [i for i, trace in enumerate(loaded) if trace is None]
        if missing:
            raise ValueError(
                f"Missing persisted traces for task indices: {missing[:10]}"
                f"{'...' if len(missing) > 10 else ''}"
            )
        return loaded  # type: ignore[return-value]

    def _persist_diagnosis_artifacts(
        self,
        output_dir: str | Path,
        traces: list,
        profiles: list[FMProfile],
        fm_ranking: list[tuple[str, int]],
        skill_jobs: list[dict] | None = None,
        source_trace_base: str | Path | None = None,
    ) -> dict[str, str]:
        base = Path(output_dir)
        base.mkdir(parents=True, exist_ok=True)

        trace_entries: list[dict[str, object]] = []
        grouped_entries: dict[str, list[dict[str, object]]] = {}
        for idx, (trace, profile) in enumerate(zip(traces, profiles)):
            entry = {
                "task_index": idx,
                **_trace_to_dict(trace),
                **_profile_to_dict(profile),
            }
            trace_entries.append(entry)
            for fm_id in profile.active_fm_ids():
                grouped_entries.setdefault(fm_id, []).append(entry)

        summary = {
            "n_traces": len(traces),
            "fm_ranking": [{"fm_id": fm_id, "count": count} for fm_id, count in fm_ranking],
            "source_trace_base": str(source_trace_base) if source_trace_base else "",
        }
        written = {
            "summary": _write_json(base / "summary.json", summary),
            "profiles": _write_json(base / "profiles.json", trace_entries),
        }

        fm_group_dir = base / "fm_groups"
        for fm_id, entries in grouped_entries.items():
            _write_json(fm_group_dir / f"{fm_id}.json", entries)

        if skill_jobs is not None:
            job_dir = base / "skill_jobs"
            for job in skill_jobs:
                proposal_idx = job["proposal_idx"]
                validation_idx = job["validation_idx"]
                payload = {
                    "fm_id": job["fm_id"],
                    "proposal_idx": proposal_idx,
                    "validation_idx": validation_idx,
                    "proposal_traces": [trace_entries[i] for i in proposal_idx],
                    "validation_traces": [trace_entries[i] for i in validation_idx],
                }
                _write_json(job_dir / f"{job['fm_id']}.json", payload)

        return written

    def _load_persisted_diagnosis(
        self,
        diagnose_dir: str | Path,
        tasks: list[str],
        target_fm: str | None = None,
    ) -> tuple[list[MASTrace], list[FMProfile], list[dict], list[tuple[str, int]]]:
        base = Path(diagnose_dir)
        if not base.exists():
            raise FileNotFoundError(f"Persisted diagnose directory does not exist: {base}")

        profiles_path = base / "profiles.json"
        if not profiles_path.exists():
            raise FileNotFoundError(f"Persisted diagnose profiles are missing: {profiles_path}")

        raw_entries = json.loads(profiles_path.read_text(encoding="utf-8"))
        if not isinstance(raw_entries, list):
            raise ValueError(f"Persisted diagnose profiles must be a JSON list: {profiles_path}")

        traces: list[MASTrace | None] = [None] * len(tasks)
        profiles: list[FMProfile | None] = [None] * len(tasks)
        for raw in raw_entries:
            if not isinstance(raw, dict):
                continue
            idx = int(raw.get("task_index", -1))
            if idx < 0 or idx >= len(tasks):
                continue
            trace_path = str(raw.get("trace_path", ""))
            if not trace_path:
                raise ValueError(f"Persisted diagnose entry missing trace_path for task index {idx}")
            trace_file = Path(trace_path)
            if not trace_file.exists():
                raise FileNotFoundError(f"Persisted trace file referenced by diagnose artifacts is missing: {trace_file}")

            traces[idx] = MASTrace(
                trace_id=int(raw.get("trace_id", idx)),
                mas_name=str(raw.get("mas_name", getattr(self.dag, "dag_id", "OptPilot"))),
                llm_name=str(raw.get("llm_name", getattr(self.runner, "model", ""))),
                benchmark_name=str(raw.get("benchmark_name", "")),
                trajectory=trace_file.read_text(encoding="utf-8"),
                trace_path=str(trace_file),
                task_key=str(raw.get("task_key", tasks[idx][:50])),
                task_success=raw.get("task_success"),
                task_score=raw.get("task_score"),
                latency_s=raw.get("latency_s"),
            )
            profiles[idx] = _profile_from_dict(raw)

        missing = [i for i, trace in enumerate(traces) if trace is None or profiles[i] is None]
        if missing:
            raise ValueError(
                f"Persisted diagnose artifacts missing task indices: {missing[:10]}"
                f"{'...' if len(missing) > 10 else ''}"
            )

        summary_path = base / "summary.json"
        summary_data: dict[str, object] = {}
        if summary_path.exists():
            loaded = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                summary_data = loaded

        summary_ranking = summary_data.get("fm_ranking", [])
        fm_ranking = [
            (str(item.get("fm_id", "")), int(item.get("count", 0)))
            for item in summary_ranking
            if isinstance(item, dict) and item.get("fm_id")
        ]
        if target_fm:
            fm_ranking = [item for item in fm_ranking if item[0] == target_fm]
        if not fm_ranking:
            fm_ranking = rank_fm_groups(profiles, target_fm)

        skill_jobs_dir = base / "skill_jobs"
        skill_jobs: list[dict] = []
        if skill_jobs_dir.exists():
            for job_path in sorted(skill_jobs_dir.glob("*.json")):
                payload = json.loads(job_path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    continue
                fm_id = str(payload.get("fm_id", ""))
                if not fm_id:
                    continue
                if target_fm and fm_id != target_fm:
                    continue
                skill_jobs.append({
                    "fm_id": fm_id,
                    "proposal_idx": [int(i) for i in payload.get("proposal_idx", [])],
                    "validation_idx": [int(i) for i in payload.get("validation_idx", [])],
                    "prior_negatives": self.negatives_store.load(fm_id),
                })

        if not skill_jobs:
            for fm_id, _count in fm_ranking:
                split = split_proposal_validation_indices(fm_id, profiles)
                if split is None:
                    continue
                proposal_idx, validation_idx = split
                skill_jobs.append({
                    "fm_id": fm_id,
                    "proposal_idx": proposal_idx,
                    "validation_idx": validation_idx,
                    "prior_negatives": self.negatives_store.load(fm_id),
                })

        return traces, profiles, skill_jobs, fm_ranking

    def optimize(
        self,
        tasks: list[str],
        max_rounds: int = 5,
        target_fm: str | None = None,
        budget: SkillBudget | None = None,
    ) -> dict:
        """Run the full online optimization loop with Skill Workflows.

        Skills for different FM groups can run in parallel since they each
        work on independent copies of the DAG.
        """
        current_dag = self.dag

        print(f"=== OptPilot Skill Workflow Optimization ===")
        print(f"Tasks: {len(tasks)}, Max rounds: {max_rounds}, Parallel: {self.parallel}")
        print()

        results: list[dict] = []

        for round_i in range(max_rounds):
            print(f"=== Round {round_i + 1}/{max_rounds} ===")

            # 1. Run MAS + Diagnose
            print(f"  [1/2] Running MAS on {len(tasks)} tasks + diagnosing...")
            traces = self.runner.run_batch(tasks, dag=current_dag)
            profiles = self.diagnoser.diagnose_batch(traces)

            # 2. Rank FM groups
            fm_ranking = rank_fm_groups(profiles, target_fm)
            if not fm_ranking:
                print(f"  No active FM groups with ≥2 traces. Done!")
                break

            print(f"  Active FM groups: {', '.join(f'{fid}({c})' for fid, c in fm_ranking)}")

            # 3. Prepare skill dispatches
            skill_jobs: list[dict] = []
            for fm_id, _fm_count in fm_ranking:
                split = split_proposal_validation_indices(fm_id, profiles)
                if split is None:
                    print(f"  Skipping Group-{fm_id}: cannot split proposal/validation.")
                    continue

                proposal_idx, validation_idx = split
                prior_negatives = self.negatives_store.load(fm_id)

                skill_jobs.append({
                    "fm_id": fm_id,
                    "proposal_idx": proposal_idx,
                    "validation_idx": validation_idx,
                    "prior_negatives": prior_negatives,
                })

            if not skill_jobs:
                print(f"  No dispatchable skills. Done!")
                break

            # 4. Dispatch skills (parallel or sequential)
            if self.parallel and len(skill_jobs) > 1:
                skill_results = self._run_parallel(
                    skill_jobs, current_dag, tasks, traces, profiles, budget,
                )
            else:
                skill_results = self._run_sequential(
                    skill_jobs, current_dag, tasks, traces, profiles, budget,
                )

            # 5. Collect results — pick the best successful skill
            best_result: SkillResult | None = None
            for fm_id, result in skill_results:
                fm_name = GROUP_NAMES.get(fm_id, fm_id)
                round_entry = {
                    "round": round_i + 1,
                    "fm_id": fm_id,
                    "fm_name": fm_name,
                    "success": result.success,
                    "inner_iterations": result.inner_iterations,
                    "outer_rounds": result.outer_rounds,
                    "final_fm_rate": result.final_fm_rate,
                    "final_pass_rate": result.final_pass_rate,
                    "n_negatives": len(result.negatives),
                }
                results.append(round_entry)
                self.tracker.log(round_entry, step=len(results) - 1)

                if result.success:
                    print(f"  Group-{fm_id} ({fm_name}): REPAIRED "
                          f"(fm={result.final_fm_rate:.2f}, pass={result.final_pass_rate:.3f})")
                    if best_result is None or result.final_pass_rate > best_result.final_pass_rate:
                        best_result = result
                else:
                    print(f"  Group-{fm_id} ({fm_name}): FAILED after {result.outer_rounds} rounds")
                    # Persist negatives for future runs
                    self.negatives_store.extend(fm_id, result.negatives)
                    self.evolver.record_failure(fm_id)

                    # Meta-evolution check
                    if self.evolver.should_evolve(fm_id):
                        print(f"  Triggering meta-evolution for Skill-{fm_id}...")
                        all_neg = self.negatives_store.load(fm_id)
                        skill = get_skill(fm_id)
                        evolved_path = self.evolver.evolve_skill(fm_id, skill, all_neg)
                        if evolved_path:
                            print(f"    Evolved skill saved: {evolved_path}")
                            load_evolved_skill(fm_id, evolved_path)

            # 6. Update DAG if any skill succeeded
            if best_result and best_result.dag:
                current_dag = best_result.dag
                self.dag = current_dag
                self.evolver.reset_failures(best_result.fm_id)
                print(f"  Adopting DAG from Skill-{best_result.fm_id}.")
            else:
                print(f"  No successful repairs this round.")

            print()

        # Summary
        summary = {
            "total_rounds": len(set(r["round"] for r in results)),
            "results": results,
        }
        self.tracker.save_local("skill_optimization.json")
        return summary

    def _run_skill(
        self,
        job: dict,
        dag: MASDAG,
        tasks: list[str],
        traces: list,
        profiles: list[FMProfile],
        budget: SkillBudget | None,
    ) -> tuple[str, SkillResult]:
        """Run a single skill job."""
        fm_id = job["fm_id"]
        proposal_idx = job["proposal_idx"]
        validation_idx = job["validation_idx"]
        prior_negatives = job["prior_negatives"]

        # Each skill gets its own DAG copy
        skill_dag = copy.deepcopy(dag)

        try:
            skill = get_skill(fm_id)
        except KeyError:
            return fm_id, SkillResult(success=False, fm_id=fm_id)

        # Inject prior negatives into the skill run
        skill_budget = budget or SkillBudget()

        result = skill.run(
            original_dag=skill_dag,
            proposal_traces=_take_by_index(traces, proposal_idx),
            proposal_profiles=_take_by_index(profiles, proposal_idx),
            proposal_tasks=_take_by_index(tasks, proposal_idx),
            validation_tasks=_take_by_index(tasks, validation_idx),
            runner=self.runner,
            diagnoser=self.diagnoser,
            budget=SkillBudget(
                max_llm_calls=skill_budget.max_llm_calls,
                max_batch_runs=skill_budget.max_batch_runs,
                max_wall_time_s=skill_budget.max_wall_time_s,
            ),
            prior_negatives=prior_negatives,
        )

        return fm_id, result

    async def aoptimize(
        self,
        tasks: list[str],
        max_rounds: int = 1,
        target_fm: str | None = None,
        budget: SkillBudget | None = None,
        concurrency: int = 256,
        trace_output_base: str | Path | None = None,
        dag_output_base: str | Path | None = None,
    ) -> dict:
        """Async optimization: diagnose → parallel skills → forge all successes.

        Default max_rounds=1: each skill's inner loop handles convergence.
        Outer rounds are available but rarely needed.
        """
        current_dag = self.dag

        print(f"=== OptPilot Skill Workflow Optimization (async) ===")
        print(f"Tasks: {len(tasks)}, Max rounds: {max_rounds}, Concurrency: {concurrency}")
        print()

        results: list[dict] = []
        dag_versions: list[dict[str, str | int]] = []
        if dag_output_base:
            initial_path = _save_dag_if_possible(current_dag, Path(dag_output_base) / "initial.yaml")
            if initial_path:
                dag_versions.append({"stage": "initial", "path": initial_path})

        for round_i in range(max_rounds):
            print(f"=== Round {round_i + 1}/{max_rounds} ===")
            round_trace_dir = (
                Path(trace_output_base) / f"round_{round_i + 1}" / "train"
                if trace_output_base else None
            )
            round_dag_dir = Path(dag_output_base) / f"round_{round_i + 1}" if dag_output_base else None
            if round_dag_dir:
                start_path = _save_dag_if_possible(current_dag, round_dag_dir / "start.yaml")
                if start_path:
                    dag_versions.append({"stage": "round_start", "round": round_i + 1, "path": start_path})

            # 1. Run MAS + Diagnose
            print(f"  [1/3] Running MAS on {len(tasks)} tasks + diagnosing...")
            traces = await self.runner.arun_batch(
                tasks,
                dag=current_dag,
                output_base=round_trace_dir,
                max_concurrency=concurrency,
            )
            profiles = await self.diagnoser.adiagnose_batch(traces)

            # 2. Rank FM groups
            fm_ranking = rank_fm_groups(profiles, target_fm)
            if not fm_ranking:
                print(f"  No active FM groups with ≥2 traces. Done!")
                break

            print(f"  Active FM groups: {', '.join(f'{fid}({c})' for fid, c in fm_ranking)}")

            # 3. Prepare and dispatch skills
            skill_jobs: list[dict] = []
            for fm_id, _fm_count in fm_ranking:
                split = split_proposal_validation_indices(fm_id, profiles)
                if split is None:
                    print(f"  Skipping Group-{fm_id}: cannot split proposal/validation.")
                    continue
                proposal_idx, validation_idx = split
                prior_negatives = self.negatives_store.load(fm_id)
                skill_jobs.append({
                    "fm_id": fm_id,
                    "proposal_idx": proposal_idx,
                    "validation_idx": validation_idx,
                    "prior_negatives": prior_negatives,
                })

            if not skill_jobs:
                print(f"  No dispatchable skills. Done!")
                break

            print(f"  [2/3] Dispatching {len(skill_jobs)} skills in parallel...")
            skill_coros = [
                self._arun_skill(job, current_dag, tasks, traces, profiles, budget, concurrency)
                for job in skill_jobs
            ]
            skill_results_raw = await asyncio.gather(*skill_coros)

            # 4. Collect results
            successful_results: list[SkillResult] = []
            for fm_id, result in skill_results_raw:
                fm_name = GROUP_NAMES.get(fm_id, fm_id)
                round_entry = {
                    "round": round_i + 1,
                    "fm_id": fm_id,
                    "fm_name": fm_name,
                    "success": result.success,
                    "inner_iterations": result.inner_iterations,
                    "outer_rounds": result.outer_rounds,
                    "final_fm_rate": result.final_fm_rate,
                    "final_pass_rate": result.final_pass_rate,
                    "n_negatives": len(result.negatives),
                }
                results.append(round_entry)
                self.tracker.log(round_entry, step=len(results) - 1)

                if result.success:
                    print(f"  Group-{fm_id} ({fm_name}): REPAIRED "
                          f"(fm={result.final_fm_rate:.2f}, pass={result.final_pass_rate:.3f})")
                    successful_results.append(result)
                    if round_dag_dir and result.dag is not None:
                        skill_path = _save_dag_if_possible(
                            result.dag,
                            round_dag_dir / f"skill_{fm_id}_success.yaml",
                        )
                        if skill_path:
                            dag_versions.append({
                                "stage": "skill_success",
                                "round": round_i + 1,
                                "fm_id": fm_id,
                                "path": skill_path,
                            })
                else:
                    print(f"  Group-{fm_id} ({fm_name}): FAILED after {result.outer_rounds} rounds")
                    self.negatives_store.extend(fm_id, result.negatives)
                    self.evolver.record_failure(fm_id)

            # 5. Forge: merge ALL successful skill changes
            if successful_results:
                if len(successful_results) == 1:
                    merged_dag = successful_results[0].dag
                    print(f"  [3/3] Single success — adopting DAG from Skill-{successful_results[0].fm_id}.")
                else:
                    print(f"  [3/3] Forging {len(successful_results)} successful repairs...")
                    merged_dag = await forge(current_dag, successful_results)
                    print(f"  Forge complete — merged changes from: "
                          f"{', '.join(r.fm_id for r in successful_results)}")

                current_dag = merged_dag
                self.dag = current_dag
                for r in successful_results:
                    self.evolver.reset_failures(r.fm_id)
            else:
                print(f"  No successful repairs this round.")

            if round_dag_dir:
                final_path = _save_dag_if_possible(current_dag, round_dag_dir / "final.yaml")
                if final_path:
                    dag_versions.append({"stage": "round_final", "round": round_i + 1, "path": final_path})

            print()

        if dag_output_base:
            final_path = _save_dag_if_possible(current_dag, Path(dag_output_base) / "final.yaml")
            if final_path:
                dag_versions.append({"stage": "final", "path": final_path})

        summary = {
            "total_rounds": len(set(r["round"] for r in results)),
            "results": results,
            "dag_versions": dag_versions,
        }
        self.tracker.save_local("skill_optimization.json")
        return summary

    async def aoptimize_from_traces(
        self,
        tasks: list[str],
        trace_base: str | Path,
        target_fm: str | None = None,
        budget: SkillBudget | None = None,
        concurrency: int = 256,
        trace_output_base: str | Path | None = None,
        dag_output_base: str | Path | None = None,
    ) -> dict:
        """Warm-start optimization from persisted train traces.

        This re-runs diagnosis on existing traces and dispatches skills without
        re-executing the train set.
        """
        current_dag = self.dag
        results: list[dict] = []
        dag_versions: list[dict[str, str | int]] = []
        if dag_output_base:
            initial_path = _save_dag_if_possible(current_dag, Path(dag_output_base) / "initial.yaml")
            if initial_path:
                dag_versions.append({"stage": "initial", "path": initial_path})

        round_root = Path(trace_output_base) / "round_1" if trace_output_base else None
        round_dag_dir = Path(dag_output_base) / "round_1" if dag_output_base else None
        if round_dag_dir:
            start_path = _save_dag_if_possible(current_dag, round_dag_dir / "start.yaml")
            if start_path:
                dag_versions.append({"stage": "round_start", "round": 1, "path": start_path})

        print("=== OptPilot Skill Workflow Warm Start (async) ===")
        print(f"Tasks: {len(tasks)}, Trace source: {trace_base}, Concurrency: {concurrency}")
        print()
        print("  [1/3] Loading persisted train traces + re-diagnosing...")
        traces = self._load_persisted_traces(trace_base, tasks)
        if round_root:
            _write_json(
                round_root / "reused_train_manifest.json",
                {
                    "source_trace_base": str(trace_base),
                    "n_traces": len(traces),
                    "traces": [
                        {
                            "task_index": idx,
                            "trace_path": trace.trace_path,
                            "task_key": trace.task_key,
                            "benchmark_name": trace.benchmark_name,
                        }
                        for idx, trace in enumerate(traces)
                    ],
                },
            )
        profiles = await self.diagnoser.adiagnose_batch(traces)

        fm_ranking = rank_fm_groups(profiles, target_fm)
        if not fm_ranking:
            print("  No active FM groups with ≥2 traces. Done!")
            return {"total_rounds": 0, "results": [], "dag_versions": dag_versions}

        print(f"  Active FM groups: {', '.join(f'{fid}({c})' for fid, c in fm_ranking)}")

        skill_jobs: list[dict] = []
        for fm_id, _fm_count in fm_ranking:
            split = split_proposal_validation_indices(fm_id, profiles)
            if split is None:
                print(f"  Skipping Group-{fm_id}: cannot split proposal/validation.")
                continue
            proposal_idx, validation_idx = split
            skill_jobs.append({
                "fm_id": fm_id,
                "proposal_idx": proposal_idx,
                "validation_idx": validation_idx,
                "prior_negatives": self.negatives_store.load(fm_id),
            })

        diagnosis_dir = round_root / "diagnose" if round_root else None
        if diagnosis_dir:
            self._persist_diagnosis_artifacts(
                output_dir=diagnosis_dir,
                traces=traces,
                profiles=profiles,
                fm_ranking=fm_ranking,
                skill_jobs=skill_jobs,
                source_trace_base=trace_base,
            )

        if not skill_jobs:
            print("  No dispatchable skills. Done!")
            return {"total_rounds": 1, "results": [], "dag_versions": dag_versions}

        print(f"  [2/3] Dispatching {len(skill_jobs)} skills in parallel...")
        skill_coros = [
            self._arun_skill(job, current_dag, tasks, traces, profiles, budget, concurrency)
            for job in skill_jobs
        ]
        skill_results_raw = await asyncio.gather(*skill_coros)

        successful_results: list[SkillResult] = []
        for fm_id, result in skill_results_raw:
            fm_name = GROUP_NAMES.get(fm_id, fm_id)
            round_entry = {
                "round": 1,
                "fm_id": fm_id,
                "fm_name": fm_name,
                "success": result.success,
                "inner_iterations": result.inner_iterations,
                "outer_rounds": result.outer_rounds,
                "final_fm_rate": result.final_fm_rate,
                "final_pass_rate": result.final_pass_rate,
                "n_negatives": len(result.negatives),
            }
            results.append(round_entry)
            self.tracker.log(round_entry, step=len(results) - 1)

            if result.success:
                print(
                    f"  Group-{fm_id} ({fm_name}): REPAIRED "
                    f"(fm={result.final_fm_rate:.2f}, pass={result.final_pass_rate:.3f})"
                )
                successful_results.append(result)
                if round_dag_dir and result.dag is not None:
                    skill_path = _save_dag_if_possible(
                        result.dag,
                        round_dag_dir / f"skill_{fm_id}_success.yaml",
                    )
                    if skill_path:
                        dag_versions.append({
                            "stage": "skill_success",
                            "round": 1,
                            "fm_id": fm_id,
                            "path": skill_path,
                        })
            else:
                print(f"  Group-{fm_id} ({fm_name}): FAILED after {result.outer_rounds} rounds")
                self.negatives_store.extend(fm_id, result.negatives)
                self.evolver.record_failure(fm_id)

        if successful_results:
            if len(successful_results) == 1:
                merged_dag = successful_results[0].dag
                print(f"  [3/3] Single success — adopting DAG from Skill-{successful_results[0].fm_id}.")
            else:
                print(f"  [3/3] Forging {len(successful_results)} successful repairs...")
                merged_dag = await forge(current_dag, successful_results)
                print(
                    "  Forge complete — merged changes from: "
                    f"{', '.join(r.fm_id for r in successful_results)}"
                )

            current_dag = merged_dag
            self.dag = current_dag
            for r in successful_results:
                self.evolver.reset_failures(r.fm_id)
        else:
            print("  No successful repairs this round.")

        if round_dag_dir:
            final_round_path = _save_dag_if_possible(current_dag, round_dag_dir / "final.yaml")
            if final_round_path:
                dag_versions.append({"stage": "round_final", "round": 1, "path": final_round_path})
        if dag_output_base:
            final_path = _save_dag_if_possible(current_dag, Path(dag_output_base) / "final.yaml")
            if final_path:
                dag_versions.append({"stage": "final", "path": final_path})

        summary = {
            "total_rounds": 1,
            "results": results,
            "dag_versions": dag_versions,
            "trace_source": str(trace_base),
            "diagnose_dir": str(diagnosis_dir) if diagnosis_dir else "",
        }
        self.tracker.save_local("skill_optimization.json")
        return summary

    async def aoptimize_from_diagnose(
        self,
        tasks: list[str],
        diagnose_dir: str | Path,
        target_fm: str | None = None,
        budget: SkillBudget | None = None,
        concurrency: int = 256,
        dag_output_base: str | Path | None = None,
    ) -> dict:
        """Warm-start optimization directly from persisted diagnose artifacts."""
        current_dag = self.dag
        results: list[dict] = []
        dag_versions: list[dict[str, str | int]] = []
        if dag_output_base:
            initial_path = _save_dag_if_possible(current_dag, Path(dag_output_base) / "initial.yaml")
            if initial_path:
                dag_versions.append({"stage": "initial", "path": initial_path})

        round_dag_dir = Path(dag_output_base) / "round_1" if dag_output_base else None
        if round_dag_dir:
            start_path = _save_dag_if_possible(current_dag, round_dag_dir / "start.yaml")
            if start_path:
                dag_versions.append({"stage": "round_start", "round": 1, "path": start_path})

        print("=== OptPilot Skill Workflow Diagnose Reuse (async) ===")
        print(f"Tasks: {len(tasks)}, Diagnose source: {diagnose_dir}, Concurrency: {concurrency}")
        print()
        print("  [1/3] Loading persisted diagnose artifacts...")
        traces, profiles, skill_jobs, fm_ranking = self._load_persisted_diagnosis(
            diagnose_dir=diagnose_dir,
            tasks=tasks,
            target_fm=target_fm,
        )

        if not fm_ranking:
            print("  No active FM groups with ≥2 traces. Done!")
            return {
                "total_rounds": 0,
                "results": [],
                "dag_versions": dag_versions,
                "diagnose_source": str(diagnose_dir),
            }

        print(f"  Active FM groups: {', '.join(f'{fid}({c})' for fid, c in fm_ranking)}")

        if not skill_jobs:
            print("  No dispatchable skills. Done!")
            return {
                "total_rounds": 1,
                "results": [],
                "dag_versions": dag_versions,
                "diagnose_source": str(diagnose_dir),
            }

        print(f"  [2/3] Dispatching {len(skill_jobs)} skills in parallel...")
        skill_coros = [
            self._arun_skill(job, current_dag, tasks, traces, profiles, budget, concurrency)
            for job in skill_jobs
        ]
        skill_results_raw = await asyncio.gather(*skill_coros)

        successful_results: list[SkillResult] = []
        for fm_id, result in skill_results_raw:
            fm_name = GROUP_NAMES.get(fm_id, fm_id)
            round_entry = {
                "round": 1,
                "fm_id": fm_id,
                "fm_name": fm_name,
                "success": result.success,
                "inner_iterations": result.inner_iterations,
                "outer_rounds": result.outer_rounds,
                "final_fm_rate": result.final_fm_rate,
                "final_pass_rate": result.final_pass_rate,
                "n_negatives": len(result.negatives),
            }
            results.append(round_entry)
            self.tracker.log(round_entry, step=len(results) - 1)

            if result.success:
                print(
                    f"  Group-{fm_id} ({fm_name}): REPAIRED "
                    f"(fm={result.final_fm_rate:.2f}, pass={result.final_pass_rate:.3f})"
                )
                successful_results.append(result)
                if round_dag_dir and result.dag is not None:
                    skill_path = _save_dag_if_possible(
                        result.dag,
                        round_dag_dir / f"skill_{fm_id}_success.yaml",
                    )
                    if skill_path:
                        dag_versions.append({
                            "stage": "skill_success",
                            "round": 1,
                            "fm_id": fm_id,
                            "path": skill_path,
                        })
            else:
                print(f"  Group-{fm_id} ({fm_name}): FAILED after {result.outer_rounds} rounds")
                self.negatives_store.extend(fm_id, result.negatives)
                self.evolver.record_failure(fm_id)

        if successful_results:
            if len(successful_results) == 1:
                merged_dag = successful_results[0].dag
                print(f"  [3/3] Single success — adopting DAG from Skill-{successful_results[0].fm_id}.")
            else:
                print(f"  [3/3] Forging {len(successful_results)} successful repairs...")
                merged_dag = await forge(current_dag, successful_results)
                print(
                    "  Forge complete — merged changes from: "
                    f"{', '.join(r.fm_id for r in successful_results)}"
                )

            current_dag = merged_dag
            self.dag = current_dag
            for r in successful_results:
                self.evolver.reset_failures(r.fm_id)
        else:
            print("  No successful repairs this round.")

        if round_dag_dir:
            final_round_path = _save_dag_if_possible(current_dag, round_dag_dir / "final.yaml")
            if final_round_path:
                dag_versions.append({"stage": "round_final", "round": 1, "path": final_round_path})
        if dag_output_base:
            final_path = _save_dag_if_possible(current_dag, Path(dag_output_base) / "final.yaml")
            if final_path:
                dag_versions.append({"stage": "final", "path": final_path})

        summary = {
            "total_rounds": 1,
            "results": results,
            "dag_versions": dag_versions,
            "diagnose_source": str(diagnose_dir),
        }
        self.tracker.save_local("skill_optimization.json")
        return summary

    async def _arun_skill(
        self,
        job: dict,
        dag: MASDAG,
        tasks: list[str],
        traces: list,
        profiles: list[FMProfile],
        budget: SkillBudget | None,
        concurrency: int = 256,
    ) -> tuple[str, SkillResult]:
        """Async version of _run_skill."""
        fm_id = job["fm_id"]
        proposal_idx = job["proposal_idx"]
        validation_idx = job["validation_idx"]
        prior_negatives = job["prior_negatives"]

        skill_dag = copy.deepcopy(dag)

        try:
            skill = get_skill(fm_id)
        except KeyError:
            return fm_id, SkillResult(success=False, fm_id=fm_id)

        skill_budget = budget or SkillBudget()

        result = await skill.arun(
            original_dag=skill_dag,
            proposal_traces=_take_by_index(traces, proposal_idx),
            proposal_profiles=_take_by_index(profiles, proposal_idx),
            proposal_tasks=_take_by_index(tasks, proposal_idx),
            validation_tasks=_take_by_index(tasks, validation_idx),
            runner=self.runner,
            diagnoser=self.diagnoser,
            budget=SkillBudget(
                max_llm_calls=skill_budget.max_llm_calls,
                max_batch_runs=skill_budget.max_batch_runs,
                max_wall_time_s=skill_budget.max_wall_time_s,
            ),
            prior_negatives=prior_negatives,
            concurrency=concurrency,
        )

        return fm_id, result

    def _run_parallel(
        self, jobs, dag, tasks, traces, profiles, budget,
    ) -> list[tuple[str, SkillResult]]:
        """Run multiple skills in parallel using threads."""
        results: list[tuple[str, SkillResult]] = []

        with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
            futures = {
                executor.submit(self._run_skill, job, dag, tasks, traces, profiles, budget): job
                for job in jobs
            }
            for future in as_completed(futures):
                job = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    fm_id = job["fm_id"]
                    print(f"  Skill-{fm_id} failed with error: {e}")
                    results.append((fm_id, SkillResult(success=False, fm_id=fm_id)))

        return results

    def _run_sequential(
        self, jobs, dag, tasks, traces, profiles, budget,
    ) -> list[tuple[str, SkillResult]]:
        """Run skills sequentially."""
        results: list[tuple[str, SkillResult]] = []
        for job in jobs:
            result = self._run_skill(job, dag, tasks, traces, profiles, budget)
            results.append(result)
        return results
