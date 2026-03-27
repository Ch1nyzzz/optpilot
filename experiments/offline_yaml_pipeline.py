"""Offline YAML Pipeline - 高并发版本。

MAST-Data → Diagnose (all FMs, 并行) → YAML Optimize → Judge (并行) → Library.

并发策略 (RPM=3000):
- trace 间: asyncio 并发处理多条 trace
- FM 诊断: 同一 trace 内多个 FM 异步 localize
- Judge: 同一 trace 内多个 FM 异步评估

Usage:
    python -m experiments.offline_yaml_pipeline --yaml dags/ag2_mathchat.yaml --max-traces 10
    python -m experiments.offline_yaml_pipeline --yaml dags/ag2_mathchat.yaml --group B --benchmark GSM --workers 30
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optpilot.config import LIBRARY_DIR, OFFLINE_YAML_MAX_WORKERS, RESULTS_DIR
from optpilot.data.fm_taxonomy_6group import GROUP_NAMES
from optpilot.data.loader import load_traces, print_fm_stats
from optpilot.library.repair_library import RepairLibrary
from optpilot.modules.diagnoser import Diagnoser
from optpilot.modules.judge import Judge
from optpilot.modules.yaml_optimizer import YAMLOptimizer
from optpilot.models import (
    FMProfile, JudgeVerdict, MASTrace,
    RepairCandidate, RepairAction, RepairType,
)


def _yaml_diff_summary(original: str, modified: str) -> str:
    orig_lines = set(original.strip().splitlines())
    mod_lines = set(modified.strip().splitlines())
    added = mod_lines - orig_lines
    removed = orig_lines - mod_lines
    parts = []
    if added:
        parts.append(f"+{len(added)} lines")
    if removed:
        parts.append(f"-{len(removed)} lines")
    return ", ".join(parts) if parts else "no diff"


async def _judge_one_fm(
    judge: Judge, trace: MASTrace, fm_id: str,
    candidate: RepairCandidate, profile: FMProfile,
) -> tuple[str, JudgeVerdict]:
    """Judge 单个 FM，返回 (fm_id, verdict)。"""
    verdict = await judge.aevaluate(trace, fm_id, candidate, profile)
    return fm_id, verdict


async def _process_one_trace(
    trace: MASTrace,
    yaml_path: Path,
    diagnoser: Diagnoser,
    optimizer: YAMLOptimizer,
    judge: Judge,
    judge_workers: int,
) -> dict | None:
    """处理单条 trace 的完整流程: diagnose → optimize → judge。"""
    active_fms = trace.active_fm_ids()
    if not active_fms:
        return None

    t0 = time.time()

    # 1. Diagnose (内部已异步化)
    profile = await diagnoser.adiagnose(trace)

    diag_time = time.time() - t0

    # 2. Optimize YAML
    t1 = time.time()
    opt_result = await optimizer.aoptimize(yaml_path, profile, trace)
    opt_time = time.time() - t1

    # 3. Judge (并行评估所有 FM)
    candidate = RepairCandidate(
        fm_id=",".join(profile.active_fm_ids()),
        actions=[RepairAction(
            repair_type=RepairType.CONFIG_CHANGE,
            target="YAML",
            description=opt_result["analysis"][:200],
        )],
        description=opt_result["analysis"][:200],
        source="yaml_optimizer",
    )

    t2 = time.time()
    verdicts: dict[str, JudgeVerdict] = {}
    fm_ids = profile.active_fm_ids()
    judge_semaphore = asyncio.Semaphore(min(judge_workers, len(fm_ids)))

    async def judge_one(fm_id: str) -> None:
        async with judge_semaphore:
            try:
                _, verdict = await _judge_one_fm(judge, trace, fm_id, candidate, profile)
                verdicts[fm_id] = verdict
            except Exception as e:
                print(f"    Judge failed for FM-{fm_id}: {e}")
                verdicts[fm_id] = JudgeVerdict(
                    trace_id=trace.trace_id, fm_id=fm_id,
                    would_fix=False, confidence=0.0, reasoning=str(e),
                )

    await asyncio.gather(*(judge_one(fm_id) for fm_id in fm_ids))

    judge_time = time.time() - t2
    total_time = time.time() - t0

    # 汇总
    n_fix = sum(1 for v in verdicts.values() if v.would_fix)
    print(
        f"  trace_{trace.trace_id}: {len(active_fms)} FMs, "
        f"judge={n_fix}/{len(verdicts)} fix, "
        f"yaml_valid={opt_result['yaml_valid']}, "
        f"time={total_time:.1f}s (diag={diag_time:.1f}s opt={opt_time:.1f}s judge={judge_time:.1f}s)"
    )

    # 诊断详情
    for fm_id in fm_ids:
        loc = profile.localization.get(fm_id)
        v = verdicts.get(fm_id)
        loc_str = f"agent={loc.agent}, cause={loc.root_cause[:60]}..." if loc and loc.agent != "unknown" else "localization failed"
        judge_str = f"{'PASS' if v and v.would_fix else 'FAIL'}({v.confidence:.2f})" if v else "N/A"
        print(f"    FM-{fm_id}: {loc_str} | {judge_str}")

    return {
        "trace_id": trace.trace_id,
        "benchmark": trace.benchmark_name,
        "active_fms": active_fms,
        "diagnosis": {
            fm_id: {
                "agent": loc.agent,
                "step": loc.step,
                "root_cause": loc.root_cause,
            } if (loc := profile.localization.get(fm_id)) else None
            for fm_id in active_fms
        },
        "optimization": {
            "analysis": opt_result["analysis"],
            "yaml_valid": opt_result["yaml_valid"],
            "diff": _yaml_diff_summary(opt_result["original_yaml"], opt_result["modified_yaml"]),
            "modified_yaml": opt_result["modified_yaml"] if opt_result["yaml_valid"] else None,
        },
        "judgments": {
            fm_id: {
                "would_fix": v.would_fix,
                "confidence": v.confidence,
                "reasoning": v.reasoning,
            }
            for fm_id, v in verdicts.items()
        },
        "timing": {
            "diagnose_s": round(diag_time, 1),
            "optimize_s": round(opt_time, 1),
            "judge_s": round(judge_time, 1),
            "total_s": round(total_time, 1),
        },
    }


async def _run_offline_yaml_pipeline_async(
    yaml_path: str,
    mas_name: str = "AG2",
    group_filter: str | None = None,
    benchmark: str | None = None,
    max_traces: int | None = None,
    max_workers: int = OFFLINE_YAML_MAX_WORKERS,
):
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        print(f"ERROR: YAML file not found: {yaml_path}")
        return

    print(f"=== OptPilot Offline YAML Pipeline (Concurrent) ===")
    print(f"MAS: {mas_name}, YAML: {yaml_path.name}, workers: {max_workers}")
    if group_filter:
        normalized = group_filter.upper()
        print(f"Group filter: Group-{normalized} ({GROUP_NAMES.get(normalized, '?')})")
    print()

    # 1. Load traces
    traces = load_traces(mas_name, fm_filter=group_filter, benchmark=benchmark)
    # 只保留有 active FM 的 trace
    traces = [t for t in traces if t.active_fm_ids()]
    if max_traces:
        traces = traces[:max_traces]
    print(f"Loaded {len(traces)} traces with active FMs")
    print_fm_stats(traces)
    print()

    # 2. Init modules
    library = RepairLibrary(LIBRARY_DIR / f"yaml_{mas_name.lower()}_library.json")
    diagnoser = Diagnoser(max_workers=max_workers)
    optimizer = YAMLOptimizer(library=library)
    judge = Judge()

    # 3. 并行处理所有 trace
    t_start = time.time()
    results = []

    print(f"Processing {len(traces)} traces with {max_workers} workers...")
    print()

    trace_semaphore = asyncio.Semaphore(min(max_workers, len(traces)))

    async def process_trace(idx: int, trace: MASTrace) -> None:
        async with trace_semaphore:
            try:
                result = await _process_one_trace(
                    trace,
                    yaml_path,
                    diagnoser,
                    optimizer,
                    judge,
                    judge_workers=max_workers,
                )
                if result:
                    results.append(result)
            except Exception as e:
                print(f"  ERROR trace index {idx}: {e}")

    await asyncio.gather(
        *(process_trace(i, trace) for i, trace in enumerate(traces))
    )

    total_time = time.time() - t_start

    # 4. Save valid YAMLs
    out_dir = RESULTS_DIR / f"yaml_opt_{mas_name.lower()}"
    out_dir.mkdir(exist_ok=True)
    saved = 0
    for r in results:
        if r["optimization"].get("modified_yaml"):
            out_path = out_dir / f"trace_{r['trace_id']}_optimized.yaml"
            out_path.write_text(r["optimization"]["modified_yaml"], encoding="utf-8")
            saved += 1
            # 不在 JSON 结果中保存完整 YAML（太大）
            r["optimization"]["modified_yaml"] = f"saved to {out_path.name}"

    # 5. Summary
    print()
    print(f"=== Pipeline Complete ({total_time:.1f}s) ===")
    print(f"Traces processed: {len(results)}")

    total_fms = sum(len(r["active_fms"]) for r in results)
    fixed_fms = sum(
        sum(1 for v in r["judgments"].values() if v["would_fix"])
        for r in results
    )
    valid_yamls = sum(1 for r in results if r["optimization"]["yaml_valid"])

    print(f"Total FMs diagnosed: {total_fms}")
    if total_fms:
        print(f"Judge would-fix: {fixed_fms}/{total_fms} ({fixed_fms/total_fms*100:.0f}%)")
    print(f"Valid YAML outputs: {valid_yamls}/{len(results)} (saved: {saved})")
    if results:
        avg_time = sum(r["timing"]["total_s"] for r in results) / len(results)
        print(f"Avg time per trace: {avg_time:.1f}s")

    # Save results
    suffix = f"_{group_filter.lower()}" if group_filter else ""
    out_file = RESULTS_DIR / f"offline_yaml_{mas_name.lower()}{suffix}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved: {out_file}")


def run_offline_yaml_pipeline(
    yaml_path: str,
    mas_name: str = "AG2",
    group_filter: str | None = None,
    benchmark: str | None = None,
    max_traces: int | None = None,
    max_workers: int = OFFLINE_YAML_MAX_WORKERS,
):
    asyncio.run(
        _run_offline_yaml_pipeline_async(
            yaml_path=yaml_path,
            mas_name=mas_name,
            group_filter=group_filter,
            benchmark=benchmark,
            max_traces=max_traces,
            max_workers=max_workers,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OptPilot Offline YAML Pipeline (Concurrent)")
    parser.add_argument("--yaml", required=True, help="Path to MAS YAML config")
    parser.add_argument("--mas", default="AG2", help="MAS name for trace loading")
    parser.add_argument("--group", default=None, help="Filter traces by failure group (optional)")
    parser.add_argument("--benchmark", default=None, help="Filter by benchmark")
    parser.add_argument("--max-traces", type=int, default=None, help="Max traces")
    parser.add_argument(
        "--workers",
        type=int,
        default=OFFLINE_YAML_MAX_WORKERS,
        help=f"Max concurrent workers (default: {OFFLINE_YAML_MAX_WORKERS})",
    )
    args = parser.parse_args()

    run_offline_yaml_pipeline(
        yaml_path=args.yaml,
        mas_name=args.mas,
        group_filter=args.group,
        benchmark=args.benchmark,
        max_traces=args.max_traces,
        max_workers=args.workers,
    )
