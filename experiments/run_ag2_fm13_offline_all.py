"""Run ALL AG2 FM-1.3 traces through the offline pipeline.

Features:
- Async concurrency (diagnose/judge async, optimizer/distiller via to_thread)
- Batch processing with wrap-up after each batch
- Resume support: skips already-processed traces based on results file
- Incremental save after each batch

Usage:
    python -u -m experiments.run_ag2_fm13_offline_all
    python -u -m experiments.run_ag2_fm13_offline_all --batch-size 30 --workers 32
    python -u -m experiments.run_ag2_fm13_offline_all --resume  # skip already done
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optpilot.config import DAG_DIR, OFFLINE_HINTS_DIR, OFFLINE_SKILLS_DIR
from optpilot.dag.core import MASDAG
from optpilot.data.fm_taxonomy import FM_NAMES
from optpilot.data.loader import load_traces
from optpilot.library.repair_library import RepairLibrary
from optpilot.modules.diagnoser import Diagnoser
from optpilot.modules.distiller import Distiller
from optpilot.modules.judge import Judge
from optpilot.modules.optimizer import Optimizer
from optpilot.modules.wrap_up import WrapUp

FM_ID = "1.3"
MAS_NAME = "AG2"
RESULTS_FILE = Path(__file__).resolve().parents[1] / "results" / f"offline_{MAS_NAME}_{FM_ID}.json"


async def _process_one_trace(
    idx: int,
    trace,
    fm_id: str,
    diagnoser: Diagnoser,
    optimizer: Optimizer,
    judge: Judge,
    distiller: Distiller,
    dag,
) -> dict | None:
    """Process a single trace: diagnose → optimize → judge → distill."""
    t0 = time.time()
    label = f"[{idx}] trace_{trace.trace_id} ({trace.llm_name}/{trace.benchmark_name})"

    try:
        # 1. Diagnose (async)
        profile = await diagnoser.adiagnose(trace, target_fm=fm_id)
        diag_time = time.time() - t0

        loc = profile.localization.get(fm_id)
        loc_str = f"agent={loc.agent}, step={loc.step}" if loc else "localization failed"

        # 2. Optimize (sync → to_thread)
        t1 = time.time()
        candidate = await asyncio.to_thread(
            optimizer.generate_repair, fm_id, profile, trace, dag
        )
        opt_time = time.time() - t1

        # 3. Judge (async)
        t2 = time.time()
        verdict = await judge.aevaluate(trace, fm_id, candidate, profile)
        judge_time = time.time() - t2

        # 4. Distill (sync → to_thread)
        t3 = time.time()
        entry = await asyncio.to_thread(
            distiller.distill_offline, fm_id, candidate, verdict, MAS_NAME
        )
        distill_time = time.time() - t3

        total = time.time() - t0
        fix_str = "PASS" if verdict.would_fix else "FAIL"
        print(
            f"  {label}: {loc_str} | {fix_str}({verdict.confidence:.2f}) | "
            f"{total:.1f}s (D={diag_time:.1f} O={opt_time:.1f} J={judge_time:.1f} S={distill_time:.1f})"
        )

        return {
            "trace_index": idx,
            "trace_id": trace.trace_id,
            "llm_name": trace.llm_name,
            "benchmark": trace.benchmark_name,
            "profile": asdict(profile),
            "candidate": asdict(candidate),
            "verdict": asdict(verdict),
            "entry_id": entry.entry_id,
            "would_fix": verdict.would_fix,
            "confidence": verdict.confidence,
            "timing": {
                "diagnose_s": round(diag_time, 1),
                "optimize_s": round(opt_time, 1),
                "judge_s": round(judge_time, 1),
                "distill_s": round(distill_time, 1),
                "total_s": round(total, 1),
            },
        }

    except Exception as e:
        elapsed = time.time() - t0
        print(f"  {label}: ERROR ({elapsed:.1f}s) - {e}")
        return {
            "trace_index": idx,
            "trace_id": trace.trace_id,
            "llm_name": trace.llm_name,
            "benchmark": trace.benchmark_name,
            "error": str(e),
            "would_fix": False,
            "confidence": 0.0,
            "timing": {"total_s": round(elapsed, 1)},
        }


async def run_batch(
    batch_idx: int,
    batch_traces: list[tuple[int, object]],
    diagnoser: Diagnoser,
    optimizer: Optimizer,
    judge: Judge,
    distiller: Distiller,
    dag,
    max_workers: int,
) -> list[dict]:
    """Run a batch of traces concurrently."""
    semaphore = asyncio.Semaphore(max_workers)
    results: list[dict] = []
    lock = asyncio.Lock()

    async def process(idx: int, trace) -> None:
        async with semaphore:
            result = await _process_one_trace(
                idx, trace, FM_ID, diagnoser, optimizer, judge, distiller, dag,
            )
            if result:
                async with lock:
                    results.append(result)

    await asyncio.gather(*(process(idx, tr) for idx, tr in batch_traces))
    # Sort by trace_index for deterministic output
    results.sort(key=lambda r: r["trace_index"])
    return results


async def main(
    batch_size: int = 30,
    max_workers: int = 32,
    resume: bool = True,
):
    fm_name = FM_NAMES.get(FM_ID, "?")
    print(f"=== AG2 FM-{FM_ID} ({fm_name}) Full Offline Pipeline ===")
    print(f"Batch size: {batch_size}, Workers: {max_workers}, Resume: {resume}")
    print()

    # 1. Load all traces
    all_traces = load_traces(MAS_NAME, fm_filter=FM_ID)
    print(f"Total traces: {len(all_traces)}")

    # 2. Load existing results for resume
    done_indices: set[int] = set()
    existing_results: list[dict] = []
    if resume and RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            existing_results = json.load(f)
        done_indices = {r["trace_index"] for r in existing_results if "trace_index" in r}
        print(f"Resuming: {len(done_indices)} traces already done, skipping them")

    # 3. Build work list (index, trace) pairs
    work = [(i, t) for i, t in enumerate(all_traces) if i not in done_indices]
    if not work:
        print("All traces already processed!")
        return
    print(f"Remaining: {len(work)} traces")
    print()

    # 4. Init modules
    dag_path = DAG_DIR / "chatdev.yaml"
    dag = MASDAG.load(str(dag_path)) if dag_path.exists() else None
    if dag:
        print(f"DAG: {dag.dag_id} ({len(dag.agent_nodes)} agents, {len(dag.edges)} edges)")

    hints_path = OFFLINE_HINTS_DIR / MAS_NAME.lower() / f"fm_{FM_ID.replace('.', '_')}_hints.json"
    skills_path = OFFLINE_SKILLS_DIR / MAS_NAME.lower() / f"fm_{FM_ID.replace('.', '_')}_skills.json"
    hints_library = RepairLibrary(hints_path)
    skills_library = RepairLibrary(skills_path)

    diagnoser = Diagnoser(max_workers=max_workers)
    optimizer = Optimizer(hints_library)
    judge = Judge()
    distiller = Distiller(hints_library)
    wrap_up = WrapUp(hints_library, output_library=skills_library)

    # 5. Split into batches and process
    batches = [work[i:i + batch_size] for i in range(0, len(work), batch_size)]
    all_results = list(existing_results)
    total_start = time.time()

    for b_idx, batch in enumerate(batches):
        print(f"\n{'='*60}")
        print(f"Batch {b_idx + 1}/{len(batches)} ({len(batch)} traces)")
        print(f"{'='*60}")
        t_batch = time.time()

        batch_results = await run_batch(
            b_idx, batch, diagnoser, optimizer, judge, distiller, dag, max_workers,
        )
        all_results.extend(batch_results)

        batch_time = time.time() - t_batch
        n_fix = sum(1 for r in batch_results if r.get("would_fix"))
        n_err = sum(1 for r in batch_results if "error" in r)
        print(
            f"\nBatch {b_idx + 1} done: {len(batch_results)} traces in {batch_time:.1f}s, "
            f"would_fix={n_fix}/{len(batch_results)}, errors={n_err}"
        )

        # Wrap up after each batch
        print(f"Wrapping skills for FM-{FM_ID}...")
        wrapped = wrap_up.wrap_fm(FM_ID, source_mas=MAS_NAME)
        print(f"  Wrapped: {len(wrapped)} skills")

        # Flush libraries and save results incrementally
        hints_library.flush()
        skills_library.flush()
        RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_FILE, "w") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"  Saved {len(all_results)} results to {RESULTS_FILE.name}")

    # 6. Final summary
    total_time = time.time() - total_start
    total_fix = sum(1 for r in all_results if r.get("would_fix"))
    total_err = sum(1 for r in all_results if "error" in r)
    stats = hints_library.get_stats()

    print(f"\n{'='*60}")
    print(f"=== ALL DONE ({total_time:.1f}s) ===")
    print(f"Total traces: {len(all_results)}/{len(all_traces)}")
    print(f"Would-fix: {total_fix}/{len(all_results)} ({total_fix/max(1,len(all_results))*100:.1f}%)")
    print(f"Errors: {total_err}")
    print(f"Hint library stats: {stats}")
    print(f"Results: {RESULTS_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AG2 FM-1.3 Full Offline Pipeline")
    parser.add_argument("--batch-size", type=int, default=30, help="Traces per batch (default: 30)")
    parser.add_argument("--workers", type=int, default=64, help="Max concurrent workers (default: 64)")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh, ignore existing results")
    args = parser.parse_args()

    asyncio.run(main(
        batch_size=args.batch_size,
        max_workers=args.workers,
        resume=not args.no_resume,
    ))
