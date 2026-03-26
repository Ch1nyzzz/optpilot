"""Exp 3: Experience Accumulation.

Tests whether the Repair Library improves over time by processing
traces in sequential batches and measuring repair speed improvement.

Usage:
    python -m experiments.exp3_accumulation --fm 1.3 --batch-size 10 --dag dags/chatdev.yaml
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optpilot.config import DAG_DIR, LIBRARY_DIR, RESULTS_DIR
from optpilot.dag.core import MASDAG
from optpilot.data.fm_taxonomy import FM_NAMES
from optpilot.data.loader import load_chatdev_traces
from optpilot.library.repair_library import RepairLibrary
from optpilot.modules.diagnoser import Diagnoser
from optpilot.modules.distiller import Distiller
from optpilot.modules.judge import Judge
from optpilot.modules.optimizer import Optimizer
from optpilot.tracking import Tracker


def run_accumulation(fm_id: str = "1.3", batch_size: int = 10, dag_path: str | None = None, use_wandb: bool = False):
    """Process traces in batches, measuring library accumulation effect."""
    traces = load_chatdev_traces(fm_filter=fm_id)
    print(f"=== Exp 3: Accumulation for FM-{fm_id} ({FM_NAMES.get(fm_id)}) ===")
    print(f"Total traces: {len(traces)}, Batch size: {batch_size}")

    # Fresh library for this experiment
    lib_path = LIBRARY_DIR / f"exp3_accumulation_fm{fm_id}.json"
    library = RepairLibrary(lib_path)

    # Load DAG for architecture context
    dag = None
    if dag_path:
        dag = MASDAG.load(dag_path)
    else:
        default_dag = DAG_DIR / "chatdev.yaml"
        if default_dag.exists():
            dag = MASDAG.load(default_dag)

    diagnoser = Diagnoser()
    optimizer = Optimizer(library)
    judge = Judge()
    distiller = Distiller(library)
    tracker = Tracker(f"exp3_fm{fm_id}", use_wandb=use_wandb)

    # Process in batches
    batches = [traces[i:i+batch_size] for i in range(0, len(traces), batch_size)]
    batch_results = []

    for b_idx, batch in enumerate(batches):
        print(f"\n--- Batch {b_idx + 1}/{len(batches)} ({len(batch)} traces) ---")
        library_from_count = sum(1 for e in library.entries if e.status != "failed")
        n_library_hits = 0
        n_would_fix = 0

        for trace in batch:
            profile = diagnoser.diagnose(trace)
            candidate = optimizer.generate_repair(fm_id, profile, trace, dag)
            if candidate.source == "library":
                n_library_hits += 1
            verdict = judge.evaluate(trace, fm_id, candidate, profile)
            if verdict.would_fix:
                n_would_fix += 1
            distiller.distill_offline(fm_id, candidate, verdict)

        result = {
            "batch": b_idx + 1,
            "batch_size": len(batch),
            "library_size_before": library_from_count,
            "library_size_after": len(library.entries),
            "library_hit_rate": n_library_hits / len(batch),
            "would_fix_rate": n_would_fix / len(batch),
        }
        batch_results.append(result)
        tracker.log(result, step=b_idx)

        print(f"  Library size: {library_from_count} → {len(library.entries)}")
        print(f"  Library hit rate: {n_library_hits}/{len(batch)} ({n_library_hits/len(batch)*100:.0f}%)")
        print(f"  Would-fix rate: {n_would_fix}/{len(batch)} ({n_would_fix/len(batch)*100:.0f}%)")

    # Summary
    print(f"\n=== Accumulation Summary ===")
    for r in batch_results:
        print(f"  Batch {r['batch']}: lib_size={r['library_size_after']}, "
              f"hit_rate={r['library_hit_rate']:.0%}, fix_rate={r['would_fix_rate']:.0%}")

    out_path = RESULTS_DIR / f"exp3_accumulation_fm{fm_id}.json"
    out_path.write_text(json.dumps(batch_results, indent=2))
    library.flush()
    tracker.save_local(f"exp3_fm{fm_id}.json")
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fm", default="1.3")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--dag", default=None, help="Path to MASDAG YAML file")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    run_accumulation(args.fm, args.batch_size, args.dag, args.wandb)
