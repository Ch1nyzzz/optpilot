"""Offline Pipeline - Phase A: MAST-Data → Diagnose → Optimize → Judge → Library.

Usage:
    python -m experiments.offline_pipeline --mas AG2 --fm 1.3 --max-traces 5
    python -m experiments.offline_pipeline --mas ChatDev --dag dags/chatdev.yaml --fm 2.6
"""

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optpilot.config import DAG_DIR, OFFLINE_HINTS_DIR, OFFLINE_SKILLS_DIR
from optpilot.dag.core import MASDAG
from optpilot.data.fm_taxonomy import FM_NAMES
from optpilot.data.loader import load_traces, print_fm_stats
from optpilot.library.repair_library import RepairLibrary
from optpilot.modules.diagnoser import Diagnoser
from optpilot.modules.distiller import Distiller
from optpilot.modules.judge import Judge
from optpilot.modules.optimizer import Optimizer
from optpilot.modules.wrap_up import WrapUp
from optpilot.tracking import Tracker


def run_offline_pipeline(
    mas_name: str = "AG2",
    fm_id: str = "1.3",
    dag_path: str | None = None,
    benchmark: str | None = None,
    max_traces: int | None = None,
    use_wandb: bool = False,
):
    """Run offline analysis pipeline on MAST-Data traces."""
    print(f"=== OptPilot Offline Pipeline ===")
    print(f"MAS: {mas_name}, Target FM: FM-{fm_id} ({FM_NAMES.get(fm_id, '?')})")

    # 1. Load traces
    traces = load_traces(mas_name, fm_filter=fm_id, benchmark=benchmark)
    if max_traces:
        traces = traces[:max_traces]
    print(f"Loaded {len(traces)} traces with FM-{fm_id}")
    if traces:
        avg_len = sum(len(t.trajectory) for t in traces) / len(traces)
        print(f"Avg trajectory length: {avg_len:.0f} chars")
    print()

    hints_path = OFFLINE_HINTS_DIR / mas_name.lower() / f"fm_{fm_id.replace('.', '_')}_hints.json"
    skills_path = OFFLINE_SKILLS_DIR / mas_name.lower() / f"fm_{fm_id.replace('.', '_')}_skills.json"

    # 2. Initialize modules with optional DAG context
    hints_library = RepairLibrary(hints_path)
    skills_library = RepairLibrary(skills_path)
    diagnoser = Diagnoser()
    optimizer = Optimizer(hints_library)
    judge = Judge()
    distiller = Distiller(hints_library)
    tracker = Tracker(f"offline_{mas_name}_{fm_id}", use_wandb=use_wandb)

    # Load DAG if path provided
    dag = None
    if dag_path:
        dag = MASDAG.load(dag_path)
        print(f"DAG loaded: {dag.dag_id} ({len(dag.agent_nodes)} agents, {len(dag.edges)} edges)")
        print()

    # 3. Process each trace
    for i, trace in enumerate(traces):
        print(f"--- Trace {i+1}/{len(traces)} (id={trace.trace_id}) ---")

        # Diagnose
        print(f"  [1/4] Diagnosing...")
        profile = diagnoser.diagnose(trace, target_fm=fm_id)
        loc = profile.localization.get(fm_id)
        if loc:
            print(f"    Agent: {loc.agent}, Step: {loc.step}")
            print(f"    Root cause: {loc.root_cause[:120]}...")

        # Generate repair (pass DAG if available for architecture context)
        print(f"  [2/4] Generating repair...")
        candidate = optimizer.generate_repair(fm_id, profile, trace, dag=dag)
        print(f"    Repair: {candidate.description[:100]}")
        print(f"    Actions: {len(candidate.actions)}, Source: {candidate.source}")

        # Judge evaluation
        print(f"  [3/4] Judging (counterfactual)...")
        verdict = judge.evaluate(trace, fm_id, candidate, profile)
        print(f"    Would fix: {verdict.would_fix} (confidence: {verdict.confidence:.2f})")

        # Distill
        print(f"  [4/4] Distilling to library...")
        entry = distiller.distill_offline(fm_id, candidate, verdict, source_mas=mas_name)
        print(f"    Entry: {entry.entry_id} status={entry.status}")

        # Track
        tracker.log({
            "trace_id": trace.trace_id,
            "fm_id": fm_id,
            "would_fix": verdict.would_fix,
            "confidence": verdict.confidence,
            "repair_source": candidate.source,
            "n_actions": len(candidate.actions),
        }, step=i)

        tracker.log_result({
            "trace_id": trace.trace_id,
            "profile": asdict(profile),
            "candidate": asdict(candidate),
            "verdict": asdict(verdict),
            "entry_id": entry.entry_id,
        })

        print()

    # 4. Summary
    stats = hints_library.get_stats()
    print(f"=== Pipeline Complete ===")
    print(f"Traces processed: {len(traces)}")
    print(f"Hint stats: {stats}")

    would_fix_count = sum(1 for r in tracker.results if isinstance(r, dict) and r.get("would_fix"))
    print(f"Judge positive rate: {would_fix_count}/{len(traces)}")

    wrap_up = WrapUp(hints_library, output_library=skills_library)
    wrapped = wrap_up.wrap_fm(fm_id, source_mas=mas_name)
    print(f"Wrapped skills for FM-{fm_id}: {len(wrapped)}")

    hints_library.flush()
    skills_library.flush()
    tracker.save_local(f"offline_{mas_name}_{fm_id}.json")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OptPilot Offline Pipeline")
    parser.add_argument("--mas", default="AG2", help="MAS framework (default: AG2)")
    parser.add_argument("--fm", default="1.3", help="Target FM id (default: 1.3)")
    parser.add_argument("--dag", default=None, help="Path to MASDAG YAML file")
    parser.add_argument("--benchmark", default=None, help="Filter by benchmark")
    parser.add_argument("--max-traces", type=int, default=None, help="Max traces to process")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B tracking")
    args = parser.parse_args()

    run_offline_pipeline(
        mas_name=args.mas, fm_id=args.fm, dag_path=args.dag,
        benchmark=args.benchmark, max_traces=args.max_traces, use_wandb=args.wandb,
    )
