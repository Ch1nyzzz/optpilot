"""Exp 4: Ablation Study.

Tests four ablation variants to identify critical components:
1. w/o library: never retrieve, always generate fresh
2. w/o distillation: generate but don't save to library
3. w/o diagnosis: no FM localization, just use FM label
4. w/o prioritization: random FM selection instead of highest frequency

Usage:
    python -m experiments.exp4_ablation --group B --max-traces 20 --dag dags/chatdev.yaml
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optpilot.config import DAG_DIR, LIBRARY_DIR, RESULTS_DIR
from optpilot.dag.core import MASDAG
from optpilot.data.fm_taxonomy_6group import GROUP_NAMES
from optpilot.data.loader import load_traces
from optpilot.library.repair_library import RepairLibrary
from optpilot.models import FMLocalization
from optpilot.modules.diagnoser import Diagnoser
from optpilot.modules.distiller import Distiller
from optpilot.modules.judge import Judge
from optpilot.modules.optimizer import Optimizer
from optpilot.tracking import Tracker


def run_ablation_variant(
    variant: str,
    group_id: str,
    traces,
    dag,
    use_wandb: bool = False,
) -> dict:
    """Run a single ablation variant."""
    lib_path = LIBRARY_DIR / f"exp4_{variant}_group_{group_id.lower()}.json"
    library = RepairLibrary(lib_path)
    diagnoser = Diagnoser()
    optimizer = Optimizer(library)
    judge = Judge()
    distiller = Distiller(library)

    n_would_fix = 0

    for i, trace in enumerate(traces):
        # Diagnose (may be skipped for w/o diagnosis)
        profile = diagnoser.diagnose(trace)

        if variant == "wo_diagnosis":
            # Remove localization - only keep FM labels
            profile.localization = {
                fm: FMLocalization("unknown", "unknown", "", "")
                for fm in profile.active_fm_ids()
            }

        # Generate repair
        if variant == "wo_library":
            # Force generate new (bypass library search)
            candidate = optimizer._generate_new(group_id, profile, trace, dag)
        else:
            candidate = optimizer.generate_repair(group_id, profile, trace, dag)

        # Judge
        verdict = judge.evaluate(trace, group_id, candidate, profile)
        if verdict.would_fix:
            n_would_fix += 1

        # Distill
        if variant != "wo_distillation":
            distiller.distill_offline(group_id, candidate, verdict)

    library.flush()
    return {
        "variant": variant,
        "fm_id": group_id,
        "n_traces": len(traces),
        "would_fix_count": n_would_fix,
        "would_fix_rate": n_would_fix / len(traces) if traces else 0,
        "library_size": len(library.entries),
    }


def run_ablation(group_id: str = "B", max_traces: int = 20, dag_path: str | None = None, use_wandb: bool = False):
    """Run all ablation variants."""
    group_id = group_id.upper()
    traces = load_traces("ChatDev", fm_filter=group_id)[:max_traces]

    # Load DAG for architecture context
    dag = None
    if dag_path:
        dag = MASDAG.load(dag_path)
    else:
        default_dag = DAG_DIR / "chatdev.yaml"
        if default_dag.exists():
            dag = MASDAG.load(default_dag)

    print(f"=== Exp 4: Ablation for Group-{group_id} ({GROUP_NAMES.get(group_id)}) ===")
    print(f"Traces: {len(traces)}")

    variants = ["full", "wo_library", "wo_distillation", "wo_diagnosis"]
    results = []

    for variant in variants:
        print(f"\n--- Variant: {variant} ---")
        r = run_ablation_variant(variant, group_id, traces, dag, use_wandb)
        results.append(r)
        print(f"  Would-fix rate: {r['would_fix_rate']:.0%} ({r['would_fix_count']}/{r['n_traces']})")
        print(f"  Library size: {r['library_size']}")

    # Summary
    print(f"\n=== Ablation Summary ===")
    print(f"{'Variant':<20} {'Fix Rate':>10} {'Library':>10}")
    print("-" * 42)
    for r in results:
        print(f"{r['variant']:<20} {r['would_fix_rate']:>9.0%} {r['library_size']:>10}")

    out_path = RESULTS_DIR / f"exp4_ablation_group_{group_id.lower()}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", default="B")
    parser.add_argument("--max-traces", type=int, default=20)
    parser.add_argument("--dag", default=None, help="Path to MASDAG YAML file")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    run_ablation(args.group, args.max_traces, args.dag, args.wandb)
