"""Taxonomy stability experiment.

Uses 100 MAST trajectories, diagnoses each 3 times with our 6-group taxonomy,
and measures inter-run consistency to validate classifier stability.

Usage:
    python -m experiments.exp_taxonomy_stability [--n-traces 100] [--n-runs 3] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download

from optpilot.data.fm_taxonomy_6group import GROUP_IDS, mast_annotation_to_groups
from optpilot.models import MASTrace
from optpilot.modules.diagnoser import Diagnoser

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "taxonomy_stability"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_mast_traces(n_traces: int, seed: int) -> list[MASTrace]:
    """Download MAST dataset and sample n_traces with stratification by MAS."""
    print(f"Downloading MAST dataset from HuggingFace...")
    path = hf_hub_download(
        repo_id="mcemri/MAD",
        filename="MAD_full_dataset.json",
        repo_type="dataset",
    )
    with open(path) as f:
        raw_data = json.load(f)

    # Keep only traces with at least one failure annotation
    failed = [d for d in raw_data if any(v == 1 for v in d["mast_annotation"].values())]
    print(f"  Total traces: {len(raw_data)}, with failures: {len(failed)}")

    # Stratified sampling by mas_name
    rng = random.Random(seed)
    by_mas: dict[str, list] = defaultdict(list)
    for d in failed:
        by_mas[d["mas_name"]].append(d)

    # Proportional allocation
    total_failed = len(failed)
    sampled: list = []
    remaining = n_traces
    mas_names = sorted(by_mas.keys())
    for i, mas in enumerate(mas_names):
        pool = by_mas[mas]
        if i == len(mas_names) - 1:
            count = remaining
        else:
            count = max(1, round(n_traces * len(pool) / total_failed))
            count = min(count, remaining, len(pool))
        rng.shuffle(pool)
        sampled.extend(pool[:count])
        remaining -= count
        if remaining <= 0:
            break

    # If still under target, fill from remaining pool
    if len(sampled) < n_traces:
        used_ids = {(d["mas_name"], d["trace_id"]) for d in sampled}
        extras = [d for d in failed if (d["mas_name"], d["trace_id"]) not in used_ids]
        rng.shuffle(extras)
        sampled.extend(extras[: n_traces - len(sampled)])

    sampled = sampled[:n_traces]
    rng.shuffle(sampled)

    # Print sampling distribution
    sample_counts = Counter(d["mas_name"] for d in sampled)
    print(f"  Sampled {len(sampled)} traces: {dict(sample_counts)}")

    # Convert to MASTrace
    traces = []
    for i, d in enumerate(sampled):
        trajectory = d["trace"]["trajectory"] if isinstance(d["trace"], dict) else str(d["trace"])
        ground_truth = mast_annotation_to_groups(d["mast_annotation"])
        traces.append(MASTrace(
            trace_id=i,
            mas_name=d["mas_name"],
            llm_name=d.get("llm_name", "unknown"),
            benchmark_name=d.get("benchmark_name", "unknown"),
            trajectory=trajectory,
            mast_annotation=ground_truth,
        ))

    return traces


# ---------------------------------------------------------------------------
# Stability metrics
# ---------------------------------------------------------------------------

def compute_per_label_agreement(
    all_runs: list[list[dict[str, bool]]],
) -> dict[str, float]:
    """For each label, fraction of traces where all runs agree."""
    n_traces = len(all_runs[0])
    n_runs = len(all_runs)
    agreement = {}
    for gid in GROUP_IDS:
        agree_count = 0
        for i in range(n_traces):
            vals = [all_runs[r][i][gid] for r in range(n_runs)]
            if len(set(vals)) == 1:
                agree_count += 1
        agreement[gid] = agree_count / n_traces
    return agreement


def compute_exact_match_rate(all_runs: list[list[dict[str, bool]]]) -> float:
    """Fraction of traces where ALL labels match across ALL runs."""
    n_traces = len(all_runs[0])
    n_runs = len(all_runs)
    exact = 0
    for i in range(n_traces):
        vectors = [tuple(all_runs[r][i][gid] for gid in GROUP_IDS) for r in range(n_runs)]
        if len(set(vectors)) == 1:
            exact += 1
    return exact / n_traces


def compute_fleiss_kappa(all_runs: list[list[dict[str, bool]]]) -> dict[str, float]:
    """Fleiss' kappa per label (treating each run as a rater)."""
    n_runs = len(all_runs)
    n_traces = len(all_runs[0])
    kappas = {}

    for gid in GROUP_IDS:
        # For each trace, count how many raters said True vs False
        p_bar_sum = 0.0
        n_true_total = 0
        for i in range(n_traces):
            n_true = sum(1 for r in range(n_runs) if all_runs[r][i][gid])
            n_false = n_runs - n_true
            n_true_total += n_true
            p_bar_sum += (n_true * (n_true - 1) + n_false * (n_false - 1))

        N = n_traces
        n = n_runs
        P_bar = p_bar_sum / (N * n * (n - 1)) if N * n * (n - 1) > 0 else 0

        p_true = n_true_total / (N * n) if N * n > 0 else 0
        p_false = 1 - p_true
        P_e = p_true ** 2 + p_false ** 2

        if abs(1 - P_e) < 1e-10:
            kappas[gid] = 1.0  # perfect agreement
        else:
            kappas[gid] = (P_bar - P_e) / (1 - P_e)

    return kappas


def compute_majority_deviation(all_runs: list[list[dict[str, bool]]]) -> dict[str, float]:
    """Fraction of (trace, label) pairs where a run deviates from majority vote."""
    n_runs = len(all_runs)
    n_traces = len(all_runs[0])
    deviations = {}

    for gid in GROUP_IDS:
        dev_count = 0
        total = 0
        for i in range(n_traces):
            vals = [all_runs[r][i][gid] for r in range(n_runs)]
            majority = sum(vals) > n_runs / 2
            for v in vals:
                if v != majority:
                    dev_count += 1
                total += 1
        deviations[gid] = dev_count / total if total > 0 else 0

    return deviations


def compute_ground_truth_alignment(
    run_results: list[dict[str, bool]],
    traces: list[MASTrace],
) -> dict[str, dict[str, float]]:
    """Compare one run's results against MAST ground-truth annotations."""
    n = len(traces)
    metrics: dict[str, dict[str, float]] = {}

    for gid in GROUP_IDS:
        tp = fp = fn = tn = 0
        for i in range(n):
            pred = run_results[i][gid]
            gt = bool(traces[i].mast_annotation.get(gid, 0))
            if pred and gt:
                tp += 1
            elif pred and not gt:
                fp += 1
            elif not pred and gt:
                fn += 1
            else:
                tn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / n if n > 0 else 0.0
        metrics[gid] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Taxonomy stability experiment")
    parser.add_argument("--n-traces", type=int, default=100)
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    print(f"=== Taxonomy Stability Experiment ===")
    print(f"  n_traces={args.n_traces}, n_runs={args.n_runs}, seed={args.seed}")
    traces = load_mast_traces(args.n_traces, args.seed)

    # 2. Run diagnosis n_runs times
    diagnoser = Diagnoser()
    all_runs: list[list[dict[str, bool]]] = []

    for run_idx in range(args.n_runs):
        print(f"\n--- Run {run_idx + 1}/{args.n_runs} ---")
        t0 = time.time()
        profiles = diagnoser.classify_batch(traces)
        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.1f}s")

        run_results = []
        for p in profiles:
            labels = {gid: p.labels[gid].present if gid in p.labels else False for gid in GROUP_IDS}
            run_results.append(labels)
        all_runs.append(run_results)

        # Print FM rates for this run
        fm_rates = {}
        for gid in GROUP_IDS:
            fm_rates[gid] = sum(1 for r in run_results if r[gid]) / len(run_results)
        print(f"  FM rates: {json.dumps({k: f'{v:.1%}' for k, v in fm_rates.items()})}")

    # 3. Compute stability metrics
    print(f"\n{'='*60}")
    print(f"=== STABILITY RESULTS ===")
    print(f"{'='*60}")

    # 3a. Per-label agreement
    agreement = compute_per_label_agreement(all_runs)
    print(f"\nPer-label agreement (fraction of traces where all {args.n_runs} runs agree):")
    for gid in GROUP_IDS:
        print(f"  {gid}: {agreement[gid]:.1%}")
    avg_agreement = sum(agreement.values()) / len(agreement)
    print(f"  Average: {avg_agreement:.1%}")

    # 3b. Exact match rate
    exact_match = compute_exact_match_rate(all_runs)
    print(f"\nExact match rate (all 6 labels identical across {args.n_runs} runs): {exact_match:.1%}")

    # 3c. Fleiss' kappa
    kappas = compute_fleiss_kappa(all_runs)
    print(f"\nFleiss' kappa (inter-run agreement beyond chance):")
    for gid in GROUP_IDS:
        print(f"  {gid}: {kappas[gid]:.3f}")
    avg_kappa = sum(kappas.values()) / len(kappas)
    print(f"  Average: {avg_kappa:.3f}")

    # 3d. Majority vote deviation
    deviations = compute_majority_deviation(all_runs)
    print(f"\nMajority vote deviation (fraction of individual judgments differing from majority):")
    for gid in GROUP_IDS:
        print(f"  {gid}: {deviations[gid]:.1%}")
    avg_dev = sum(deviations.values()) / len(deviations)
    print(f"  Average: {avg_dev:.1%}")

    # 4. Ground-truth alignment (using majority vote)
    print(f"\n{'='*60}")
    print(f"=== GROUND-TRUTH ALIGNMENT (majority vote vs MAST annotation) ===")
    print(f"{'='*60}")

    # Build majority vote
    majority_results = []
    for i in range(len(traces)):
        majority = {}
        for gid in GROUP_IDS:
            votes = [all_runs[r][i][gid] for r in range(args.n_runs)]
            majority[gid] = sum(votes) > args.n_runs / 2
        majority_results.append(majority)

    gt_metrics = compute_ground_truth_alignment(majority_results, traces)
    print(f"\n{'Group':<6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Acc':>6}  (TP/FP/FN/TN)")
    for gid in GROUP_IDS:
        m = gt_metrics[gid]
        print(f"  {gid:<4}  {m['precision']:>5.1%} {m['recall']:>5.1%} {m['f1']:>5.1%} {m['accuracy']:>5.1%}  ({m['tp']}/{m['fp']}/{m['fn']}/{m['tn']})")

    # Also compute per-run alignment
    print(f"\n--- Per-run ground-truth F1 ---")
    for run_idx in range(args.n_runs):
        gt_run = compute_ground_truth_alignment(all_runs[run_idx], traces)
        f1s = {gid: gt_run[gid]["f1"] for gid in GROUP_IDS}
        avg_f1 = sum(f1s.values()) / len(f1s)
        print(f"  Run {run_idx + 1}: {json.dumps({k: f'{v:.1%}' for k, v in f1s.items()})}  avg={avg_f1:.1%}")

    # 5. Save results
    results = {
        "config": {
            "n_traces": args.n_traces,
            "n_runs": args.n_runs,
            "seed": args.seed,
            "model": "MiniMaxAI/MiniMax-M2.5",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "sample_distribution": dict(Counter(t.mas_name for t in traces)),
        "stability": {
            "per_label_agreement": agreement,
            "exact_match_rate": exact_match,
            "fleiss_kappa": kappas,
            "majority_deviation": deviations,
            "avg_agreement": avg_agreement,
            "avg_kappa": avg_kappa,
        },
        "ground_truth_alignment": {
            "majority_vote": gt_metrics,
            "per_run": [],
        },
        "per_run_fm_rates": [],
        "raw_results": [],
    }

    # Per-run details
    for run_idx in range(args.n_runs):
        fm_rates = {}
        for gid in GROUP_IDS:
            fm_rates[gid] = sum(1 for r in all_runs[run_idx] if r[gid]) / len(all_runs[run_idx])
        results["per_run_fm_rates"].append(fm_rates)

        gt_run = compute_ground_truth_alignment(all_runs[run_idx], traces)
        results["ground_truth_alignment"]["per_run"].append(gt_run)

    # Raw per-trace results
    for i, trace in enumerate(traces):
        entry = {
            "trace_id": i,
            "mas_name": trace.mas_name,
            "benchmark": trace.benchmark_name,
            "ground_truth": trace.mast_annotation,
            "runs": [all_runs[r][i] for r in range(args.n_runs)],
        }
        results["raw_results"].append(entry)

    out_path = RESULTS_DIR / f"stability_{args.n_traces}traces_{args.n_runs}runs_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
