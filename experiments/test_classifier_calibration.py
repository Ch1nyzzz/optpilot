"""FM-1.3 Classifier Calibration Test.

Evaluate classify_fm precision/recall on held-out AG2 traces with MAST annotations.
Uses stratified sampling: equal positive and negative examples per benchmark.

Usage:
    python -u -m experiments.test_classifier_calibration
    python -u -m experiments.test_classifier_calibration --n-per-class 20 --workers 32
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optpilot.config import RESULTS_DIR
from optpilot.data.loader import load_traces
from optpilot.modules.diagnoser import Diagnoser

FM_ID = "1.3"
MAS_NAME = "AG2"


def sample_balanced(
    n_per_class: int = 15,
    seed: int = 42,
    benchmarks: list[str] | None = None,
) -> list[tuple[object, bool]]:
    """Sample balanced positive/negative traces per benchmark."""
    if benchmarks is None:
        benchmarks = ["MMLU", "Olympiad"]

    rng = random.Random(seed)
    samples = []

    for bench in benchmarks:
        all_traces = load_traces(MAS_NAME, benchmark=bench)
        positives = [t for t in all_traces if t.mast_annotation.get(FM_ID, 0) == 1]
        negatives = [t for t in all_traces if t.mast_annotation.get(FM_ID, 0) == 0]

        n_pos = min(n_per_class, len(positives))
        n_neg = min(n_per_class, len(negatives))
        sampled_pos = rng.sample(positives, n_pos)
        sampled_neg = rng.sample(negatives, n_neg)

        samples.extend((t, True) for t in sampled_pos)
        samples.extend((t, False) for t in sampled_neg)
        print(f"  {bench}: sampled {n_pos} pos + {n_neg} neg")

    rng.shuffle(samples)
    return samples


def run_calibration(
    n_per_class: int = 15,
    max_workers: int = 32,
    seed: int = 42,
    model: str | None = None,
):
    print(f"=== FM-{FM_ID} Classifier Calibration ===")
    print(f"Model: {model or 'default (SYSTEM_MODEL)'}")
    print(f"Sampling {n_per_class} per class per benchmark, workers={max_workers}")
    print()

    samples = sample_balanced(n_per_class=n_per_class, seed=seed)
    print(f"\nTotal samples: {len(samples)}")
    print()

    diagnoser = Diagnoser(max_workers=max_workers)

    # Run classification
    traces = [t for t, _ in samples]
    ground_truth = [label for _, label in samples]

    print(f"Classifying {len(traces)} traces...")
    t0 = time.time()

    # Use classify_batch to populate mast_annotation, but we need to
    # clear the existing annotation first to test the classifier fairly
    for t in traces:
        original_label = t.mast_annotation.get(FM_ID, 0)
        t._gt_label = original_label  # stash ground truth
        t.mast_annotation = {}  # clear so classifier runs blind

    classify_model = model or None
    diagnoser.classify_batch(traces, FM_ID, model=classify_model)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s ({elapsed/len(traces):.1f}s/trace)")
    print()

    # Compute metrics
    tp = fp = tn = fn = 0
    per_trace = []
    for trace, gt_label in zip(traces, ground_truth):
        predicted = trace.mast_annotation.get(FM_ID, 0) == 1
        gt = gt_label

        if predicted and gt:
            tp += 1
        elif predicted and not gt:
            fp += 1
        elif not predicted and gt:
            fn += 1
        else:
            tn += 1

        per_trace.append({
            "trace_id": trace.trace_id,
            "benchmark": trace.benchmark_name,
            "llm": trace.llm_name,
            "ground_truth": gt,
            "predicted": predicted,
            "correct": predicted == gt,
        })

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(traces) if traces else 0.0

    print(f"=== Results ===")
    print(f"  TP={tp}  FP={fp}")
    print(f"  FN={fn}  TN={tn}")
    print()
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1:        {f1:.3f}")
    print(f"  Accuracy:  {accuracy:.3f}")

    # Per-benchmark breakdown
    print()
    for bench in ["MMLU", "Olympiad"]:
        bench_traces = [r for r in per_trace if r["benchmark"] == bench]
        if not bench_traces:
            continue
        b_tp = sum(1 for r in bench_traces if r["predicted"] and r["ground_truth"])
        b_fp = sum(1 for r in bench_traces if r["predicted"] and not r["ground_truth"])
        b_fn = sum(1 for r in bench_traces if not r["predicted"] and r["ground_truth"])
        b_tn = sum(1 for r in bench_traces if not r["predicted"] and not r["ground_truth"])
        b_prec = b_tp / (b_tp + b_fp) if (b_tp + b_fp) > 0 else 0.0
        b_rec = b_tp / (b_tp + b_fn) if (b_tp + b_fn) > 0 else 0.0
        b_f1 = 2 * b_prec * b_rec / (b_prec + b_rec) if (b_prec + b_rec) > 0 else 0.0
        print(f"  {bench}: P={b_prec:.3f} R={b_rec:.3f} F1={b_f1:.3f} (TP={b_tp} FP={b_fp} FN={b_fn} TN={b_tn})")

    # Misclassifications
    errors = [r for r in per_trace if not r["correct"]]
    if errors:
        print(f"\n  Misclassifications ({len(errors)}):")
        for r in errors[:10]:
            label = "FP" if r["predicted"] else "FN"
            print(f"    [{label}] trace_{r['trace_id']} ({r['benchmark']}/{r['llm']})")

    # Save results
    results = {
        "fm_id": FM_ID,
        "n_samples": len(traces),
        "n_per_class": n_per_class,
        "seed": seed,
        "metrics": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        },
        "elapsed_s": round(elapsed, 1),
        "per_trace": per_trace,
    }
    out_path = RESULTS_DIR / "classifier_calibration_fm13.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FM-1.3 Classifier Calibration")
    parser.add_argument("--n-per-class", type=int, default=15, help="Samples per class per benchmark (default: 15)")
    parser.add_argument("--workers", type=int, default=32, help="Max concurrent workers (default: 32)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--model", default=None, help="Override LLM model for classification")
    args = parser.parse_args()

    run_calibration(
        n_per_class=args.n_per_class,
        max_workers=args.workers,
        seed=args.seed,
        model=args.model,
    )
