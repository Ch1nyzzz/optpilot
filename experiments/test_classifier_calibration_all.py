"""Full FM Classifier Calibration - all FMs × all models × both prompts.

Evaluates classify_all_fms on held-out AG2 traces with MAST annotations.
Reports per-FM precision/recall/F1 for each (model, prompt_style) combination.

Usage:
    python -u -m experiments.test_classifier_calibration_all
    python -u -m experiments.test_classifier_calibration_all --n-samples 40 --workers 64
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optpilot.config import RESULTS_DIR
from optpilot.data.fm_taxonomy import FM_IDS, FM_NAMES
from optpilot.data.loader import load_traces
from optpilot.modules.diagnoser import Diagnoser

MAS_NAME = "AG2"

MODELS = [
    ("MiniMax-M2.5", "MiniMaxAI/MiniMax-M2.5"),
    ("GLM-5", "zai-org/GLM-5"),
    ("Kimi-K2.5", "moonshotai/Kimi-K2.5"),
    ("DeepSeek-R1", "deepseek-ai/DeepSeek-R1"),
]

PROMPT_STYLES = ["short", "mast"]


def sample_traces(n_samples: int = 40, seed: int = 42) -> list:
    """Sample traces stratified by benchmark, mixing pos and neg for all FMs."""
    rng = random.Random(seed)
    samples = []
    for bench in ["MMLU", "Olympiad"]:
        all_traces = load_traces(MAS_NAME, benchmark=bench)
        n = min(n_samples // 2, len(all_traces))
        samples.extend(rng.sample(all_traces, n))
        print(f"  {bench}: sampled {n}/{len(all_traces)} traces")
    rng.shuffle(samples)
    return samples


def compute_metrics(ground_truth: list[int], predicted: list[int]) -> dict:
    """Compute precision/recall/F1 from binary lists."""
    tp = sum(g == 1 and p == 1 for g, p in zip(ground_truth, predicted))
    fp = sum(g == 0 and p == 1 for g, p in zip(ground_truth, predicted))
    fn = sum(g == 1 and p == 0 for g, p in zip(ground_truth, predicted))
    tn = sum(g == 0 and p == 0 for g, p in zip(ground_truth, predicted))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(ground_truth) if ground_truth else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "accuracy": round(accuracy, 3),
        "support_pos": tp + fn,
        "support_neg": fp + tn,
    }


def run_one_combination(
    traces: list,
    model_id: str,
    prompt_style: str,
    max_workers: int,
) -> dict[str, dict]:
    """Run classification for one (model, prompt) combo, return per-FM metrics."""
    diagnoser = Diagnoser(max_workers=max_workers)

    # Classify all traces concurrently
    results: dict[int, dict[str, bool]] = {}

    def classify_one(idx: int, trace) -> None:
        r = diagnoser.classify_all_fms(trace, model=model_id, prompt_style=prompt_style)
        results[idx] = r

    with ThreadPoolExecutor(max_workers=min(max_workers, len(traces))) as pool:
        futures = {
            pool.submit(classify_one, i, t): i for i, t in enumerate(traces)
        }
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                idx = futures[fut]
                print(f"    Error trace {idx}: {e}")
                results[idx] = {}

    # Compute per-FM metrics
    per_fm = {}
    for fm_id in FM_IDS:
        gt = [t.mast_annotation.get(fm_id, 0) for t in traces]
        pred = [1 if results.get(i, {}).get(fm_id, False) else 0 for i in range(len(traces))]
        per_fm[fm_id] = compute_metrics(gt, pred)

    return per_fm


def run_calibration(
    n_samples: int = 40,
    max_workers: int = 64,
    seed: int = 42,
):
    print(f"=== Full FM Classifier Calibration ===")
    print(f"Samples: {n_samples}, Workers: {max_workers}")
    print()

    traces = sample_traces(n_samples=n_samples, seed=seed)
    print(f"\nTotal: {len(traces)} traces")

    # Show FM distribution in sample
    print("\nGround truth FM distribution:")
    for fm_id in FM_IDS:
        n_pos = sum(1 for t in traces if t.mast_annotation.get(fm_id, 0) == 1)
        if n_pos > 0:
            print(f"  FM-{fm_id} ({FM_NAMES[fm_id]}): {n_pos}/{len(traces)}")
    print()

    all_results = {}

    for model_name, model_id in MODELS:
        for prompt_style in PROMPT_STYLES:
            combo = f"{model_name}+{prompt_style}"
            print(f"\n{'='*60}")
            print(f"Running: {combo}")
            print(f"{'='*60}")

            t0 = time.time()
            per_fm = run_one_combination(traces, model_id, prompt_style, max_workers)
            elapsed = time.time() - t0

            all_results[combo] = {
                "model": model_id,
                "prompt_style": prompt_style,
                "elapsed_s": round(elapsed, 1),
                "per_fm": per_fm,
            }

            # Print summary for this combo
            print(f"\n  Done in {elapsed:.1f}s")
            print(f"  {'FM':<6} {'P':>6} {'R':>6} {'F1':>6} {'Acc':>6} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}")
            print(f"  {'-'*54}")
            for fm_id in FM_IDS:
                m = per_fm[fm_id]
                if m["support_pos"] > 0 or m["tp"] + m["fp"] > 0:
                    print(
                        f"  {fm_id:<6} {m['precision']:>6.3f} {m['recall']:>6.3f} "
                        f"{m['f1']:>6.3f} {m['accuracy']:>6.3f} "
                        f"{m['tp']:>4} {m['fp']:>4} {m['fn']:>4} {m['tn']:>4}"
                    )

            # Macro-average F1 (only FMs with positive support)
            fms_with_support = [fm for fm in FM_IDS if per_fm[fm]["support_pos"] > 0]
            macro_f1 = sum(per_fm[fm]["f1"] for fm in fms_with_support) / len(fms_with_support) if fms_with_support else 0
            print(f"  {'MACRO':>6} F1={macro_f1:.3f} (over {len(fms_with_support)} FMs with positive support)")

    # Final comparison table
    print(f"\n\n{'='*80}")
    print("COMPARISON: Macro-F1 by (model, prompt)")
    print(f"{'='*80}")
    print(f"{'Combo':<30} {'Macro-F1':>10} {'FM-1.3 F1':>10} {'Time(s)':>10}")
    print(f"{'-'*60}")
    for combo, data in all_results.items():
        fms_with_support = [fm for fm in FM_IDS if data["per_fm"][fm]["support_pos"] > 0]
        macro_f1 = sum(data["per_fm"][fm]["f1"] for fm in fms_with_support) / len(fms_with_support) if fms_with_support else 0
        fm13_f1 = data["per_fm"]["1.3"]["f1"]
        print(f"  {combo:<28} {macro_f1:>10.3f} {fm13_f1:>10.3f} {data['elapsed_s']:>10.1f}")

    # Save
    out_path = RESULTS_DIR / "classifier_calibration_all_fms.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full FM Classifier Calibration")
    parser.add_argument("--n-samples", type=int, default=40, help="Total samples (split across benchmarks, default: 40)")
    parser.add_argument("--workers", type=int, default=64, help="Max concurrent workers (default: 64)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    run_calibration(
        n_samples=args.n_samples,
        max_workers=args.workers,
        seed=args.seed,
    )
