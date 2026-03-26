#!/usr/bin/env python3
"""Evaluate coarse 3-label FM classifier against the existing trace dataset."""

import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai


CATEGORY_MAP = {
    "C1": ["1.1", "1.2", "1.3", "1.4", "1.5"],
    "C2": ["2.1", "2.2", "2.3", "2.4", "2.5", "2.6"],
    "C3": ["3.1", "3.2", "3.3"],
}
ALL_CATEGORY_IDS = ["C1", "C2", "C3"]


def _load_api_key() -> str:
    key = (
        os.environ.get("TOGETHER_API_KEY")
        or os.environ.get("TOGETHER_AI_API_KEY")
        or os.environ.get("together_ai_api")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    )
    if not key:
        env_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env")
        if os.path.exists(env_path):
            for line in open(env_path):
                line = line.strip()
                if (
                    line.startswith("TOGETHER_API_KEY=")
                    or line.startswith("TOGETHER_AI_API_KEY=")
                    or line.startswith("together_ai_api=")
                ):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not key:
        raise RuntimeError(
            "Together AI API key not found. Set TOGETHER_API_KEY, TOGETHER_AI_API_KEY, "
            "together_ai_api, or OPENAI_API_KEY, or add it to .env."
        )
    return key


API_KEY = _load_api_key()
API_BASE = (
    os.environ.get("TOGETHER_API_BASE")
    or os.environ.get("OPENAI_API_BASE")
    or os.environ.get("OPENAI_BASE_URL")
    or "https://api.together.xyz/v1"
)
EVAL_MODEL = "openai/gpt-oss-120b"
MAX_WORKERS = int(os.environ.get("EVAL_MAX_WORKERS", "24"))
CLIENT = openai.OpenAI(api_key=API_KEY, base_url=API_BASE)
EVAL_DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "evolve_classifier",
    "evaluator",
    "eval_traces.json",
)


def llm_call(prompt: str) -> str:
    """Call Together AI LLM."""
    for attempt in range(3):
        try:
            response = CLIENT.chat.completions.create(
                model=EVAL_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=2048,
            )
            return response.choices[0].message.content or ""
        except openai.AuthenticationError:
            raise RuntimeError("Together AI API returned 401 - invalid API key")
        except Exception:
            if attempt < 2:
                time.sleep(2**attempt)
            else:
                return ""


def compute_f1(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def collapse_annotation(mast_annotation: dict) -> dict[str, bool]:
    return {
        cat_id: any(mast_annotation.get(fm_id, 0) == 1 for fm_id in fm_ids)
        for cat_id, fm_ids in CATEGORY_MAP.items()
    }


def evaluate(program_path: str) -> dict:
    """Evaluate a candidate 3-label classifier program."""
    spec = importlib.util.spec_from_file_location("candidate", program_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    with open(EVAL_DATA_PATH) as f:
        traces = json.load(f)

    results = {}

    def classify_one(idx: int, trace: dict):
        try:
            pred = mod.classify(trace["trajectory"], llm_call)
            results[idx] = pred
        except Exception:
            results[idx] = {}

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(traces))) as pool:
        futures = {pool.submit(classify_one, i, t): i for i, t in enumerate(traces)}
        for fut in as_completed(futures):
            fut.result()

    per_cat = {}
    f1_values = []
    exact_match_correct = 0

    for i, trace in enumerate(traces):
        gt = collapse_annotation(trace["mast_annotation"])
        pred = {cat_id: bool(results.get(i, {}).get(cat_id, False)) for cat_id in ALL_CATEGORY_IDS}
        if pred == gt:
            exact_match_correct += 1

    for cat_id in ALL_CATEGORY_IDS:
        tp = fp = fn = tn = 0
        for i, trace in enumerate(traces):
            gt = collapse_annotation(trace["mast_annotation"]).get(cat_id, False)
            pred = bool(results.get(i, {}).get(cat_id, False))
            if pred and gt:
                tp += 1
            elif pred and not gt:
                fp += 1
            elif not pred and gt:
                fn += 1
            else:
                tn += 1

        f1 = compute_f1(tp, fp, fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        support = tp + fn

        per_cat[cat_id] = {
            "f1": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "support": support,
        }
        f1_values.append(f1)

    macro_f1 = sum(f1_values) / len(f1_values) if f1_values else 0.0
    exact_match_accuracy = exact_match_correct / len(traces) if traces else 0.0
    label_accuracy = sum(
        per_cat[cat_id]["tp"] + per_cat[cat_id]["tn"] for cat_id in ALL_CATEGORY_IDS
    ) / (len(traces) * len(ALL_CATEGORY_IDS))

    lines = [f"Macro-F1: {macro_f1:.4f}", f"Exact-match accuracy: {exact_match_accuracy:.4f}"]
    lines.append(f"Label accuracy: {label_accuracy:.4f}")
    lines.append("")
    lines.append(f"{'CAT':<4} {'F1':>6} {'P':>6} {'R':>6} {'TP':>4} {'FP':>4} {'FN':>4} {'sup':>4}")
    lines.append("-" * 46)
    for cat_id in ALL_CATEGORY_IDS:
        m = per_cat[cat_id]
        lines.append(
            f"{cat_id:<4} {m['f1']:>6.3f} {m['precision']:>6.3f} {m['recall']:>6.3f} "
            f"{m['tp']:>4} {m['fp']:>4} {m['fn']:>4} {m['support']:>4}"
        )

    worst = sorted(
        [(cat_id, per_cat[cat_id]) for cat_id in ALL_CATEGORY_IDS],
        key=lambda x: x[1]["f1"],
    )
    lines.append("")
    lines.append("Worst categories:")
    for cat_id, m in worst:
        lines.append(f"  {cat_id}: F1={m['f1']:.3f} (TP={m['tp']}, FP={m['fp']}, FN={m['fn']})")

    return {
        "status": "success",
        "combined_score": round(max(0.0, macro_f1), 4),
        "exact_match_accuracy": round(exact_match_accuracy, 4),
        "label_accuracy": round(label_accuracy, 4),
        "metrics": per_cat,
        "artifacts": {"feedback": "\n".join(lines)},
    }


if __name__ == "__main__":
    result = evaluate(sys.argv[1])
    print(json.dumps(result))
