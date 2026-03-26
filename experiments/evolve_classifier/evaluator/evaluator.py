#!/usr/bin/env python3
"""Evaluate FM classifier prompt against MAST-annotated AG2 traces.

Uses gpt-oss-120b as the classification model.
"""

import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai

# --- LLM setup (Together AI, gpt-oss-120b) ---
# Load API key: try env vars, then fall back to .env file
def _load_api_key() -> str:
    """Load Together AI API key from env vars or .env file. Fail fast if missing."""
    key = (
        os.environ.get("TOGETHER_API_KEY")
        or os.environ.get("TOGETHER_AI_API_KEY")
        or os.environ.get("together_ai_api")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    )
    if not key:
        _env_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env")
        if os.path.exists(_env_path):
            for line in open(_env_path):
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
MAX_WORKERS = int(os.environ.get("EVAL_MAX_WORKERS", "32"))

# Preflight check: verify API credentials with a minimal request
client = openai.OpenAI(api_key=API_KEY, base_url=API_BASE)
try:
    client.models.list()
except openai.AuthenticationError:
    raise RuntimeError(
        f"Together AI authentication failed (401). Check your API key. "
        f"API base: {API_BASE}"
    )
except Exception:
    pass  # Other errors (network, etc.) — let the actual evaluation handle them

EVAL_DATA_PATH = os.path.join(os.path.dirname(__file__), "eval_traces.json")

ALL_FM_IDS = [
    "1.1", "1.2", "1.3", "1.4", "1.5",
    "2.1", "2.2", "2.3", "2.4", "2.5", "2.6",
    "3.1", "3.2", "3.3",
]


def llm_call(prompt: str) -> str:
    """Call Together AI LLM (gpt-oss-120b)."""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=EVAL_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=4096,
            )
            return response.choices[0].message.content or ""
        except openai.AuthenticationError:
            raise RuntimeError("Together AI API returned 401 — invalid API key")
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return ""


def compute_f1(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate(program_path: str) -> dict:
    """Evaluate a candidate classifier program. Called by skydiscover."""
    # Load candidate module
    spec = importlib.util.spec_from_file_location("candidate", program_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Load evaluation traces
    with open(EVAL_DATA_PATH) as f:
        traces = json.load(f)

    # Classify all traces concurrently
    results = {}

    def classify_one(idx: int, trace: dict):
        try:
            pred = mod.classify(trace["trajectory"], llm_call)
            results[idx] = pred
        except Exception:
            results[idx] = {}

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(traces))) as pool:
        futures = {
            pool.submit(classify_one, i, t): i for i, t in enumerate(traces)
        }
        for fut in as_completed(futures):
            fut.result()

    # Compute per-FM metrics
    per_fm = {}
    fm_f1_values = []
    for fm_id in ALL_FM_IDS:
        tp = fp = fn = tn = 0
        for i, trace in enumerate(traces):
            gt = trace["mast_annotation"].get(fm_id, 0) == 1
            pred = results.get(i, {}).get(fm_id, False)
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

        per_fm[fm_id] = {
            "f1": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "support": support,
        }
        if support > 0:
            fm_f1_values.append(f1)

    macro_f1 = sum(fm_f1_values) / len(fm_f1_values) if fm_f1_values else 0.0

    # Build feedback
    lines = [f"Macro-F1: {macro_f1:.4f}", ""]
    lines.append(f"{'FM':<6} {'F1':>6} {'P':>6} {'R':>6} {'TP':>4} {'FP':>4} {'FN':>4} {'sup':>4}")
    lines.append("-" * 48)
    for fm_id in ALL_FM_IDS:
        m = per_fm[fm_id]
        if m["support"] > 0 or m["tp"] + m["fp"] > 0:
            lines.append(
                f"{fm_id:<6} {m['f1']:>6.3f} {m['precision']:>6.3f} "
                f"{m['recall']:>6.3f} {m['tp']:>4} {m['fp']:>4} {m['fn']:>4} {m['support']:>4}"
            )

    worst = sorted(
        [(fm_id, per_fm[fm_id]) for fm_id in ALL_FM_IDS if per_fm[fm_id]["support"] > 0],
        key=lambda x: x[1]["f1"],
    )[:3]
    lines.append("")
    lines.append("Worst FMs (focus improvement here):")
    for fm_id, m in worst:
        lines.append(f"  FM-{fm_id}: F1={m['f1']:.3f} (TP={m['tp']}, FP={m['fp']}, FN={m['fn']})")

    return {
        "status": "success",
        "combined_score": round(max(0.0, macro_f1), 4),
        "metrics": per_fm,
        "artifacts": {"feedback": "\n".join(lines)},
    }


if __name__ == "__main__":
    result = evaluate(sys.argv[1])
    print(json.dumps(result))
