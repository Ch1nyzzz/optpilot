"""Run blind 6-group labeling on the 100-trace AG2 annotation pack.

Example:
  python3 -m experiments.run_ag2_6group_blind_models
  python3 -m experiments.run_ag2_6group_blind_models --workers 24
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optpilot.llm import acall_llm_json

PACK_PATH = Path("data/annotations/ag2_6group_eval_100_blind.jsonl")
OUT_DIR = Path("results/annotation_runs/ag2_6group_eval_100")

MODEL_MAP = {
    "glm5": "zai-org/GLM-5",
    "minimax": "MiniMaxAI/MiniMax-M2.5",
    "deepseekr1": "deepseek-ai/DeepSeek-R1",
    "kimi": "moonshotai/Kimi-K2.5",
}

GROUP_IDS = ["A", "B", "C", "D", "E", "F"]

SYSTEM_PROMPT = """\
You are a careful evaluator for multi-agent system execution traces.

Your job is to label a trace using the following 6-group taxonomy.

Definitions:

A = Instruction Non-Compliance
The agent violates an explicit task requirement, constraint, or role boundary.

B = Execution Loop / Stuck
The system repeats the same step or keeps going after it should stop, without meaningful progress.

C = Context Loss
The agent loses previously established context, forgets prior progress, or the conversation effectively resets.

D = Communication Failure
Critical information is not shared, ignored, or clarification is not requested when needed between agents.

E = Task Drift / Reasoning Error
The execution path deviates from the intended task, or the agent's action does not match its stated reasoning.

F = Verification Failure
The system fails to verify properly, verifies too weakly, stops too early, or concludes success incorrectly.

Important rules:
1. A trace may have multiple true labels.
2. A trace may also have NO failure labels at all.
3. If none of A-F clearly applies, output all false.
4. Do not assume a trace is faulty just because it comes from a benchmark dataset.
5. Label only what is clearly supported by the trace.
6. Prefer precision over recall when evidence is weak.
7. Focus on root cause, not just surface symptoms.
8. Keep rationale brief and evidence-based.

Return ONLY valid JSON with this exact schema:
{
  "A": false,
  "B": false,
  "C": false,
  "D": false,
  "E": false,
  "F": false,
  "primary_turning_point": "",
  "evidence_snippet": "",
  "rationale": ""
}
"""

USER_PROMPT_TEMPLATE = """\
Label the following AG2 trace using the 6-group taxonomy.

Instructions:
- Read the full trace carefully.
- Decide whether each label A-F is true or false.
- If the trace completes normally and no clear failure appears, return all false.
- `primary_turning_point` should be the first step where failure becomes clearly visible. If no failure exists, write "none".
- `evidence_snippet` should be a short quote or short summary of the key evidence.
- `rationale` should be 1-3 sentences only.
- Before assigning any positive label, ask whether a careful human reviewer would agree that the failure is clearly present from the trace alone.

Trace metadata:
- sample_id: {sample_id}
- benchmark_name: {benchmark_name}
- trace_id: {trace_id}

Trace:
{trajectory}
"""


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no", ""}:
            return False
    return False


def load_pack(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


async def label_one(sample: dict, model_id: str) -> dict:
    user_prompt = USER_PROMPT_TEMPLATE.format(
        sample_id=sample["sample_id"],
        benchmark_name=sample["benchmark_name"],
        trace_id=sample["trace_id"],
        trajectory=sample["trajectory"],
    )
    result = await acall_llm_json(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        model=model_id,
        temperature=0.0,
        max_tokens=2048,
    )
    labels = {gid: _coerce_bool(result.get(gid, False)) for gid in GROUP_IDS}
    return {
        "sample_id": sample["sample_id"],
        "source_uid": sample["source_uid"],
        "trace_id": sample["trace_id"],
        "benchmark_name": sample["benchmark_name"],
        "llm_name": sample["llm_name"],
        "task_key": sample["task_key"],
        "labels": labels,
        "primary_turning_point": str(result.get("primary_turning_point", "")).strip(),
        "evidence_snippet": str(result.get("evidence_snippet", "")).strip(),
        "rationale": str(result.get("rationale", "")).strip(),
    }


async def run_model(alias: str, model_id: str, samples: list[dict], workers: int) -> dict:
    print(f"=== Running {alias} ({model_id}) on {len(samples)} traces ===")
    t0 = time.time()
    semaphore = asyncio.Semaphore(workers)
    results: dict[str, dict] = {}
    failures: dict[str, str] = {}
    done = 0

    async def run_one(sample: dict) -> None:
        nonlocal done
        async with semaphore:
            try:
                results[sample["sample_id"]] = await label_one(sample, model_id=model_id)
            except Exception as e:
                failures[sample["sample_id"]] = str(e)
            done += 1
            if done % 10 == 0:
                print(f"  {alias}: {done}/{len(samples)} done")

    await asyncio.gather(*(run_one(sample) for sample in samples))

    ordered = []
    for sample in samples:
        row = results.get(sample["sample_id"])
        if row is None:
            row = {
                "sample_id": sample["sample_id"],
                "source_uid": sample["source_uid"],
                "trace_id": sample["trace_id"],
                "benchmark_name": sample["benchmark_name"],
                "llm_name": sample["llm_name"],
                "task_key": sample["task_key"],
                "labels": {gid: False for gid in GROUP_IDS},
                "primary_turning_point": "",
                "evidence_snippet": "",
                "rationale": "",
                "error": failures[sample["sample_id"]],
            }
        ordered.append(row)

    label_counts = Counter()
    nonempty = 0
    all_false = 0
    for row in ordered:
        active = [gid for gid, val in row["labels"].items() if val]
        if active:
            nonempty += 1
        else:
            all_false += 1
        for gid in active:
            label_counts[gid] += 1

    elapsed = round(time.time() - t0, 1)
    payload = {
        "model_alias": alias,
        "model_id": model_id,
        "n_samples": len(samples),
        "elapsed_s": elapsed,
        "workers": workers,
        "label_counts": {gid: label_counts.get(gid, 0) for gid in GROUP_IDS},
        "nonempty_predictions": nonempty,
        "all_false_predictions": all_false,
        "errors": failures,
        "predictions": ordered,
    }
    print(
        f"  {alias} done in {elapsed}s | "
        f"nonempty={nonempty} all_false={all_false} errors={len(failures)}"
    )
    return payload


async def main_async(workers: int, models: list[str]) -> None:
    samples = load_pack(PACK_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    selected_models = {alias: MODEL_MAP[alias] for alias in models}
    summary = {
        "pack_path": str(PACK_PATH),
        "n_samples": len(samples),
        "workers_per_model": workers,
        "models": {},
    }

    tasks = {
        asyncio.create_task(
            run_model(alias=alias, model_id=model_id, samples=samples, workers=workers)
        ): alias
        for alias, model_id in selected_models.items()
    }

    for task in asyncio.as_completed(tasks):
        payload = await task
        alias = payload["model_alias"]
        out_path = OUT_DIR / f"{alias}.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["models"][alias] = {
            "model_id": payload["model_id"],
            "elapsed_s": payload["elapsed_s"],
            "label_counts": payload["label_counts"],
            "nonempty_predictions": payload["nonempty_predictions"],
            "all_false_predictions": payload["all_false_predictions"],
            "n_errors": len(payload["errors"]),
            "output_path": str(out_path),
        }

    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run blind AG2 6-group labels across models.")
    parser.add_argument("--workers", type=int, default=16, help="Per-model async concurrency.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=list(MODEL_MAP.keys()),
        choices=list(MODEL_MAP.keys()),
        help="Model aliases to run.",
    )
    args = parser.parse_args()
    asyncio.run(main_async(workers=args.workers, models=args.models))


if __name__ == "__main__":
    main()
