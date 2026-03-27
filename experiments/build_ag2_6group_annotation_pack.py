"""Build a 100-trace blind annotation pack for AG2 6-group evaluation.

Outputs:
  - data/annotations/ag2_6group_eval_100_blind.jsonl
  - data/annotations/ag2_6group_eval_100_index.csv
  - data/annotations/ag2_6group_eval_100_template.csv
  - data/annotations/ag2_6group_eval_100_manifest.json

Sampling policy:
  - 100 AG2 traces total from MMLU + Olympiad
  - 20 weak-clean traces (10 per benchmark)
  - 80 additional traces selected greedily to improve weak 6-group coverage
  - loader-normalized 6-group labels are used only for sampling balance, not as gold labels
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optpilot.data.fm_taxonomy_6group import GROUP_IDS
from optpilot.data.loader import load_traces

MAS_NAME = "AG2"
BENCHMARKS = ("MMLU", "Olympiad")
DEFAULT_TOTAL = 100
DEFAULT_CLEAN = 20
DEFAULT_SEED = 42
BENCHMARK_TARGET = {"MMLU": 50, "Olympiad": 50}
GROUP_TARGETS = {
    "A": 15,
    "B": 18,
    "C": 12,
    "D": 15,
    "E": 18,
    "F": 18,
}


def _source_uid(trace) -> str:
    return f"{trace.benchmark_name}:{trace.trace_id}"


def _weak_groups(trace) -> dict[str, int]:
    return dict(trace.mast_annotation)


def _active_groups(trace) -> list[str]:
    return [gid for gid, val in _weak_groups(trace).items() if val]


def _load_ag2_traces() -> list:
    traces = []
    for bench in BENCHMARKS:
        traces.extend(load_traces(MAS_NAME, benchmark=bench))
    return traces


def _sample_clean(
    traces: list,
    rng: random.Random,
    clean_total: int,
) -> list:
    clean_per_bench = clean_total // len(BENCHMARKS)
    selected = []
    for bench in BENCHMARKS:
        pool = [
            trace for trace in traces
            if trace.benchmark_name == bench and not any(_weak_groups(trace).values())
        ]
        if len(pool) < clean_per_bench:
            raise ValueError(
                f"Not enough clean traces in {bench}: need {clean_per_bench}, found {len(pool)}"
            )
        selected.extend(rng.sample(pool, clean_per_bench))
    return selected


def _score_candidate(
    trace,
    current_group_counts: Counter[str],
    bench_counts: Counter[str],
    focus_group: str | None,
) -> tuple[float, ...]:
    active = _active_groups(trace)
    deficits = {
        gid: max(GROUP_TARGETS[gid] - current_group_counts.get(gid, 0), 0)
        for gid in GROUP_IDS
    }
    group_gain = sum(deficits[gid] for gid in active if deficits[gid] > 0)
    rarity_gain = sum(1.0 / GROUP_TARGETS[gid] for gid in active if deficits[gid] > 0)
    bench_need = max(BENCHMARK_TARGET[trace.benchmark_name] - bench_counts[trace.benchmark_name], 0)
    focus_hit = 1.0 if focus_group and focus_group in active else 0.0
    clean_penalty = 1.0 if not active else 0.0
    return (
        focus_hit,
        group_gain,
        bench_need,
        rarity_gain,
        -clean_penalty,
        -len(active),
    )


def sample_annotation_pack(
    total: int = DEFAULT_TOTAL,
    clean_total: int = DEFAULT_CLEAN,
    seed: int = DEFAULT_SEED,
) -> tuple[list, dict]:
    if total != sum(BENCHMARK_TARGET.values()):
        raise ValueError(
            f"This pack is configured for {sum(BENCHMARK_TARGET.values())} total traces, got {total}."
        )
    if clean_total % len(BENCHMARKS) != 0:
        raise ValueError("clean_total must split evenly across benchmarks.")

    rng = random.Random(seed)
    traces = _load_ag2_traces()
    selected = _sample_clean(traces, rng=rng, clean_total=clean_total)
    selected_ids = {_source_uid(trace) for trace in selected}

    group_counts: Counter[str] = Counter()
    bench_counts: Counter[str] = Counter(trace.benchmark_name for trace in selected)
    for trace in selected:
        for gid in _active_groups(trace):
            group_counts[gid] += 1

    while len(selected) < total:
        remaining = [
            trace for trace in traces
            if _source_uid(trace) not in selected_ids
            and bench_counts[trace.benchmark_name] < BENCHMARK_TARGET[trace.benchmark_name]
        ]
        if not remaining:
            raise RuntimeError("Ran out of candidate traces before reaching target size.")

        deficits = {
            gid: GROUP_TARGETS[gid] - group_counts.get(gid, 0)
            for gid in GROUP_IDS
        }
        unmet = [gid for gid, deficit in deficits.items() if deficit > 0]
        focus_group = max(unmet, key=lambda gid: deficits[gid]) if unmet else None

        rng.shuffle(remaining)
        if focus_group:
            focused = [trace for trace in remaining if focus_group in _active_groups(trace)]
            if focused:
                remaining = focused

        best = max(
            remaining,
            key=lambda trace: _score_candidate(
                trace,
                current_group_counts=group_counts,
                bench_counts=bench_counts,
                focus_group=focus_group,
            ),
        )
        selected.append(best)
        selected_ids.add(_source_uid(best))
        bench_counts[best.benchmark_name] += 1
        for gid in _active_groups(best):
            group_counts[gid] += 1

    selected.sort(key=lambda trace: (trace.benchmark_name, trace.trace_id))

    manifest = {
        "seed": seed,
        "total": total,
        "clean_total": clean_total,
        "benchmark_counts": dict(Counter(trace.benchmark_name for trace in selected)),
        "weak_group_counts": {
            gid: sum(_weak_groups(trace)[gid] for trace in selected)
            for gid in GROUP_IDS
        },
        "group_targets": GROUP_TARGETS,
        "selection_policy": {
            "clean": f"{clean_total // len(BENCHMARKS)} clean traces per benchmark",
            "non_clean": "greedy selection for weak 6-group coverage with benchmark balancing",
        },
        "note": (
            "weak_group_counts come from loader-normalized 6-group annotations. "
            "They are for sampling audit only, not gold labels."
        ),
    }
    return selected, manifest


def write_outputs(selected: list, manifest: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    blind_path = out_dir / "ag2_6group_eval_100_blind.jsonl"
    index_path = out_dir / "ag2_6group_eval_100_index.csv"
    template_path = out_dir / "ag2_6group_eval_100_template.csv"
    manifest_path = out_dir / "ag2_6group_eval_100_manifest.json"

    blind_rows = []
    index_rows = []
    template_rows = []

    for idx, trace in enumerate(selected, start=1):
        sample_id = f"ag2_6g_{idx:03d}"
        source_uid = _source_uid(trace)
        prompt_preview = trace.trajectory[:180].replace("\n", " ")

        blind_rows.append({
            "sample_id": sample_id,
            "source_uid": source_uid,
            "trace_id": trace.trace_id,
            "benchmark_name": trace.benchmark_name,
            "llm_name": trace.llm_name,
            "task_key": trace.task_key,
            "trajectory_chars": len(trace.trajectory),
            "trajectory": trace.trajectory,
        })

        index_rows.append({
            "sample_id": sample_id,
            "source_uid": source_uid,
            "trace_id": trace.trace_id,
            "benchmark_name": trace.benchmark_name,
            "llm_name": trace.llm_name,
            "task_key": trace.task_key,
            "trajectory_chars": len(trace.trajectory),
            "prompt_preview": prompt_preview,
        })

        template_rows.append({
            "sample_id": sample_id,
            "source_uid": source_uid,
            "trace_id": trace.trace_id,
            "benchmark_name": trace.benchmark_name,
            "llm_name": trace.llm_name,
            "task_key": trace.task_key,
            "A": "",
            "B": "",
            "C": "",
            "D": "",
            "E": "",
            "F": "",
            "primary_turning_point": "",
            "evidence_snippet": "",
            "rationale": "",
            "annotator": "",
            "review_status": "",
            "notes": "",
        })

    with blind_path.open("w", encoding="utf-8") as f:
        for row in blind_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with index_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
        writer.writeheader()
        writer.writerows(index_rows)

    with template_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(template_rows[0].keys()))
        writer.writeheader()
        writer.writerows(template_rows)

    manifest_with_sources = dict(manifest)
    manifest_with_sources["samples"] = [
        {
            "sample_id": row["sample_id"],
            "source_uid": row["source_uid"],
            "benchmark_name": row["benchmark_name"],
            "trace_id": row["trace_id"],
        }
        for row in index_rows
    ]
    manifest_path.write_text(
        json.dumps(manifest_with_sources, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AG2 6-group annotation pack.")
    parser.add_argument("--total", type=int, default=DEFAULT_TOTAL)
    parser.add_argument("--clean-total", type=int, default=DEFAULT_CLEAN)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/annotations"),
        help="Output directory for generated files.",
    )
    args = parser.parse_args()

    selected, manifest = sample_annotation_pack(
        total=args.total,
        clean_total=args.clean_total,
        seed=args.seed,
    )
    write_outputs(selected, manifest, out_dir=args.out_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
