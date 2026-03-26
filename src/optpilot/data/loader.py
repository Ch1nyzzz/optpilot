"""MAST-Data loader - loads traces from local cached JSON files."""

import json
from collections import Counter
from pathlib import Path

from optpilot.config import MAST_DATA_CACHE
from optpilot.models import MASTrace


def _find_full_dataset() -> Path:
    """Find MAD_full_dataset.json in HuggingFace cache."""
    for snap_dir in MAST_DATA_CACHE.iterdir():
        full = snap_dir / "MAD_full_dataset.json"
        if full.exists():
            return full
    raise FileNotFoundError(
        f"MAD_full_dataset.json not found in {MAST_DATA_CACHE}. "
        "Run: from datasets import load_dataset; load_dataset('mcemri/MAST-Data')"
    )


def load_all_traces() -> list[MASTrace]:
    """Load all 1242 traces from MAST-Data."""
    path = _find_full_dataset()
    with open(path) as f:
        raw = json.load(f)

    traces = []
    for item in raw:
        trace = MASTrace(
            trace_id=item["trace_id"],
            mas_name=item["mas_name"],
            llm_name=item["llm_name"],
            benchmark_name=item["benchmark_name"],
            trajectory=item["trace"]["trajectory"],
            task_key=item["trace"].get("key", ""),
            mast_annotation=item["mast_annotation"],
        )
        traces.append(trace)
    return traces


def load_traces(
    mas_name: str = "AG2",
    fm_filter: str | None = None,
    benchmark: str | None = None,
    llm_filter: str | None = None,
) -> list[MASTrace]:
    """Load traces filtered by MAS framework, FM, benchmark, and LLM."""
    traces = [t for t in load_all_traces() if t.mas_name == mas_name]
    if benchmark:
        traces = [t for t in traces if t.benchmark_name == benchmark]
    if llm_filter:
        traces = [t for t in traces if t.llm_name == llm_filter]
    if fm_filter:
        traces = [t for t in traces if t.mast_annotation.get(fm_filter, 0) == 1]
    return traces


# Backward compat
def load_chatdev_traces(fm_filter=None, benchmark=None):
    return load_traces("ChatDev", fm_filter=fm_filter, benchmark=benchmark)


def print_fm_stats(traces: list[MASTrace]) -> None:
    """Print FM distribution statistics."""
    total = len(traces)
    fm_counts: Counter[str] = Counter()
    for t in traces:
        for fm_id, val in t.mast_annotation.items():
            if val == 1:
                fm_counts[fm_id] += 1

    clean = sum(1 for t in traces if not t.active_fm_ids())
    print(f"Total traces: {total}, Clean: {clean} ({clean/total*100:.1f}%)")
    print("FM distribution:")
    for fm_id, cnt in fm_counts.most_common():
        print(f"  FM-{fm_id}: {cnt} ({cnt/total*100:.1f}%)")
