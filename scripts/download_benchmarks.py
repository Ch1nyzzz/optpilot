#!/usr/bin/env python3
"""Warm the cache for the official benchmarks used by online experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optpilot.data.benchmarks import (  # noqa: E402
    AIME2025_CONFIGS,
    AIME2025_DATASET,
    MMLU_DATASET,
    MMLU_DEFAULT_SUBJECTS,
    OLYMPIADBENCH_DATASET,
    OLYMPIADBENCH_DEFAULT_CONFIG,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download official online benchmark datasets")
    parser.add_argument(
        "--mmlu-subject",
        action="append",
        dest="mmlu_subjects",
        default=None,
        help="Repeat to override the default MMLU subject list.",
    )
    parser.add_argument(
        "--olympiad-config",
        default=OLYMPIADBENCH_DEFAULT_CONFIG,
        help="OlympiadBench config to cache (default: maths_en_no_proof).",
    )
    args = parser.parse_args()

    subjects = tuple(args.mmlu_subjects or MMLU_DEFAULT_SUBJECTS)
    print(f"Caching MMLU subjects: {subjects}")
    for subject in subjects:
        dataset = load_dataset(MMLU_DATASET, subject, split="test")
        print(f"  MMLU/{subject}: {len(dataset)} test examples")

    print(f"Caching AIME 2025 configs: {AIME2025_CONFIGS}")
    for config in AIME2025_CONFIGS:
        dataset = load_dataset(AIME2025_DATASET, config, split="test")
        print(f"  AIME2025/{config}: {len(dataset)} test examples")

    olympiad = load_dataset(OLYMPIADBENCH_DATASET, args.olympiad_config, split="test")
    print(f"OlympiadBench/{args.olympiad_config}: {len(olympiad)} test examples")


if __name__ == "__main__":
    main()
