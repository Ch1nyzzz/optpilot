"""GAIA-style general agent benchmark.

GAIA is gated on HuggingFace. We use a combination of:
1. Multi-step reasoning questions requiring tool use
2. If GAIA access is available, load from HF directly

Tasks require agents to search, calculate, and synthesize information
from multiple sources - matching Magentic-One's general-purpose design.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Iterable

from optpilot.data.benchmarks import BenchmarkExample


# Synthetic GAIA-style tasks (multi-step, require tools)
_TASKS: list[dict[str, Any]] = [
    {
        "task_id": "gaia_001",
        "prompt": (
            "What is the population of the capital city of the country that "
            "won the most gold medals in the 2024 Paris Olympics? "
            "Provide the approximate population in millions."
        ),
        "gold_answer": "14",  # Beijing, China ≈ 14 million (urban)
        "context_docs": {
            "2024 Olympics Medal Count": "The 2024 Paris Olympics gold medal count: USA 40, China 40, GB 14, France 16. China won 40 gold medals tied with USA but had more total medals.",
            "World Capitals": "China capital: Beijing. USA capital: Washington DC. France capital: Paris.",
            "City Populations": "Beijing population: approximately 21.5 million (metro), 14 million (urban core). Washington DC: 700,000. Paris: 2.1 million.",
        },
    },
    {
        "task_id": "gaia_002",
        "prompt": (
            "Calculate the total area in square kilometers of the three largest "
            "countries in South America. Round to the nearest million."
        ),
        "gold_answer": "13",  # Brazil 8.5M + Argentina 2.8M + Peru 1.3M ≈ 13M
        "context_docs": {
            "South American Countries by Area": "Largest countries in South America by area: 1. Brazil: 8,515,767 km². 2. Argentina: 2,780,400 km². 3. Peru: 1,285,216 km². 4. Colombia: 1,141,748 km².",
        },
    },
    {
        "task_id": "gaia_003",
        "prompt": (
            "Who wrote the novel that was adapted into the 1994 movie 'The Shawshank "
            "Redemption'? In what year was that novel first published? "
            "Return the author's name and the year, separated by a comma."
        ),
        "gold_answer": "Stephen King,1982",
        "context_docs": {
            "The Shawshank Redemption": "The Shawshank Redemption (1994) is based on Stephen King's novella 'Rita Hayworth and Shawshank Redemption', published in the 1982 collection 'Different Seasons'.",
        },
    },
    {
        "task_id": "gaia_004",
        "prompt": (
            "If you invest $10,000 at 5% annual compound interest, how much will "
            "you have after 20 years? Round to the nearest dollar."
        ),
        "gold_answer": "26533",
        "context_docs": {},
    },
    {
        "task_id": "gaia_005",
        "prompt": (
            "What is the chemical formula of the mineral that is the primary "
            "component of sandstone? Also, what is its hardness on the Mohs scale?"
        ),
        "gold_answer": "SiO2,7",
        "context_docs": {
            "Sandstone Composition": "Sandstone is a sedimentary rock composed mainly of quartz (silicon dioxide, SiO2) grains. Quartz has a Mohs hardness of 7.",
        },
    },
    {
        "task_id": "gaia_006",
        "prompt": (
            "How many prime numbers are there between 100 and 200? "
            "List them and return the count."
        ),
        "gold_answer": "21",
        "context_docs": {},
    },
    {
        "task_id": "gaia_007",
        "prompt": (
            "The Eiffel Tower is 330 meters tall. If a ball is dropped from the top "
            "(ignoring air resistance), how many seconds does it take to reach the "
            "ground? Use g=9.8 m/s². Round to one decimal place."
        ),
        "gold_answer": "8.2",
        "context_docs": {
            "Physics Formulas": "Free fall: h = 0.5 * g * t^2, so t = sqrt(2h/g). g = 9.8 m/s².",
        },
    },
    {
        "task_id": "gaia_008",
        "prompt": (
            "What is the GDP per capita (in USD) of the country that hosted "
            "the 2022 FIFA World Cup? Round to the nearest thousand."
        ),
        "gold_answer": "84000",
        "context_docs": {
            "2022 FIFA World Cup": "The 2022 FIFA World Cup was hosted by Qatar.",
            "Qatar Economy": "Qatar GDP per capita: approximately $84,514 USD (2023 estimate), one of the highest in the world.",
        },
    },
    {
        "task_id": "gaia_009",
        "prompt": (
            "Convert 72 degrees Fahrenheit to Celsius, then to Kelvin. "
            "Return the Kelvin value rounded to one decimal place."
        ),
        "gold_answer": "295.4",
        "context_docs": {},
    },
    {
        "task_id": "gaia_010",
        "prompt": (
            "What language has the most native speakers in the world? "
            "Approximately how many native speakers does it have (in millions)?"
        ),
        "gold_answer": "Mandarin Chinese,920",
        "context_docs": {
            "World Languages": "Languages by native speakers: 1. Mandarin Chinese: ~920 million. 2. Spanish: ~475 million. 3. English: ~373 million. 4. Hindi: ~344 million.",
        },
    },
    {
        "task_id": "gaia_011",
        "prompt": (
            "A train leaves Station A at 9:00 AM traveling at 80 km/h. "
            "Another train leaves Station B (400 km away) at 10:00 AM traveling "
            "toward Station A at 120 km/h. At what time do they meet?"
        ),
        "gold_answer": "10:36 AM",
        "context_docs": {},
    },
    {
        "task_id": "gaia_012",
        "prompt": (
            "What is the tallest building in the world as of 2024? "
            "How tall is it in meters and how many floors does it have?"
        ),
        "gold_answer": "Burj Khalifa,828,163",
        "context_docs": {
            "Tallest Buildings": "The Burj Khalifa in Dubai is the tallest building in the world at 828 meters (2,717 ft) with 163 floors, completed in 2010.",
        },
    },
]


_STRICT_UNSUPPORTED_TERMS = (
    "attached",
    "attachment",
    "image",
    "photo",
    "screenshot",
    "video",
    "youtube",
    "audio",
    "listen",
    "pdf",
    "spreadsheet",
    "excel",
    "csv",
    "docx",
    "pptx",
    "xlsx",
    "zip",
    "png",
    "jpg",
    "jpeg",
    "mp3",
)


def _question_mentions_unsupported_modality(question: str) -> bool:
    lowered = question.lower()
    return any(term in lowered for term in _STRICT_UNSUPPORTED_TERMS)


def is_strict_supported_gaia_row(question: str, file_name: str | None = None) -> bool:
    """Return whether the current GAIA tool stack can solve the task at all."""
    if file_name:
        return False
    return not _question_mentions_unsupported_modality(question)


def split_gaia_examples_evenly(
    examples: Iterable[BenchmarkExample],
) -> tuple[list[BenchmarkExample], list[BenchmarkExample]]:
    """Split GAIA examples evenly while balancing difficulty levels when present."""
    by_level: dict[str, list[BenchmarkExample]] = defaultdict(list)
    for example in examples:
        level = str(example.metadata.get("level", "unknown"))
        by_level[level].append(example)

    train_examples: list[BenchmarkExample] = []
    test_examples: list[BenchmarkExample] = []
    for level in sorted(by_level):
        level_examples = by_level[level]
        for idx, example in enumerate(level_examples):
            if idx % 2 == 0:
                train_examples.append(example)
            else:
                test_examples.append(example)
    return train_examples, test_examples


def load_gaia_examples(limit: int = 165, *, strict_supported_only: bool = False) -> list[BenchmarkExample]:
    """Load GAIA examples. Tries HF first, falls back to synthetic."""
    # Try loading real GAIA dataset
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "gaia-benchmark/GAIA", "2023_all", split="validation",
            trust_remote_code=True,
        )
        examples = []
        for i, item in enumerate(ds):
            question = item["Question"]
            file_name = str(item.get("file_name", "") or "")
            if strict_supported_only and not is_strict_supported_gaia_row(
                question,
                file_name=file_name,
            ):
                continue
            examples.append(BenchmarkExample(
                benchmark_name="GAIA",
                task_id=item.get("task_id", f"gaia_real_{i}"),
                prompt=question,
                gold_answers=(str(item.get("Final answer", "")),),
                answer_type="exact",
                metadata={
                    "context_docs": {},
                    "level": item.get("Level", ""),
                    "file_name": file_name,
                    "file_path": str(item.get("file_path", "") or ""),
                },
            ))
            if len(examples) >= limit:
                break
        if examples:
            return examples
    except Exception:
        pass

    # Fallback to synthetic
    examples = []
    for task in _TASKS[:limit]:
        if strict_supported_only and not is_strict_supported_gaia_row(
            task["prompt"],
            file_name=None,
        ):
            continue
        examples.append(BenchmarkExample(
            benchmark_name="GAIA",
            task_id=task["task_id"],
            prompt=task["prompt"],
            gold_answers=(task["gold_answer"],),
            answer_type="exact",
            metadata={"context_docs": task.get("context_docs", {})},
        ))
    return examples


def score_gaia(prediction: str, gold_answer: str) -> float:
    """Score GAIA task: flexible matching on the answer."""
    if not prediction:
        return 0.0
    pred = prediction.strip().lower().replace(",", "").replace(" ", "")
    gold = gold_answer.strip().lower().replace(",", "").replace(" ", "")

    if pred == gold or gold in pred:
        return 1.0

    # Check if all parts of a multi-part answer are present
    gold_parts = gold_answer.strip().lower().split(",")
    if len(gold_parts) > 1:
        matches = sum(1 for part in gold_parts if part.strip() in prediction.lower())
        return matches / len(gold_parts)

    return 0.0
