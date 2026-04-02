"""HotpotQA benchmark loader and exact-match scorer.

Loads multi-hop QA problems from hotpot_qa/distractor on HuggingFace.
Each example includes supporting context paragraphs so agents can reason
over provided passages without needing external retrieval.
"""

from __future__ import annotations

import re
import string
from collections import Counter

from optpilot.data.benchmarks import BenchmarkExample


def load_hotpotqa_examples(limit: int = 200) -> list[BenchmarkExample]:
    """Load HotpotQA distractor-setting problems as BenchmarkExamples.

    Uses the validation split (the test split has no gold answers).
    Each example provides context paragraphs + question; agents must
    produce a short answer.
    """
    from datasets import load_dataset

    ds = load_dataset("hotpot_qa", "distractor", split="validation")
    examples: list[BenchmarkExample] = []
    for item in ds:
        if len(examples) >= limit:
            break

        # Build context from the provided paragraphs
        context_parts: list[str] = []
        for title, sentences in zip(item["context"]["title"], item["context"]["sentences"]):
            paragraph = "".join(sentences)
            context_parts.append(f"[{title}]\n{paragraph}")
        context_block = "\n\n".join(context_parts)

        task_prompt = (
            "Answer the following question based on the provided context. "
            "Give ONLY the answer, no explanation.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {item['question']}\n\n"
            "Answer:"
        )

        examples.append(BenchmarkExample(
            benchmark_name="HotpotQA",
            task_id=item["id"],
            prompt=task_prompt,
            gold_answers=(item["answer"],),
            answer_type="short_answer",
            metadata={
                "question": item["question"],
                "type": item["type"],
                "level": item["level"],
                "supporting_facts_title": list(item["supporting_facts"]["title"]),
            },
        ))
    return examples


def _normalize_answer(s: str) -> str:
    """Normalize answer string for comparison (from HotpotQA official eval)."""
    s = s.lower()
    # Remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # Remove punctuation
    s = s.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    s = " ".join(s.split())
    return s


def _f1_score(prediction: str, gold: str) -> float:
    """Token-level F1 between prediction and gold answer."""
    pred_tokens = _normalize_answer(prediction).split()
    gold_tokens = _normalize_answer(gold).split()
    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def score_hotpotqa(prediction: str, gold_answer: str) -> float:
    """Score a HotpotQA prediction.

    Returns 1.0 for exact match, otherwise token-level F1.
    This matches the official HotpotQA evaluation protocol.
    """
    if not prediction:
        return 0.0
    # Extract just the answer if the agent included extra text
    # Take the first line as the answer
    answer_line = prediction.strip().split("\n")[0].strip()
    if _normalize_answer(answer_line) == _normalize_answer(gold_answer):
        return 1.0
    return _f1_score(answer_line, gold_answer)
