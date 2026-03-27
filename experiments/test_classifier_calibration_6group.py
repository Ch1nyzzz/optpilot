"""6-Group FM Classifier Calibration Test (v2 — CoT + Disambiguation + Few-shot).

Evaluate the simplified 6-group taxonomy classification on AG2 traces.
Ground truth: normalized 6-group annotations from the loader.

Usage:
    python -u -m experiments.test_classifier_calibration_6group
    python -u -m experiments.test_classifier_calibration_6group --n-samples 60 --workers 64
    python -u -m experiments.test_classifier_calibration_6group --model "MiniMaxAI/MiniMax-M2.5"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optpilot.config import RESULTS_DIR
from optpilot.data.fm_taxonomy_6group import (
    GROUP_DEFINITIONS, GROUP_IDS, GROUP_NAMES,
)
from optpilot.data.loader import load_traces
from optpilot.llm import acall_llm_json
from optpilot.modules.diagnoser import _prepare_trace_content

MAS_NAME = "AG2"

# ---------------------------------------------------------------------------
# v2.0 Prompt: CoT + Disambiguation + Few-shot
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert in diagnosing failures in Multi-Agent LLM Systems (MAS).
You will analyze an execution trace and classify failures into 6 groups.

## ANALYSIS METHODOLOGY

Read the trace in three passes:
1. **Scan for outcome**: Did the system complete the task? What was the final output?
2. **Identify turning points**: Where did things start going wrong? Look for:
   - Repeated messages or actions (→ possible Group B)
   - Agent ignoring or misunderstanding instructions (→ possible Group A)
   - Information that one agent had but didn't share (→ possible Group D)
   - Agent doing something different from what it said it would do (→ possible Group E)
   - Missing or superficial review steps (→ possible Group F)
   - Agent "forgetting" earlier context (→ possible Group C)
3. **Root cause attribution**: For each turning point, ask "WHY did this happen?" and classify by the root cause, not the surface symptom.

## FAILURE GROUP DEFINITIONS

### Group A — Instruction Non-Compliance
The agent's output directly violates explicit constraints or role boundaries stated in the system/task prompt.

**Positive indicators (classify as A):**
- Task says "implement using React" but agent uses Vue
- Task says "do not use placeholder code" but agent writes `# TODO: implement`
- A "Reviewer" agent starts writing code instead of reviewing
- Agent produces output that ignores a specific stated requirement

**Negative indicators (do NOT classify as A):**
- Agent produces wrong code that happens to not work — this is capability failure, not disobedience
- Agent misunderstands an ambiguous requirement — only count if the requirement was clearly stated
- Agent's output is low quality but doesn't violate any explicit constraint

**Decision rule**: Would a human reviewer say "the instruction clearly said X, but the agent did Y"? If yes → A.

---

### Group B — Execution Loop / Stuck
The system enters a repetitive cycle or fails to recognize that execution should stop.

**Positive indicators (classify as B):**
- Same reasoning/action appears 3+ times with no new information
- Agent says "let me try again" and repeats identical steps
- Task is already solved but agent keeps going ("Continue", "Let me check further...")
- System hits max iterations without convergence

**Negative indicators (do NOT classify as B):**
- Agent retries with a DIFFERENT approach after failure — this is recovery, not looping
- Agent repeats a step once due to context loss — classify as C instead
- Agent explores multiple solutions sequentially — not a loop if each attempt is distinct

**Decision rule**: Remove the repeated segment — would anything be lost? If no → B.

---

### Group C — Context Loss
Previously established information is lost, forgotten, or the conversation state resets.

**Positive indicators (classify as C):**
- Agent re-introduces a problem that was already resolved earlier in the trace
- Agent asks a question that was already answered
- Conversation seems to "restart" — agent loses awareness of prior progress
- Agent contradicts its own earlier statements due to forgotten context

**Negative indicators (do NOT classify as C):**
- Agent deliberately re-examines a previous decision — this is review, not context loss
- Agent works on a new sub-task — switching topics ≠ forgetting
- Information was never clearly established — can't lose what you never had

**Decision rule**: Was the information clearly present earlier in the trace AND is the agent now acting as if it never existed? If yes → C.

---

### Group D — Communication Failure
Critical information fails to flow between agents due to withholding, ignoring, or not seeking clarification.

**Positive indicators (classify as D):**
- Agent A has information that Agent B needs, but doesn't share it
- Agent receives a question/request but ignores it in its response
- Agent faces ambiguous input and proceeds with assumptions instead of asking
- Agent B's recommendation is present in the trace but Agent A acts as if it doesn't exist

**Negative indicators (do NOT classify as D):**
- Single-agent system (no inter-agent communication to fail)
- Agent misunderstands information that WAS communicated — this may be E (reasoning error)
- Agent lacks information that no other agent has — nobody can share what doesn't exist

**Decision rule**: Was there a point where Agent X could have communicated information to Agent Y, and failure to do so caused downstream problems? If yes → D.

---

### Group E — Task Drift / Reasoning Error
The agent's execution trajectory deviates from what correct execution would look like, or its actions don't match its stated reasoning.

**Positive indicators (classify as E):**
- Agent says "I will examine the source code" but instead writes new code from scratch
- Agent starts solving a different problem than what was asked
- Agent's chain of reasoning is correct but its final action contradicts the reasoning
- Agent progressively drifts from the original objective over multiple turns
- Agent creates a simplified/toy version instead of working with the actual codebase

**Negative indicators (do NOT classify as E):**
- Agent follows a correct but suboptimal approach — not drift if it's still on-task
- Agent makes a factual/coding error while staying on-task — capability limitation, not drift
- Agent loops on the same step — classify as B instead

**Decision rule**: Compare the agent's ACTUAL execution path against the INTENDED task objective. Is there a clear divergence in direction? If yes → E.

---

### Group F — Verification Failure
The system's quality control process is absent, premature, superficial, or reaches wrong conclusions.

**Positive indicators (classify as F):**
- Agent declares "task complete" without running tests or checking output
- Agent runs tests, they fail, but agent says "looks good"
- Agent abandons the task after first error without attempting to fix
- Code compiles but has obvious runtime bugs that a basic test would catch
- Agent reviews code but misses critical logical errors

**Negative indicators (do NOT classify as F):**
- Agent tries to verify but the verification tool/environment fails — infrastructure issue
- No verification was expected for the task type (e.g., simple Q&A)
- Agent correctly identifies that the task is impossible and stops — this is correct termination

**Decision rule**: Was there an opportunity for verification that was missed, skipped, or done incorrectly? If yes → F.

---

## DISAMBIGUATION RULES (use these when multiple groups seem applicable)

| Situation | Looks like... | Actually is... | Why |
|---|---|---|---|
| Agent repeats same step AND drifts from goal | B + E | **B** (primary), E only if drift happens BEFORE the loop | The loop is the dominant failure; drift may be a consequence |
| Agent ignores instruction AND produces wrong output | A + E | **A** if instruction was explicit; **E** if instruction was vague | A requires clear violation of stated constraint |
| Agent forgets context AND starts looping | C + B | **C** (primary) if context loss caused the loop | Root cause is context loss; loop is the symptom |
| Agent doesn't verify AND task drifts | F + E | Mark **both** — these are independent failures | Drift and verification failure have different repair paths |
| Agent doesn't communicate AND task fails | D + F | Mark **both** if both are independently present | Communication and verification are orthogonal |
| Agent says "done" too early | B or F? | **F** (premature termination) | B is about NOT stopping; F is about stopping TOO EARLY |

## IMPORTANT NOTES
- Multiple groups can be TRUE simultaneously — most failed traces have 2-3 failure groups
- Focus on ROOT CAUSE, not surface symptoms
- A trace with no failures should return all false (but this is rare in failure datasets)
- When in doubt between two groups, check the disambiguation table above

## FEW-SHOT EXAMPLES

### Example 1 (AG2 MathChat):

Student: Please solve: What is the sum of 1+1?
Assistant: I'll solve this step by step.
```python
result = 1 + 1
print(result)
```
Output: 2
The answer is 2.

Student: TERMINATE
Assistant: Let me verify this further. I'll try another approach.
```python
a = 1
b = 1
print(a + b)
```
Output: 2
Yes, confirmed the answer is 2.

Student: TERMINATE
Assistant: Let me also check using a different method...

**Analysis:**
1. Outcome: The correct answer (2) was found immediately, but the system didn't stop.
2. Turning points: After the Student said "TERMINATE" twice, the Assistant continued with redundant verification.
3. Root cause: The Assistant is unaware of or ignoring termination conditions. This is a loop/stuck issue (B), not a reasoning error.

**Final Answer:** {"A": false, "B": true, "C": false, "D": false, "E": false, "F": false}

### Example 2 (ChatDev):

CEO: Create a snake game with score display and speed increase every 5 points.
CPO: I'll design the game with basic snake movement.
Programmer: Here's the implementation:
```python
# snake_game.py
import pygame
# ... [full implementation with snake movement, food, collision]
# Note: score display and speed increase not implemented yet
# TODO: add score feature
```
Reviewer: The code looks clean and well-structured. Approved!

**Analysis:**
1. Outcome: The snake game was created but missing two explicit requirements: score display and speed increase.
2. Turning points: Programmer left TODO for score feature → violates explicit task constraint (A). Reviewer approved code without checking requirements → verification failure (F).
3. Root cause: Two independent failures.

**Final Answer:** {"A": true, "B": false, "C": false, "D": false, "E": false, "F": true}

### Example 3 (HyperAgent):

Planner: We need to fix the Line3D._verts3d attribute error. Let me examine the source.
Navigator: I found the issue in art3d.py line 203. The set_3d_properties method needs to initialize _verts3d.
Planner: Let me examine the source code to understand the issue.
Navigator: As I mentioned, the issue is in art3d.py. Here's the relevant code: [shows code]
Planner: I think we should look at how Line3D works. Let me search for it.
Navigator: I already showed you this. The fix should be in set_3d_properties.
Planner: Let me start by searching for the Line3D class in the codebase...

**Analysis:**
1. Outcome: Task not completed — system kept restarting the investigation.
2. Turning points: Navigator provided the answer, but Planner repeatedly ignored it and restarted investigation. Same search-examine cycle repeated 3+ times.
3. Root cause: Planner ignored Navigator's input → D. Same steps repeated → B. Planner lost awareness of earlier findings → C.

**Final Answer:** {"A": false, "B": true, "C": true, "D": true, "E": false, "F": false}
"""

USER_PROMPT_TEMPLATE = """\
## Trace to Analyze

{trace_content}

## Your Task

Analyze this trace step by step:

1. **Outcome assessment**: Did the system achieve the task goal? What went wrong at a high level?

2. **Turning point identification**: Identify the key moments where failures occurred. For each, quote the relevant trace segment briefly.

3. **Root cause classification**: For each turning point, determine which failure group(s) apply using the definitions and disambiguation rules.

4. **Final classification**: Output your final answer.

Think through your analysis carefully before giving the final JSON.

### Analysis:
[Your step-by-step reasoning here]

### Final Answer:
```json
{{
    "A": <true/false>,
    "B": <true/false>,
    "C": <true/false>,
    "D": <true/false>,
    "E": <true/false>,
    "F": <true/false>
}}
```"""


# ---------------------------------------------------------------------------
# Classification function
# ---------------------------------------------------------------------------

async def classify_6group(trajectory: str, model: str | None = None) -> dict[str, bool]:
    """Classify a trace into 6 failure groups via async CoT LLM call."""
    trace_content = _prepare_trace_content(trajectory)
    user_prompt = USER_PROMPT_TEMPLATE.format(trace_content=trace_content)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    kwargs = {}
    if model:
        kwargs["model"] = model
    result = await acall_llm_json(
        messages,
        max_tokens=8192,
        **kwargs,
    )
    return {gid: bool(result.get(gid, False)) for gid in GROUP_IDS}


# ---------------------------------------------------------------------------
# Metrics & sampling
# ---------------------------------------------------------------------------

def compute_metrics(gt: list[int], pred: list[int]) -> dict:
    tp = sum(g == 1 and p == 1 for g, p in zip(gt, pred))
    fp = sum(g == 0 and p == 1 for g, p in zip(gt, pred))
    fn = sum(g == 1 and p == 0 for g, p in zip(gt, pred))
    tn = sum(g == 0 and p == 0 for g, p in zip(gt, pred))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(gt) if gt else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "accuracy": round(accuracy, 3),
        "support_pos": tp + fn,
        "support_neg": fp + tn,
    }


def sample_traces(n_samples: int = 60, seed: int = 42) -> list:
    rng = random.Random(seed)
    samples = []
    for bench in ["MMLU", "Olympiad"]:
        all_traces = load_traces(MAS_NAME, benchmark=bench)
        n = min(n_samples // 2, len(all_traces))
        samples.extend(rng.sample(all_traces, n))
        print(f"  {bench}: sampled {n}/{len(all_traces)} traces")
    rng.shuffle(samples)
    return samples


# ---------------------------------------------------------------------------
# Main calibration
# ---------------------------------------------------------------------------

async def run_calibration_async(
    n_samples: int = 60,
    max_workers: int = 16,
    seed: int = 42,
    model: str | None = None,
):
    model_display = model or "default (SYSTEM_MODEL)"
    print(f"=== 6-Group FM Classifier Calibration (v2 CoT+Few-shot, async) ===")
    print(f"Model: {model_display}")
    print(f"Samples: {n_samples}, Concurrency: {max_workers}")
    print()

    traces = sample_traces(n_samples=n_samples, seed=seed)
    print(f"\nTotal: {len(traces)} traces")

    # Ground truth: loader already normalizes dataset annotations to 6 groups
    gt_groups = [dict(t.mast_annotation) for t in traces]

    print("\nGround truth group distribution:")
    for gid in GROUP_IDS:
        n_pos = sum(g[gid] for g in gt_groups)
        print(f"  Group {gid} ({GROUP_NAMES[gid]}): {n_pos}/{len(traces)} ({100*n_pos/len(traces):.1f}%)")
    print()

    # Classify all traces with async semaphore
    print(f"Classifying {len(traces)} traces...")
    t0 = time.time()

    pred_groups: dict[int, dict[str, bool]] = {}
    semaphore = asyncio.Semaphore(max_workers)
    done_count = 0

    async def classify_one(idx: int) -> None:
        nonlocal done_count
        async with semaphore:
            try:
                pred_groups[idx] = await classify_6group(traces[idx].trajectory, model=model)
            except Exception as e:
                print(f"    Error trace {idx}: {e}")
                pred_groups[idx] = {}
            done_count += 1
            if done_count % 10 == 0:
                print(f"  ... {done_count}/{len(traces)} done")

    await asyncio.gather(*(classify_one(i) for i in range(len(traces))))

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s ({elapsed/len(traces):.1f}s/trace)")
    print()

    # Compute per-group metrics
    print(f"{'='*70}")
    print(f"{'Group':<8} {'Name':<30} {'P':>6} {'R':>6} {'F1':>6} {'Acc':>6} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}")
    print(f"{'-'*70}")

    per_group = {}
    for gid in GROUP_IDS:
        gt = [gt_groups[i][gid] for i in range(len(traces))]
        pred = [1 if pred_groups.get(i, {}).get(gid, False) else 0 for i in range(len(traces))]
        m = compute_metrics(gt, pred)
        per_group[gid] = m
        print(
            f"  {gid:<6} {GROUP_NAMES[gid]:<30} "
            f"{m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f} {m['accuracy']:>6.3f} "
            f"{m['tp']:>4} {m['fp']:>4} {m['fn']:>4} {m['tn']:>4}"
        )

    # Macro averages
    groups_with_support = [g for g in GROUP_IDS if per_group[g]["support_pos"] > 0]
    macro_p = sum(per_group[g]["precision"] for g in groups_with_support) / len(groups_with_support) if groups_with_support else 0
    macro_r = sum(per_group[g]["recall"] for g in groups_with_support) / len(groups_with_support) if groups_with_support else 0
    macro_f1 = sum(per_group[g]["f1"] for g in groups_with_support) / len(groups_with_support) if groups_with_support else 0
    print(f"{'-'*70}")
    print(f"  {'MACRO':<6} {'':30} {macro_p:>6.3f} {macro_r:>6.3f} {macro_f1:>6.3f}")
    print()

    # Per-trace detail: misclassification analysis
    print("Misclassification details (first 15):")
    errors = []
    for i in range(len(traces)):
        for gid in GROUP_IDS:
            gt_val = gt_groups[i][gid]
            pred_val = 1 if pred_groups.get(i, {}).get(gid, False) else 0
            if gt_val != pred_val:
                label = "FP" if pred_val else "FN"
                errors.append((label, gid, traces[i].trace_id, traces[i].benchmark_name))
    for label, gid, tid, bench in errors[:15]:
        print(f"  [{label}] Group {gid} — trace_{tid} ({bench})")
    if len(errors) > 15:
        print(f"  ... and {len(errors) - 15} more")
    print()

    # Save results
    results = {
        "prompt_version": "v2.0_cot_fewshot",
        "model": model or "default",
        "n_samples": len(traces),
        "seed": seed,
        "elapsed_s": round(elapsed, 1),
        "per_group": per_group,
        "macro": {"precision": round(macro_p, 3), "recall": round(macro_r, 3), "f1": round(macro_f1, 3)},
        "per_trace": [
            {
                "trace_id": traces[i].trace_id,
                "benchmark": traces[i].benchmark_name,
                "gt_groups": gt_groups[i],
                "pred_groups": {gid: pred_groups.get(i, {}).get(gid, False) for gid in GROUP_IDS},
            }
            for i in range(len(traces))
        ],
    }
    out_path = RESULTS_DIR / "classifier_calibration_6group_v2.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="6-Group FM Classifier Calibration (v2)")
    parser.add_argument("--n-samples", type=int, default=60, help="Total samples (split across benchmarks, default: 60)")
    parser.add_argument("--workers", type=int, default=16, help="Max concurrency (default: 16)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--model", default=None, help="Override LLM model (default: SYSTEM_MODEL / MiniMax-M2.5)")
    args = parser.parse_args()

    asyncio.run(run_calibration_async(
        n_samples=args.n_samples,
        max_workers=args.workers,
        seed=args.seed,
        model=args.model,
    ))
