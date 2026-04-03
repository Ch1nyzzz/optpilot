"""Case Study: Why Evolved DAGs Pass Shadow Gate but Degrade on Test

Analysis of 9 cases where OpenEvolve selected an evolved candidate over the
original DAG (shadow gate passed), yet test-set accuracy decreased.

Run: python -m experiments.case_study_shadow_gate_leakage
"""

import json
import glob
import os
import re
from collections import defaultdict
from pathlib import Path


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def load_experiments():
    """Load all OpenEvolve experiment results and classify them."""
    experiments = []
    for f in sorted(glob.glob(str(RESULTS_DIR / "*openevolve*.json"))):
        if "_reeval" in f:
            continue
        try:
            data = json.load(open(f))
        except Exception:
            continue

        shadow = data.get("shadow_selection", {})
        selected = shadow.get("selected_candidate", "")
        tb_acc = data.get("test_baseline", {}).get("overall_accuracy")
        tf_acc = data.get("test_final", {}).get("overall_accuracy")
        if tb_acc is None or tf_acc is None:
            continue

        evo = data.get("evolution", {}).get("best_metrics", {})

        experiments.append({
            "file": os.path.basename(f).replace(".json", ""),
            "data": data,
            "selected": selected,
            "delta": tf_acc - tb_acc,
            "train_acc": evo.get("accuracy", 0),
            "train_combined": evo.get("combined_score", 0),
            "shadow_base": shadow.get("baseline_shadow_accuracy", 0),
            "shadow_sel": shadow.get("selected_shadow_accuracy", 0),
            "test_base": tb_acc,
            "test_final": tf_acc,
            "num_agents": evo.get("num_agents", 0),
            "train_total_fm": evo.get("total_failures", 0),
            "test_total_fm": evo.get("test_total_failures", 0),
            "evo_metrics": evo,
        })

    return experiments


def count_agents_in_code(filepath):
    """Count agent nodes in a build_dag() Python file."""
    if not os.path.exists(filepath):
        return None
    code = open(filepath).read()
    return len(re.findall(r'"type":\s*"agent"', code))


def classify(experiments):
    """Split into degraded (evolved selected, test worse) vs improved."""
    degraded = [e for e in experiments if e["selected"] != "original_dag" and e["delta"] < -0.01]
    improved = [e for e in experiments if e["selected"] != "original_dag" and e["delta"] > 0.01]
    fallback = [e for e in experiments if e["selected"] == "original_dag"]
    return degraded, improved, fallback


def print_report(experiments):
    degraded, improved, fallback = classify(experiments)

    print("=" * 80)
    print("CASE STUDY: Shadow Gate Leakage in OpenEvolve Blind Cold-Start")
    print("=" * 80)

    # ─────────────────────────────────────────────────────────────
    # Section 1: Overview
    # ─────────────────────────────────────────────────────────────
    print(f"\n{'─' * 80}")
    print("1. OVERVIEW")
    print(f"{'─' * 80}")
    print(f"  Total experiments analyzed: {len(experiments)}")
    print(f"  Evolved selected & test improved:  {len(improved)}")
    print(f"  Evolved selected & test degraded:  {len(degraded)}")
    print(f"  Shadow gate fell back to original:  {len(fallback)}")
    print(f"  Degradation rate (of evolved selections): "
          f"{len(degraded)}/{len(degraded) + len(improved)} = "
          f"{len(degraded) / max(1, len(degraded) + len(improved)):.0%}")

    # ─────────────────────────────────────────────────────────────
    # Section 2: Aggregate Comparison
    # ─────────────────────────────────────────────────────────────
    print(f"\n{'─' * 80}")
    print("2. AGGREGATE COMPARISON: Improved vs Degraded")
    print(f"{'─' * 80}")

    for label, group in [("Improved", improved), ("Degraded", degraded)]:
        if not group:
            continue
        n = len(group)
        avg = lambda key: sum(e[key] for e in group) / n
        print(f"\n  {label} ({n} cases):")
        print(f"    Avg train accuracy (on 20 hard tasks):  {avg('train_acc'):.3f}")
        print(f"    Avg agent count:                        {avg('num_agents'):.1f}")
        print(f"    Avg shadow margin (sel - base):         "
              f"{sum(e['shadow_sel'] - e['shadow_base'] for e in group) / n:+.3f}")
        print(f"    Avg FM inflation (test - train):        "
              f"{sum(e['test_total_fm'] - e['train_total_fm'] for e in group) / n:+.1f}")
        print(f"    Avg test baseline accuracy:             {avg('test_base'):.3f}")

    # ─────────────────────────────────────────────────────────────
    # Section 3: Root Cause Analysis
    # ─────────────────────────────────────────────────────────────
    print(f"\n{'─' * 80}")
    print("3. ROOT CAUSE ANALYSIS")
    print(f"{'─' * 80}")

    print("""
  Finding 1: Shadow Gate Composition Bias
  ────────────────────────────────────────
  The shadow set is constructed from train tasks that the BASELINE already
  solves correctly. By design, evolution trains on the 20 hardest failures,
  and shadow = remaining ~80 train tasks (mostly easy).

  Consequence: shadow gate only tests "don't break easy tasks". It cannot
  detect degradation on UNSEEN hard tasks that differ from the 20 training
  hard tasks.

  Evidence:
    - Degraded cases: avg shadow margin = +0.022 (barely above baseline)
    - Improved cases: avg shadow margin = +0.224 (clearly above baseline)
    - When shadow margin ≈ 0, the evolved DAG neither helps nor hurts on
      easy tasks — but may be overfit to the specific 20 hard tasks.

  Finding 2: FM Inflation (Failure Mode Explosion on Test)
  ────────────────────────────────────────────────────────
  Evolved DAGs show significantly higher failure mode counts on test vs
  train. This is the smoking gun of overfitting: the DAG learns to suppress
  failures on the specific 20 training tasks but the fixes don't generalize.

  Evidence:
    - Degraded: avg FM inflation = +6.7 failures (train → test)
    - Improved: avg FM inflation = +2.4 failures
    - Worst case (linear_loop_gaia): +17 FM inflation, driven by
      Execution Loop (+25%), Communication Failure (+10%),
      Task Drift (+15%), Verification Failure (+30%)

  Most inflated FMs across degraded cases:
    - Verification Failure (Group F): appears in 7/9 degraded cases
    - Task Drift (Group E): appears in 5/9 degraded cases
    - Communication Failure (Group D): appears in 4/9 degraded cases

  Finding 3: Agent Proliferation Without Architectural Guards
  ────────────────────────────────────────────────────────────
  Evolution tends to add agents (Planner, Verifier, Formatter, Refiner)
  to address specific failures. But unlike MAST's "Locked Verifier" approach,
  our search space has no constraints preventing:
    - Added agents that increase latency and context dilution
    - Verification agents that are overly lenient (always pass)
    - Routing complexity that breaks on unfamiliar inputs

  Evidence:
    - linear_gaia: 2 → 6 agents (+Planner, Refiner, Formatter, Verifier)
      Result: test accuracy dropped from 16.6% to 9.2%
    - star_math: 4 → 6 agents (+Verifier, PlanValidator)
      Result: test accuracy dropped from 36.0% to 28.0%

  Finding 4: The "High Baseline" Trap
  ────────────────────────────────────
  Degradation is most severe when the baseline is already strong.

  Evidence:
    - Degraded avg test baseline: 0.452
    - Improved avg test baseline: 0.330

  When baseline accuracy is high (e.g., HumanEval 95.3%), there's little
  room to improve but much room to break. The 20 "hard" training tasks
  represent edge cases that may require very different strategies from the
  bulk of tasks. Optimizing for those edge cases can harm general
  performance — classic overfitting to the tail of the distribution.

  Worst example: linear_loop_humaneval evolved a Planner agent that helped
  on 20 hard HumanEval tasks but produced irrelevant plans for the simpler
  tasks, confusing the Solver (test: 95.3% → 79.7%, -15.6pp).
""")

    # ─────────────────────────────────────────────────────────────
    # Section 4: Detailed Case Studies
    # ─────────────────────────────────────────────────────────────
    print(f"\n{'─' * 80}")
    print("4. DETAILED CASE STUDIES")
    print(f"{'─' * 80}")

    for e in sorted(degraded, key=lambda x: x["delta"]):
        print(f"\n  {'━' * 70}")
        print(f"  {e['file']}")
        print(f"  Delta: {e['delta']:+.3f}  |  Selected: {e['selected']}")
        print(f"  {'━' * 70}")

        evo = e["evo_metrics"]
        print(f"    Train (20 hard): acc={e['train_acc']:.2f}  combined={e['train_combined']:.3f}  FMs={e['train_total_fm']}")
        print(f"    Shadow: base={e['shadow_base']:.3f}  selected={e['shadow_sel']:.3f}  margin={e['shadow_sel'] - e['shadow_base']:+.3f}")
        print(f"    Test: base={e['test_base']:.3f}  final={e['test_final']:.3f}  delta={e['delta']:+.3f}")

        # FM breakdown
        fm_names = {
            "fm_A": "Instruction Non-Compliance",
            "fm_B": "Execution Loop / Stuck",
            "fm_C": "Context Loss",
            "fm_D": "Communication Failure",
            "fm_E": "Task Drift",
            "fm_F": "Verification Failure",
        }
        print(f"\n    FM train→test inflation:")
        for fm_key, fm_name in fm_names.items():
            train_v = evo.get(fm_key, 0)
            test_v = evo.get(f"test_{fm_key}", 0)
            if test_v > train_v + 0.03:
                print(f"      {fm_name:35s} {train_v:5.0%} → {test_v:5.0%} (+{test_v - train_v:.0%})")

        # Task-level flips
        tb_details = e["data"].get("test_baseline", {}).get("details", [])
        tf_details = e["data"].get("test_final", {}).get("details", [])
        if tb_details and tf_details and len(tb_details) == len(tf_details):
            gained = sum(1 for b, f in zip(tb_details, tf_details) if b["score"] <= 0 and f["score"] > 0)
            lost = sum(1 for b, f in zip(tb_details, tf_details) if b["score"] > 0 and f["score"] <= 0)
            print(f"\n    Task flips: gained={gained}, lost={lost}, net={gained - lost}")

        # Structural changes
        artifacts_dir = RESULTS_DIR / f"{e['file']}_artifacts"
        evolved_path = artifacts_dir / "dag_versions" / "shadow_selected.py"
        if evolved_path.exists():
            code = evolved_path.read_text()
            agents = re.findall(r'"id":\s*"(Agent_[^"]+)"', code)
            has_loop = "loop_counter" in code.lower() or "LoopCounter" in code
            print(f"\n    Evolved agents ({len(agents)}): {', '.join(agents)}")
            print(f"    Has loop: {has_loop}")

    # ─────────────────────────────────────────────────────────────
    # Section 5: Recommendations
    # ─────────────────────────────────────────────────────────────
    print(f"\n{'─' * 80}")
    print("5. RECOMMENDATIONS")
    print(f"{'─' * 80}")
    print("""
  R1. Stratified Shadow Gate
  ──────────────────────────
  Current shadow set is biased toward easy tasks. Instead:
    - Include a mix of baseline-correct AND baseline-incorrect tasks
    - Or use a held-out set that mirrors the difficulty distribution of
      the full test set (stratified by benchmark difficulty tier)
    - Shadow gate should require IMPROVEMENT on hard tasks, not just
      non-regression on easy tasks

  R2. FM Inflation Threshold
  ──────────────────────────
  Add a shadow gate check that compares FM rates on shadow set between
  evolved and baseline. If total_failures increases by >X% on shadow,
  reject the candidate even if accuracy is maintained. This catches
  "fragile" candidates that happen to get right answers by luck on shadow
  but have worse internal behavior.

  R3. Architectural Guardrails (cf. MAST "Locked Verifier")
  ──────────────────────────────────────────────────────────
  Constrain the search space:
    - Cap maximum agent count (e.g., initial_agents + 2)
    - Require core nodes (e.g., Solver/Verifier) to be preserved
    - Penalize routing complexity (number of conditional edges)
  This prevents the "agent proliferation" pattern that inflates context
  and increases failure surface area.

  R4. Multi-Run Shadow Gate
  ─────────────────────────
  LLM outputs are stochastic. A single shadow evaluation is noisy.
  Run shadow evaluation 2-3 times and use the MINIMUM accuracy as the
  gate signal. This reduces false-positive shadow passes caused by
  lucky runs.

  R5. Difficulty-Aware Training Set
  ──────────────────────────────────
  Currently evolving on the 20 hardest tasks. This creates strong
  selection pressure for edge-case strategies. Consider:
    - Using a mixed set (e.g., 10 hard + 10 medium) for evolution
    - Or running periodic "regression checks" on easy tasks during
      evolution itself (not just at the end via shadow gate)

  R6. Reeval-Based Selection (Already Partially Implemented)
  ──────────────────────────────────────────────────────────
  The reeval data confirms high variance in test results:
    - linear_loop_humaneval_022625: baseline runs = [0.875, 0.906, 0.953]
    - Same case evolved runs = [0.797, 0.703, 0.812]
  3-run reeval helps but the shadow gate decision is still made on a
  single run. Move multi-run evaluation BEFORE the shadow gate decision.
""")


def main():
    experiments = load_experiments()
    print_report(experiments)


if __name__ == "__main__":
    main()
