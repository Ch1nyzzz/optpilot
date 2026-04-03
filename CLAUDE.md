# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**OptPilot** is a research project investigating whether **data-driven priors** extracted from evolutionary search traces can improve the efficiency and quality of Multi-Agent System (MAS) optimization.

Core experiment: **prior-guided OpenEvolve vs blind OpenEvolve**. We run blind evolutionary search (via SkyDiscover/OpenEvolve) on MAS DAGs, extract structural priors from effective mutations, then compare guided search (with priors) against blind search (without priors) on the same tasks.

## Architecture

**Two-phase pipeline:**

1. **Blind Cold-Start** (`run_openevolve.py`): SkyDiscover runs MAP-Elites evolutionary search on a target MAS DAG. No priors, no diagnosis — pure blind mutation.
2. **Prior Extraction** (`analyze_openevolve_traces.py`): Analyzes effective mutations from phase 1 — posterior-filters on held-out test set, classifies structural changes, outputs data-driven priors (Jacobian warmup, recipes, pattern catalog updates).

**Core modules:**

- **DAG Abstraction** (`dag/core.py`, `dag/executor.py`): MASDAG representation + lightweight BFS executor
- **Runner** (`modules/runner.py`): Executes MASDAG on benchmark tasks, collects traces with ground-truth scoring
- **Diagnoser** (`modules/diagnoser.py`): 6-group FM classification (A-F) via LLM
- **Evaluator** (`openevolve_evaluator_multi.py`): Fitness evaluator for SkyDiscover, supports multiple target MAS
- **Prior Store** (`skills/jacobian.py`, `skills/recipes.py`, `skills/repair_patterns.py`): Accumulated structural priors

MAS-as-DAG abstraction: any MAS = MASDAG(Nodes, Edges).
- Node types: agent (LLM call), literal (fixed text), loop_counter (iteration control), passthrough
- Edge attributes: trigger, condition (keyword matching), carry_data, loop (continue/exit)
- Optimization = YAML-level DAG modification by evolutionary search

## Experience Storage

Experience is stored **globally** in `library_store/`. Topology differentiation is automatic:

- `MASDAG.extract_topology_features()` detects `has_hub` from DAG structure
- `FailureSignature.signature_key()` embeds this feature: e.g. `"B:hub=0"`
- Jacobian matrix rows naturally separate hub vs non-hub topology families
- No hardcoded topology mapping — new target MAS automatically finds matching experience
- Loop presence is not tracked: evolution adds loops autonomously regardless of initial topology

Storage layout:
- `library_store/jacobian/` — Repair effectiveness matrix + data-driven priors
- `library_store/recipes/` — Distilled repair recipes (per FM group)
- `library_store/negatives/` — Lessons from failed repairs
- `library_store/pattern_catalog.json` — Evolved pattern catalog

## Target Systems

| target_mas | Benchmark | Topology |
|---|---|---|
| ag2 | MMLU + AIME 2025 + OlympiadBench | no-hub (linear) |
| agentcoder | HumanEval | no-hub (linear) |
| simple_star | GAIA | hub (star) |
| simple_hier | SWE-bench Lite | no-hub (linear) |
| appworld | AppWorld | hub (star) |
| hyperagent | SWE-bench Lite | no-hub (linear) |
| magentic | GAIA | hub (star) |

- Model: MiniMax M2.5 via Together AI (unified for execution + diagnosis)
- Initial DAG builders: `experiments/openevolve_initial_dag_*.py`

## DAG Executor Conventions

- Agent system prompts: executor reads `node.prompt` first, falls back to `node.role`
- Literal content: executor reads `config.content` first, falls back to `node.prompt`
- Agent params: supports both flat `config.temperature` and nested `config.params.temperature`
- Loop counter edges: use explicit `loop: exit` or `loop: continue` annotations; topology inference as fallback

## Key Entry Points

- `python -m experiments.run_openevolve --target-mas agentcoder --iterations 50` — Blind cold-start
- `python -m experiments.run_openevolve --target-mas agentcoder --iterations 50 --with-priors` — Prior-guided
- `python -m experiments.analyze_openevolve_traces --openevolve-dir <dir> --target-mas agentcoder` — Extract priors
- `python -m experiments.run_ag2_mathchat_baseline --tasks 20` — Baseline measurement

## Project Memory

Maintain persistent project memory under `memory_bank/`.

- `memory_bank/project_goal.md` records the project mission and current objectives.
- `memory_bank/progress.md` records current status and latest milestones.
- `memory_bank/architecture.md` records repository structure and architecture decisions.

Whenever the architecture, repository structure, or system decomposition changes, update
`memory_bank/architecture.md` in the same change. Keep `memory_bank/progress.md`
current as major milestones are completed.
