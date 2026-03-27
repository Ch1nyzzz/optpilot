# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**OptPilot** is a research project building an **Experience-Driven Repair System for Multi-Agent Systems (MAS)**. It diagnoses MAS failures using the 6-group FM taxonomy (A-F), generates targeted YAML-level repair actions via Skill Workflows, validates repairs against ground-truth benchmarks, and accumulates experience (negatives + meta-evolution) across runs.

Core thesis: **diagnosis-driven targeted repair, not blind evolution**. Competitors (MAST+OpenEvolve) use taxonomy as a better fitness signal for evolutionary search; we use diagnostic results to drive precise, targeted fixes.

## Architecture

**Skill Workflow closed-loop** (replaced the original Optimizer+Distiller+WrapUp):

- **Orchestrator** (`orchestrator.py`): runs MAS → diagnoses → ranks FM groups → dispatches Skill Workflows (parallel)
- **Runner** (`modules/runner.py`): executes MASDAG via built-in DAGExecutor, collects traces with ground-truth scoring
- **Diagnoser** (`modules/diagnoser.py`): 6-group FM classification (MiniMax M2.5) + agent/step localization
- **Skill Workflows** (`skills/`): 6 Python classes (A-F), each a complete repair agent: analyze → evolve (inner loop, YAML-level) → validate → reflect (outer loop)
- **SkillEvolver** (`skills/evolution.py`): when a Skill fails ≥3 times, LLM modifies Skill's own Python source

MAS-as-DAG abstraction: any MAS = MASDAG(Nodes, Edges).
- Node types: agent (LLM call), literal (fixed text), loop_counter (iteration control), passthrough
- Edge attributes: trigger, condition (keyword matching), carry_data, loop (continue/exit)
- Repair = YAML-level DAG modification by LLM

## Target System

**AG2 MathChat** (3-agent GroupChat) on official benchmarks.
- DAG: `dags/ag2_mathchat.yaml` — Agent_Problem_Solver + Agent_Code_Executor + Agent_Verifier
- Benchmarks: MMLU + AIME 2025 + OlympiadBench (ground-truth scoring)
- Model: MiniMax M2.5 via Together AI (unified for execution + diagnosis)
- Entry point: `python -m experiments.run_ag2_mathchat_skill --tasks 9 --rounds 3`

## DAG Executor Conventions

- Agent system prompts: executor reads `node.prompt` first, falls back to `node.role`
- Literal content: executor reads `config.content` first, falls back to `node.prompt`
- Agent params: supports both flat `config.temperature` and nested `config.params.temperature`
- Loop counter edges: use explicit `loop: exit` or `loop: continue` annotations; topology inference as fallback

## Key Comparison Points

| Competitor | What they do | Our difference |
|---|---|---|
| MAST/AgentFail | Taxonomy + dataset, no repair | We close the loop with automated repair |
| AgentDebug | Inference-time re-rollout for single agent | We do design-time MAS optimization |
| MAST+OpenEvolve | Taxonomy as fitness for blind evolution | We do diagnosis → targeted repair |

## Project Memory

Maintain persistent project memory under `memory_bank/`.

- `memory_bank/project_goal.md` records the project mission and current objectives.
- `memory_bank/progress.md` records current status and latest milestones.
- `memory_bank/architecture.md` records repository structure and architecture decisions.

Whenever the architecture, repository structure, or system decomposition changes, update
`memory_bank/architecture.md` in the same change. Keep `memory_bank/progress.md`
current as major milestones are completed.
