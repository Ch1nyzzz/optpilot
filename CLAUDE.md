# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**OptPilot** is a research project building an **Experience-Driven Repair System for Multi-Agent Systems (MAS)**. It diagnoses MAS failures using the MAST taxonomy, generates targeted repair actions on the MAS's DAG structure, validates repairs, and distills successful fixes into a reusable Repair Library.

Core thesis: **diagnosis-driven targeted repair, not blind evolution**. Competitors (MAST+OpenEvolve) use taxonomy as a better fitness signal for evolutionary search; we use diagnostic results to drive precise, targeted fixes.

## Architecture

Five-module closed-loop:
- **Orchestrator**: coordinates the loop, prioritizes which FM to fix
- **Runner**: runs MAS, collects traces, manages DAG adapter (YAML read/write for ChatDev)
- **Diagnoser**: MAST-based FM classification, fine-grained localization to agent + step
- **Optimizer**: retrieves from Repair Library or generates new DAG repair actions via LLM
- **Distiller**: validates repairs, distills successful fixes into Repair Library

MAS-as-DAG abstraction: any MAS = DAG(Nodes, Edges). Repair actions = DAG operations (node mutation/add/delete, edge mutation/rewire).

## Target System

**ChatDev v2** on ProgramDev benchmark with GPT-4o.
- ChatDev is natively a YAML-driven DAG executor — the entire MAS is defined in `ChatDev_v1.yaml`
- MAST-Data has 130 ChatDev traces (93 with failures), top FM is Step Repetition (36.2%)
- Modifications only require YAML changes, no Python code changes needed

## Key Comparison Points

| Competitor | What they do | Our difference |
|---|---|---|
| MAST/AgentFail | Taxonomy + dataset, no repair | We close the loop with automated repair |
| AgentDebug | Inference-time re-rollout for single agent | We do design-time MAS optimization |
| MAST+OpenEvolve | Taxonomy as fitness for blind evolution | We do diagnosis → targeted repair |

## Current State

Research phase with concrete experiment design. Data (MAST-Data) analyzed, target system (ChatDev) source code analyzed. Ready for MVP implementation.

## Project Memory

Maintain persistent project memory under `memory_bank/`.

- `memory_bank/project_goal.md` records the project mission and current objectives.
- `memory_bank/progress.md` records current status and latest milestones.
- `memory_bank/architecture.md` records repository structure and architecture decisions.

Whenever the architecture, repository structure, or system decomposition changes, update
`memory_bank/architecture.md` in the same change. Keep `memory_bank/progress.md`
current as major milestones are completed.
