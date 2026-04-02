# Project Goal

## Mission

Build **OptPilot**, a system that investigates whether data-driven structural priors can improve evolutionary optimization of Multi-Agent Systems (MAS).

## Core Experiment

**Prior-guided OpenEvolve vs Blind OpenEvolve**

1. Run blind evolutionary search (SkyDiscover/OpenEvolve) on MAS DAGs
2. Extract structural priors from effective mutations (posterior-filtered)
3. Run blind and guided in parallel for fair comparison
4. Distill from guided results to iteratively improve priors

Key question: Do extracted priors improve sample efficiency and generalization?

## Scope Decision (2026-03-25)

**Target domain: Multi-Agent Systems (MAS)**

理由：
- MAS failure pattern 丰富（agent 间通信、角色分配、协调策略）
- 现有 taxonomy 数据可直接借鉴（MAST 1642 条 trace，AgentFail 307 条）
- 学术热度高

## Current Targets

| target_mas | Benchmark | Topology (auto-detected) |
|---|---|---|
| ag2 | MMLU + AIME 2025 + OlympiadBench | hub=0, loop=1 |
| agentcoder | HumanEval | hub=0, loop=1 |
| simple_star | GAIA | hub=1, loop=1 |
| simple_hier | SWE-bench Lite | hub=0, loop=1 |
| appworld | AppWorld | hub=1, loop=0 |
| hyperagent | SWE-bench Lite | hub=0, loop=1 |
| magentic | GAIA | hub=1, loop=1 |

## Model Stack

- MAS execution + FM diagnosis: MiniMax M2.5 via Together AI
- OpenEvolve experiments: `openai/gpt-oss-120b`
