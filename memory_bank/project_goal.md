# Project Goal

## Mission

Build **OptPilot**, a diagnosis-driven repair system for Multi-Agent Systems (MAS) that:
1. Diagnoses MAS failures using 6-group FM taxonomy (A-F)
2. Generates targeted YAML-level repairs via Skill Workflows
3. Validates repairs against ground-truth benchmarks
4. Accumulates experience (negatives + meta-evolution) across runs

## Core Thesis

**Diagnosis → targeted repair**, not blind evolution.

Competitors use taxonomy as a better fitness signal for evolutionary search (MAST+OpenEvolve).
We use diagnostic results (which FM group, which agent, which step, what root cause) to drive precise, targeted fixes.

## Scope Decision (2026-03-25)

**Target domain: Multi-Agent Systems (MAS)**

理由：
- MAS failure pattern 丰富（agent 间通信、角色分配、协调策略）
- 现有 taxonomy 数据可直接借鉴（MAST 1642 条 trace，AgentFail 307 条）
- 学术热度高

## 核心差异化定位

### vs MAST / AgentFail
- 它们只做诊断分析（"哪里坏了"），不做修复
- 我们自动修复 + 蒸馏经验

### vs AgentDebug
- inference-time re-rollout（单 agent）
- 我们做 design-time MAS 架构优化

### vs MAST + OpenEvolve
- 用 MAST taxonomy 作为 fitness function 驱动 OpenEvolve 盲进化
- 我们用诊断信号做定向修复——不是更好的 reward，而是更好的 search

## Current Target

- **MAS**: AG2 MathChat (3-agent GroupChat)
- **Benchmarks**: MMLU + AIME 2025 + OlympiadBench (ground-truth scoring)
- **Pipeline**: Skill Workflows (analyze → evolve → validate → reflect)
- **Model**: MiniMax M2.5 via Together AI
