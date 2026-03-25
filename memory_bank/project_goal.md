# Project Goal

## Mission

Build **OptPilot**, a prior-guided, taxonomy-driven meta-optimizer that can improve
different classes of optimizers by combining prior knowledge, structured failure
diagnosis, and targeted repair.

## Core Objectives

1. Define a unified interface for wrapping different optimizers.
2. Build a reusable prior library that can warm-start optimization.
3. Design a two-layer error taxonomy that maps failures to repair policies.
4. Close the loop from diagnosis to action to re-evaluation.
5. Validate cross-optimizer and cross-domain knowledge transfer.

## Scope Decision (2026-03-25)

**Target domain: Multi-Agent Systems (MAS)**，不再追求泛化到所有 optimizer 类型。

理由：
- MAS failure pattern 丰富（agent 间通信、角色分配、协调策略）
- 现有 taxonomy 数据可直接借鉴（MAST 1642 条 trace，AgentFail 307 条）
- 学术热度高

Target MAS 框架候选：AutoGen, CrewAI, ChatDev, MetaGPT

## 核心差异化定位 (2026-03-25)

### vs MAST / AgentFail
- 它们只做诊断分析（"哪里坏了"），不做修复
- MAST 的 case study 手动改 ChatDev 得到 +9.4% / +15.6%，但没有自动化

### vs AgentDebug
- AgentDebug 做 inference-time debugging：发现错误 → feedback → re-rollout，但 agent 本身没变
- 只针对单 agent，不涉及 MAS 协调问题
- 类比："帮你重考一次" vs 我们的 "改教材、改课程设计"

### vs MAST + OpenEvolve (UC Berkeley ADRS blog, 2026-03-25 发现)
**这是最近的竞品**。他们用 MAST taxonomy 作为 fitness function 驱动 OpenEvolve 进化 MAS：
- 实验：OlympiadBench 数学推理，Binary feedback 25% vs MAST feedback **33%**
- 3 agent 进化到 7 agent，FM-1.1 减少 62%
- 核心发现："Accuracy 进化产生专家，MAST 进化产生架构师"

**他们的局限（= 我们的空间）：**
1. 搜索策略仍是盲进化（OpenEvolve random mutation），不是 diagnosis → targeted repair
2. 只在一个场景验证（OlympiadBench, 20 题训练）
3. 没有跨 MAS 框架、跨任务的知识积累和迁移
4. Reward hacking 问题严重，需手动锁定架构
5. 搜索空间受限于预定义"可变块"

**我们的核心论点：他们用更好的信号做盲搜，我们用诊断信号做定向修复。不是更好的 reward，而是更好的 search。**

## Current Focus

- 选定具体 target MAS 框架，跑起来收集真实 failure pattern
- 设计 diagnosis → targeted repair 的闭环机制
- Keep project memory in `memory_bank/` up to date as the repository evolves.
