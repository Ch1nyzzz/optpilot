# Architecture

## Current Repository Architecture

The repository is currently documentation-first and organized around research notes.

- `related_papers.md`: primary literature synthesis and project framing
- `papers/`: supporting paper files and source material (21 papers)
- `experiment_design.md`: 实验设计文档 (v2)，包含 pipeline、baselines、metrics、open questions
- `memory_bank/`: persistent project memory for goals, progress, and architecture
- `AGENTS.md` and `CLAUDE.md`: contributor and agent instructions

## Core Architecture Decision: MAS-as-DAG (2026-03-25)

任何 MAS 系统统一表示为 DAG（有向无环图）：
- **Node** = Agent (role_prompt, tools, config)
- **Edge** = Communication (source→target, message_schema, routing_condition)
- **Repair Action** = DAG 操作 (node mutation/add/delete, edge mutation/rewire)

每个 MAS 框架需要一个 DAG adapter：`to_dag()`, `from_dag()`, `apply_repair(action)`。
这解决了跨 MAS 的统一修改接口问题，也使 repair action 天然可迁移。

## ChatDev 源码分析 (2026-03-25)

ChatDev v2 **原生就是 YAML-driven DAG executor**：
- 源码 ~28K 行 Python，核心是 DAG 执行引擎（`workflow/`）
- 整个 MAS 定义在一个 YAML 文件（`yaml_instance/ChatDev_v1.yaml`，~1000 行）
- YAML 结构：`graph.nodes[]` + `graph.edges[]`，天然匹配我们的 DAG 抽象
- **DAG adapter 几乎不用写**——直接修改 YAML 即可

ChatDev DAG 节点类型：
- `agent`: LLM agent（Programmer, Code Reviewer, CEO, CPO, Test Engineer）
- `literal`: 固定 prompt 文本（phase instruction）
- `loop_counter`: 循环控制（code review max 10 轮，test max 3 轮）
- `passthrough`: 路由节点

可修改点（对应 repair action）：
- **Node mutation**: 改 agent 的 `role` 字段（prompt）→ 直接改 YAML
- **Node add/delete**: 增删 node + 对应 edges → 改 YAML
- **Edge rewire**: 改 `from`/`to` → 改 YAML
- **Loop config**: 改 `loop_counter.max_iterations`（如 code review 从 10 改为 5）→ 改 YAML
- **Edge condition**: 改 keyword-based 终止条件 → 改 YAML

## OptPilot Agent 架构 (2026-03-25)

五个模块组成 closed-loop：

| 模块 | 职责 |
|------|------|
| **Orchestrator** | 协调整个 loop，决定优先修哪个 FM，何时停止 |
| **Runner** | 运行 MAS 收集轨迹，管理 DAG adapter (to_dag / from_dag / apply_repair) |
| **Diagnoser** | 基于 MAST 的 FM 分类标注，细化定位到具体 agent 和 step |
| **Optimizer** | 检索 Repair Library 历史方案或用 LLM 生成新的 DAG repair action |
| **Distiller** | 验证修复有效后蒸馏方案存入 Repair Library |

数据流：Runner → Diagnoser → Optimizer → Runner (apply) → Diagnoser (verify) → Distiller

**MVP 数据策略**：Phase A 直接用 MAST-Data 已有 trace（跳过 Runner），Phase B 再启用 Runner 验证。

## Architecture Change Policy

Any meaningful architecture decision or repository-structure change must be recorded in
this file. Examples include:

- adding new top-level directories such as `src/`, `tests/`, or `scripts/`
- changing the system decomposition or optimizer interface boundaries
- introducing new storage, evaluation, or workflow components

Each update should briefly state what changed, why it changed, and which files or
modules were affected.
