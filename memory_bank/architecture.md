# Architecture

## Current Repository Structure

```
optpilot/
├── src/optpilot/              # Core library
│   ├── config.py              # Global configuration (env, models, paths)
│   ├── models.py              # Core data models (MASTrace, FMProfile, RepairAction, etc.)
│   ├── llm.py                 # LLM interface (Together AI)
│   ├── orchestrator.py        # Online optimization loop controller
│   ├── registry.py            # Runner factory
│   ├── tracking.py            # Experiment tracking (W&B integration)
│   ├── dag/                   # DAG abstraction and execution
│   │   ├── core.py            # MASDAG, DAGNode, DAGEdge + serialization
│   │   └── executor.py        # DAGExecutor: lightweight workflow engine
│   ├── modules/               # Core optimization modules
│   │   ├── base_runner.py     # MASRunner abstract base
│   │   ├── runner.py          # OptPilotRunner (built-in DAG execution)
│   │   ├── diagnoser.py       # FM diagnosis via LLM
│   │   ├── optimizer.py       # RepairCandidate generation
│   │   ├── judge.py           # Offline counterfactual evaluation
│   │   ├── distiller.py       # Distill repairs into reusable skills
│   │   └── yaml_optimizer.py  # YAML-based optimization
│   ├── library/
│   │   └── repair_library.py  # Persistent repair library (JSON)
│   └── data/
│       ├── fm_taxonomy.py     # FM definitions and taxonomy
│       └── loader.py          # Load MAST-Data traces
├── dags/                      # MASDAG workflow definitions (YAML)
│   └── chatdev.yaml           # ChatDev v1 workflow
├── experiments/               # Experiment pipelines
├── library_store/             # Repair library storage
├── results/                   # Experiment results
└── memory_bank/               # Project documentation
```

## Core Architecture Decision: MAS-as-DAG

任何 MAS 系统统一表示为 MASDAG（有向图，支持循环）：
- **DAGNode** = Agent (role, prompt, config) | Literal | LoopCounter | Passthrough
- **DAGEdge** = Communication (source→target, trigger, condition, carry_data)
- **RepairAction** = DAG 操作 (node mutation/add/delete, edge mutation/rewire, config change)

MASDAG 是唯一的核心表示，既用于定义、又用于执行、又用于修复。

## DAG Executor (2026-03-25)

自研轻量执行引擎，完全脱离 ChatDev v2 依赖：
- 基于 MASDAG 直接执行工作流（无需外部 subprocess）
- 支持四种节点类型：agent (LLM调用), literal (固定文本), loop_counter (循环控制), passthrough (透传)
- 支持 keyword 条件边和 trigger/non-trigger 边
- 执行产出 ExecutionTrace，转为 MASTrace 供诊断

执行流程：
1. 确定 start nodes，注入 task prompt
2. BFS 式执行 ready queue 中的节点
3. 每个 agent 节点通过 llm_fn 调用 LLM
4. 根据边条件传播输出到下游节点
5. loop_counter 节点管理循环次数和退出

## OptPilot Agent 架构

五个模块组成 closed-loop：

| 模块 | 职责 |
|------|------|
| **Orchestrator** | 协调整个 loop，决定优先修哪个 FM，何时停止 |
| **Runner** | 用 DAGExecutor 直接执行 MASDAG，收集执行轨迹 |
| **Diagnoser** | 基于 MAST 的 FM 分类标注，细化定位到具体 agent 和 step |
| **Optimizer** | 检索 Repair Library 历史方案或用 LLM 生成新的 DAG repair action |
| **Distiller** | 验证修复有效后蒸馏方案存入 Repair Library |

数据流：
```
Runner(dag) → traces → Diagnoser → profiles → Optimizer → candidate
                                                              ↓
                                                         apply_repair → new_dag
                                                              ↓
                                                    Runner(new_dag) → new_traces
                                                              ↓
                                                         Distiller → library
```

## Architecture Change Policy

Any meaningful architecture decision or repository-structure change must be recorded in
this file. Examples include:

- adding new top-level directories
- changing the system decomposition or interface boundaries
- introducing new storage, evaluation, or workflow components
