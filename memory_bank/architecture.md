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
├── tests/                     # Regression tests for executor and persistence behavior
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

Loop counter 约定：
- 优先读取 edge config 中的 `loop: continue|exit`
- 若未显式标注，则执行器按 trigger-edge 拓扑判断：目标节点若能回到当前 loop counter，则视为 continue edge，否则视为 exit edge
- 因此 loop counter 不再在同一轮同时触发“继续循环”和“退出下一阶段”

## OptPilot Agent 架构

五个模块组成 closed-loop：

| 模块 | 职责 |
|------|------|
| **Orchestrator** | 协调整个 loop，决定优先修哪个 FM，何时停止 |
| **Runner** | 用 DAGExecutor 直接执行 MASDAG，收集执行轨迹 |
| **Diagnoser** | 基于 MAST 的 FM 分类标注，细化定位到具体 agent 和 step |
| **Optimizer** | 检索 Repair Library 历史方案或用 LLM 生成新的 DAG repair action |
| **Distiller** | 验证修复有效后蒸馏方案存入 Repair Library |

## Repair Library Persistence (2026-03-25)

Repair Library 仍然使用 JSON 持久化，但写盘策略改为 buffered flush：
- `add()` / `update_stats()` 默认只更新内存并标记 dirty
- 流程末尾显式调用 `flush()` 落盘，减少实验中每条 entry 都全量重写 JSON 的开销
- 进程退出时也会执行兜底 flush

## LLM Concurrency Control (2026-03-25)

LLM 调用层现在按模型家族做全局限流：
- 每个模型家族共享一个全局 `Semaphore`，限制进程内并发请求数
- 同时维护 60 秒滑动窗口，限制该模型家族的 RPM
- 当前内置家族是 `MiniMax M2.5`、`GLM-5` 和默认兜底配置
- 具体阈值通过环境变量控制，而不是散落在各个 `ThreadPoolExecutor` 里

高并发链路补充：
- `llm.py` 现在同时提供同步 `OpenAI` 和异步 `AsyncOpenAI` 客户端
- 高并发实验主链路优先走异步接口 `acall_llm` / `acall_llm_json`
- `offline_yaml_pipeline` 已从 `ThreadPoolExecutor` 迁移到 `asyncio.gather`
- `Diagnoser`、`Judge`、`YAMLOptimizer` 提供 async 方法，作为高并发路径的默认实现

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

## Online Validation Protocol (2026-03-26)

在线蒸馏现在不再是“同一批任务提案、同一批任务验证”的自闭环，而是显式拆成 proposal / validation 两个子集：
- 先在当前 DAG 上跑整批任务并诊断，选择出现次数至少为 2 的目标 FM
- proposal 子集只使用部分命中该 FM 的 trace，用于聚合证据并生成 repair
- validation 子集与 proposal 完全不重叠，且至少保留一个命中该 FM 的 holdout trace
- repaired DAG 只在 validation 子集上重跑并判定成败

验证标准也从“目标 FM 计数下降”收紧为：
- 目标 FM 在 holdout validation 上下降
- `pass` rate 上升

其中 `pass` 优先来自外部 benchmark scorer 或评测器；若未提供显式 scorer，则 Runner 退化为 success proxy：
- 到达 DAG `metadata.success_nodes` 指定节点即视为成功
- 若未显式配置且存在 `FINAL` 节点，则执行到 `FINAL` 视为成功
- 默认 score = `1.0`（成功）/ `0.0`（失败）

同时保留复杂度观测指标但不作为主判据：
- 平均运行时间 `latency_s`
- `validation_metrics.runtime_delta_s`

相关影响：
- `Optimizer` 在 online repair 时改为基于多个 trace/profile 聚合 fault evidence，而不是单条代表 trace
- `Distiller` 在线蒸馏 pattern 时会看 before/after holdout evidence，不再只看 repair 文本
- `RepairLibrary.search()` 默认排除 `failed` entries，减少坏 skill 被再次检索到的概率

## Skill Wrap-Up (2026-03-26)

Repair Library 现在区分两类条目：
- `hint`: 原始离线/在线蒸馏结果，保留正例和反例
- `wrapped`: 批次级 wrap-up 后的 canonical skill，用于优先检索

wrap-up 规则：
- 按 `fm_id` 聚合 raw hints
- validated hints 和高置信 unvalidated hints 作为支持证据
- failed hints 和低置信 unvalidated hints 作为反例边界
- 输出少量 canonical skills，每条都同时包含：
  - `root_cause_pattern`: when to use
  - `when_not_to_use`: failure boundary
  - `validation_metrics.recommended_actions`
  - `avoid_actions`

检索策略：
- 若某个 FM 已有 `wrapped` skills，则优先只检索 wrapped entries
- 若还没有 wrap-up 产物，再回退到 raw hints

存储边界：
- offline hints 单独保存在 `library_store/offline_hints/`
- offline wrapped skills 单独保存在 `library_store/offline_skills/`
- online 产物不与 offline 共用这两类路径
- 因此 offline 不再把 raw hints 和 wrapped skills 混在同一个 JSON 文件里

## Architecture Change Policy

Any meaningful architecture decision or repository-structure change must be recorded in
this file. Examples include:

- adding new top-level directories
- changing the system decomposition or interface boundaries
- introducing new storage, evaluation, or workflow components

## Experiment Variants (2026-03-26)

The repository now keeps two separate prompt-optimization experiment variants under `experiments/`:
- `experiments/evolve_classifier/`: fine-grained 14-label FM prediction
- `experiments/evolve_classifier_3label/`: coarse 3-label multi-label prediction (`C1/C2/C3`)

The 3-label variant reuses the same trace dataset as the 14-label variant but applies a
different evaluator and output schema. This keeps coarse-category experiments isolated
from the fine-grained setup while avoiding duplicate label files.
