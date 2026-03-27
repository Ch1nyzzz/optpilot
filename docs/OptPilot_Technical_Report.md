# OptPilot: Experience-Driven Repair System for Multi-Agent Systems

> 基于诊断驱动的多智能体系统自动修复框架

---

## 1. 研究动机与问题定义

### 1.1 背景

多智能体系统（Multi-Agent Systems, MAS）在复杂任务协作中表现出强大能力，但在实际运行中经常出现各类失败模式（Failure Modes）：

- 智能体不遵循指令或角色规范
- 系统陷入无限循环或死锁
- 上下文信息在传递过程中丢失
- 智能体间通信中断或信息不完整
- 任务偏离或推理错误
- 验证环节缺失或判断错误

现有研究（MAST、AgentFail）在失败模式的分类和诊断方面做出了重要贡献，但**缺乏从诊断到自动修复的闭环能力**。

### 1.2 核心论题

**诊断驱动的定向修复（Diagnosis → Targeted Repair），而非盲目进化（Blind Evolution）。**

| 对比方 | 做法 | OptPilot 的差异 |
|--------|------|-----------------|
| **MAST / AgentFail** | 失败分类 + 数据集，不做修复 | 我们闭合了"诊断→修复"的自动化环路 |
| **AgentDebug** | 推理时单智能体重跑 | 我们做设计时的 MAS 架构级优化 |
| **MAST + OpenEvolve** | 用失败分类作为进化搜索的适应度信号 | 我们用诊断结果做定向修复——不是更好的 reward，而是更好的 search |

---

## 2. 系统架构

### 2.1 整体流程

```
┌──────────────────────────────────────────────────────────────────┐
│                      OptPilot Pipeline                          │
│                                                                  │
│  ┌─────────┐    ┌───────────┐    ┌──────────────┐               │
│  │ Runner  │───▶│ Diagnoser │───▶│ Orchestrator │               │
│  │(执行MAS)│    │(FM诊断)   │    │(排名+分发)    │               │
│  └─────────┘    └───────────┘    └──────┬───────┘               │
│                                         │                        │
│                    ┌────────────────────┬┴──────────────────┐    │
│                    ▼                    ▼                   ▼    │
│              ┌──────────┐        ┌──────────┐        ┌──────────┐│
│              │ Skill-A  │        │ Skill-B  │  ...   │ Skill-F  ││
│              │(指令遵循) │        │(执行循环) │        │(验证失败) ││
│              └────┬─────┘        └────┬─────┘        └────┬─────┘│
│                   │                   │                   │      │
│                   └───────────┬───────┘───────────────────┘      │
│                               ▼                                  │
│                    ┌─────────────────────┐                       │
│                    │ Forge（多修复合并）  │                        │
│                    └──────────┬──────────┘                       │
│                               ▼                                  │
│                    ┌─────────────────────┐                       │
│                    │  Updated MASDAG     │                       │
│                    └─────────────────────┘                       │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 核心抽象：MAS-as-DAG

任何多智能体系统统一表示为 **MASDAG**（有向图，支持循环）：

| 概念 | 描述 |
|------|------|
| **DAGNode** | 节点类型：`agent`（LLM 调用）、`literal`（固定文本）、`loop_counter`（迭代控制）、`passthrough` |
| **DAGEdge** | 通信边：`source→target`，属性包括 `trigger`、`condition`（关键词匹配）、`carry_data`、`loop`（continue/exit） |
| **RepairAction** | DAG 操作：节点变更/添加/删除、边变更/重连、配置修改 |

**MASDAG 是唯一的核心表示**：定义、执行、修复共用同一数据结构。修复操作 = YAML 级别的 DAG 修改。

### 2.3 DAG Executor

自研轻量级 BFS 工作流引擎，完全脱离外部框架（如 ChatDev v2）依赖：

1. 确定起始节点（`metadata.start`），注入任务 prompt
2. BFS 执行 ready queue 中的节点
3. Agent 节点：使用 `node.prompt` 或 `node.role` 作为系统提示词
4. Literal 节点：使用 `config.content` 或 `node.prompt` 作为输出
5. 根据边条件（关键词匹配）传播输出到下游节点
6. `loop_counter` 节点管理循环次数和退出条件

---

## 3. 六组失败模式分类体系（6-Group FM Taxonomy）

基于 MAST 论文的 17 类 FM 简化合并为 6 组，每组对应一种修复策略：

| 组别 | 名称 | 原始 FM | 修复策略 |
|------|------|---------|----------|
| **A** | Instruction Non-Compliance（指令遵循） | FM-1.1, 1.2 | 优化 prompt 设计，明确角色定义和约束 |
| **B** | Execution Loop / Stuck（执行循环/卡死） | FM-1.3, 1.5 | 添加循环检测、最大迭代限制、显式终止条件 |
| **C** | Context Loss（上下文丢失） | FM-1.4, 2.1 | 改进状态管理、上下文窗口管理、检查点机制 |
| **D** | Communication Failure（通信失败） | FM-2.2, 2.4, 2.5 | 改进通信协议，强制信息共享机制 |
| **E** | Task Drift / Reasoning Error（任务漂移/推理错误） | FM-2.3, 2.6 | 添加目标追踪、CoT 验证、行动-推理一致性检查 |
| **F** | Verification Failure（验证失败） | FM-3.1, 3.2, 3.3 | 多级验证：底层代码编译 + 高层目标检查 |

**FM 分类器**：MiniMax M2.5，经 100 条 blind trace 校准，与人类专家标注容忍率 **89.9%–98.0%**。

---

## 4. Skill Workflow 架构

### 4.1 模块职责

| 模块 | 职责 |
|------|------|
| **Orchestrator** | 诊断 → 按 FM 频率排名 → 分发给 Skill Workflows（支持并行） |
| **Runner** | 用 DAGExecutor 执行 MASDAG，收集 trace + benchmark 评分 |
| **Diagnoser** | 6-group FM 分类 + agent/step 定位（并发诊断） |
| **Skill Workflows (A-F)** | 6 个 Python 类，每个 FM group 一个完整的修复 agent |
| **SkillEvolver** | Skill 连续失败达阈值 → LLM 修改 Skill 源码 → 动态加载 |
| **Forger** | 多个 Skill 同时成功时，合并所有修复到一个 DAG |

### 4.2 Skill 内部闭环

每个 Skill 包含**双层循环**：

```
for outer_round in range(MAX_OUTER_ROUNDS=3):       # 外循环：反思
    analyze(dag, traces, profiles, negatives) → AnalysisResult

    for inner_iter in range(MAX_INNER_ITERS=5):      # 内循环：迭代修复
        evolve(dag, analysis, negatives) → EvolveResult (YAML级修改)
        run_batch(proposal_tasks) → new_traces
        diagnose → fm_rate
        if fm_rate < 0.2 or no_improvement: break

    # 在线验证
    run_batch(validation_tasks, original_dag) → before_val   # 基线（缓存）
    run_batch(validation_tasks, repaired_dag) → after_val
    if judge(before_val, after_val): return success

    reflect(...) → ReflectInsight → negatives.append(insight)  # 经验积累
```

### 4.3 关键设计决策

| 设计 | 说明 |
|------|------|
| **Validation 基线缓存** | `before_val_traces` 在整个 `run()` 生命周期只计算一次 |
| **pass_rate 使用 task_score** | 优先使用 ground-truth benchmark 评分（而非 `task_success` 完成率） |
| **Judge 条件** | FM rate 必须严格下降；无真实 DAG diff 的 repair 不允许判 success |
| **SkillBudget** | `max_llm_calls=30` / `max_batch_runs=10` / `max_wall_time_s=600` |
| **失败快照** | 即使因 no-op repair 或预算耗尽提前结束，也生成合成 ReflectInsight |

### 4.4 Proposal / Validation 分离

- **Proposal 子集**：命中目标 FM 的 trace，用于聚合证据并生成修复方案
- **Validation 子集**：所有剩余 trace（不与 proposal 重叠），保留至少一个命中 FM 的 holdout
- **验证标准**：目标 FM 在 validation 上下降 + repair 后的 DAG 必须与原 DAG 有真实差异

### 4.5 Meta-Evolution（元进化）

当某个 Skill 连续失败达到阈值（默认 3 次）时触发：

1. SkillEvolver 准备临时工作区：Skill 源文件 + 上下文索引 + 失败摘要
2. LLM 作为 meta-agent，使用 tool-calling（bash、file read）自主检查代码库
3. 在更宽松预算下（最多 30 轮 tool-calling，32K tokens）修改 Skill Python 源码
4. 语法检查（`compile()`）→ 保存到 `library_store/evolved_skills/`
5. `importlib` 动态加载替换注册表中的旧版本

### 4.6 并行 Skill 执行

- 不同 FM group 的 Skill 可以并行优化（`asyncio.gather`）
- 各自工作在 DAG 的独立 `deepcopy` 上，互不干扰
- Orchestrator 收集所有结果，多个成功时通过 Forger 合并

---

## 5. 目标系统：AG2 MathChat

### 5.1 系统概述

基于 MAST 论文 Appendix L 的官方 3-agent GroupChat：

```yaml
Agents:
  - Agent_Problem_Solver: 独立解题，步骤化方案
  - Agent_Code_Executor:  Python 代码求解（含 SymPy）
  - Agent_Verifier:       对比两个 agent 的结果，判定最终答案

Control:
  - Turn Counter: max_iterations=5，循环控制
  - SOLUTION_FOUND 关键词触发终止 → FINAL
```

### 5.2 数据流

```
USER (task prompt)
  ├──context──▶ Agent_Problem_Solver ──trigger──▶ Agent_Verifier
  ├──context──▶ Agent_Code_Executor  ──trigger──▶ Agent_Verifier
  └──context──▶ Agent_Verifier
                    │
                    ├── SOLUTION_FOUND ──▶ FINAL
                    ├── 无 SOLUTION_FOUND ──▶ Agent_Code_Executor（委派继续）
                    └── 每轮触发 Turn Counter
                              └── max iterations ──exit──▶ FINAL
```

### 5.3 评测基准

| Benchmark | 类型 | 评分方式 |
|-----------|------|----------|
| **cais/mmlu** | 多选知识题 | 精确匹配 |
| **opencompass/AIME2025** | 竞赛数学（整数答案） | 从 `\boxed{...}` 提取答案，精确匹配 |
| **lscpku/OlympiadBench** (`maths_en_no_proof`) | 开放式数学题 | 精确匹配 + SymPy 符号匹配 |

---

## 6. 技术实现

### 6.1 仓库结构

```
optpilot/
├── src/optpilot/              # 核心库
│   ├── config.py              # 全局配置（环境变量、模型、路径、限流）
│   ├── models.py              # 核心数据模型
│   ├── llm.py                 # LLM 接口（Together AI，同步+异步，限流）
│   ├── orchestrator.py        # Skill 分发循环（诊断→并行Skill→采纳DAG）
│   ├── skills/                # Skill Workflows（每个 FM group 一个）
│   │   ├── base.py            # BaseSkill ABC + GenericSkill + run() 模板
│   │   ├── registry.py        # FM group → Skill 类映射 + 动态加载
│   │   ├── negatives.py       # ReflectInsight 持久化（JSON per FM group）
│   │   ├── evolution.py       # Skill 元进化（LLM 修改 Skill 代码）
│   │   ├── forger.py          # 多 Skill 成功时合并修复
│   │   └── skill_{a-f}.py     # 6 个具体 Skill（A-F）
│   ├── dag/                   # DAG 抽象和执行
│   │   ├── core.py            # MASDAG, DAGNode, DAGEdge + YAML 序列化
│   │   └── executor.py        # DAGExecutor: 轻量级 BFS 工作流引擎
│   ├── modules/               # 核心模块
│   │   ├── runner.py          # OptPilotRunner（DAG 执行 + benchmark 评分）
│   │   └── diagnoser.py       # FM 诊断（6-group，并发）
│   └── data/
│       ├── fm_taxonomy_6group.py  # A-F 失败组定义
│       └── benchmarks.py      # 官方 benchmark 加载器 + ground-truth 评分器
├── dags/                      # MASDAG 工作流定义（YAML）
│   └── ag2_mathchat.yaml      # AG2 MathChat 3-agent GroupChat
├── experiments/               # 实验入口
│   └── run_ag2_mathchat_skill.py  # 主实验：AG2 × benchmarks × Skill Workflows
├── library_store/             # 持久化存储
│   ├── negatives/             # 每 FM group 的 ReflectInsight 历史
│   ├── evolved_skills/        # 元进化后的 Skill 源文件
│   └── skill_agent_traces/    # Skill agent 的 tool-calling 记录
└── results/                   # 实验结果和 trace artifacts
```

### 6.2 LLM 并发控制

按模型家族做全局限流：

- 每个模型家族共享 `Semaphore`（限并发）+ 60 秒滑动窗口（限 RPM）
- 内置家族：MiniMax M2.5（96 并发）、GLM-5（48 并发）、default（64 并发）
- 同时提供同步和异步客户端

### 6.3 持久化与追溯

| 类别 | 存储位置 | 内容 |
|------|----------|------|
| **Trace Artifacts** | `results/*_artifacts/optimization/round_k/train/` | 完整执行 trace + sidecar metadata |
| **Diagnose Artifacts** | `results/*_artifacts/optimization/round_k/diagnose/` | FM profiles + 按 FM 分组 + skill jobs |
| **DAG Versions** | `results/*_artifacts/dag_versions/` | 每轮 DAG 快照（input → round_start → skill_success → final） |
| **Tool Traces** | `library_store/skill_agent_traces/` | Skill agent 的 LLM tool-calling 记录 |
| **Negatives** | `library_store/negatives/` | 每 FM group 的失败反思 |
| **Evolved Skills** | `library_store/evolved_skills/` | 元进化后的 Skill 源代码 |

---

## 7. 实验设计

### 7.1 实验入口

```bash
python -m experiments.run_ag2_mathchat_skill \
    --model openai/gpt-oss-120b \
    --train 100 --test 100 \
    --rounds 1 \
    --concurrency 512 \
    --timeout 600
```

### 7.2 实验流程

1. **加载 benchmarks**：从 MMLU + AIME2025 + OlympiadBench 按比例抽样
2. **Train/Test 分割**：按 benchmark 等比分层抽样
3. **在 train 上优化**：Orchestrator 运行 Skill Workflow 闭环
4. **在 test 上评测**：对比 baseline DAG 和 optimized DAG 的准确率
5. **结果持久化**：JSON 汇总 + 完整 trace artifacts

### 7.3 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--train` | 100 | 训练集大小 |
| `--test` | 100 | 测试集大小 |
| `--rounds` | 1 | 外部优化轮数（Skill 内部自行收敛） |
| `--concurrency` | 512 | 最大并发任务数 |
| `--timeout` | 600s | 单任务超时 |
| `--group` | None | 聚焦特定 FM group（A-F） |
| `--reuse-traces-dir` | None | 复用已有 train traces，跳过重新执行 |

---

## 8. 研究进展

### Phase 1: 文献调研 + 设计 (已完成)
- 调研 21 篇相关论文
- 确定 MAS 优化方向
- 完成差异化定位和实验设计文档

### Phase 2: 核心实现 (已完成)
- 完成核心 pipeline 代码
- MVP Target: AG2 MathChat（597 条 traces，短 trace ~5K chars）
- 离线 pipeline 验证成功

### Phase 3: 架构脱耦 (已完成)
- 自研 DAGExecutor，完全脱离 chatdev_v2
- MASDAG 自有序列化格式

### Phase 4: FM Classifier 校准 (已完成)
- 100 条 AG2 blind trace，6 位标注者
- MiniMax M2.5 与人类容忍率 89.9%–98.0%

### Phase 5: Online 6-Group Pipeline (已完成)
- 全面切换 6-group taxonomy (A-F)
- AG2 MathChat DAG 基于 MAST 论文 Appendix L
- 统一 MiniMax M2.5 模型

### Phase 6: Skill Workflow Architecture (已完成)
- 6 个 Skill Workflows (A-F)
- 内循环收敛 + 外循环反思 + 并行执行
- Meta-evolution: Skill 连续失败后 LLM 修改 Skill 源码

### Phase 7: Bug Fixes + Refinement (已完成)
- 修复 DAGExecutor 系统提示词丢失
- 修复 literal 内容读取
- 修复 pass_rate 评分（task_score vs task_success）
- 修复 Turn Counter 循环逻辑

---

## 9. 关键研究决策

| 日期 | 决策 | 理由 |
|------|------|------|
| 2026-03-25 | 聚焦 MAS 而非通用优化 | failure pattern 丰富，有现成 taxonomy 数据 |
| 2026-03-25 | 核心差异化 = targeted repair vs blind evolution | MAST+OpenEvolve 已做了 taxonomy-as-reward |
| 2026-03-25 | MVP 切换到 AG2 (MathChat) | 597 条 trace，短 trace，3-agent 架构 |
| 2026-03-25 | Together AI: MiniMax M2.5（统一） | 统一 API，校准验证通过 |
| 2026-03-26 | pass_rate 必须用 task_score | task_success 只是"是否跑完"，几乎恒为 True |
| 2026-03-26 | Loop counter 需显式 `loop: exit` 标注 | 避免被动计数器每轮误触下游 |

---

## 10. 下一步计划

1. **端到端实验验证**：在完整 benchmark 上跑 Skill Workflow pipeline，收集 baseline vs optimized 对比数据
2. **Skill 特化**：为每个 FM group 定制更精准的修复策略 prompt
3. **Meta-evolution 验证**：验证 SkillEvolver 连续失败后的自我改进效果
4. **Ablation 实验**：验证各组件（diagnosis、targeted repair、negatives、meta-evolution）的独立贡献
5. **论文撰写**：整理实验数据，撰写学术论文
