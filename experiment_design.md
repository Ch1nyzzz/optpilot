# OptPilot Experiment Design (v2)

## 核心思路

**Experience-Driven Repair Library**：在真实 MAS trace 上诊断 failure mode，生成修复方案，验证有效后蒸馏存入 library。对新 trace 的相同 FM 直接检索历史有效方案应用。

## MAS-as-DAG 统一抽象

任何 MAS 系统统一表示为 DAG（有向无环图），这是所有修改操作的基础：

```
MAS = DAG(Nodes, Edges)

Node = Agent
  - role_prompt: str          # 角色定义和指令
  - tools: List[Tool]         # 可用工具
  - config: Dict              # 框架特定配置 (max_turns, temperature, etc.)

Edge = Communication
  - source → target           # 通信方向
  - message_schema: Dict      # 消息格式约束
  - routing_condition: str    # 何时触发这条边
```

### 统一修改操作（Repair Action Space）

| 操作类型 | 说明 | 示例 |
|---------|------|------|
| **Node Mutation** | 修改 agent 的 prompt / config | 重写 Reviewer 的 role prompt |
| **Node Add** | 增加新 agent | 插入 RepetitionDetector agent |
| **Node Delete** | 删除冗余 agent | 移除无效的 QA agent |
| **Edge Mutation** | 修改通信协议 / 消息格式 | 要求结构化 JSON 传递而非自由文本 |
| **Edge Rewire** | 改变拓扑结构 | 加反馈环路、改串行为并行 |

### 框架适配示例

```
ChatDev → DAG:
  CEO → ProductManager → Programmer → Reviewer → Tester
  (线性 pipeline，每条 edge 是多轮对话)

MetaGPT → DAG:
  ProductManager → Architect → ProjectManager → Engineer → QA
  (SOP 驱动，edge 上有文档产物约束)

AutoGen → DAG:
  动态 GroupChat，edge 由 speaker selection 决定
  (需要在 DAG 中表示动态路由)
```

这样 repair action 天然跨 MAS 可迁移：同一个 "加 verification agent" 操作在不同 DAG 上实例化为不同的具体实现。

## 核心 Claims

| Claim | 说明 | 对标 |
|-------|------|------|
| **C1**: Targeted repair > Blind evolution | 诊断驱动的定向修复比进化搜索收敛更快、效果更好 | vs MAST+OpenEvolve |
| **C2**: Design-time optimization > Inference-time debugging | 修改 MAS 架构本身比 re-rollout 更有效 | vs AgentDebug |
| **C3**: Cross-MAS knowledge transfer | Repair library 在 MAS-A 上积累的经验能直接加速 MAS-B | 独有贡献 |
| **C4**: Experience accumulation | 系统随使用变强，第 N 次优化比第 1 次快 | 独有贡献 |

---

## 实验设置

### Target MAS 框架

- **ChatDev** — 结构化 pipeline 式，MAST-Data 有现成 trace（30 条 GPT-4o + 100 条 ProgramDev-v2）
- **MetaGPT** — 结构化 SOP 式，MAST-Data 同样有 trace

### Benchmark

- **ProgramDev / ProgramDev-v2** — MAST 论文的主要 benchmark，代码生成任务，方便对比
- 可选扩展：GAIA（通用 agent）、SWE-bench Lite（真实 SE 任务）

### Base LLM

- GPT-4o（与 MAST 论文一致，方便对比）
- 可选：Claude 3.7 Sonnet（MAST 也测了）

---

## OptPilot Agent 架构

```
┌──────────────────────────────────────────────────┐
│                  Orchestrator                     │
│  控制整个 loop 的生命周期和状态                      │
│  决定：继续优化 / 停止 / 切换目标 FM                │
└──────────┬───────────────────────────┬────────────┘
           │                           │
     ┌─────▼─────┐              ┌──────▼──────┐
     │  Runner   │              │  Diagnoser  │
     │  运行 MAS  │──traces──→  │  MAST 诊断   │
     │  收集轨迹  │              │  FM 分类标注  │
     └───────────┘              └──────┬──────┘
                                       │
                                  FM profile
                                       │
                                ┌──────▼──────┐
                                │  Optimizer  │
                                │  生成 repair │◄── Repair Library
                                │  DAG action  │     (检索历史方案)
                                └──────┬──────┘
                                       │
                                 repair applied
                                       │
                                ┌──────▼──────┐
                                │  Distiller  │
                                │  验证+蒸馏   │──→ Repair Library
                                │  存入 library│     (写入新方案)
                                └─────────────┘
```

### 各模块职责

| 模块 | 输入 | 输出 | 职责 |
|------|------|------|------|
| **Orchestrator** | FM profile + budget | 控制信号 | 协调整个 loop，决定优先修哪个 FM，何时停止 |
| **Runner** | MAS config + tasks | traces | 运行 MAS 收集轨迹，管理 DAG adapter |
| **Diagnoser** | traces | FM labels + detail | MAST 分类标注，定位到具体 agent 和 step |
| **Optimizer** | FM + trace context + library | DAG repair actions | 检索历史方案或生成新方案 |
| **Distiller** | repair + before/after FM profile | library entry | 验证修复有效后蒸馏存入 library |

### Orchestrator 核心逻辑

```python
while budget > 0:
    traces = runner.run(mas, tasks)
    fm_profile = diagnoser.diagnose(traces)
    if fm_profile.is_clean(): break
    top_fm = fm_profile.highest_frequency()
    repair = optimizer.generate_repair(top_fm, traces)
    runner.apply_repair(repair)
    new_traces = runner.run(mas, tasks)
    new_fm_profile = diagnoser.diagnose(new_traces)
    distiller.evaluate_and_store(top_fm, repair, fm_profile, new_fm_profile)
```

## OptPilot Pipeline

### 数据来源：直接使用 MAST-Data

MAST-Data (HuggingFace `mcemri/MAST-Data`) 包含：
- ✅ 完整执行轨迹 (`trace.trajectory`)
- ✅ 14 个 FM 的 0/1 标签 (`mast_annotation`)
- ✅ task 成功/失败标签
- ❌ 不含具体哪个 agent / 哪个 step 出错（需要 Diagnoser 细化）

**MVP 阶段可以跳过 Runner 的"收集"功能，直接在已有 trace 上做 diagnose → optimize → distill。验证修复效果时再启用 Runner 重新跑 MAS。**

### Phase A: 离线分析（不需要 Runner）

```
MAST-Data traces (ChatDev, ProgramDev-v2, GPT-4o)
    ↓
[DIAGNOSER] 在 MAST 标签基础上细化：
    - 定位到具体哪个 agent 出错
    - 定位到具体哪个 step 出错
    - 分析 root cause context
    ↓
[OPTIMIZER] 针对每个 FM 生成候选 repair action:
    - 检索 library（初始为空）
    - LLM 根据 FM + trace context 生成 DAG repair
    ↓
[DISTILLER] 蒸馏候选方案（待验证）
    → 存入 library，标记为 "unvalidated"
```

### Phase B: 在线验证（需要 Runner）

```
[RUNNER] 应用 Phase A 的 repair action 到 ChatDev
    ↓
[RUNNER] 重新跑相同 task batch
    ↓
[DIAGNOSER] 对比修复前后的 FM profile
    ↓
[DISTILLER] 验证结果：
    - FM 消除 + task 成功 → 标记为 "validated"，更新 success_rate
    - FM 未消除 → 标记为 "failed"，反馈给 Optimizer 重新生成
    - 引入新 FM → 记录 side effect
```

### Phase C: Cross-MAS Transfer

```
Repair Library (从 ChatDev 积累的 validated 方案)
    ↓
新 MAS: MetaGPT on ProgramDev-v2
    ↓
[DIAGNOSER] 诊断 MetaGPT traces → FM-1.3 detected
    ↓
[OPTIMIZER] 检索 library → 找到 ChatDev 的 FM-1.3 方案
    → 适配到 MetaGPT 的 DAG（LLM 做 translation）
    ↓
[RUNNER] 应用 + 验证
    ↓
[DISTILLER] 更新 library（跨 MAS 验证数据）
```

---

## Repair Library Schema (草案，待 MVP 后修订)

```json
{
  "fm_id": "FC1.3",
  "fm_name": "Step Repetition",
  "source_mas": "ChatDev",
  "source_task_type": "code_generation",
  "dag_actions": [
    {"type": "node_add", "description": "Insert RepetitionDetector between Programmer and Reviewer"},
    {"type": "edge_mutation", "description": "Add early-stop condition on Programmer→Reviewer edge"}
  ],
  "repair_description": "高层文本描述，可跨 MAS 复用（蒸馏格式待 MVP 后确定）",
  "success_rate": 0.85,
  "n_validated": 12,
  "side_effects": ["FM-3.1 increased by 3%"],
  "created_at": "2026-03-25"
}
```

关键设计：repair 操作用 DAG action 序列表示，天然跨 MAS 可迁移。蒸馏的具体粒度（文本描述 vs diff vs workflow）待 MVP 跑出真实修复过程后确定。

---

## Baselines

| Baseline | 描述 | 来源 |
|---|---|---|
| **Original** | 原始 MAS，不做任何修改 | 直接跑 |
| **Random Search** | 随机修改 MAS 配置，相同修改预算 | 控制组 |
| **OpenEvolve + Binary** | 进化搜索，只用 pass/fail 反馈 | 复现 ADRS blog |
| **OpenEvolve + MAST** | 进化搜索，用 MAST failure mode 作为 fitness | 复现 ADRS blog |
| **AgentDebug-style** | inference-time re-rollout with feedback | 复现 AgentDebug |
| **OptPilot (cold)** | 我们的方法，空 library 冷启动 | 证明 C1, C2 |
| **OptPilot (warm)** | 我们的方法，用其他 MAS 积累的 library | 证明 C3, C4 |

---

## 评估指标

### 主指标

- **Task Success Rate** — 任务完成率
- **Sample Efficiency** — 达到 X% 性能需要的优化轮数 / LLM 调用次数
- **Failure Mode Reduction** — 每轮优化后 failure profile 变化

### 辅助指标

- **Repair Precision** — 针对 FM-x 的 repair 是否真的减少了 FM-x（targeted 有效性）
- **Repair Side Effect Rate** — 修复一个 FM 引入新 FM 的比例
- **Transfer Success Rate** — library 中方案直接应用到新 MAS 的成功率
- **Library Growth Curve** — library 大小 vs 修复效率的关系
- **LLM Cost** — 总 token 消耗

---

## 实验矩阵

### Exp 1: 主实验 — Targeted Repair vs Baselines（证明 C1 + C2）

**设置**：ChatDev + ProgramDev-v2，GPT-4o，所有 7 个方法

**产出**：
- Convergence curve（x = 优化轮数, y = success rate）
- Failure mode 瀑布图：每轮后各 FM 占比变化
- Repair precision 热力图：repair action 对应 FM 的消除率

### Exp 2: Cross-MAS Transfer（证明 C3）

**设置**：
```
条件 A: OptPilot cold start 优化 MetaGPT
条件 B: OptPilot warm start（用 ChatDev library）优化 MetaGPT
条件 C: OpenEvolve+MAST 优化 MetaGPT（无 transfer 能力的对照）
```

**产出**：
- A vs B 的收敛速度对比
- Library 中方案的 transfer success rate by FM type
- 哪些 FM 的方案 transfer 效果好，哪些需要 adaptation

### Exp 3: Experience Accumulation（证明 C4）

**设置**：在 ChatDev 上按顺序优化 task batch 1, 2, 3...，观察 library 积累效应

**产出**：
- 第 N 个 batch 的平均修复轮数 vs batch 序号（预期下降）
- Library size vs repair success rate 曲线
- 与 OpenEvolve 对比：他们每次从头，我们越来越快

### Exp 4: Ablation Study

| 变体 | 去掉什么 | 验证什么 |
|------|---------|---------|
| w/o library retrieval | 每次都从头生成方案，不检索 | library 是否加速 |
| w/o distillation | 不蒸馏保存成功方案 | 积累是否关键 |
| w/o MAST diagnosis | 有 library 但不诊断 FM，按 task failure 类型检索 | 细粒度诊断是否关键 |
| w/o prioritization | 不按频率排序，随机选 FM 修复 | 优先级策略是否关键 |
| w/o adaptation | transfer 时直接复制方案，不做 MAS-specific 适配 | adaptation 是否必要 |

### Exp 5: 泛化实验（可选）

在其他 MAS（AutoGen, CrewAI）或其他 benchmark（GAIA, SWE-bench Lite）上验证。

---

## 预期结果

```
Figure 1: Convergence curve (Exp 1)
  → OptPilot cold 在 3-5 轮达到 OpenEvolve+MAST 50 轮的水平
  → OptPilot 的曲线是阶梯状的（每修一个 FM 跳一级）

Figure 2: Failure mode waterfall (Exp 1)
  → 每轮精准消除 1 个高频 FM，profile 逐步变干净

Figure 3: Transfer experiment (Exp 2)
  → Warm-start 比 cold-start 快 2-3x 收敛
  → FC1 类方案 transfer 最好（设计问题跨 MAS 通用）
  → FC2 类方案需要 adaptation（协调方式 MAS-specific）

Figure 4: Accumulation curve (Exp 3)
  → 第 5 个 batch 的修复速度是第 1 个的 3x+

Figure 5: Ablation (Exp 4)
  → Library retrieval 和 MAST diagnosis 是最关键的两个组件

Table 1: Final performance comparison
  → OptPilot warm > OptPilot cold > OpenEvolve+MAST > OpenEvolve+Binary > AgentDebug > Original
```

---

## 最小可行实验（MVP）

### Target System: ChatDev + ProgramDev + GPT-4o

选择理由：
- MAST-Data 有 **130 条 ChatDev trace**（全部是 ProgramDev + GPT-4o），可直接用
- 其中 93 条有 failure（71.5%），37 条 clean
- MAST 论文有 case study baseline（手改 role spec +9.4%，加 verification +15.6%）
- 线性 pipeline DAG 最简单，adapter 易实现
- ProgramDev 有明确的 pass/fail 判定

### ChatDev Failure Profile（来自 MAST-Data 分析）

| FM | 名称 | 数量 | 占比 | 优先级 |
|----|------|------|------|--------|
| **1.3** | **Step Repetition** | **47** | **36.2%** | MVP 首选 |
| 1.5 | Unaware Termination | 38 | 29.2% | 高 |
| 2.6 | Reasoning-Action Mismatch | 37 | 28.5% | 高 |
| 3.3 | Incorrect Verification | 33 | 25.4% | 中 |
| 1.1 | Disobey Task Spec | 32 | 24.6% | 中 |
| 2.2 | Fail Ask Clarification | 27 | 20.8% | 中 |

Trace 粒度：每条 trace 包含完整 agent 对话日志（~236K chars，~78 pages），可定位到具体 agent 和 step。
例：FM-1.3 的 trace 中 Programmer↔Code Reviewer 循环了 4 轮（Phase 4-15）。

### 数据标注粒度

- **Full dataset (1242 traces)**: LLM 标注，14 个 FM 的 0/1 标签，无 agent/step 级别定位
- **Human dataset (19 traces)**: 3 个标注员独立标注，含 FM 定义文本，但量太少
- **Diagnoser 的工作**：在已有 FM 标签基础上，读 trace 内容，细化定位到具体 agent 和 step

### MVP 目标
用最小代价验证两个核心假设：
1. 针对特定 FM 的 targeted repair 有效
2. 成功方案可以在同 FM 的其他 trace 上复用

### MVP 步骤

**Week 1: 离线分析（Phase A，不需要跑 MAS）**
1. 下载 MAST-Data 中 ChatDev + ProgramDev 的 traces
2. 已有 FM 标签，用 Diagnoser 细化：定位到具体 agent 和 step
3. 统计 failure profile，确认 top-3 FM
4. 用 Optimizer 对 top-1 FM 生成候选 repair action（DAG 操作）

**Week 2: 在线验证（Phase B，需要跑 ChatDev）**
5. 搭建 ChatDev Runner + DAG adapter
6. 应用 repair action，重新跑相同 tasks
7. Diagnoser 对比修复前后 FM profile
8. 同时做一组 random repair 作为对照
9. Distiller 蒸馏有效方案存入 library

**Week 3: 复用验证**
10. 把 validated 方案应用到同 FM 的其他 trace 对应的 task
11. 看方案是否可直接复用（不需要重新生成）
12. 如果可以 → 核心假设成立 → 进入完整实验

---

## 待解决的实现问题

### P1: Repair 的 Action Space 定义

**问题**：修复方案的具体形式是什么？有几种可能：
- (a) **Prompt-level**: 只修改 agent 的 system prompt / role description
- (b) **Config-level**: 修改 MAS 框架的配置（termination condition, max_turns, tools）
- (c) **Agent-level**: 增删 agent（加一个 checker, 删一个冗余 agent）
- (d) **Workflow-level**: 修改 agent 间的通信拓扑和执行顺序
- (e) **Code-level**: 修改 MAS 框架的源代码

**需要决策**：我们支持哪些级别？全部支持实现复杂度高，只支持 prompt-level 则实验效果可能不够强。

**建议**：MVP 先做 (a) + (b)，完整实验加 (c) + (d)。

### P2: MAST 诊断的粒度和准确性

**问题**：MAST LLM annotator 的准确率是 94%（accuracy），但 Cohen's κ = 0.77。这意味着：
- 大约 6% 的 trace 会被误诊
- 误诊会导致错误的 repair action（修了不该修的东西）

**需要决策**：
- 是否需要在 MAST 之上加一层人工审核？
- 误诊导致的错误修复怎么处理？（validate 阶段能 catch 吗？）
- 是否需要 confidence threshold：低置信度的诊断不修？

### P3: 蒸馏的粒度 [DEFERRED]

**决策**：等 MVP 跑出真实的修复过程后再定。修复可能是单步 prompt edit，也可能是多步 workflow 变更序列，过早设计 schema 会限制灵活性。MVP 阶段先手动记录所有修复操作的原始信息，后续再决定蒸馏格式。

### P4: 检索匹配的粒度

**问题**：同一个 FM（如 FC1.3 Step Repetition）在不同上下文中表现不同：
- ChatDev 的 Step Repetition 可能是 Programmer 和 Reviewer 之间的死循环
- MetaGPT 的 Step Repetition 可能是 ProductManager 反复修改 PRD

同一个 FM label 但 root cause 不同，历史方案可能不适用。

**需要决策**：
- 检索 key 只用 FM id？还是 FM id + trace context embedding？
- 怎么判断一个历史方案"适用于"当前 trace？
- 检索到多个方案时怎么排序？

### P5: Side Effect 处理

**问题**：修复 FM-A 可能引入 FM-B。例如：
- 修 FC1.3（Step Repetition）加了 early termination → 可能引入 FC3.1（Premature Termination）
- 修 FC2.6（Reasoning-Action Mismatch）加了 checker agent → 可能引入 FC1.3（多了一轮对话导致 repetition）

**需要决策**：
- 修复后如果出现新 FM，是回滚还是继续修？
- 是否需要一个 side effect predictor（根据历史数据预测修 FM-A 可能引入什么）？
- 修复顺序是否重要？（先修哪个后修哪个）

### P6: Validate 的标准

**问题**：怎么定义"修复成功"？
- (a) 目标 FM 在该 trace 上消失 → 但 task 可能还是 fail（因为有其他 FM）
- (b) task 成功 → 但可能不是因为修了这个 FM（可能是 LLM 随机性）
- (c) 目标 FM 在多条 trace 上消失 → 更可靠但需要更多样本

**需要决策**：单 trace 验证还是 batch 验证？需要多少样本才算"validated"？

### P7: 计算预算

**问题**：每一轮 loop 需要：
- 跑一次 MAS（MAS 自身的 LLM 调用）
- MAST 诊断（至少一次 LLM 调用）
- 生成 repair 方案（LLM 调用）
- 重新跑 MAS 验证（又一次 MAS 调用）

一轮 loop 在 30 个 task 上可能需要 100+ 次 LLM 调用。

**需要决策**：
- 总预算限制？（如 10 轮优化 × 30 tasks = 300 次 MAS 运行）
- 是否需要 task sampling（不是每轮都跑全部 30 题）？
- 跟 OpenEvolve baseline 怎么公平对比？（按 LLM 调用次数还是按轮数？）

### P8: MAS 框架的可修改性 [RESOLVED → DAG 抽象]

**决策**：采用 MAS-as-DAG 统一抽象（见文档顶部）。任何 MAS 表示为 DAG(Nodes, Edges)，修改操作统一为 node mutation / add / delete + edge mutation / rewire。

**仍需实现**：每个 MAS 框架需要一个 DAG adapter：
- ChatDev adapter：解析 phase config → DAG，修改后写回 config
- MetaGPT adapter：解析 SOP 定义 → DAG，修改后生成 Python 代码
- 通用 adapter 接口：`to_dag()`, `from_dag()`, `apply_repair(action)`
