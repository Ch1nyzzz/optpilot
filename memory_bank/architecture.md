# Architecture

## Current Repository Structure

```
optpilot/
├── src/optpilot/              # Core library
│   ├── config.py              # Global configuration (env, models, paths, rate limits)
│   ├── models.py              # Core data models (MASTrace, FMProfile, RepairAction,
│   │                          #   EvolveResult, ReflectInsight, SkillBudget,
│   │                          #   AnalysisResult, SkillResult)
│   ├── llm.py                 # LLM interface (Together AI, sync + async, rate limiting)
│   ├── orchestrator.py        # Jacobian-driven single repair loop
│   ├── registry.py            # Runner factory
│   ├── tracking.py            # Experiment tracking (W&B integration)
│   ├── skills/                # Repair loop and pattern catalog
│   │   ├── repair_loop.py     # Stateless functions: analyze(), multi-candidate
│   │   │                      #   aevolve(), reflect()
│   │   ├── tools.py           # ToolContext, search_and_replace/bash tools,
│   │   │                      #   dag_to_python(), python_source_to_dag()
│   │   ├── repair_patterns.py # FailureSignature, RepairPattern, PatternCatalog (JSON-persisted)
│   │   ├── jacobian.py        # RepairJacobian matrix (failure×pattern → success rate)
│   │   ├── negatives.py       # ReflectInsight persistence (JSON per FM group)
│   │   ├── evolution.py       # CatalogEvolver (LLM modifies pattern catalog)
│   │   └── base.py            # [DEPRECATED] Old BaseSkill/GenericSkill classes
│   ├── dag/                   # DAG abstraction and execution
│   │   ├── core.py            # MASDAG, DAGNode, DAGEdge + YAML/Python serialization
│   │   └── executor.py        # DAGExecutor: lightweight BFS workflow engine
│   ├── modules/               # Core modules
│   │   ├── base_runner.py     # MASRunner abstract base
│   │   ├── runner.py          # OptPilotRunner (DAG execution + benchmark scoring)
│   │   ├── diagnoser.py       # FM diagnosis via LLM (6-group, concurrent)
│   │   ├── judge.py           # Offline counterfactual evaluation (optional)
│   │   ├── yaml_optimizer.py  # YAML-based optimization (reference)
│   │   └── _legacy/           # Replaced by Skill Workflows
│   ├── library/
│   │   └── repair_library.py  # Persistent repair library (JSON, buffered flush)
│   └── data/
│       ├── fm_taxonomy_6group.py # Canonical A-F failure-group taxonomy (+ analyze_hint)
│       ├── loader.py          # Load MAST-Data traces (normalizes to A-F)
│       └── benchmarks.py      # Official benchmark loaders + ground-truth scorers
│                              # (MMLU, AIME 2025, OlympiadBench)
├── dags/                      # MASDAG workflow definitions (YAML)
│   ├── ag2_mathchat.yaml      # AG2 MathChat 3-agent GroupChat (primary)
│   ├── simple_star_gaia.yaml  # Simplified 3-agent star topology for GAIA
│   ├── simple_hierarchical_swebench.yaml # Simplified 3-agent hierarchy for SWE-bench
│   └── chatdev.yaml           # ChatDev v1 workflow (reference)
├── experiments/               # Experiment pipelines
│   ├── run_skill.py           # Unified multi-topology online repair entry
│   ├── run_openevolve.py      # Unified multi-topology OpenEvolve cold-start entry
│   ├── analyze_openevolve_traces.py # Offline prior extraction from OpenEvolve traces
│   ├── run_ag2_mathchat_skill.py  # Older AG2-specific entry
│   └── ...                    # offline/online pipelines, ablations
├── data/annotations/          # Blind/manual labeling packs for 6-group evaluation
├── tests/                     # Regression tests
├── library_store/             # Persistent experience
│   ├── negatives/             # Per-FM-group ReflectInsight history (JSON)
│   ├── jacobian/              # Jacobian matrix + outcomes (JSON/JSONL)
│   ├── pattern_catalog.json   # Evolved pattern catalog (JSON, runtime-modified)
│   ├── skill_agent_traces/    # Per-FM-group tool-calling transcripts
│   ├── meta_evolve_traces/    # Catalog evolution transcripts
│   ├── ag2/                   # Per-topology isolated experience store
│   │   ├── negatives/         # Topology-specific ReflectInsight history
│   │   ├── jacobian/          # Topology-specific Jacobian warmup + outcomes
│   │   ├── recipes/           # Topology-specific offline repair recipes
│   │   ├── meta_evolve_traces/# Topology-specific catalog evolution traces
│   │   └── pattern_catalog.json
│   ├── simple_star/           # Same isolated layout for simple GAIA topology
│   ├── simple_hier/           # Same isolated layout for simple SWE-bench topology
│   ├── offline_hints/         # Offline repair hints
│   ├── offline_skills/        # Offline wrapped skills
│   ├── online_hints/          # Online repair hints
│   └── online_skills/         # Online wrapped skills
├── results/                   # Experiment results, metrics, and per-run trace artifacts
└── memory_bank/               # Project documentation
```

版本管理约定：
- `logs/`、`results/`、`library_store/skill_agent_traces/`、`library_store/meta_evolve_traces/` 为运行生成物，默认不纳入 Git。
- `library_store/pattern_catalog.json` 属于可演化算法知识库，可按需要随代码一并版本化。

## Core Architecture: MAS-as-DAG

任何 MAS 系统统一表示为 MASDAG（有向图，支持循环）：
- **DAGNode** = Agent (role, prompt, config) | Literal (config.content) | LoopCounter (config.max_iterations) | Passthrough
- **DAGEdge** = Communication (source→target, trigger, condition, carry_data, loop)
- **RepairAction** = DAG 操作 (node mutation/add/delete, edge mutation/rewire, config change)

MASDAG 是唯一的核心表示，既用于定义、又用于执行、又用于修复。

## DAG Executor

自研轻量执行引擎，完全脱离 ChatDev v2 依赖：

执行流程：
1. 确定 start nodes（metadata.start），注入 task prompt
2. BFS 式执行 ready queue 中的节点
3. Agent 节点：`node.prompt or node.role` 作为 system prompt，支持 `config.params` 嵌套参数
4. Literal 节点：`config.content` 或 `node.prompt` 作为输出
5. 根据边条件（keyword matching）传播输出到下游节点
6. loop_counter 节点管理循环次数和退出

Loop counter 约定：
- 优先读取 edge config 中的 `loop: continue|exit` 显式标注
- 若未显式标注，则拓扑推断：目标节点若能经 trigger edge 回到当前 loop counter，则为 continue edge，否则为 exit edge
- 当 continue_edges 和 exit_edges 都为空时（无法识别循环结构），才 fallback 到所有 trigger edge
- 仅有 exit_edges 而无 continue_edges 时（被动计数器），count < max_iter 时不触发任何下游

## Jacobian-Driven Single Repair Loop (2026-03-28)

取代了原来的 Skill 分层架构（6 个 Skill 类 + parallel dispatch + forger merge）。
现在是 OpenEvolve 风格的单循环，加上 Jacobian 经验累积：

```
Orchestrator.aoptimize():
  incumbent_traces = run(initial_dag, fixed_eval_tasks)     # 只初始化一次
  incumbent_profiles = classify(incumbent_traces)
  incumbent_score = fitness(incumbent_traces, incumbent_profiles)

  for round in max_rounds:
      signature = extract_dominant_failure(incumbent_profiles)  # 提取最严重的 FM group

      pattern = jacobian.recommend(signature)                   # 经验驱动选修复方向
      analysis = analyze(dag, incumbent_traces, incumbent_profiles)
      candidates = aevolve(dag, analysis, pattern)             # 生成多候选局部 mutation
      best = argmax_fitness(candidates, fixed_eval_tasks)      # minibatch 选最优候选

      jacobian.update(outcome)                                 # 累积经验

      if best.score > incumbent_score:
          dag = best.dag
          incumbent_traces = best.traces
          incumbent_profiles = best.profiles
          incumbent_score = best.score
      else:
          insight = reflect(...)                               # 反思教训
          negatives.append(insight)
          if should_evolve(): catalog_evolve()
```

### 关键组件

| 模块 | 文件 | 职责 |
|------|------|------|
| **Orchestrator** | `orchestrator.py` | 单循环控制器：诊断 → 推荐 → 修复 → 评估 |
| **repair_loop** | `skills/repair_loop.py` | 无状态函数：`analyze()`, 多候选 diff-mutation `aevolve()`, `reflect()` |
| **PatternCatalog** | `skills/repair_patterns.py` | 动态修复模式目录，JSON 持久化，支持运行时增删改 |
| **RepairJacobian** | `skills/jacobian.py` | (FM group × 修复模式) → 成功率矩阵 |
| **CatalogEvolver** | `skills/evolution.py` | 当 pattern 反复失败 → LLM 改进 catalog |
| **NegativesStore** | `skills/negatives.py` | 按 FM group 持久化失败教训 |

### Online Eval Protocol

- **在线 active eval**: 若设置 `eval_tasks_per_round`，每轮从 train task 中按 benchmark 做均匀平衡抽样，避免固定前缀子集导致的系统性过拟合
- **候选池评估**: 默认每轮 1 个候选，直接绑定 Jacobian top-1 pattern。这样一轮只评估一个新 DAG，再决定是否采纳，轮次语义与 OpenEvolve 的单 candidate iteration 对齐
- **Shadow gate**: 每 5 轮额外抽一个不与 active 重叠的 balanced shadow batch；shadow 只比较正确率（accuracy），不再比较 combined fitness
- **Shadow-triggered meta evolve**: 若同一 FM group 连续 3 次在 shadow gate 被拒绝，触发一次 catalog meta-evolution；普通 active-only 失败不再直接触发 meta evolve
- **Diagnosis-driven meta evolve**: shadow gate 被拒绝时，会对回退样本生成结构化 diagnosis bundle（primary FM、root cause、agent、dag_component、候选改动摘要）；CatalogEvolver 必须先读这份 bundle，再判断为什么候选没成功、应如何调整 pattern catalog
- **Adopt 准则**: 仅当 best candidate fitness 严格高于 incumbent fitness 时才采纳
- **与 OpenEvolve 对齐**: online 阶段不做 “同轮 before/after 双跑”，而是 baseline 初始化一次 + 后续每轮只评估新候选；OpenEvolve baseline 侧显式将 `max_parallel_iterations` 设为 1，避免并发 iteration 带来的吞吐对齐但轮次语义不对齐
- **OpenEvolve split 对齐**: `run_ag2_mathchat_openevolve.py` 显式把 train-side eval prompts 传给 evaluator，避免 evaluator 自己回退到全集前缀样本

### Diagnosis Semantics

- **单题唯一主因**: `Diagnoser` 仍保留 A-F 多标签作为辅助诊断信息，但会额外为每条失败 trace 选出唯一一个 `primary_fm_id`
- **单题唯一 root cause**: localization 默认只对这个 `primary_fm_id` 执行，因此每道错题只有一个主 root cause，保存在 `FMProfile.primary_localization`
- **优化目标对齐**: `rank_fm_groups()` 和 `extract_failure_signatures()` 现在按 `primary_fm_id` 计数，不再让同一道题同时给多个 FM 贡献支持度
- **持久化兼容**: diagnose artifact 会额外保存 `primary_fm_id` 和 `primary_localization`，复用 diagnose 目录时保持同样语义

### Repair Jacobian 系统

用结构化的 (FM group × RepairPattern) → 成功率矩阵：

- **FailureSignature**: online key 只用 `fm_group`；`dag_component` / `agent` 仅保留作离线元数据
- **RepairPattern**: 可枚举的修复策略（13 个默认 pattern + 运行时可增加）
- **PatternCatalog**: 动态 pattern 目录，加载/保存 JSON，支持 `add_pattern()` / `update_pattern()` / `effective` 标记
- **RepairJacobian**: 矩阵存储 + 推荐 + 更新，持久化到 `library_store/jacobian/`
- **冷启动**: 优先加载 `library_store/jacobian/data_driven_priors.json`（离线 OpenEvolve trace 分析产出），无则回退到手写启发式 prior
- **推荐冷却**: 同一 FM group 下若某个 assigned pattern 连续失败达到阈值，下一轮临时 cooldown 1 轮，避免反复推荐同一方向
- **Observed attribution**: online 不依赖 LLM 解释编辑；而是对 `old_dag -> new_dag` 做确定性结构差分，再映射成有限 pattern family
- **数据驱动**: 随经验积累，贝叶斯混合逐渐从 prior 过渡到经验 success_rate

### Catalog Meta-Evolution

当 pattern 连续失败达到阈值（默认 `META_EVOLVE_FAILURE_THRESHOLD=3`）：
1. CatalogEvolver 为 LLM 准备 catalog.json + failure_summary.md + meta_context.md
2. LLM 通过 `add_pattern` / `update_pattern` / `bash` 工具修改目录
3. 可添加新 pattern、修改 description、标记无效 pattern
4. 保存更新后的 catalog 到当前 topology 的 catalog 路径（旧入口默认 `library_store/pattern_catalog.json`）

### Topology-Isolated Experience Store

为防止不同 MAS topology 之间的经验污染，`run_skill.py` / `run_openevolve.py` /
`analyze_openevolve_traces.py` 现在支持按 topology 隔离持久化状态：

- `library_store/<topology>/negatives/`
- `library_store/<topology>/jacobian/`
- `library_store/<topology>/recipes/`
- `library_store/<topology>/meta_evolve_traces/`
- `library_store/<topology>/pattern_catalog.json`

隔离规则：
- `Orchestrator(topology=...)` 会把在线 repair、Jacobian 更新、catalog evolve 都绑定到对应 topology 子目录
- offline analysis 写入的 recipe / Jacobian warmup 必须被同 topology 的 online repair 读取
- 顶层全局目录 `library_store/negatives/`、`library_store/jacobian/`、`library_store/pattern_catalog.json` 仍保留给旧入口，作为向后兼容路径

## LLM Concurrency Control

按模型家族做全局限流：
- 每个模型家族共享 Semaphore（限并发）+ 60 秒滑动窗口（限 RPM）
- 内置家族：MiniMax M2.5 (96 并发)、GLM-5 (48 并发)、default (64 并发)
- 同时提供同步和异步客户端

## 6-Group Taxonomy

唯一 active taxonomy：
- **A**: Instruction Non-Compliance
- **B**: Execution Loop / Stuck
- **C**: Context Loss
- **D**: Communication Failure
- **E**: Task Drift / Reasoning Error
- **F**: Verification Failure

定义: `src/optpilot/data/fm_taxonomy_6group.py`（含 `analyze_hint` 字段）
FM Classifier: MiniMax M2.5（100 条 blind trace 校准，与人类专家容忍率 89.9%–98.0%）

## AG2 MathChat DAG

基于 MAST Appendix L 改成更稳定的 context-preserving 3-agent workflow：
- `dags/ag2_mathchat.yaml`
- Agents: Agent_Problem_Solver, Agent_Code_Executor, Agent_Verifier
- Routing: `Problem_Solver -> Code_Executor -> Verifier`
- Context preservation: USER 原题面通过非触发边预加载到三个 agent；Verifier 不再把 speaker-routing 文本回灌给下游 agent
- 终止: `SOLUTION_FOUND` 关键词 → FINAL

## Official Benchmarks

在线评测使用官方 benchmark + ground-truth scoring：
- `cais/mmlu`: 多选知识题
- `opencompass/AIME2025`: 竞赛数学（整数答案）
- `lscpku/OlympiadBench-official` (`maths_en_no_proof`): 开放式数学题

评分: 从 `\boxed{...}` 提取答案，精确匹配或 SymPy 符号匹配。

## Offline Cold-Start Pipeline (2026-03-29)

离线阶段用 OpenEvolve 盲进化作为冷启动，然后分析有效 trace 归纳数据驱动先验：

```
离线:
  OpenEvolve 100 iter (train set only, 盲进化)
       ↓
  analyze_openevolve_traces.py:
    1. 从 programs/*.json 提取 parent→child mutation
    2. 筛选 train score 提升的 mutation
    3. 在 held-out test set 上后验评估（防 overfit）
    4. 只保留 test 也提升的 "真正有效" mutation
    5. diff DAG → infer_observed_pattern_from_dags() 分类
    6. 输出: data_driven_priors.json + jacobian_warmup.jsonl
       ↓
  暖启动 Jacobian matrix + 替换手写冷启动 prior

在线:
  Jacobian.recommend() 从第一轮起就有数据驱动的推荐
```

### Current Multi-Topology Cold-Start Setup

当前用两条简化 topology 做低成本冷启动验证：

- `simple_star`:
  - benchmark = GAIA
  - role = 简化版 star topology，用于搜集 general-task MAS 结构先验
- `simple_hier`:
  - benchmark = SWE-bench Lite
  - role = 简化版 hierarchical topology，用于搜集 code-repair MAS 结构先验

实验顺序是：

```text
run_openevolve.py
  -> analyze_openevolve_traces.py
  -> run_skill.py
```

设计意图：
- 先在更小、更稳定、更便宜的 topology 上积累结构先验
- 再把这些先验变成 topology-isolated 的 Jacobian warmup / recipes / catalog updates
- 最终验证 diagnosis-driven repair 是否比 blind evolution 更高效、更可迁移

### Role Of OpenEvolve

在当前系统里，OpenEvolve 不是最终优化器，而是 cold-start prior generator：

- 它负责在大而离散的 DAG 设计空间里做盲搜索
- 它的输出要经过 offline posterior filtering 才会进入 OptPilot 经验库
- 最终在线优化仍由 `Orchestrator + Diagnoser + RepairJacobian + CatalogEvolver`
  驱动，而不是继续依赖 blind search

### Repair Recipe Library

从有效 mutation 中归纳的修复原则，存储在 `library_store/recipes/`：

```python
RepairRecipe:
    precondition: "Verifier 不加验证就接受 Solver 的答案"
    action: "要求 Verifier 独立重新求解，再对比"
    root_cause: "Verifier 对 Solver 形成信任依赖"
    n_effective: 8    # 证据强度
    avg_test_delta: 0.12
```

三级推荐：
1. **Jacobian**: FM group → change_type (粗) + applied decay (防重复)
2. **Recipe**: (fm_group, change_type) → 具体修复原则 (中)
3. **Evolve prompt**: 注入 top-3 recipe 作为经验参考

### Applied Decay

已成功应用的 change_type 在后续推荐中 score × 0.3，鼓励多样性。
跟踪粒度：session 级别（`session_applied_patterns`），不跨 session 累积。

关键设计决策：
- **后验过滤而非双评估**: 进化过程不接触 test set，无 leakage
- **test set 必须提升才算有效**: 对齐优化方向，防止 train overfit
- **先验来源分离**: `data_driven_priors.json` 可独立更新，不需要改代码
- **Recipe 粒度**: 比 raw diff 抽象（可复用）、比 change type 具体（可操作）
- **Recipe/Meta-evolve 隔离**: topology-aware 实验读取和写入 `library_store/<topology>/...`，避免 GAIA / SWE-bench / MathChat 互相污染经验

## Architecture Change Policy

Any meaningful architecture decision or repository-structure change must be recorded in
this file.
