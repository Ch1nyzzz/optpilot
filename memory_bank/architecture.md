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
│   │   ├── repair_loop.py     # Stateless functions: analyze(), aevolve(), reflect()
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
│   └── chatdev.yaml           # ChatDev v1 workflow (reference)
├── experiments/               # Experiment pipelines
│   ├── run_ag2_mathchat_skill.py  # Primary entry: AG2 × benchmarks × Jacobian loop
│   └── ...                    # offline/online pipelines, ablations
├── data/annotations/          # Blind/manual labeling packs for 6-group evaluation
├── tests/                     # Regression tests
├── library_store/             # Persistent experience
│   ├── negatives/             # Per-FM-group ReflectInsight history (JSON)
│   ├── jacobian/              # Jacobian matrix + outcomes (JSON/JSONL)
│   ├── pattern_catalog.json   # Evolved pattern catalog (JSON, runtime-modified)
│   ├── skill_agent_traces/    # Per-FM-group tool-calling transcripts
│   ├── meta_evolve_traces/    # Catalog evolution transcripts
│   ├── offline_hints/         # Offline repair hints
│   ├── offline_skills/        # Offline wrapped skills
│   ├── online_hints/          # Online repair hints
│   └── online_skills/         # Online wrapped skills
├── results/                   # Experiment results, metrics, and per-run trace artifacts
└── memory_bank/               # Project documentation
```

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
      result = aevolve(dag, analysis, pattern)                 # LLM 修 DAG

      candidate_traces = run(result.dag, fixed_eval_tasks)     # 每轮只评估新候选一次
      candidate_profiles = classify(candidate_traces)
      candidate_score = fitness(candidate_traces, candidate_profiles)

      jacobian.update(outcome)                                 # 累积经验

      if candidate_score > incumbent_score:
          dag = result.dag
          incumbent_traces = candidate_traces
          incumbent_profiles = candidate_profiles
          incumbent_score = candidate_score
      else:
          insight = reflect(...)                               # 反思教训
          negatives.append(insight)
          if should_evolve(): catalog_evolve()
```

### 关键组件

| 模块 | 文件 | 职责 |
|------|------|------|
| **Orchestrator** | `orchestrator.py` | 单循环控制器：诊断 → 推荐 → 修复 → 评估 |
| **repair_loop** | `skills/repair_loop.py` | 无状态函数：`analyze()`, `aevolve()`, `reflect()` |
| **PatternCatalog** | `skills/repair_patterns.py` | 动态修复模式目录，JSON 持久化，支持运行时增删改 |
| **RepairJacobian** | `skills/jacobian.py` | (FM group × 修复模式) → 成功率矩阵 |
| **CatalogEvolver** | `skills/evolution.py` | 当 pattern 反复失败 → LLM 改进 catalog |
| **NegativesStore** | `skills/negatives.py` | 按 FM group 持久化失败教训 |

### Online Eval Protocol

- **固定 eval 子集**: 若设置 `eval_tasks_per_round`，取 train task 的固定前缀子集，所有 round 复用同一批 online eval task
- **单次候选评估**: 每轮只运行一次 candidate DAG；当前 incumbent 的 traces / profiles / score 在 round 之间缓存
- **Adopt 准则**: 仅当 candidate fitness 严格高于 incumbent fitness 时才采纳
- **与 OpenEvolve 对齐**: online 阶段不做 “同轮 before/after 双跑”，而是 baseline 初始化一次 + 后续每轮只评估新候选

### Repair Jacobian 系统

用结构化的 (FM group × RepairPattern) → 成功率矩阵：

- **FailureSignature**: online key 只用 `fm_group`；`dag_component` / `agent` 仅保留作离线元数据
- **RepairPattern**: 可枚举的修复策略（13 个默认 pattern + 运行时可增加）
- **PatternCatalog**: 动态 pattern 目录，加载/保存 JSON，支持 `add_pattern()` / `update_pattern()` / `effective` 标记
- **RepairJacobian**: 矩阵存储 + 推荐 + 更新，持久化到 `library_store/jacobian/`
- **冷启动**: 按 FM group 使用启发式 prior 打分（例如 B 偏 loop，C/D 偏 context/edge，F 偏 verification）
- **Observed attribution**: online 不依赖 LLM 解释编辑；而是对 `old_dag -> new_dag` 做确定性结构差分，再映射成有限 pattern family
- **数据驱动**: 随经验积累，贝叶斯混合逐渐从 prior 过渡到经验 success_rate

### Catalog Meta-Evolution

当 pattern 连续失败达到阈值（默认 `META_EVOLVE_FAILURE_THRESHOLD=3`）：
1. CatalogEvolver 为 LLM 准备 catalog.json + failure_summary.md + meta_context.md
2. LLM 通过 `add_pattern` / `update_pattern` / `bash` 工具修改目录
3. 可添加新 pattern、修改 description、标记无效 pattern
4. 保存更新后的 catalog 到 `library_store/pattern_catalog.json`

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

基于 MAST 论文 Appendix L 的官方 3-agent GroupChat：
- `dags/ag2_mathchat.yaml`
- Agents: Agent_Problem_Solver, Agent_Code_Executor, Agent_Verifier
- Turn Counter (max=5)，仅 `loop: exit` 边到 FINAL（被动计数器，count < max 时不触发下游）
- 终止: `SOLUTION_FOUND` 关键词 → FINAL

## Official Benchmarks

在线评测使用官方 benchmark + ground-truth scoring：
- `cais/mmlu`: 多选知识题
- `opencompass/AIME2025`: 竞赛数学（整数答案）
- `lscpku/OlympiadBench-official` (`maths_en_no_proof`): 开放式数学题

评分: 从 `\boxed{...}` 提取答案，精确匹配或 SymPy 符号匹配。

## Architecture Change Policy

Any meaningful architecture decision or repository-structure change must be recorded in
this file.
