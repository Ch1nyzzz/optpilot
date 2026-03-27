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
│   ├── orchestrator.py        # Skill dispatch loop (diagnose → parallel skills → adopt DAG)
│   ├── registry.py            # Runner factory
│   ├── tracking.py            # Experiment tracking (W&B integration)
│   ├── skills/                # Skill Workflows (one per FM group)
│   │   ├── base.py            # BaseSkill ABC + GenericSkill + run() template
│   │   ├── registry.py        # FM group → Skill class mapping + dynamic loading
│   │   ├── negatives.py       # ReflectInsight persistence (JSON per FM group)
│   │   ├── evolution.py       # Skill meta-evolution (LLM modifies Skill code)
│   │   └── skill_{a-f}.py     # 6 concrete skills (A-F)
│   ├── dag/                   # DAG abstraction and execution
│   │   ├── core.py            # MASDAG, DAGNode, DAGEdge + YAML serialization
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
│       ├── fm_taxonomy_6group.py # Canonical A-F failure-group taxonomy
│       ├── loader.py          # Load MAST-Data traces (normalizes to A-F)
│       └── benchmarks.py      # Official benchmark loaders + ground-truth scorers
│                              # (MMLU, AIME 2025, OlympiadBench)
├── dags/                      # MASDAG workflow definitions (YAML)
│   ├── ag2_mathchat.yaml      # AG2 MathChat 3-agent GroupChat (primary)
│   └── chatdev.yaml           # ChatDev v1 workflow (reference)
├── experiments/               # Experiment pipelines
│   ├── run_ag2_mathchat_skill.py  # Primary entry: AG2 × benchmarks × Skill Workflows
│   └── ...                    # offline/online pipelines, ablations
├── data/annotations/          # Blind/manual labeling packs for 6-group evaluation
├── tests/                     # Regression tests
├── library_store/             # Repair library + negatives + evolved skills
│   ├── negatives/             # Per-FM-group ReflectInsight history (JSON)
│   ├── evolved_skills/        # Meta-evolved Skill source files
│   ├── offline_hints/         # Offline repair hints
│   ├── offline_skills/        # Offline wrapped skills
│   ├── online_hints/          # Online repair hints
│   └── online_skills/         # Online wrapped skills
├── results/                   # Experiment results and metrics
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

## Skill Workflow Architecture (2026-03-26)

取代了原来的 Optimizer+Distiller+WrapUp：

| 模块 | 职责 |
|------|------|
| **Orchestrator** | 诊断 → 按 FM 频率分发给 Skill Workflows（支持并行） |
| **Runner** | 用 DAGExecutor 执行 MASDAG，收集 trace + benchmark scoring |
| **Diagnoser** | 6-group FM 分类 (MiniMax M2.5) + agent/step 定位 |
| **Skill Workflows (A-F)** | 6 个 Python 类，每个 FM group 一个完整的修复 agent |
| **SkillEvolver** | Skill 连续失败 ≥3 次 → LLM 修改 Skill 源码 → 动态加载 |

### Skill 内部闭环

```
for outer_round in range(MAX_OUTER_ROUNDS=3):
    analyze(dag, traces, profiles, negatives) → AnalysisResult
    for inner_iter in range(MAX_INNER_ITERS=5):
        evolve(dag, analysis, negatives, history) → EvolveResult (YAML-level)
        run_batch(proposal_tasks) → new_traces
        diagnose → fm_rate
        if fm_rate < 0.2 or no_improvement: break
    run_batch(validation_tasks, original_dag) → before_val  # cached at start
    run_batch(validation_tasks, repaired_dag) → after_val
    if judge(before_val, after_val): return success
    reflect(...) → ReflectInsight → negatives.append(insight)
```

关键设计：
- **Validation 基线缓存**：before_val_traces 在整个 run() 生命周期只算一次
- **pass_rate 使用 task_score**：优先使用 ground-truth benchmark 评分（而非 task_success 完成率）
- **Judge 条件**：FM rate 下降 AND pass_rate 上升
- **SkillBudget**：max_llm_calls=30 / max_batch_runs=10 / max_wall_time_s=600

### 并行 Skill 执行

不同 FM group 的 Skill 可以并行优化（ThreadPoolExecutor），各自工作在 DAG 的独立 deepcopy 上。
Orchestrator 收集所有结果，选最佳成功结果采纳到主 DAG。

### Meta-Evolution

当 Skill 连续失败 ≥3 次：
1. SkillEvolver 读取 Skill 的 Python 源码 + 累积 ReflectInsight
2. LLM 生成修改版代码
3. compile() 语法检查 → 保存到 library_store/evolved_skills/
4. importlib 动态加载替换注册表中的旧版本

## Online Validation Protocol

proposal / validation 分离：
- proposal 子集：命中目标 FM 的 trace 的前半部分，用于聚合证据并生成 repair
- validation 子集：所有剩余 trace（不与 proposal 重叠），至少保留一个命中该 FM 的 holdout

验证标准：
- 目标 FM 在 validation 上下降
- pass_rate 上升（使用 ground-truth benchmark scorer）

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

定义: `src/optpilot/data/fm_taxonomy_6group.py`
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
