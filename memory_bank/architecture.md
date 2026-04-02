# Architecture

## Current Repository Structure

```
optpilot/
├── src/optpilot/              # Core library
│   ├── config.py              # Global configuration (env, models, paths, rate limits)
│   ├── models.py              # Core data models
│   ├── llm.py                 # LLM interface (Together AI, sync + async, rate limiting)
│   ├── dag/                   # DAG abstraction and execution
│   │   ├── core.py            # MASDAG, DAGNode, DAGEdge + topology feature detection
│   │   └── executor.py        # DAGExecutor: lightweight BFS workflow engine
│   ├── modules/               # Core modules
│   │   ├── base_runner.py     # MASRunner abstract base
│   │   ├── runner.py          # OptPilotRunner (DAG execution + benchmark scoring)
│   │   └── diagnoser.py       # FM diagnosis via LLM (6-group, concurrent)
│   ├── skills/                # Prior store and pattern analysis
│   │   ├── jacobian.py        # RepairJacobian matrix (failure×pattern → success rate)
│   │   ├── recipes.py         # Repair recipe library
│   │   └── repair_patterns.py # PatternCatalog + mutation classification
│   ├── data/                  # Benchmarks and taxonomy
│   │   ├── fm_taxonomy_6group.py  # 6-group FM taxonomy (A-F)
│   │   ├── benchmarks.py         # MMLU, AIME 2025, OlympiadBench
│   │   ├── benchmarks_humaneval.py # HumanEval (code generation)
│   │   ├── benchmarks_appworld.py
│   │   ├── benchmarks_gaia.py
│   │   └── benchmarks_swebench.py
│   └── tools/                 # External tool integrations per target MAS
│       ├── agentcoder_tools.py  # Python code execution
│       ├── appworld_tools.py    # AppWorld API
│       ├── hyperagent_tools.py  # Git repo + shell execution
│       └── magentic_tools.py    # Web search + Python exec
├── experiments/               # Experiment entry points
│   ├── run_openevolve.py            # Blind / prior-guided OpenEvolve
│   ├── analyze_openevolve_traces.py # Prior extraction from traces
│   ├── openevolve_evaluator_multi.py # Multi-target fitness evaluator
│   ├── openevolve_evaluator.py       # Single-target evaluator (AG2)
│   ├── openevolve_initial_dag*.py    # Initial DAG builders per target
│   ├── run_ag2_mathchat_baseline.py  # Baseline measurement
│   └── openevolve_config.yaml        # SkyDiscover configuration
├── library_store/             # Global experience store
│   ├── jacobian/              # Repair effectiveness matrix
│   ├── recipes/               # Repair recipes (per FM group)
│   ├── negatives/             # Lessons from failed repairs
│   └── pattern_catalog.json   # Evolved pattern catalog
├── tests/                     # Regression tests (pytest)
└── memory_bank/               # Project documentation
```

版本管理约定：
- `logs/`、`results/` 为运行生成物，默认不纳入 Git。

## Core Architecture: MAS-as-DAG

任何 MAS 系统统一表示为 MASDAG（有向图，支持循环）：
- **DAGNode** = Agent (role, prompt, config) | Literal (config.content) | LoopCounter (config.max_iterations) | Passthrough
- **DAGEdge** = Communication (source→target, trigger, condition, carry_data, loop)
- **Optimization** = YAML-level DAG modification by evolutionary search

MASDAG 是唯一的核心表示，既用于定义、又用于执行、又用于进化。

## DAG Executor

自研轻量执行引擎：

1. 确定 start nodes（metadata.start），注入 task prompt
2. BFS 式执行 ready queue 中的节点
3. Agent 节点：`node.prompt or node.role` 作为 system prompt，支持 `config.params` 嵌套参数
4. Literal 节点：`config.content` 或 `node.prompt` 作为输出
5. 根据边条件（keyword matching）传播输出到下游节点
6. loop_counter 节点管理循环次数和退出

Loop counter 约定：
- 优先读取 edge config 中的 `loop: continue|exit` 显式标注
- 若未显式标注，则拓扑推断
- 仅有 exit_edges 而无 continue_edges 时（被动计数器），count < max_iter 时不触发任何下游

## Two-Phase Pipeline

### Phase 1: Blind Cold-Start (OpenEvolve)

```
run_openevolve.py:
  load initial DAG (openevolve_initial_dag_*.py)
  SkyDiscover MAP-Elites evolutionary search
    → openevolve_evaluator_multi.py: evaluate candidate DAGs on benchmark tasks
    → iterate N generations (blind mutation, no priors)
  output: programs/*.json (evolved DAG population)
```

### Phase 2: Prior Extraction

```
analyze_openevolve_traces.py:
  1. 从 programs/*.json 提取 parent→child mutation
  2. 筛选 train score 提升的 mutation
  3. 在 held-out test set 上后验评估（防 overfit）
  4. 只保留 test 也提升的 "真正有效" mutation
  5. diff DAG → infer_observed_pattern_from_dags() 分类
  6. 从 DAG 自动提取拓扑特征 (has_hub, has_loop)
  7. 输出:
     - data_driven_priors.json → Jacobian warmup
     - jacobian_warmup.jsonl → 带拓扑特征的经验记录
     - recipes.json → 蒸馏的修复 recipes
```

### Phase 3: Guided vs Blind Comparison (主实验)

并行对比两组：
- **Blind**: `run_openevolve.py` 不带 `--with-priors`
- **Guided**: `run_openevolve.py --with-priors`，evaluator feedback 注入先验

比较指标：sample efficiency, held-out test accuracy, FM reduction

## Global Experience Store

经验存储在全局 `library_store/` 目录下，**不按 target MAS 或 control flow 分目录**。

拓扑区分通过自动检测实现：
- `MASDAG.extract_topology_features()` 从 DAG 结构自动检测 `has_hub` 和 `has_loop`
  - `has_hub`: 某个 agent 的传递出度（经 trigger edge）≥ 半数其他 agent
  - `has_loop`: 存在 `loop_counter` 节点或 `loop:` 边标注
- `FailureSignature.signature_key()` = `"B:hub=0:loop=1"` — 不同拓扑自然分到不同 Jacobian 行
- 新增 target MAS 无需手动映射，拓扑特征自动匹配已有经验

经验层级：
1. **Jacobian matrix** (`library_store/jacobian/matrix.json`): (FM group × 拓扑 × change family) → 成功率
2. **Data-driven priors** (`library_store/jacobian/data_driven_priors.json`): 冷启动先验（离线提取）
3. **Recipes** (`library_store/recipes/*.json`): 按 FM group 的修复 recipe
4. **Negatives** (`library_store/negatives/`): 失败教训
5. **Pattern catalog** (`library_store/pattern_catalog.json`): 修复模式目录

## 6-Group FM Taxonomy

唯一 active taxonomy：
- **A**: Instruction Non-Compliance
- **B**: Execution Loop / Stuck
- **C**: Context Loss
- **D**: Communication Failure
- **E**: Task Drift / Reasoning Error
- **F**: Verification Failure

定义: `src/optpilot/data/fm_taxonomy_6group.py`
FM Classifier: MiniMax M2.5（100 条 blind trace 校准，与人类专家容忍率 89.9%–98.0%）

## LLM Concurrency Control

按模型家族做全局限流：
- 每个模型家族共享 Semaphore（限并发）+ 60 秒滑动窗口（限 RPM）
- 内置家族：MiniMax M2.5 (96 并发)、default (64 并发)
- 同时提供同步和异步客户端

## Architecture Change Policy

Any meaningful architecture decision or repository-structure change must be recorded in
this file.
