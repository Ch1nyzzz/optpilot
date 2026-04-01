# 代码变更汇总：简化拓扑 + 多拓扑支持

**日期**: 2026-03-31
**目标**: 构建简化版 Star/Hierarchical DAG (3-agent)，并为整个实验流程添加多拓扑隔离支持。

---

## 一、背景与动机

原有的 Magentic Star (5-agent) 和 HyperAgent Hierarchical (4-agent) 复杂度过高：
- Magentic: 5 个 agent 顺序执行，carry_data 全开，单轮 context ~160K tokens
- HyperAgent: 4-agent chain + 无界 feedback loop，SWE-bench prompt 超 131K token limit

需要：
1. 简化 DAG 保留核心信息流模式，在自然匹配的 benchmark 上快速实验
2. 多拓扑之间 Jacobian 矩阵、negatives、pattern catalog 存储隔离，防止经验污染

---

## 二、新建文件 (6 files)

### 1. `dags/simple_star_gaia.yaml`
**目的**: 简化版 Star 拓扑 DAG 定义

- 3 agents: Orchestrator (hub) + Researcher (spoke, web_search/read_document) + Solver (spoke, calculator/python_exec)
- 显式 LoopCounter (max_iterations=2)，替代原版隐式无界循环
- USER context 只给 Orchestrator（原版广播到 5 个 agent）
- max_tokens=4096/agent（原版 16384），worst case ~36K tokens
- 匹配 GAIA benchmark

### 2. `dags/simple_hierarchical_swebench.yaml`
**目的**: 简化版 Hierarchical 拓扑 DAG 定义

- 3 agents: Planner (top, search_code/list_files/read_file) + Coder (middle, read_file/edit_file) + Verifier (bottom, run_command/read_file)
- 显式 LoopCounter (max_iterations=3)，适配 SWE-bench 的 fix-test-refix 循环
- USER context 只给 Planner（原版广播到 4 个 agent，SWE-bench prompt 长达 900 words）
- Planner 合并了原 Navigator 的职责（减少 1 个 agent）
- max_tokens=4096/agent，worst case ~43K tokens
- 匹配 SWE-bench Lite benchmark

### 3. `experiments/openevolve_initial_dag_simple_star.py`
**目的**: OpenEvolve 进化搜索的初始 Python 程序（Star）

- `build_dag()` 函数返回与 YAML 等价的 dict 结构
- 遵循 `EVOLVE-BLOCK-START` / `EVOLVE-BLOCK-END` 标记（SkyDiscover 要求）

### 4. `experiments/openevolve_initial_dag_simple_hier.py`
**目的**: OpenEvolve 进化搜索的初始 Python 程序（Hierarchical）

- 同上

### 5. `experiments/run_skill.py`
**目的**: 统一多拓扑 Jacobian-driven repair 实验入口（替代 `run_ag2_mathchat_skill.py`）

- 支持 6 种拓扑: ag2, appworld, hyperagent, magentic, simple_star, simple_hier
- `_load_topology()` 函数根据 topology 参数加载对应的 benchmark + scorer + tool_setup_fn
- 构建 Orchestrator 时传入 `topology` 参数，自动使用拓扑隔离的存储路径
- simple_star 复用 GAIA benchmark + magentic_tools
- simple_hier 复用 SWE-bench benchmark + hyperagent_tools

### 6. `docs/changelog_simplified_topologies.md`
**目的**: 本文件，记录所有变更

---

## 三、修改文件 (7 files)

### 1. `src/optpilot/config.py`
**目的**: 添加拓扑隔离的路径助手函数

新增 5 个函数：
- `topology_library_dir(topology)` → `library_store/<topology>/`
- `topology_jacobian_dir(topology)` → `library_store/<topology>/jacobian/`
- `topology_negatives_dir(topology)` → `library_store/<topology>/negatives/`
- `topology_catalog_path(topology)` → `library_store/<topology>/pattern_catalog.json`
- `topology_recipes_dir(topology)` → `library_store/<topology>/recipes/`

纯增量改动，原有的全局 `JACOBIAN_DIR`、`NEGATIVES_DIR` 等常量不变，向后兼容。

### 2. `src/optpilot/orchestrator.py`
**目的**: Orchestrator 支持按拓扑隔离存储

`__init__` 新增 `topology: str | None = None` 参数：
- 传入 topology 时：NegativesStore、PatternCatalog、RepairJacobian 使用拓扑专属路径
- 不传时（`None`）：保持原有全局路径行为，向后兼容

### 3. `experiments/run_openevolve.py`
**目的**: OpenEvolve 实验支持简化拓扑

- `TOPOLOGIES` dict 新增 `simple_star` 和 `simple_hier` 两项
- benchmark loading elif 链新增两个分支：
  - `simple_star`: 复用 GAIA loader + magentic_tools
  - `simple_hier`: 复用 SWE-bench loader + hyperagent_tools

### 4. `experiments/openevolve_evaluator_multi.py`
**目的**: OpenEvolve evaluator 支持简化拓扑

`_LOADERS` dict 新增：
- `"simple_star": _load_magentic` （复用 GAIA benchmark + magentic tools）
- `"simple_hier": _load_hyperagent` （复用 SWE-bench + hyperagent tools）

### 5. `experiments/analyze_openevolve_traces.py`
**目的**: 离线分析脚本支持多拓扑

- 新增 `--topology` 参数 (argparse choices 从 4 项扩展到 6 项)
- 新增 `_load_benchmark_for_topology()` 函数，统一各拓扑的 benchmark 加载逻辑
- `simple_star` / `simple_hier` 分支递归调用对应的 magentic / hyperagent 加载器
- `apply_jacobian_warmup` 改为写入拓扑隔离的 Jacobian 目录
- Recipe 安装改为写入拓扑隔离的 recipes 目录
- 输出目录名包含 topology: `offline_analysis_{topology}_{timestamp}`

### 6. `dags/hyperagent_hierarchical.yaml`
**目的**: 修复原版 HyperAgent context 爆炸问题

- `max_tokens` 从 16384 降到 4096（4 个 agent 全部）
- 原因：4-agent chain 每个 16K，carry_data 累积后超 131K limit

---

## 四、存储隔离架构

改动前（全局共享）：
```
library_store/
  jacobian/          ← 所有拓扑共用
  negatives/         ← 所有拓扑共用
  pattern_catalog.json
  recipes/
```

改动后（按拓扑隔离）：
```
library_store/
  ag2/jacobian/、negatives/、pattern_catalog.json、recipes/
  simple_star/jacobian/、negatives/、...
  simple_hier/jacobian/、negatives/、...
  hyperagent/...
  magentic/...
  jacobian/           ← 原有全局目录保留，向后兼容
  negatives/          ← 同上
```

---

## 五、简化拓扑设计对比

| 属性 | AG2 Linear | 原 Star (5-agent) | 简化 Star (3-agent) | 原 Hier (4-agent) | 简化 Hier (3-agent) |
|---|---|---|---|---|---|
| Agents | 3 | 5 | 3 | 4 | 3 |
| Edges | 8 | 15 | 10 | 10 | 8 |
| Loop bound | 无 | 无 (隐式) | 2 (显式) | 无 (隐式) | 3 (显式) |
| USER 广播 | 3 agents | 5 agents | 1 agent | 4 agents | 1 agent |
| max_tokens | 16384 | 16384 | 4096 | 4096 | 4096 |
| Worst-case | ~50K | ~160K+ | ~36K | ~130K+ | ~43K |
| Benchmark | Math | GAIA | GAIA | SWE-bench | SWE-bench |

---

## 六、完整实验流程

```bash
# Step 1: OpenEvolve 冷启动盲进化
python -m experiments.run_openevolve --topology simple_star --train 80 --test 80 --iterations 50
python -m experiments.run_openevolve --topology simple_hier --train 100 --test 100 --iterations 50

# Step 2: 离线分析 → 提取先验 + Jacobian warmup + repair recipes
python -m experiments.analyze_openevolve_traces \
    --openevolve-dir results/<openevolve_output_dir> \
    --topology simple_star
python -m experiments.analyze_openevolve_traces \
    --openevolve-dir results/<openevolve_output_dir> \
    --topology simple_hier

# Step 3: Jacobian-driven targeted repair（利用积累的先验）
python -m experiments.run_skill --topology simple_star --rounds 50
python -m experiments.run_skill --topology simple_hier --rounds 50

# 未来: 将先验迁移到原版复杂拓扑验证
```
