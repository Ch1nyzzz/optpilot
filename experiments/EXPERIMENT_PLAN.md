# Cold-Start Experiment Plan

## 实验目标

验证 **data-driven priors 能否提升 MAS 进化搜索的效率和质量**。

具体来说，我们要回答：
1. 从盲进化 trace 中提取的结构先验（Jacobian warmup、recipes、pattern catalog）能否提升 guided evolution 的 sample efficiency？
2. 不同 DAG 拓扑（linear, star, 有/无循环）在不同任务类型上的进化表现是否有显著差异？
3. 先验的跨拓扑迁移能力如何？

## 实验设计

### Phase 1: Blind Cold-Start（当前阶段）

4 种简化拓扑 × 4 种 benchmark 的全组合盲进化实验（共 16 组），收集进化 trace 用于后续 prior extraction。

**拓扑：**

| 拓扑 | 结构 | 节点数 | 特点 |
|---|---|---|---|
| `linear` | A → B | 2 | 最简单的串行 pipeline |
| `linear_loop` | A → B → LoopCounter → A | 2+loop | 串行 + 迭代改进 |
| `star` | Hub → {Worker1, Worker2, ...} → Hub | 4-5 | 中心调度 + 并行 worker |
| `star_loop` | star + LoopCounter | 4-5+loop | 中心调度 + 迭代 |

**Benchmark：**

| Benchmark | 类型 | 来源 | 难度 | Baseline 正确率 |
|---|---|---|---|---|
| `math` | 数学推理 | AIME 2024/2025 + OlympiadBench + MMLU | 中-高 | 14-36% |
| `livecodebench` | 代码生成 | Codeforces/AtCoder/LeetCode 竞赛题 (v5, 880题) | 高 | 1-3% |
| `gaia` | 通用推理 | GAIA benchmark (165题, Level 1-3) | 中-高 | 17-23% |
| `swebench` | 代码修复 | SWE-bench Lite (GitHub issue 修复) | 高 | 24-41% |

**实验参数：**
- 每组 100 iterations
- Train set: 100 tasks, Test set: 100 tasks（不足时取全部）
- Eval tasks per iteration: 20（从 baseline 全错任务中选取）
- 执行模型: `openai/gpt-oss-120b`（MiniMax M2.5 via Together AI）
- 优化模型（OpenEvolve mutation）: `openai/gpt-oss-120b`
- 执行 temperature: 0.0（确定性输出，避免随机波动干扰评估）
- 优化 temperature: 0.7（保持搜索多样性）
- SkyDiscover 配置: 5 islands, population 40, MAP-Elites

**命令模板：**
```bash
python -u -m experiments.run_openevolve \
  --topology <topo> \
  --benchmark <bench> \
  --iterations 100 \
  --train 100 \
  --test 100 \
  --eval-tasks 20
```

### Phase 2: Prior Extraction（待 Phase 1 完成）

从 Phase 1 的进化 trace 中提取先验：
```bash
python -m experiments.analyze_openevolve_traces \
  --openevolve-dir results/<run_dir>/openevolve_output \
  --target-mas <topology>_<benchmark>
```

提取内容：
- Jacobian warmup：FM group → 修复操作的有效性矩阵
- Recipes：蒸馏后的修复方案（per FM group）
- Pattern catalog：有效的 DAG 结构模式

### Phase 3: Guided vs Blind 对比（待 Phase 2 完成）

使用 `--with-priors` 标志运行 guided evolution，与 Phase 1 的 blind 结果对比：
- Sample efficiency：达到相同 fitness 所需的 iteration 数
- Final fitness：100 iteration 后的最优分数
- Held-out test accuracy：在 test set 上的泛化能力
- FM reduction：各 failure mode 的减少程度

## 当前实验状态

**启动时间：** 2026-04-01 21:36

**运行状态：** Phase 1 — 16 组盲进化实验进行中

| 拓扑 \ Benchmark | math | livecodebench | gaia | swebench |
|---|---|---|---|---|
| linear | ▶ 运行中 | ▶ 运行中 | ▶ 运行中 | ▶ 运行中 |
| linear_loop | ▶ 运行中 | ▶ 运行中 | ▶ 运行中 | ▶ 运行中 |
| star | ▶ 运行中 | ▶ 运行中 | ▶ 运行中 | ▶ 运行中 |
| star_loop | ▶ 运行中 | ▶ 运行中 | ▶ 运行中 | ▶ 运行中 |

**Baseline 正确率（Phase 1 初始 DAG，temperature=0.0）：**

| 拓扑 \ Benchmark | math | livecodebench | gaia | swebench |
|---|---|---|---|---|
| linear | 32/100 | 3/100 | 17/100 | 39/100 |
| linear_loop | TBD | 2/100 | 21/100 | 39/100 |
| star | 36/100 | 1/100 | 23/100 | 41/100 |
| star_loop | TBD | TBD | TBD | TBD |

**结果目录：** `results/<topology>_<benchmark>_openevolve_gpt-oss-120b_<timestamp>_blind_artifacts/`

**日志目录：** `logs/cold_start_20260401_213610/`

## 关键设计决策

| 决策 | 理由 |
|---|---|
| temperature=0.0 执行 | 之前 temp=0.2 导致同一 DAG 同一任务重跑结果差异巨大（HumanEval baseline 33% 但 iter 0 eval 90%），无法区分 DAG 改进和随机波动 |
| HumanEval → LiveCodeBench | HumanEval baseline 42% 太简单，gpt-oss-120b 级别模型 pass@1 很高；LiveCodeBench 是竞赛题，baseline 1-3% |
| HotpotQA → GAIA | HotpotQA 在 temp=0 后 baseline 高达 67-80%，全错任务不足 20 个；GAIA 17-23% 更合适 |
| 4 种简化拓扑 | 比原来的 7 种 target_mas 更干净，纯拓扑变量，排除 prompt/tool 差异 |
| 统一模型 gpt-oss-120b | 执行和优化用同一模型，减少变量 |
