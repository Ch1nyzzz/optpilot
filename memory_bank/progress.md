# Progress

## Current Status

- Core OptPilot pipeline implemented and running on AG2 (MathChat) traces.
- Offline pipeline validated: Diagnoser → Optimizer → Judge → Distiller → Library.
- Models: GLM-5 (OptPilot system) + gpt-oss-120b (target MAS) via Together AI.

## Latest Update (2026-03-25)

### Phase 1: 文献 + 设计 (完成)
- 完成文献下载（21 篇论文）
- 确定 scope：聚焦 MAS 优化
- 明确差异化：diagnosis-driven targeted repair vs blind evolution
- 实验设计文档完成（5 个实验 + ablation）

### Phase 2: 实现 (进行中)
- **核心 pipeline 代码完成**:
  - `src/optpilot/`: config, llm (Together AI), models, tracking, data loader, fm_taxonomy
  - `src/optpilot/dag/`: DAG 抽象 + ChatDev YAML adapter
  - `src/optpilot/modules/`: Diagnoser, Optimizer, Judge, Distiller
  - `src/optpilot/library/`: RepairLibrary (JSON 持久化)
  - `experiments/`: offline_pipeline, online_pipeline, exp1-5

- **MVP Target 切换为 AG2 (MathChat)**:
  - AG2 在 MAST-Data 中有 597 条 traces（最多），trace 短（~5K chars）
  - 2-agent 简单架构（Student + Assistant），无复杂 DAG
  - 3 个 benchmark: GSM (223), Olympiad (206), MMLU (168)
  - ChatDev 太复杂（trace 240K chars，YAML 1000行），暂时搁置

- **离线 pipeline 验证成功**:
  - 在 3 条 AG2 FM-1.3 traces 上完整跑通
  - Diagnoser: GLM-5 成功定位到 agent + step + root cause
  - Optimizer: 生成具体的 node_mutation / config_change repair actions
  - Judge: 反事实评估 3/3 would_fix=True，confidence 0.90-0.95
  - Library: 4 entries 存入 offline_ag2_library.json

### 关键发现
- GLM-5 是 reasoning model，需要大 max_tokens（8192+）来容纳 reasoning + output
- AG2 trace ~5K chars 可直接全量发送 LLM，不需要 chunking
- ChatDev trace ~240K chars 需要 chunking/摘要，GLM-5 处理极慢

## Key Research Decisions

| 日期 | 决策 | 理由 |
|------|------|------|
| 2026-03-25 | 聚焦 MAS 而非 any optimizer | failure pattern 丰富，有现成 taxonomy 数据 |
| 2026-03-25 | 核心差异化 = targeted repair vs blind evolution | MAST+OpenEvolve 做了 taxonomy-as-reward，我们走 diagnosis→action |
| 2026-03-25 | MVP 切换到 AG2 (MathChat) | 597 条 trace，短 trace (5K chars)，简单 2-agent 架构 |
| 2026-03-25 | 双层实验: offline + online | 对比反事实评估 vs 实跑验证的 gap |
| 2026-03-25 | Together AI: GLM-5 (system) + gpt-oss-120b (target) | 统一 API |

### Phase 3: 架构脱耦 (完成 2026-03-25)
- **完全脱离 chatdev_v2 依赖**:
  - 自研 DAGExecutor 轻量执行引擎 (`dag/executor.py`)
  - MASDAG 自有序列化格式 (YAML)，直接 `to_dict()`/`from_dict()`/`save()`/`load()`
  - OptPilotRunner 替代 ChatDevRunner，内置 Python 执行（无 subprocess）
  - 删除所有 adapter (chatdev_adapter, ag2_adapter, appworld_adapter, hyperagent_adapter)
  - ChatDev 工作流转为 `dags/chatdev.yaml` (MASDAG 格式)
  - 更新所有实验管线和 orchestrator

## Next Steps

1. **扩大离线实验**: 在更多 AG2 traces 上跑 offline pipeline (FM-1.3, FM-2.6, FM-1.1)
2. **在线 Pipeline 验证**: 用 DAGExecutor 实际执行 MASDAG，收集在线 trace
3. **Exp 5**: 对比 offline Judge 预测 vs online 实际结果
4. **Exp 3**: 经验积累实验 (library hit rate 随 batch 增长)
5. **更多 DAG 定义**: 为 AG2 MathChat 等创建 MASDAG YAML
