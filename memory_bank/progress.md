# Progress

## Current Status

- Core OptPilot Skill Workflow pipeline implemented on AG2 MathChat.
- Skill Workflows (A-F) with iterative convergence + reflection + meta-evolution.
- Official benchmark scoring: MMLU + AIME 2025 + OlympiadBench.
- Model: MiniMax M2.5 (unified for execution + diagnosis) via Together AI.
- Multi-topology cold-start pipeline implemented for `simple_star` (GAIA) and
  `simple_hier` (SWE-bench Lite).
- `run_skill.py` now serves as the unified Jacobian-driven repair entrypoint for
  multiple topologies.
- Topology-isolated storage added for Jacobian warmup, recipes, negatives, and
  catalog evolution traces.

## Latest Update (2026-04-01)

### Phase 9: Simplified Topologies + Multi-Topology Cold Start (进行中)
- 新增 `simple_star_gaia.yaml` 和 `simple_hierarchical_swebench.yaml` 两个 3-agent DAG
- `run_openevolve.py` / `openevolve_evaluator_multi.py` / `analyze_openevolve_traces.py`
  已支持 `simple_star` 和 `simple_hier`
- `run_skill.py` 已替代单一 AG2 skill 入口，统一支持多 topology
- 修复多拓扑实验中的两个关键问题：
  - OpenEvolve evaluator 先截断样本再按 prompt 过滤，导致 eval set 被静默缩小
  - recipe / meta-evolve 仍读写全局目录，导致 topology 隔离只做了一半
- 当前并行实验状态：
  - `simple_star`: OpenEvolve 50 iter 已完成，进入 `analyze_openevolve_traces`
  - `simple_hier`: OpenEvolve 50 iter 已完成，正在收尾 test evaluation / pipeline 切换
- 当前结论趋势：
  - blind OpenEvolve 可用于冷启动搜到更好的 DAG
  - 真正的长期优化目标仍是 diagnosis-driven targeted repair，而不是继续依赖 blind evolution

## Previous Update (2026-03-26)

### Phase 1: 文献 + 设计 (完成)
- 21 篇论文，scope 确定为 MAS 优化
- 差异化：diagnosis-driven targeted repair vs blind evolution
- 实验设计文档完成（5 个实验 + ablation）

### Phase 2: 实现 (完成)
- 核心 pipeline 代码完成
- MVP Target: AG2 MathChat（597 条 traces，短 trace ~5K chars）
- 离线 pipeline 验证成功

### Phase 3: 架构脱耦 (完成)
- 自研 DAGExecutor，完全脱离 chatdev_v2
- MASDAG 自有序列化格式，OptPilotRunner 内置执行

### Phase 4: FM Classifier 校准 (完成)
- 100 条 AG2 blind trace，6 位标注者
- MiniMax M2.5 选定（与人类容忍率 89.9%–98.0%）

### Phase 5: Online 6-Group Pipeline (完成)
- 全面切换 6-group taxonomy (A-F)
- AG2 MathChat DAG 基于 MAST 论文 Appendix L
- 统一 MiniMax M2.5

### Phase 6: Skill Workflow Architecture (完成)
- 6 个 Skill Workflows (A-F)：内循环收敛 + 外循环反思 + 并行执行
- Meta-evolution: Skill 连续失败后 LLM 修改 Skill 源码
- 旧模块 → _legacy/

### Phase 7: Critical Bug Fixes (完成 2026-03-26)
- **修复 DAGExecutor agent 系统提示词丢失**: executor 现在回退读 `node.role`（YAML 用 `role` 而非 `prompt`）
- **修复 DAGExecutor literal 内容丢失**: executor 现在回退读 `config.content`
- **修复 `_pass_rate` 使用 `task_success` 而非 `task_score`**: 现在优先使用 benchmark ground-truth 评分
- **修复 Turn Counter 每轮误触 FINAL**: 添加 `loop: exit` 标注，修复 fallback 逻辑（仅在无任何循环标注时才 fallback）
- **修复 agent config.params 嵌套**: executor 支持 `config.params.temperature` 和 `config.temperature` 两种格式

## Key Research Decisions

| 日期 | 决策 | 理由 |
|------|------|------|
| 2026-03-25 | 聚焦 MAS 而非 any optimizer | failure pattern 丰富，有现成 taxonomy 数据 |
| 2026-03-25 | 核心差异化 = targeted repair vs blind evolution | MAST+OpenEvolve 做了 taxonomy-as-reward |
| 2026-03-25 | MVP 切换到 AG2 (MathChat) | 597 条 trace，短 trace，3-agent 架构 |
| 2026-03-25 | Together AI: MiniMax M2.5 (统一) | 统一 API，校准验证通过 |
| 2026-03-26 | pass_rate 必须用 task_score | task_success 只是"是否跑完"，几乎恒为 True |
| 2026-03-26 | Loop counter 需显式 loop: exit 标注 | 避免被动计数器每轮误触下游 |

### Phase 8: Offline Cold-Start Pipeline (进行中 2026-03-29)
- 离线 OpenEvolve 100 iter 盲进化作为冷启动
- `analyze_openevolve_traces.py`: 提取有效 mutation → test set 后验过滤 → 归纳先验
- Jacobian `_cold_start_score` 支持从 `data_driven_priors.json` 加载数据驱动先验
- 替换原有手写 prior 和 `_DEFAULT_CATALOG`

## Next Steps

1. **完成并汇总 `simple_star` / `simple_hier` 冷启动实验**:
   - OpenEvolve result
   - offline prior extraction
   - online `run_skill` warm-start validation
2. **比较 blind evolution vs diagnosis-driven repair**:
   - sample efficiency
   - held-out test accuracy
   - FM reduction
3. **验证 prior 迁移价值**:
   - 从 simplified topology 迁移到更复杂的原始 Star / Hierarchical DAG
4. **回到 AG2 主线做端到端验证**:
   - 用冷启动 prior + Jacobian + recipes 运行完整在线 OptPilot pipeline
