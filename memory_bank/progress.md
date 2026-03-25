# Progress

## Current Status

- Repository is in the early research stage.
- Literature review has been collected in `related_papers.md` and `papers/`.
- Contributor guidance files `AGENTS.md` and `CLAUDE.md` are in place.
- `memory_bank/` has been initialized to track goals, progress, and architecture.

## Latest Update (2026-03-25)

- 完成文献下载（21 篇论文到 `papers/`）
- 深入分析了 AgentDebug、MAST、MAST+OpenEvolve blog
- 确定 scope：聚焦 MAS 优化，不再追求 any-optimizer
- 明确差异化定位：diagnosis-driven targeted repair（vs 盲进化 / 纯诊断）
- 发现最近竞品：UC Berkeley ADRS 的 MAST + OpenEvolve 工作
- 下载并分析 MAST-Data 数据集（1242 traces，其中 ChatDev 130 条）
- 分析 ChatDev failure profile：top FM 是 Step Repetition (36.2%)
- 分析 ChatDev v2 源码：**原生 YAML-driven DAG**，~28K 行 Python
  - 整个 MAS 定义在一个 YAML 文件中（`ChatDev_v1.yaml`，~1000 行）
  - nodes + edges 结构天然匹配我们的 DAG 抽象
  - 修改只需改 YAML，不需要改 Python 代码
  - DAG adapter 几乎不用写

## Key Research Decisions

| 日期 | 决策 | 理由 |
|------|------|------|
| 2026-03-25 | 聚焦 MAS 而非 any optimizer | failure pattern 丰富，有现成 taxonomy 数据 |
| 2026-03-25 | 不训练分类器 | 数据量太小，taxonomy 粒度不匹配，与 optimizer-agnostic 定位矛盾 |
| 2026-03-25 | 核心差异化 = targeted repair vs blind evolution | MAST+OpenEvolve 已做了 taxonomy-as-reward，我们需要走 diagnosis→action 路线 |
| 2026-03-25 | Target system = ChatDev (ProgramDev, GPT-4o) | MAST-Data 有 130 条 trace，源码是 YAML DAG 天然适配，MAST 论文有 case study baseline |
| 2026-03-25 | Experience-Driven Repair Library 架构 | 蒸馏有效修复方案，跨 trace / 跨 MAS 检索复用 |
| 2026-03-25 | 蒸馏粒度 deferred | 等 MVP 跑出真实修复过程后决定 |

## Next Steps

1. **MVP Phase A**：在 MAST-Data 的 130 条 ChatDev trace 上做离线分析
   - Diagnoser 细化 FM-1.3 (Step Repetition) 的 47 条 trace，定位到具体 agent 和 step
   - Optimizer 生成候选 repair action
2. **MVP Phase B**：搭建 ChatDev Runner，应用 repair 并验证
3. **MVP Phase C**：验证方案跨 trace 复用
4. 自动化 OptPilot pipeline
5. 完整实验（Exp 1-5）+ 论文撰写
