# Progress

## Current Status

主实验：**先验指导 OpenEvolve vs 盲进化 OpenEvolve**

- 两阶段 pipeline 已实现：blind cold-start → prior extraction
- 多目标 MAS 支持：ag2, agentcoder, simple_star, simple_hier, appworld, hyperagent, magentic
- 经验存储全局化，通过 DAG 拓扑特征 (has_hub) 自动区分（只分 hub vs no-hub，不分 loop）
- Official benchmark scoring: MMLU + AIME 2025 + OlympiadBench + HumanEval + GAIA + SWE-bench Lite
- AgentCoder target 已实现（Programmer → TestDesigner → TestExecutor pipeline）

## Latest Update (2026-04-01)

### 彻底清理 control flow 残留
- 删除了旧的 control flow 分桶机制（pipeline / centralized_dispatch）
- 删除了 merge_experience_by_control_flow.py 和 merge_experience_to_global.py
- 清理了 config.py 中大量废弃变量（OFFLINE_HINTS_DIR, EVOLVED_SKILLS_DIR, META_EVOLVE_TRACES_DIR 等）
- 经验存储改为全局 + 拓扑特征自动匹配
- 修复了 analyze_openevolve_traces.py 蒸馏时不传 has_hub 的 bug
- 更新了全部文档（CLAUDE.md, AGENTS.md, memory_bank/）

### 新增 AgentCoder target
- `benchmarks_humaneval.py`: HumanEval 加载 + subprocess 代码执行评分
- `agentcoder_tools.py`: python_exec 工具
- `openevolve_initial_dag_agentcoder.py`: 3-agent pipeline DAG
- 注册到 run_openevolve.py 和 evaluator
- 端到端验证通过（score=1.0 on canonical solution）

### 代码库大清理
- 删除了整个 Skill Workflow 架构
- 删除了旧模块（_legacy/, judge, yaml_optimizer, repair_library 等）
- 删除了废弃实验脚本和测试

## Historical Phases

| Phase | 状态 | 内容 |
|---|---|---|
| 1: 文献 + 设计 | 完成 | 21 篇论文，scope 确定为 MAS 优化 |
| 2: 实现 | 完成 | 核心 pipeline 代码，MVP: AG2 MathChat |
| 3: 架构脱耦 | 完成 | 自研 DAGExecutor，脱离 chatdev_v2 |
| 4: FM Classifier 校准 | 完成 | MiniMax M2.5 选定，与人类容忍率 89.9%–98.0% |
| 5: Online 6-Group Pipeline | 完成 | 全面切换 6-group taxonomy |
| 6: Skill Workflow | 完成→废弃 | 已删除 |
| 7: Critical Bug Fixes | 完成 | DAGExecutor 系统提示词/literal/loop 修复 |
| 8: Offline Cold-Start | 完成 | OpenEvolve 盲进化 + analyze_traces prior extraction |
| 9: Multi-Target Cold-Start | 完成 | simple_star + simple_hier 实验 |
| 10: 代码清理 + AgentCoder | 完成 | 删除旧代码，新增 agentcoder target |
| 11: 清理 control flow 残留 | 完成 | 全局经验 + 拓扑特征自动匹配 |

## Key Research Decisions

| 日期 | 决策 | 理由 |
|------|------|------|
| 2026-03-25 | 聚焦 MAS 而非 any optimizer | failure pattern 丰富，有现成 taxonomy 数据 |
| 2026-03-25 | MVP: AG2 MathChat | 597 条 trace，短 trace，3-agent 架构 |
| 2026-03-25 | Together AI: MiniMax M2.5 | 统一 API，校准验证通过 |
| 2026-03-26 | pass_rate 必须用 task_score | task_success 只是"是否跑完"，几乎恒为 True |
| 2026-04-01 | 删除 Skill Workflow，聚焦 OpenEvolve | 主实验是 prior-guided vs blind evolution |
| 2026-04-01 | 全局经验 + 拓扑自动匹配 | 替代旧的 control flow 硬编码分桶 |

## Next Steps

1. **并行跑 agentcoder blind vs guided 对比实验**
2. **从 guided 结果蒸馏更新先验**
3. **对比分析**：sample efficiency, held-out accuracy, FM reduction
