# 冷启动实验汇报（2026-04-01）

## 1. 背景：为什么只靠 OpenEvolve 不够

我们最初用 OpenEvolve 做的是离线 blind search：在训练集上不断对 DAG 做 mutation，然后保留分数变好的结构。这个阶段有价值，但它本身并不能直接变成一个稳定、可复用的优化器，主要有四个问题：

1. OpenEvolve 只会告诉我们“某个 DAG 在 train 上变好了”，但不会直接给出可复用的修复规律。
2. 搜索结果很噪，很多候选结构只是在当前 batch 上碰巧有效，未必能泛化到 held-out 数据。
3. 搜索成本高，而且生成过程本身不稳定。日志里反复出现 `503 Service unavailable` 和 `Empty content (length limit)` 重试。
4. 搜索阶段产生的是“具体 DAG 变体”，但在线修复真正需要的是“给定某类 failure mode，优先试哪类改动，以及这类改动通常应该怎么做”。

更具体地说，OpenEvolve 暴露出来的问题不是单纯“搜得慢”，而是它缺少一个把离线搜索结果蒸馏成在线可用经验库的中间层。

## 2. 我们想做到什么

我们的目标不是把 OpenEvolve 当成最终优化器，而是把它当成冷启动数据收集器。理想 pipeline 是：

`run_openevolve -> analyze_openevolve_traces -> 产出 cold-start prior / recipe -> （可选）run_skill`

其中冷启动阶段希望得到三类产物：

1. `data_driven_priors.json`
   给定 failure mode（FM），优先尝试哪类 observable change family。
2. `jacobian_warmup.jsonl`
   作为 Jacobian 的离线 warmup 证据。
3. `recipes`
   给定 `(topology, FM, family)`，这种改动通常应该如何落到 DAG 上。

当前我们已经把顶层 Jacobian 收口成 6 个 observable family，而不是直接在更细碎的 pattern 上做在线 credit：

- `prompt_refine`
- `topology_change`
- `edge_route`
- `config_tune`
- `loop_adjust`
- `literal_update`

其中 recipe 承载更细的结构化做法；Jacobian 只学 family 级别的方向。

## 3. 当前 pipeline 的结论

现在可以明确区分两件事：

- 冷启动已经完成：`simple_star` 和 `simple_hier` 的 prior、warmup、recipe 都已经落盘。
- 全 pipeline 还没结束：两个实验后面又继续进入了 `run_skill`，这是额外的在线验证阶段，不是冷启动本身必须的一部分。

如果当前目标只是汇报 cold-start，那么分析应该停在 `analyze_openevolve_traces`；`run_skill` 可以视为附加实验。

## 4. 冷启动总体结论

### 4.1 最大结论

不同 topology 学到的 cold-start 偏好非常不一样，说明 topology-specific prior 是必要的。

- `simple_star` 学到的是“协调契约 + 最终汇总 + 路由修正 + 少量参数收敛”。
- `simple_hier` 学到的是“验证闸门 + 失败信号清洗/改写 + 路由重构”。

也就是说，我们不能指望一个全局、与 topology 无关的 repair prior 同时适配 star 和 hierarchical workflow。

### 4.2 关于 Jacobian 的结论

冷启动阶段支持了一个明确判断：

- 顶层 Jacobian 应该学 `family`，而不是学更细的 pattern。
- 更细的 pattern 应该下沉到 recipe 文本里。

原因很直接：这样在线推荐空间和在线 credit 空间是一致的，不会出现“推荐细、归因粗”的 credit leakage。

### 4.3 关于 OpenEvolve 的结论

OpenEvolve 适合作为冷启动数据源，但不适合作为最终在线策略。

它的价值主要在于：

- 发现若干能在 held-out 上复现的有效 mutation
- 给 Jacobian warmup 提供初始证据
- 给 recipe 蒸馏提供真实的 parent -> child 结构改动样本

它不擅长的部分主要是：

- 在线面向 FM 的定向修复
- 稳定地输出低噪声、可复用的结构知识
- 在服务波动和长上下文压力下维持高吞吐搜索

## 5. `simple_star` 冷启动结果

### 5.1 冷启动产物规模

- 有效 mutation：`8`
- 分类成功：`8`
- Jacobian warmup 记录：`48`
- 生成 recipe：`25`

### 5.2 学到的 prior

这些 prior 可以理解为：对每个 FM，哪类 family 在 held-out 验证后更值得优先尝试。

| FM | prompt_refine | topology_change | edge_route | config_tune |
| --- | --- | --- | --- | --- |
| A | 1.00 | 0.50 | 0.50 | 0.00 |
| B | 1.00 | 0.50 | 1.00 | 0.00 |
| C | 1.00 | 0.75 | 0.50 | 0.00 |
| D | 1.00 | 0.75 | 0.50 | 0.00 |
| E | 1.00 | 0.75 | 0.50 | 1.00 |
| F | 1.00 | 0.75 | 0.00 | 1.00 |

读法上有三个关键点：

1. `prompt_refine` 在 `simple_star` 上几乎是普遍强信号。
2. `topology_change` 是第二层通用动作，尤其在 `C/D/E/F` 上更强。
3. `config_tune` 只在 `E/F` 上体现出明显价值，说明它不是通用首选动作。

### 5.3 学到的 recipe

`simple_star` 一共 `25` 条 recipe，分布如下：

- `prompt_refine`: `6`
- `topology_change`: `9`
- `edge_route`: `6`
- `config_tune`: `4`

这些 recipe 总结出来的主题很集中：

1. 强化 orchestrator 的显式协作契约
   典型做法是要求先输出 plan，再输出明确状态词，例如 `READY_FOR_SYNTHESIS` / `NEED_MORE_INFO`，避免 orchestrator 过早产出 final answer。
2. 把“协调”和“最终答案组装”拆开
   高频 recipe 是新增 `Formatter` 或 `Synthesizer`，把最终格式化、答案汇总从 orchestrator 身上拆出去。
3. 去掉导致重复激活的回边
   多条 `edge_route` recipe 指向同一问题：specialist 的输出过早回流到 orchestrator，形成 back-edge 和循环。
4. 在少数高风险位置做参数收敛
   `config_tune` 主要集中在 verifier 或关键汇总节点上，目标是降低随机性，让最终判断更稳定。

### 5.4 对 `simple_star` 的一句话结论

`simple_star` 的冷启动经验说明：star 型结构的主要瓶颈不是“缺少更多 agent”，而是 orchestrator 角色过载、终态格式不稳定，以及错误的回流边让工作流反复重启。

## 6. `simple_hier` 冷启动结果

### 6.1 冷启动产物规模

- 有效 mutation：`4`
- 分类成功：`4`
- Jacobian warmup 记录：`24`
- 生成 recipe：`15`

### 6.2 学到的 prior

| FM | topology_change | edge_route |
| --- | --- | --- |
| A | 0.00 | 1.00 |
| B | 0.667 | 1.00 |
| C | 1.00 | 1.00 |
| D | 0.333 | 1.00 |
| E | 1.00 | 1.00 |
| F | 1.00 | 1.00 |

这个 prior 和 `simple_star` 很不一样：

- 没有学出稳定的 `prompt_refine` 冷启动信号。
- 也没有学出显式的 `config_tune` family prior。
- 真正稳定有效的是 `topology_change` 和 `edge_route`。

这说明 `simple_hier` 的失败更多是控制流和验证路径的问题，而不是 prompt 契约本身的问题。

### 6.3 学到的 recipe

`simple_hier` 一共 `15` 条 recipe，分布如下：

- `topology_change`: `9`
- `edge_route`: `6`

这些 recipe 的主要主题是：

1. 强制验证闸门
   去掉 `Planner -> FINAL` 这种捷径，只允许 verifier 的显式 `TESTS_PASS` 信号到达终态。
2. 插入 `Validator/Cleaner`
   当 verifier 或 tester 输出的是自由文本时，先加一个专门节点把它规整成可判定的单一信号。
3. 插入 `Refiner`
   不要把原始失败信号直接扔回 Planner，而是先经由 Refiner 总结失败原因、生成修正方向，再回到主规划节点。
4. 重写失败路由
   让 `TESTS_FAIL` 进入专门修复支路，而不是继续在原回路里盲目重试。

### 6.4 对 `simple_hier` 的一句话结论

`simple_hier` 的冷启动经验说明：hierarchical workflow 的主要问题不是答案生成不够强，而是验证信号不够干净、失败反馈没有被结构化处理，以及终态闸门不严格。

## 7. 新 topology 的对比结论

把两个 topology 放在一起看，可以得到几个可直接汇报的结论：

1. topology-specific cold-start 是必要的。
   `simple_star` 和 `simple_hier` 学出来的 family prior 明显不同，不能共享同一个默认 repair policy。
2. `simple_star` 更依赖 prompt-level orchestration contract。
   它学出了显著的 `prompt_refine` prior，而且 recipe 大量围绕 orchestrator 信号和最终汇总节点。
3. `simple_hier` 更依赖 control-flow sanitation。
   它几乎全部收敛到 `topology_change` 和 `edge_route`，说明核心问题在 routing、verification gate 和 failure handling。
4. 冷启动 recipe 是结构知识，不只是 prompt 模板。
   这些 recipe 本质上描述的是“什么时候该加节点、删边、改验证路径、拆职责”，而不是简单改写一句 system prompt。

## 8. AG2 的位置

作为背景，旧的 AG2 经验库已经迁到 `library_store/ag2/recipes/`，并去重成了 `3` 条 canonical shared recipes：

- `prune_intermediates`
- `add_checker`
- `restore_critical_path`

这批 AG2 recipe 主要是历史全局经验，不属于本轮 `simple_star/simple_hier` 冷启动产物本身，但它说明旧系统已经反复观察到两个稳定结构规律：

1. 多余中间节点会引入漂移和信息损耗。
2. 关键验证路径必须显式、短、可判定。

## 9. 当前实验状态

截至 `2026-04-01 10:56 EDT`：

- `simple_star` 的冷启动阶段已经完成，相关 prior 和 recipe 已经安装到 `library_store/simple_star/`。
- `simple_hier` 的冷启动阶段也已经完成，相关 prior 和 recipe 已经安装到 `library_store/simple_hier/`。
- 两个主实验后续都还在继续跑 `run_skill`，这是在线 repair 验证，不影响本次 cold-start 汇报。

换句话说：

- 如果问题是“冷启动有没有产出”，答案是有，而且已经落盘。
- 如果问题是“整个 pipeline 有没有完全结束”，答案是还没有，因为后面的在线阶段仍在跑。

## 10. 本轮最重要的结论

这轮实验已经支持下面这个工作假设：

1. OpenEvolve 最适合作为冷启动数据收集器，而不是最终在线策略。
2. 冷启动应该产出 topology-specific 的 `family prior + family-scoped recipes`。
3. 顶层 Jacobian 应该只学 6 个 observable family，细粒度 pattern 应该下沉为 recipe。
4. `simple_star` 和 `simple_hier` 的修复偏好明显不同，说明 topology 不能被忽略。
5. 如果当前目标只是冷启动，应当把主 pipeline 截断在 `analyze_openevolve_traces`，而不是默认继续跑 `run_skill`。
