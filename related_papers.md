# 相关论文整理：Meta-Optimizer for Any Optimizer

---

## 0. 我们的目标与定位

### 一句话总结

我们要构建一个 **Prior-Guided, Taxonomy-Driven Meta-Optimizer**——一个能够优化任意 optimizer 的 agent，通过注入先验知识（prior）和结构化错误分类（error taxonomy）来引导 optimizer 更高效地找到优化 target system 的最佳方向，实现比现有 blind evolutionary search 更好的泛化能力。

### 核心问题

现有的 optimizer（无论是 prompt optimizer 如 GEPA/MIPROv2，compound system optimizer 如 OPTIMAS，还是 self-improving agent 如 HyperAgents）都面临一个共同瓶颈：**它们在面对新任务或新域时缺乏系统性的指导**。HyperAgents 依赖进化搜索从零摸索 meta 能力（需要几百次迭代），GEPA 用 reflection 但没有结构化的错误诊断，OPTIMAS 的 Local Reward Function 需要从头训练。没有任何现有工作系统地利用先验知识和错误分类来加速和引导优化过程。

### 我们做什么

1. **Prior-Informed Search**：构建 optimization prior library，当面对新的 optimizer 或 target system 时，检索相关先验来 warm-start 搜索方向（例如"prompt optimization 通常对 instruction clarity 敏感"，"RL fine-tuning 对 reward shaping 敏感"），而非从零开始盲搜。
2. **Taxonomy-Guided Diagnosis → Action**：构建 optimizer-agnostic 的两层错误分类框架（上层通用 optimizer failure categories 如 premature convergence / reward hacking / exploration-exploitation imbalance；下层可按目标 optimizer 动态实例化）。每个 error category 映射到具体的 repair policy，实现 diagnose → targeted fix → re-evaluate 的 closed-loop。
3. **Optimizer-Agnostic Interface**：定义统一抽象接口 wrap 任何 optimizer（输入/输出接口、可调配置空间、evaluation protocol），使我们的 meta-optimizer 能适配 prompt optimizer、compound system optimizer、context engineering 框架等不同类型。
4. **Cross-Optimizer Knowledge Transfer**：从优化一个 optimizer 时学到的 lessons（如"Pareto selection 比 greedy selection 泛化更好"）编码为可迁移的 textual knowledge，在优化其他 optimizer 时直接复用。

### 与核心竞品的差异

| 维度 | HyperAgents | GEPA | OPTIMAS | AgentFail/MAST | 我们 |
|------|-------------|------|---------|----------------|------|
| 优化对象 | Self (code) | Prompts | Compound system | N/A (分析) | **Any optimizer** |
| 搜索策略 | 盲进化 | Reflection + Pareto | Local reward alignment | N/A | **Prior-guided + taxonomy-driven** |
| Error diagnosis | 无（自发发现） | 无结构化 | 无 | 有taxonomy，无action | **Taxonomy → repair policy → closed-loop** |
| 先验知识 | 无 | 部分（domain feedback） | 无 | 无 | **显式 prior library** |
| 跨域迁移 | 隐式（代码中） | 无 | 无 | 无 | **显式 knowledge transfer** |
| Sample efficiency | 低（~200 iters） | 中（35x fewer than RL） | 中 | N/A | **高（prior warm-start + targeted fix）** |

### 核心论点

> HyperAgents 证明了 self-improvement 可以跨域迁移，但它靠"盲进化"实现，效率低、不可解释。现有 failure taxonomy 工作（MAST, AgentFail）证明了结构化错误分类能显著提升诊断精度，但止步于分析。我们将两者结合并超越：**用结构化先验和错误 taxonomy 驱动 closed-loop 优化**，实现更快的收敛、更好的跨域泛化、以及更高的可解释性。这不是在做更好的 self-improving agent 或更好的 failure diagnosis，而是一个新的 paradigm：**knowledge-guided meta-optimization**。

### 目标泛化能力

- **跨 optimizer 类型泛化**：同一个 meta-optimizer 可以优化 GEPA（prompt optimizer）、OPTIMAS（compound system optimizer）、ACE（context engineering）等不同类型的 optimizer
- **跨任务域泛化**：从优化 QA 系统中学到的 meta-knowledge 可以加速优化 code generation 系统
- **跨 failure 类型泛化**：上层 taxonomy 的通用 categories 在面对新类型 optimizer 时无需重新设计

---

以下按研究方向分类，包含项目中已有的论文和需要额外关注的外部论文。

---

## 1. Self-Improving / Self-Evolving Agent Systems

这是你工作的核心上游方向，理解"agent如何改进自己"。

| 论文 | 链接 | 关键点 |
|------|------|--------|
| **HyperAgents** (DGM-H) — Zhang et al., 2026 | https://arxiv.org/abs/2603.19461 | 你的直接对标。Metacognitive self-modification，task agent + meta agent合为hyperagent，跨域迁移self-improvement能力。**项目中已有** |
| **Darwin Gödel Machine** (DGM) — Zhang et al., 2025 | https://arxiv.org/abs/2505.22954 | HyperAgents的前身。Self-referential coding agent在coding benchmark上自我改进，但受限于coding domain |
| **ADAS: Automated Design of Agentic Systems** — Hu et al., 2025 | https://arxiv.org/abs/2408.08435 | Meta Agent Search用固定meta agent迭代生成下游agent。ICLR 2025。HyperAgents的重要baseline |
| **Gödel Agent** — Yin et al., 2025 | https://arxiv.org/abs/2410.04444 | Self-referential agent framework for recursively self-improvement。ACL 2025 |
| **Self-Taught Optimizer (STOP)** — Zelikman et al., 2024 | https://arxiv.org/abs/2310.02304 | 递归自我改进的code generation。LM Conference 2024 |
| **A Self-Improving Coding Agent** — Robeyns et al., 2025 | https://arxiv.org/abs/2504.15228 | Agent自主编辑自己的代码，SWE-bench上17%-53%的提升 |
| **DARWIN** — 2026 | https://arxiv.org/abs/2602.05848 | Dynamic Agentically Rewriting Self-Improving Network。进化优化+容器化安全 |
| **Position: Truly Self-Improving Agents Require Intrinsic Metacognitive Learning** — ICML 2025 | https://openreview.net/forum?id=4KhDd0Ozqe | Position paper，argues intrinsic metacognition is necessary for genuine self-improvement |
| **Meta Context Engineering via Agentic Skill Evolution** — Ye et al., 2026 | https://arxiv.org/abs/2601.21557 | Agentic skill evolution用于context engineering |
| **MetaAgent: Toward Self-Evolving Agent via Tool Meta-Learning** — 2025 | https://arxiv.org/abs/2508.00271 | Learning-by-doing范式，agent持续自我反思+知识库构建 |
| **Survey: Self-Evolving AI Agents** — Fang et al., 2025 | https://arxiv.org/abs/2508.07407 | 综合survey，统一框架覆盖self-evolving agentic systems各技术 |

---

## 2. Failure Taxonomy & Diagnosis for Agent Systems

你的工作需要借鉴这些taxonomy方法论，但做法完全不同（taxonomy驱动优化，而非仅分析）。

| 论文 | 链接 | 关键点 |
|------|------|--------|
| **MAST: Why Do Multi-Agent LLM Systems Fail?** — Cemri et al., 2025 | https://arxiv.org/abs/2503.13657 | 14个failure modes，3大类（System Design / Inter-Agent Misalignment / Task Verification）。1642条trace，NeurIPS 2025 Datasets Track spotlight。**项目中已有** |
| **AgentFail: Diagnosing Failure Root Causes in Platform-Orchestrated Agentic Systems** — Ma et al. | 论文在项目中 | 16个root cause categories，3层（Agent / Workflow / Platform level）。307条failure logs，Dify/Coze平台。Counterfactual repair验证。**项目中已有** |
| **Agent Error Taxonomy + AgentDebug** — ICLR 2026 withdrawn | https://openreview.net/forum?id=PFR4E8583W | Modular failure taxonomy（memory, reflection, planning, action, system）+ debugging framework with corrective feedback。最接近closed-loop的现有工作 |
| **Microsoft: Taxonomy of Failure Mode in Agentic AI Systems** | https://cdn-dynmedia-1.microsoft.com/is/content/microsoftcorp/microsoft/final/en-us/microsoft-brand/documents/Taxonomy-of-Failure-Mode-in-Agentic-AI-Systems-Whitepaper.pdf | Safety/security导向的failure taxonomy，工业视角 |
| **Who&When Dataset** — Zhang et al., 2025 | 被AgentFail和MAST引用 | Failure logs from LLM MAS，annotated with responsible agent and failure step |
| **AgenTracer** — Zhang et al., 2025 | 被AgentFail引用 | Fine-grained error diagnosis through counterfactual replay and RL |
| **ETO: Learning from Failed Explorations** — Song et al., 2024 | 被AgentFail引用 | Agent从失败中学习提升LLM agent在复杂任务上的表现 |

---

## 3. Prompt / System Optimization（你的target optimizer类型）

这些是你的meta-optimizer需要能够优化的"下游optimizer"。

| 论文 | 链接 | 关键点 |
|------|------|--------|
| **GEPA: Reflective Prompt Evolution** — Agrawal et al. | 论文在项目中 | Genetic-Pareto prompt optimizer，用natural language reflection。比GRPO高10%，35x fewer rollouts。**项目中已有** |
| **OPTIMAS: Optimizing Compound AI Systems with Globally Aligned Local Rewards** — Wu et al. | 论文在项目中 | 为每个component维护Local Reward Function，异构配置（prompts + params + hyperparams）。ICLR 2026。**项目中已有** |
| **ACE: Agentic Context Engineering** — Zhang et al. | 论文在项目中 | Context作为evolving playbook，incremental delta updates，Generation-Reflection-Curation框架。**项目中已有** |
| **DSPy / MIPROv2** — Khattab et al., Opsahl-Ong et al. | https://arxiv.org/abs/2310.03714 / https://arxiv.org/abs/2406.11695 | 主流prompt optimization框架，joint instruction + few-shot optimization |
| **TextGrad** — Yuksekgonul et al., 2025 | https://arxiv.org/abs/2309.16421 | Text-based automatic differentiation for compound AI systems |
| **OPRO: Optimization by PROmpting** — Yang et al., 2024 | https://arxiv.org/abs/2309.03409 | 用LLM作为optimizer来优化prompt |
| **AFlow** — Zhang et al., 2024 | https://arxiv.org/abs/2410.10762 | MCTS-based workflow optimization，比ADAS更结构化的搜索 |
| **DebFlow** — Su et al., 2025 | 引用于survey | 多个debater讨论system design，judge裁决 |
| **Compound AI Systems Optimization Survey** — 2025 | https://arxiv.org/abs/2506.08234 | 综合survey覆盖所有compound system优化方法 |
| **Agent Workflow Optimization (AWO)** — 2026 | https://arxiv.org/abs/2601.22037 | Meta-tools优化agentic workflow中的冗余 |

---

## 4. Textual Equilibrium Propagation / Alternative Optimization（理论方向）

| 论文 | 链接 | 关键点 |
|------|------|--------|
| **Textual Equilibrium Propagation for Deep Learning** | 论文在项目中 | 用文本空间的equilibrium propagation做deep learning优化。**项目中已有** |

---

## 5. Alignment & Safety for Compound AI Systems

| 论文 | 链接 | 关键点 |
|------|------|--------|
| **Aligning Compound AI Systems** | 论文在项目中 | Compound AI系统的alignment问题。**项目中已有** |

---

## 6. Key Surveys & Position Papers

| 论文 | 链接 | 关键点 |
|------|------|--------|
| **Self-Evolving AI Agents Survey** — Fang et al., 2025 | https://arxiv.org/abs/2508.07407 | 最全面的self-evolving agent survey |
| **Compound AI Systems Optimization Survey** — 2025 | https://arxiv.org/abs/2506.08234 | Compound system优化方法全景 |
| **Awesome Self-Evolving Agents (GitHub)** | https://github.com/EvoAgentX/Awesome-Self-Evolving-Agents | 持续更新的paper list |
| **The Multi-Agent Trap** — Towards Data Science, 2026 | https://towardsdatascience.com/the-multi-agent-trap/ | 实践视角分析MAS的compound reliability问题 |

---

## 你的论文如何定位

```
                    Scope
                    ↑
     Any Optimizer  |  ← 你的位置：taxonomy-guided meta-optimizer
                    |     结合 prior knowledge + error diagnosis
                    |     closed-loop optimization
     Compound AI    |  OPTIMAS, ACE, GEPA
                    |
     Multi-Agent    |  MAST, AgentFail
                    |
     Single Agent   |  AgentDebug, ADAS, DGM
                    |
                    +————————————————————————→ Actionability
                    Analysis    Diagnosis    Repair    Closed-Loop
                    Only        + Feedback   Action    Optimization
```

**你独占的空间：** Broad scope (any optimizer) + Full closed-loop (taxonomy → diagnosis → targeted repair → re-evaluate)。没有任何现有工作同时覆盖这两个维度。
