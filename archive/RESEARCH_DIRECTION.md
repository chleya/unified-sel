# RESEARCH_DIRECTION.md

**Last Updated**: 2026-04-15

## 新主命题（2026-04-15 更新）

> **小模型系统如何识别自己的能力边界，并通过结构记忆、验证反馈、健康监控和调度策略，在边界区获得能力增益？**

---

## One-Line Thesis

This project should be reframed around one core research question:

**How can a small model gain large-model-like reasoning, abstraction, and code-task performance through system design rather than parameter scale alone?**

---

## 为什么这个方向更有研究价值

**不是**：
- 继续把 Unified-SEL 调到赢 EWC
- TopoMem 主打检索
- Double Helix 主打反馈

**而是**：
- 能力不是单个模型属性，而是"模型 + 结构状态 + 反馈验证 + 调度控制"共同形成的**动态系统属性**
- 小模型什么时候能靠外部结构、验证和控制机制跨过能力边界？什么时候不能？
- 系统如何知道自己处在哪个边界区？

---

## 四条研究线的价值排序

### 1. 最高价值：能力边界调度理论

**来源**：Double Helix + TopoMem + Capability Benchmark

**核心命题**（来自 `DOUBLE_HELIX_MECHANISM_NOTE_2026-04-13.md`）：
> maintain-chain utility peaks near the solver's capability boundary.

**规律假说**：
- **太弱**：反馈没用（模型无法理解反馈）
- **太强**：反馈不需要（模型自己就能解决）
- **临界区**：反馈/验证/重试最有用

**可升级为项目主命题**：
> 外部验证和反馈链不是普遍增强器，而是**边界区增强器**。

### 2. 第二价值：结构贝叶斯场

**来源**：Unified-SEL + TopoMem AdapterPool + Structural Bayesian Field

**核心思想**：
不是标准 Bayesian inference，而是**结构化信念状态**。

- 普通贝叶斯：prior + evidence → posterior
- 结构贝叶斯：structure distribution + observation + feedback → evolved structure field

**三个可测变量**：
1. **belief weight**：结构/adapter 当前可信度
2. **update operator**：反馈如何改变权重
3. **action policy**：权重分布如何决定 solve / retry / branch / escalate

### 3. 第三价值：TopoMem 作为健康信号

**核心洞察**：
TopoMem 目前不是 capability amplifier，而是 **monitor/control substrate**。

**研究命题**：
> H1/H2 不提升答案本身，但能提升**系统调度质量**。

### 4. 第四价值：Unified-SEL 作为机制实验

**核心发现**：
- shared readout drift 是遗忘来源之一
- 训练 route 和 inference route 不一致

---

## Why This Reframe Is Necessary

The repository currently contains two meaningful but different lines of work:

1. **Unified-SEL**
   - studies continual learning without explicit task-boundary labels
   - focuses on surprise-driven structure growth, reuse, pruning, and retention

2. **TopoMem**
   - studies topological signals in embedding space
   - now provides the strongest value as a health-monitoring and boundary-complexity layer rather than as a retrieval enhancer

Both lines are valid, but they do **not** directly answer the original motivation:

> how to let a small system realize stronger reasoning, abstraction, and coding ability

At the moment, the project is optimized around:

- forgetting vs EWC
- endogenous boundary formation
- health monitoring

These are useful subproblems, but they are **not the final target**.

---

## Main Project Judgment

### What the project is *not*

The main goal is **not**:

- to keep extending a toy continual-learning benchmark forever
- to prove topological retrieval beats vector retrieval
- to make `StructurePool` itself become a general code-reasoning engine

### What the project *is*

The real long-range goal is:

**to build a small-core cognitive system that can compensate for limited native model capacity through externalized memory, structured control, verification, and selective escalation**

In short:

**not a smaller giant model, but a better-organized small system**

---

## First-Principles View Of Intelligence

For this project, "intelligence" should not be treated as one vague property. It should be decomposed into at least five separable capabilities:

1. **Representation**
   - compress observations into useful internal states

2. **Working memory**
   - preserve intermediate state across multi-step reasoning

3. **Search / roll-out**
   - explore alternatives, decompose, backtrack, compare

4. **Verification / self-correction**
   - detect failure, test hypotheses, repair wrong paths

5. **Control / scheduling**
   - decide when to continue locally, when to retrieve, when to verify, when to escalate

Large models partially internalize all five in their parameters and inference dynamics.

Small models usually fail not because they lack "knowledge" alone, but because these five functions are weak, unstable, or absent.

Therefore the right question is not:

> can a small model become a large model?

The right question is:

> which parts of intelligence must be parameterized, and which parts can be externalized into the system?

---

## Current Deepest Architectural Findings

### 1. Unified-SEL has found a real retention bottleneck

The strongest current mechanistic finding is:

**shared output readout is a core source of forgetting**

From current analysis:

- some seeds forget even when structure weights barely move
- `W_out` drift tracks failure better than structure change alone

This means:

- structure birth / branch / prune solves part of the storage-isolation problem
- but it does not solve the downstream shared-readout interference problem

### 2. There is a deeper route-mismatch issue

In the current implementation:

- training updates the **active structure**
- inference uses a **utility-weighted mixture over all structures**

So the readout is trained on one representation path and evaluated on another.

That means `W_out` is not only a shared bottleneck. It is also compensating for a train/inference routing mismatch.

This makes current forgetting behavior partly architectural, not merely statistical.

### 3. TopoMem is currently a monitor, not a capability amplifier

Current evidence strongly supports:

- vector retrieval remains stronger than topological retrieval on the tested settings
- H1/H2 are most useful as health and boundary-complexity signals
- ECU modulation is promising as a monitoring/control layer

But current TopoMem does **not** yet provide:

- strong reasoning
- strong coding
- strong abstraction

So TopoMem should presently be treated as:

**a sidecar diagnostics and control substrate**

not as the main reasoning core

### 4. StructurePool is not a sufficient long-term reasoning substrate

The current `DFA + StructurePool` setup may still be scientifically useful for studying:

- internal boundary formation
- continual adaptation
- modular retention

But it is not a plausible final architecture for high-grade code reasoning or abstraction.

So it should be judged by the right standard:

- good as a controlled mechanism study
- not expected to scale into full code-intelligence by local tuning alone

---

## Core Contradiction

The deepest contradiction in the current project is:

**the repository is still optimized around "how not to forget", while the original motivation is "how a small system can become more capable"**

These are related, but they are not the same problem.

Not forgetting is only one component of becoming more capable.

If the project remains centered on anti-forgetting metrics alone, it will continue drifting away from the original goal.

---

## Research Reframe

The project should be split into two layers.

### Layer A: Mechanism line

Purpose:

- study controlled mechanisms for modular learning, routing, retention, and self-monitoring

Existing assets to keep:

- `Unified-SEL` structure lifecycle mechanisms
- mechanistic forgetting analysis
- `TopoMem` H1/H2 monitoring and ECU ideas

Role:

- produce interpretable sub-mechanisms
- not claim final strong reasoning capability

### Layer B: Capability line

Purpose:

- study how a small core can achieve stronger reasoning/code performance through system design

This becomes the actual main research target.

This line should evaluate:

- code-task success rate
- multi-step reasoning stability
- abstraction / transfer behavior
- verification success
- cost / latency / escalation tradeoffs

---

## New Primary Thesis

The new primary thesis should be:

**A small model can approach stronger reasoning and coding performance when cognition is externalized into memory, search, verification, and control loops instead of being forced to reside entirely inside model parameters.**

This thesis is much closer to the original motivation than the current EWC-centered framing.

---

## Research Questions

### RQ1. What is the main bottleneck for small-model coding/reasoning?

Possible failure classes:

- insufficient knowledge
- insufficient working memory
- insufficient search
- insufficient verification
- insufficient control policy

The project should stop treating "model weakness" as one undifferentiated problem.

### RQ2. Which cognitive functions can be externalized?

The key test:

- can external memory replace some parameter memory?
- can verification replace some model confidence?
- can search replace some latent reasoning depth?
- can scheduling replace some brute-force scale?

### RQ3. What must remain inside the model?

Likely candidates:

- local semantic compression
- tool selection priors
- decomposition heuristics
- code prior and syntax prior

This question prevents the project from falling into the opposite extreme of assuming orchestration solves everything.

### RQ4. When does a small-core system need escalation?

This is where `TopoMem` may become strategically useful:

- not as the reasoning engine itself
- but as a trigger for "local solve vs verify vs escalate"

---

## Immediate Strategic Decisions

### Decision 1: stop using "EWC vs forgetting" as the whole project definition

It remains a useful mechanism benchmark for `Unified-SEL`.

It should no longer define the project's highest goal.

### Decision 2: explicitly demote topological retrieval as a main promise

Current evidence does not support topological retrieval as the main path to stronger capability.

TopoMem should be positioned around:

- health monitoring
- boundary complexity
- control / escalation signals

### Decision 3: treat Unified-SEL as a mechanism laboratory

Unified-SEL remains valuable for studying:

- modular adaptation
- internal boundary formation
- route protection
- retention under drift

But it should not carry the burden of being the final code-reasoning architecture.

### Decision 4: create a new capability benchmark track

The project now needs tasks that match the true goal:

- code editing micro-tasks
- program repair tasks
- multi-step symbolic reasoning
- decomposition + verification tasks
- tool-mediated problem solving

---

## Most Important Near-Term Technical Work

### Track 1: fix Unified-SEL architectural self-consistency

Before more benchmark tuning:

1. align training route and inference route
2. add explicit readout-protection diagnostics
3. implement `W_out`-only protection as a mechanism study

Why:

- current results are contaminated by route mismatch
- the project needs a cleaner mechanism baseline before making bigger claims

### Track 2: define the small-core capability architecture

Introduce a new conceptual stack:

1. **small core model**
   - local generation and decomposition

2. **external working memory**
   - scratchpad, structured state, subproblem store

3. **executor**
   - code run, tests, tool calls, static checks

4. **verifier / critic**
   - evaluate outputs, detect contradictions, enforce quality gates

5. **scheduler**
   - decide retry, revise, branch, retrieve, or escalate

This is the real architecture needed for the original goal.

### Track 3: turn TopoMem into a scheduler signal source

TopoMem should evolve from:

- "retrieval method candidate"

to:

- "system-health signal for scheduling and escalation"

That means future work should ask:

- when does geometry destabilization predict failure?
- can health signals predict local reasoning collapse?
- can those signals trigger better retry or escalation policy?

---

## What To Keep vs What To Pause

### Keep

- mechanistic analysis of forgetting
- `W_out` stability studies
- H1/H2 health monitoring
- controlled modular-learning experiments

### Pause or demote

- claims that topological retrieval is a main solution path
- broad speculation that `StructurePool` can directly become strong code intelligence through more tuning
- over-centering the whole project around EWC comparison alone

---

## Proposed New North Star

The project's new north star should be:

**Build a small-core system that becomes more capable through externalized cognition, while preserving enough internal structure to remain adaptive, inspectable, and self-monitoring.**

This keeps the original ambition, while giving the project a sharper and more defensible technical direction.

---

## 90-Day Execution Outline

### Phase 1: Mechanism cleanup

- fix `Unified-SEL` route inconsistency
- add `W_out` protection experiments
- produce a clean mechanism report: what structure isolation helps, what readout protection adds

### Phase 2: Capability benchmark creation

- define a small set of code/reasoning tasks suitable for small-core systems
- establish baseline performance for:
  - bare small model
  - small model + tools
  - small model + memory + verification

### Phase 3: Small-core architecture prototype

- implement external working memory
- implement verifier/executor loop
- implement scheduler policy
- optionally use TopoMem health signals as an additional scheduling input

### Phase 4: Escalation policy study

- define clear criteria for local solve vs retry vs escalate
- measure cost/performance frontier
- test whether health-aware control improves reliability

---

## Final Position

The project should no longer be mentally framed as:

> a continual-learning system that may eventually grow into stronger intelligence

It should be framed as:

> a research program on how a small system can become more capable through modular learning, externalized cognition, verification, and control

`Unified-SEL` and `TopoMem` then become valuable components inside that broader program, rather than the final answer by themselves.
