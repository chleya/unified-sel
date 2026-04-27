# Web Research Query List for Learned Meta-Controller

Date: 2026-04-20

Status: research working note. This file records what to ask next, what was found, and how each result should constrain the minimal experiment for a learned unifying mechanism.

Related local docs:

- `META_CONTROLLER_EXPERIMENT_PROTOCOL_2026-04-20.md`
- `F_DRIVE_META_CONTROLLER_REUSE_MAP_2026-04-20.md`

## Core Question

Do not ask "how to build a unified controller" as the main question. Ask:

> How can a learned meta-controller arbitrate between habitual control, deliberative planning, memory read/write, and global broadcast in long-horizon agents under uncertainty, regime shifts, and compute cost?

Hard experimental version:

> What training signals, benchmarks, and ablations can validate that a learned meta-controller has acquired a causal control law for module arbitration, rather than only improving average task performance?

Chinese version:

> 在长程、部分可观测、会发生规则切换且带计算成本的智能体任务中，如何训练并验证一个可学习的元控制器，使它能在习惯策略、审慎规划、记忆读写和全局广播之间做有因果效力的主导权切换？

## Query List

Use these as separate searches. Do not collapse them into one broad search.

### Q1. Habit vs Deliberation Arbitration

Search terms:

- `active inference cognitive control precision optimization habits deliberation`
- `habit deliberation precision control meta cognitive level active inference`
- `expected value of control habit deliberation computational cost surprise`

Found:

- Proietti, Parr, Tessari, Friston, Pezzulo, "Active inference and cognitive control: Balancing deliberation and habits through precision optimization", Physics of Life Reviews, 2025.
- Source: https://www.sciencedirect.com/science/article/pii/S1571064525000879
- PubMed mirror: https://pubmed.ncbi.nlm.nih.gov/40424850/

Useful claim:

- Cognitive control can be framed as optimization of a precision parameter that shifts weight between habitual and deliberative action selection.
- The paper's driving simulation shows a key failure mode: a lower-level system can learn adaptive habits, but without a higher-level control layer it may fail to suspend habits when context changes.
- Relevant control signals include surprise, conflict, control cost, future outcome simulation, and mental effort.

Project implication:

- `deliberation_precision` should be an explicit meta-action, not an internal planner detail.
- Switch latency after regime shift is mandatory.
- Compute cost must be in the reward, or "always deliberate" becomes a fake solution.

### Q2. Global Workspace Selection-Broadcast

Search terms:

- `global workspace theory selection broadcast cycle real-time AI agents`
- `selection broadcast cycle global workspace robotics dynamic real-time environment`
- `global workspace selection broadcast empirical validation performance indicators`

Found:

- Nakanishi, Baba, Yoshikawa, Kamide, Ishiguro, "Hypothesis on the functional advantages of the selection-broadcast cycle structure: global workspace theory and dealing with a real-time world", Frontiers in Robotics and AI, 2025.
- Source: https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1607190/full

Useful claim:

- The paper treats selection and broadcast as a cyclic structure with potential advantages for dynamic, real-time adaptation.
- It frames the global workspace as useful for dynamic thinking adaptation, experience-based adaptation, and immediate real-time adaptation.
- It is a theory/hypothesis paper, so it does not by itself validate an engineering mechanism.

Project implication:

- Broadcast should be measured as a control action, not only assumed as architecture.
- Need an ablation where selection/broadcast is randomized or delayed.
- The project's claim should be stronger than GWT analogy: prove causal contribution through ablations.

### Q3. Agent Memory as Control Policy

Search terms:

- `agent memory control policy write manage read autonomous LLM agents survey`
- `LLM agent memory write manage read control policy taxonomy`
- `policy-learned memory management agent memory survey 2026`

Found:

- Pengfei Du, "Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers", arXiv:2603.07670, 2026.
- Source: https://arxiv.org/abs/2603.07670

Useful claim:

- The survey formalizes memory as a write-manage-read loop coupled with perception and action.
- It adds control policy as a major taxonomy axis, alongside temporal scope and representation substrate.
- It emphasizes the evaluation shift from static recall to multi-session agentic tests where memory and decision making are interleaved.

Project implication:

- Memory read and memory write must be separate meta-actions.
- Memory should not be evaluated only by recall accuracy.
- Metrics need read precision, write precision, and task effect after memory use.

### Q4. Memory Design as Meta-Learned Object

Search terms:

- `Learning to Continually Learn via Meta-learning Agentic Memory Designs ALMA`
- `meta-learning agentic memory designs retrieval update mechanisms`
- `automated meta-learning memory designs sequential decision domains`

Found:

- Xiong, Hu, Clune, "Learning to Continually Learn via Meta-learning Agentic Memory Designs", arXiv:2602.07755, 2026.
- Source: https://arxiv.org/abs/2602.07755
- HF page with project/code links: https://huggingface.co/papers/2602.07755

Useful claim:

- ALMA treats memory design itself as the target of meta-learning.
- The search space includes executable memory designs, database schemas, retrieval mechanisms, and update mechanisms.
- The reported experiments compare learned memory designs across sequential decision-making domains.

Project implication:

- First version should learn a meta-policy over fixed memory operations.
- Later version can lift this into meta-learning over memory design.
- Cross-environment transfer is required, otherwise the system may have learned a script rather than a control law.

### Q5. Bounded Persistent State and Drift Control

Search terms:

- `AI agents need memory control over more context bounded internal state drift`
- `agent cognitive compressor bounded state memory driven drift`
- `transcript replay retrieval memory poisoning drift long horizon agents`

Found:

- Bousetouane, "AI Agents Need Memory Control Over More Context", arXiv:2601.11653, 2026.
- Source: https://arxiv.org/abs/2601.11653

Useful claim:

- Long multi-turn workflows degrade through loss of constraint focus, error accumulation, and memory-induced drift.
- ACC replaces transcript replay with a bounded internal state updated online.
- It separates artifact recall from state commitment, preventing unverified content from becoming persistent memory.

Project implication:

- The shared state `g_t` should be bounded and schema-constrained.
- Drift under horizon is a primary metric, not a secondary bug.
- Memory write should distinguish "available artifact" from "committed persistent state".

### Q6. Uncertainty-Gated Memory Read and Selective Write

Search terms:

- `Oblivion self-adaptive agentic memory control decay-driven activation`
- `uncertainty gated memory retrieval write path reinforcement agentic memory`
- `agent memory read path write path uncertainty buffer sufficiency`

Found:

- Rana, Hung, Sun, Kunkel, Lawrence, "Oblivion: Self-Adaptive Agentic Memory Control through Decay-Driven Activation", arXiv:2604.00131, 2026.
- Source: https://arxiv.org/abs/2604.00131
- Code linked from arXiv: https://github.com/nec-research/oblivion

Useful claim:

- Oblivion decouples memory control into read and write paths.
- Read path decides whether to consult memory from agent uncertainty and memory buffer sufficiency.
- Write path reinforces memories that contributed to the response.
- The paper explicitly targets static and dynamic long-horizon interaction benchmarks.

Project implication:

- Add `uncertainty` and `memory_buffer_sufficiency` or equivalent to `g_t`.
- Write reward should only reinforce memories that causally helped successful outputs.
- Baseline must include `memory-always` and `memory-never`.

### Q7. Agentic Memory Benchmarks

Search terms:

- `MemoryArena benchmarking agent memory interdependent multi-session agentic tasks`
- `multi-session Memory-Agent-Environment loop benchmark memory decision making`
- `LLM agent memory benchmark decision making action interleaved`

Found:

- He et al., "MemoryArena: Benchmarking Agent Memory in Interdependent Multi-Session Agentic Tasks", arXiv:2602.16313, 2026.
- Source: https://arxiv.org/abs/2602.16313
- Project page: https://memoryarena.github.io/

Useful claim:

- Existing memory benchmarks often separate memorization from action.
- MemoryArena tests multi-session loops where agents must learn from earlier actions and feedback, distill memory, and use it in later subtasks.
- It includes web navigation, preference-constrained planning, progressive information search, and sequential formal reasoning.

Project implication:

- The minimal environment should be a Memory-Agent-Environment loop, not a pure recall benchmark.
- At least one phase must require using earlier feedback to solve a later task.
- Preference-constrained planning is a good template for long-term consistency pressure.

### Q8. HRL / Macro-Action Framing

Search terms:

- `hierarchical reinforcement learning macro actions meta controller long horizon`
- `SMDP option selection meta controller compute cost`
- `LLM augmented hierarchical reinforcement learning action primitives long-horizon`

Found:

- Jiang et al., "Hierarchical reinforcement learning based on macro actions", Complex & Intelligent Systems, 2025.
- Source: https://link.springer.com/article/10.1007/s40747-025-01895-9
- Zhang et al., "LLMs augmented hierarchical reinforcement learning with action primitives for long-horizon manipulation tasks", Scientific Reports, 2025.
- Source: https://www.nature.com/articles/s41598-025-20653-y

Useful claim:

- HRL gives a standard formulation for high-level policies selecting macro-actions or options.
- It reduces action-space complexity but can become rigid if macro-actions are fixed and hand-designed.

Project implication:

- Treat meta-controller actions as options/macros:
  - continue habit
  - call planner
  - read memory
  - broadcast interrupt
  - commit state/write memory
- But the experiment must test transfer and ablations, because HRL alone does not prove unified control.

## Recommended Deep Research Prompt

Use this exact English prompt for broader web/deep research:

```text
I am designing a minimal experiment for a learned meta-controller in long-horizon autonomous agents.

The meta-controller observes a bounded internal state and chooses among habitual policy, deliberative planner, memory read/write, and global workspace broadcast. The goal is not simply better task reward, but to show a causal, learned control law for "when to switch, who dominates, what gets broadcast, and what gets written back."

Please survey recent work from 2024-2026 on:
1. active inference / cognitive control models of habit-deliberation arbitration through precision or control cost;
2. global workspace theory and selection-broadcast cycles in AI or robotics;
3. LLM agent memory as write-manage-read control policy, including learned memory management;
4. bounded persistent state, cognitive compression, and memory-induced drift;
5. agentic memory benchmarks where memory and action are interleaved;
6. HRL/contextual-bandit formulations for high-level module arbitration.

For each thread, identify:
- the most relevant papers;
- the exact mechanism they propose;
- what state variables and actions they imply for my meta-controller;
- what metrics and ablations would validate causal control rather than average performance;
- what gaps remain that a minimal experiment could fill.

Prioritize primary sources, arXiv pages, official project pages, and peer-reviewed papers. Avoid generic blog summaries unless they link to reproducible benchmarks or code.
```

Chinese version:

```text
我正在设计一个长程自主智能体中的可学习元控制器实验。

这个元控制器读取一个有界内部状态，并在习惯策略、审慎规划器、记忆读写、全局工作空间广播之间分配主导权。目标不是单纯提高平均任务分数，而是证明系统学到了一个有因果效力的控制律：什么时候切换、切给谁、广播什么、写回什么。

请调研 2024-2026 年相关工作：
1. active inference / cognitive control 中用 precision 或控制成本调节 habit-deliberation 的模型；
2. global workspace theory 中 selection-broadcast cycle 在 AI/robotics 中的实现或假设；
3. LLM agent memory 作为 write-manage-read control policy 的研究，尤其是 learned memory management；
4. bounded persistent state、cognitive compression、memory-induced drift；
5. memory 与 action 交织的 agentic memory benchmark；
6. 用 HRL/contextual bandit 表达高层模块仲裁的工作。

每条线请输出：
- 最相关论文；
- 它提出的具体机制；
- 它暗示我的 meta-controller 应该有什么状态变量和动作；
- 它支持哪些指标和 ablation 来证明因果控制，而不只是平均性能提升；
- 还有哪些空白适合用一个最小实验补上。

优先使用 primary sources、arXiv、官方项目页和同行评审论文；避免只引用泛泛博客，除非博客链接到可复现实验或代码。
```

## Search Outcome Summary

The useful framing is now stable:

1. Active inference gives the habit/deliberation precision-control template.
2. GWT gives the selection-broadcast cycle, but still needs experimental validation.
3. Agent memory research says memory is a control policy, not only a database.
4. ACC-like bounded state gives the anti-drift constraint.
5. Oblivion gives a concrete read/write decoupling pattern.
6. MemoryArena gives a benchmark style where memory and decision making are coupled.
7. HRL gives the formal language for top-level macro-actions, but not by itself the unifying mechanism.

## Next Local Engineering Step

Convert the protocol into code in `experiments/meta_controller/` with the following first slice:

1. Environment with hidden regime shifts, sparse memory dependencies, and compute cost.
2. Fixed modules: habit policy, planner, memory read/write, predictor.
3. Meta-controller baselines:
   - learned contextual bandit
   - fixed priority rule
   - random arbitration
   - planner-always
   - memory-always
4. Metrics:
   - switch latency
   - arbitration regret
   - memory read precision
   - memory write precision
   - drift under horizon
   - recovery after surprise

Acceptance condition:

The learned meta-controller must beat fixed rules on arbitration regret and recovery after surprise under at least one held-out regime-shift pattern, while using less compute than planner-always and less memory access than memory-always.
