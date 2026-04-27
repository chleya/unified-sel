# PROJECT_MAINLINE_2026-04-09

## 核心结论

这个项目现在必须明确分成两层，而不能再把所有目标混在一起：

1. `Unified-SEL` 是当前主线，但它的角色是**机制研究主线**
2. “小模型获得更强推理、抽象、代码能力”仍然是最终目标，但它是**能力主线**
3. 这两层有关联，但不是同一个问题，不能再用一套实验同时承担两者

如果我是项目主负责人，我会明确宣布：

> 现在的主线不是“把 StructurePool 调成一个能做通用推理/代码的系统”，而是“把现有资产收束成一个清晰、可发表、可复验的机制研究结果”，同时为后续真正的能力研究搭桥。

---

## 先回答一个根问题

### 我们还在主线上吗？

在，但要说清楚是**哪条主线**。

当前真正成立的主线是：

- `Unified-SEL` 机制线：
  - endogenous boundary formation
  - modular retention
  - shared-readout interference
  - seed variance / failure modes

当前还不成立的主线是：

- “当前 `StructurePool` 会继续演化成通用 reasoning / coding core”

后者不是短期主线，只能是远期研究愿景。

---

## 我对项目现状的判断

### 已经证明的东西

1. `W_out` / shared readout 是独立遗忘源，不是附带噪声
2. 结构生灭不是假的，边界形成和失稳都能被诊断出来
3. cleaned-route 下，轻量 `W_out` 保护已经能在均值上优于 EWC
4. selective readout 是真实机制杠杆，但当前没有找到稳定的中间区间
5. TopoMem 在当前证据下更像监控/控制侧车，而不是核心推理引擎

### 已经基本证伪的东西

1. 继续靠阈值和局部控制器微调，就能把 `StructurePool` 推成通用推理系统
2. topological retrieval 会在当前设定里优于 vector retrieval
3. 只要消灭 shared `W_out`，遗忘问题就自然解决

### 当前最危险的偏航

最危险的不是结果差，而是**问题定义混乱**：

- 一会儿在做 continual learning benchmark
- 一会儿在做拓扑记忆
- 一会儿又想做 reasoning / coding
- 但这几类问题的评价标准完全不同

如果不切开，项目会一直有进展，但不会有结论。

---

## 我会怎么定义两条线

## 线 A：机制研究主线

这条线就是现在的正式主线。

### 目标

回答一个足够尖锐的问题：

> 在 modular continual learner 里，shared readout drift 是否构成独立的灾难性遗忘来源？输出隔离和读出保护分别能解释什么，不能解释什么？

### 为什么这条线值得继续

因为它已经有可写论文的资产：

- cleaned-route baseline
- `lambda=10` 的平衡点
- seed-level failure decomposition
- `exclusive_local` 负结果
- strict-gated local readout 近乎 no-op 的控制结果

这套资产已经不是“探索中随手试了几个点”，而是能组成一条机制叙事。

### 这条线的正确产出

不是“最终智能系统”，而是：

- 一篇机制论文
- 一套清晰 failure mode taxonomy
- 一组有正结果也有负结果的 readout ablation
- 一个干净、可复验的 benchmark protocol

---

## 线 B：能力研究主线

这条线仍然重要，但现在只能作为下一阶段主线，不该继续压在 `StructurePool` 身上。

### 真正要回答的问题

> 小模型的能力边界，哪些来自参数规模，哪些可以通过 memory / search / verification / control 被外化补偿？

### 这条线不等于“做个普通 agent”

如果只是拼一个常见 agent 壳子，那确实没有研究价值。

真正值得做的是：

- 不把 orchestration 当黑箱
- 明确研究哪类能力可被外化
- 明确什么时候 local solve 足够，什么时候必须 escalate
- 明确外部控制信号是否真的提升质量/成本比

也就是说，研究对象不是“agent 框架”，而是：

- control policy
- routing signal
- verification loop
- capability decomposition

这才和你最初“小模型实现大能力”的问题同一条线上。

---

## 对“别人那条建议”的判断

外部建议里，最有价值的部分不是“去做 routing”本身，而是两点：

1. 它正确地区分了“机制线”和“能力线”
2. 它正确指出 surprise / tension 更像 routing / escalation signal，而不是推理本体

但我不会完全照搬。

### 我会采纳的部分

- 停止把 `StructurePool` 当成最终通用推理架构
- 先收割 `Unified-SEL` 机制论文
- 把 TopoMem 重新定位为 monitor / signal source
- 后续能力线优先研究 routing / verification / escalation，而不是继续堆 toy continual learning

### 我不会直接跳过去的部分

- 现在立刻全面切去 LLM routing 作为唯一主线

原因很简单：

- 你手里现成、最硬、最接近论文的资产还在 `Unified-SEL`
- routing 线要重新定义任务、baseline、成本函数、评测协议
- 如果现在硬切，会把已经形成的机制发现半途丢掉

所以正确做法不是“换主线”，而是：

> 先把机制线结案，再把能力线立项。

---

## 原始初心有没有问题

没有。

“探索小模型如何获得更强推理、抽象、代码能力”这个初心本身是对的，而且是一个比“训练更大的模型”更有研究味道的问题。

真正的问题不在初心，而在两个隐含假设：

1. 误把“持续学习不遗忘”当成了“更强智能”的核心代理指标
2. 误把当前 `StructurePool` 架构当成了承载最终能力目标的候选本体

前者太窄，后者太强。

一旦这两个假设拆掉，项目就顺了：

- `Unified-SEL` 负责回答“模块化学习与遗忘机制”
- 后续能力线负责回答“外化控制如何补足小模型能力”

---

## 现在该砍掉什么

以下内容不该再作为主投入方向：

1. 继续做大量 `lambda` 扫描
2. 继续把 selective-readout 当作 benchmark default 去追分
3. 继续宣称 TopoMem 检索是核心推理路线
4. 继续把 `StructurePool` 调参当作通向代码智能的主路径

这些不是完全没价值，而是不再配得上“主线资源”。

---

## 接下来 3 个最合理的工作包

## WP1：锁定机制论文主叙事

目标：

- 把 `Unified-SEL` 的主叙事从“泛泛 anti-forgetting”收束成“shared readout interference + output isolation tradeoff”

最低交付：

- baseline cleaned-route result
- shared-readout protected result
- selective local-head tradeoff
- `exclusive_local` negative ablation
- strict-gated near-no-op control

一句话叙事：

> 共享读出漂移是真实问题，但简单的输出隔离不是充分解；遗忘来自 shared head 与 routing / specialization 的共同作用。

## WP2：只允许一个窄 probe

如果还要继续做机制实验，只做一个：

- boundary-pressure-conditioned local residual

要求：

- 只打 hard seeds `8/9`
- 只验证“是否存在比当前 strict gate 更有效、但不失控的中间区间”
- 不再做泛化 sweep

如果这个 probe 仍然不形成清晰中间区间，就停止 selective-readout 线，直接写 paper story。

## WP3：立能力线的 benchmark，而不是立能力线的系统

下一阶段不先做大系统，先做 benchmark definition：

- code edit micro-task
- short-horizon reasoning task
- decomposition + verification task
- local-only / local+verify / local+escalate 三种 protocol

先定义评价，再决定架构。

---

## 未来 4-8 周的实际路线

### 第 1 阶段

完成 `Unified-SEL` 机制线收束：

- 补一轮必要的窄 probe 或直接停止 probe
- 整理图表、结论、负结果
- 固化 paper outline

### 第 2 阶段

立能力线 benchmark：

- 不急着做完整 agent
- 先明确任务、代价、容错、调用约束
- 明确什么叫“能力提升”，而不是只看答对率

### 第 3 阶段

再决定是不是走：

- routing / escalation
- verification-centric small-core system
- 或别的更合适的 capability architecture

---

## 最终定位

如果我要给这个项目一句最准确的定义，我会这样写：

> 这是一个分层研究计划：近端用 `Unified-SEL` 研究模块化持续学习中的结构边界与共享读出遗忘机制，远端研究小模型如何通过外化记忆、验证与控制获得超出其参数规模的系统能力。

这才同时保住了两件事：

1. 不浪费你已经做出来的真实研究资产
2. 不背离你最初想追的问题
