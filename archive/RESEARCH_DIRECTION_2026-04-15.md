# 研究方向重塑（2026-04-15）

## 旧主命题（已放弃）

> Unified-SEL 如何打败 EWC？

这个方向经过充分探索后，结论是：
- Toy problem 太简单，无法展示结构池的优势
- 即使修复了 surprise 计算，仍然无法稳定击败 EWC
- 这不是正确的战场

---

## 新主命题

> **小模型系统如何识别自己的能力边界，并通过结构记忆、验证反馈、健康监控和调度策略，在边界区获得能力增益？**

这是一条统一的理论线，连接四个已有基础：
- Unified-SEL：结构状态和结构演化
- TopoMem：健康/漂移/边界复杂度信号
- Double Helix：验证反馈链和能力边界实验
- Structural Bayesian Field：统一理论语言

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

**研究问题**：
- 系统如何识别自己处于 below-boundary / near-boundary / above-boundary？
- 如何决定 local solve / retry / verify / escalate？

---

### 2. 第二价值：结构贝叶斯场

**来源**：Unified-SEL + TopoMem AdapterPool + Structural Bayesian Field

**核心思想**（来自 `STRUCTURAL_BAYESIAN_FIELD_NOTE_2026-04-15.md`）：
不是标准 Bayesian inference，而是**结构化信念状态**。

**对比**：
- 普通贝叶斯：prior + evidence → posterior
- 结构贝叶斯：structure distribution + observation + feedback → evolved structure field

**探索价值**：
> 一个系统如何在不确定环境中持续维护多个候选结构，而不是每次只选一个路径？

**三个可测变量**：
1. **belief weight**：结构/adapter 当前可信度
2. **update operator**：反馈如何改变权重
3. **action policy**：权重分布如何决定 solve / retry / branch / escalate

**现有代码雏形**：
- `core/pool.py:394`：utility 加权结构
- `topomem/adapters.py:370`：effectiveness_score 做 adapter 演化

---

### 3. 第三价值：TopoMem 作为健康信号

**来源**：TopoMem H1/H2 / SelfAwareness / HealthController

**核心洞察**（来自 `RESEARCH_DIRECTION.md:68`）：
TopoMem 目前不是 capability amplifier，而是 **monitor/control substrate**。

**这反而是好事**：
- "拓扑检索是否更准"竞争大，容易被普通向量检索打败
- "拓扑信号能否预测系统失控、领域混杂、边界复杂度、是否需要升级"更独特

**研究命题**：
> H1/H2 不提升答案本身，但能提升**系统调度质量**。

**具体问题**：
- H1 健康度下降时，是否更容易 hallucinate / retrieve wrong context / solve fail？
- H2/H1 上升时，是否表示 domain mixing，系统应该更谨慎？
- TopoMem health signal 是否能提前预测"本地解不动，应该 verify/escalate"？

---

### 4. 第四价值：Unified-SEL 作为机制实验

**来源**：Unified-SEL core + W_out/readout/routing mismatch

**核心发现**（来自 `CLAIM_EVIDENCE_MAP_UNIFIED_SEL.md`）：
- shared readout drift 是遗忘来源之一
- 训练 route 和 inference route 不一致

**机制论文/章节价值**：
- 结构池解决 representation isolation
- 但共享 readout 造成 output interference
- 训练时 active structure，推理时 utility-weighted mixture，导致 route mismatch
- 遗忘不是单纯参数漂移，而是**结构、读出头、路由三者耦合失败**

**注意**：这条线应该服务于大命题，不要继续单独沉迷"toy continual learning 能不能赢 EWC"

---

## 最值得立刻做的探索实验

### 边界感知调度实验

**问题**：
TopoMem/结构信号能不能预测某个 solver-task 是否处于 near-boundary？

**实验设计**：
对每个 task-solver pair 记录：
1. single-shot 是否成功
2. blind retry 是否提升
3. feedback retry 是否提升
4. verifier error 类型
5. TopoMem H1/H2 或 drift/health 信号
6. solver confidence / disagreement / repair-change rate

**目标**：
训练或手写一个 boundary classifier：
- below-boundary
- near-boundary
- above-boundary

**成功标准**：
分类出来的 near-boundary 区间里，feedback_retry 的收益显著高于 blind_retry。

**研究价值**：
回答核心问题：
> 系统能不能知道"什么时候值得努力，什么时候该放弃，什么时候该升级"？

---

## 统一框架图

```
                    ┌─────────────────────────────────────────┐
                    │   能力边界感知的动态结构调度系统          │
                    └──────────────┬──────────────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
┌───────▼───────┐    ┌────────────▼────────┐    ┌──────────▼──────────┐
│  Unified-SEL  │    │      TopoMem        │    │     Double Helix     │
│               │    │                     │    │                      │
│ 结构状态       │    │ 健康/漂移/边界信号   │    │ 验证反馈链            │
│ 结构演化       │    │ H1/H2 健康度       │    │ 能力边界实验           │
│               │    │                    │    │                      │
│ BELIEF WEIGHT │    │  BELIEF WEIGHT    │    │   UPDATE OPERATOR    │
└───────────────┘    └────────────────────┘    └──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │   Structural Bayesian Field      │
                    │                                  │
                    │  结构信念场                       │
                    │  belief weight + update operator │
                    │  + action policy                 │
                    └─────────────────────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │   能力边界调度（核心研究问题）     │
                    │                                  │
                    │  Below-boundary → escalate       │
                    │  Near-boundary → verify/retry    │
                    │  Above-boundary → local solve    │
                    └─────────────────────────────────┘
```

---

## 结论

最有探索价值的是**"能力边界感知的动态结构调度系统"**。

不要把项目缩成：
- Continual Learning
- TopoMem
- Double Helix

它们都只是器官。真正的生命体是：
- **结构信念场** + **拓扑健康监控** + **反馈验证链** + **能力边界调度**
