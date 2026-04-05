# AGENTS.md — Agent 操作手册

本文件是给所有接手本项目的 Agent 看的。**每次开始工作前必须先读完本文件。**

---

## 项目是什么

**Unified-SEL** 是一个研究项目，验证一个核心假设：

> 用"惊讶驱动的结构生灭"机制，在没有任务边界信号的情况下，
> 让模型自主检测环境变化并适应，效果优于需要显式任务边界的方法（如 EWC）。

用人话说：模型自己感知"我遇到了没见过的东西"，然后自动长出新的子模块来处理，
不需要有人告诉它"任务切换了"。

---

## 三个来源项目

本项目整合自三个已有项目：

| 项目 | 位置 | 贡献 |
|------|------|------|
| SEL-Lab | F:\sel-lab | 已验证的持续学习框架，DFA 学习规则，Phase 3 实验 |
| SDAS | F:\SDAS | 惊讶驱动的结构创建机制，structure_pool 的 surprise_history |
| FCRS | F:\fcrs_mis | 工程质量最好的池化实现，可插拔的进化策略 |

**不要去修改这三个项目。只读它们，不改它们。**

---

## 目录结构

```
unified-sel/
├── AGENTS.md              ← 你现在在读的文件，每次必读
├── README.md              ← 项目概述（给人看）
├── EXPERIMENT_LOG.md      ← 实验记录（每次跑完必须更新）
├── STATUS.md              ← 当前进度（每次开始前检查）
│
├── core/
│   ├── structure.py       ← 核心数据结构：Structure 类
│   ├── pool.py            ← 结构池：管理 Structure 的生灭
│   ├── learner.py         ← DFA 学习器（来自 SEL-Lab）
│   └── runtime.py         ← 路径、保存、结果管理
│
├── experiments/
│   ├── continual/
│   │   ├── no_boundary.py ← 主实验：无任务边界，环境连续变化
│   │   └── with_boundary.py ← 对照：有任务边界信号
│   └── baselines/
│       ├── ewc.py         ← EWC 基线
│       └── fixed.py       ← 固定网络基线
│
├── analysis/
│   └── compare.py         ← 读取结果，生成对比表格
│
├── results/               ← 所有实验输出放这里，不要手动编辑
└── tests/
    └── smoke_test.py      ← 快速验证代码没有崩溃
```

---

## 工作流程（每次必须按这个顺序）

### Step 1：读 STATUS.md
了解当前进度在哪里，上一个 Agent 做到了哪一步，下一步是什么。

### Step 2：读 EXPERIMENT_LOG.md
了解已经跑过的实验，避免重复，了解已知问题。

### Step 3：做一件事，只做一件事
STATUS.md 里会写明当前任务。做完这一件事，不要顺手去做其他的。

### Step 4：跑 smoke test
```bash
cd F:\unified-sel
python tests/smoke_test.py
```
如果 smoke test 不通过，不要继续，先修好。

### Step 5：更新 EXPERIMENT_LOG.md 和 STATUS.md
把你做了什么、得到了什么结果、遇到了什么问题，写进去。
**这是最重要的一步。不写记录等于没做。**

---

## 核心概念（必须理解）

### Structure（结构单元）
一个小的子模型，有自己的权重矩阵。
- `tension`：这个结构学习得有多困难（高 = 该 clone 出新结构）
- `surprise`：当前输入对这个结构有多陌生（高 = 该创建新结构）
- `utility`：这个结构有多有用（低 = 该被淘汰）

### 三种事件
- **reinforce**：输入和某个结构很像，强化那个结构
- **branch**：输入有点新，从最近的结构分裂出一个新结构
- **create**：输入完全陌生，直接创建全新结构

### 为什么比 EWC 好（假设）
EWC 需要知道"任务A结束了"才能保护权重。
本系统通过 `surprise` 自动感知"遇到新东西了"，不需要外部信号。

---

## 实验目标（终点）

成功的标准是：

```
在"无任务边界"实验中：
- Unified-SEL 的平均准确率 > EWC（有边界信号版本）
- Unified-SEL 的遗忘率 < EWC
- 统计显著性：5 个随机种子，t-test p < 0.05
```

达到这个标准 = 项目完成。

---

## 禁止事项

- 不要修改 F:\sel-lab、F:\SDAS、F:\fcrs_mis 里的文件
- 不要在没有跑 smoke test 的情况下修改 core/ 里的文件
- 不要跳过写实验记录
- 不要同时推进多个实验
- 不要在 results/ 里手动编辑 json 文件

---

## 如果遇到问题

按顺序检查：
1. 先跑 `python tests/smoke_test.py`，看哪里报错
2. 读 EXPERIMENT_LOG.md 看有没有记录过类似问题
3. 读对应的源项目文档（F:\sel-lab\README.md 等）
4. 只改最小的地方，不要大改
