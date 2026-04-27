# Weight Graph — LLM 权重矩阵的有向图分析框架

## 核心问题

> 训练好的 LLM 权重矩阵转成有向图后，会呈现出什么样的拓扑结构？
> 这些结构和模型能力、模块化、可剪枝性之间有没有关系？

## 研究假设

| 编号 | 假设 | 验证方式 |
|------|------|----------|
| H1 | 训练后的权重图具有显著的社区结构（modularity > 0.3），且高于随机初始化模型 | Louvain community detection，对比 trained vs untrained |
| H2 | 社区结构与模型能力模块化对应——剪掉某个社区主要影响特定 benchmark | 社区级剪枝 + per-benchmark 评估 |
| H3 | 跨层残差路径形成的隐式环路数量与模型推理能力正相关 | 不同规模模型的 cycle count 对比 |
| H4 | PageRank 高的神经元对应模型的"关键路径"，剪掉它们造成的性能下降最大 | PageRank pruning vs magnitude pruning vs random pruning |

## 转换方法

### 权重矩阵 → 有向图

一个线性层 `y = Wx + b`，权重矩阵 `W ∈ R^{d_out × d_in}`：

```
节点：每个神经元（输入侧 d_in 个 + 输出侧 d_out 个）
有向边：input_neuron_i → output_neuron_j，边权 = |W[j, i]|
过滤：只保留 |W[j, i]| > threshold 的边（否则图太密）
```

### 多层堆叠

```
Layer 0:  in_0 → out_0
Layer 1:  in_1 → out_1
...

跨层连接：out_k[i] → in_{k+1}[i]  （维度匹配时直接连）
残差连接：in_k[i] → in_{k+1}[i]    （skip connection，这是环路来源）
```

### Attention 层的特殊处理

Attention 有 Q/K/V/O 四个矩阵，信息流是：

```
input → Q,K,V → attention_scores → weighted_V → O → output
```

转图时把 attention head 视为一个"超节点"，内部不展开。
边权用 `||W_Q||_F * ||W_K||_F * ||W_V||_F * ||W_O||_F` 的几何平均。

## 文件结构

```
weight_graph/
├── README.md              # 本文件（实现说明书）
├── __init__.py
├── config.py              # 配置定义
├── extractor.py           # 从 HuggingFace 模型提取权重矩阵
├── graph_builder.py       # 权重矩阵 → 有向图（核心）
├── analyzers.py           # 图分析算法（社区检测、PageRank、环路检测等）
├── viz.py                 # 可视化
├── utils.py               # 工具函数（稀疏化、归一化等）
└── experiments/
    ├── __init__.py
    ├── exp01_single_layer.py     # 实验1：单层 MLP 图分析
    ├── exp02_full_model.py       # 实验2：全模型图构建 + 社区检测
    ├── exp03_trained_vs_random.py # 实验3：训练模型 vs 随机初始化对比
    ├── exp04_cross_scale.py      # 实验4：不同规模模型对比
    └── exp05_pruning.py          # 实验5：基于图的剪枝实验
```

## 依赖

```
# 核心（必须）
torch                  # 加载模型权重
transformers           # HuggingFace 模型
numpy
scipy                  # 稀疏矩阵

# 图分析（必须）
networkx               # 基础图操作（小图）
graph-tool             # 大规模图分析（可选，Linux only）
python-louvain         # 社区检测（pip install python-louvain）
# 或者用 networkx 内置的 community 模块

# 可视化（可选）
matplotlib
```

## 实现优先级

### Phase 1: 单层可行性验证（目标：2-3 天）

1. `extractor.py` — 从模型提取指定层的权重矩阵
2. `graph_builder.py` — 单层权重矩阵转有向图
3. `analyzers.py` — 基础指标（节点数、边数、degree distribution、密度）
4. `exp01_single_layer.py` — 跑一层 MLP，打印基础指标
5. `viz.py` — degree distribution 直方图

### Phase 2: 社区结构分析（目标：3-4 天）

1. `analyzers.py` 加入 Louvain 社区检测 + modularity 计算
2. `exp03_trained_vs_random.py` — trained vs untrained modularity 对比
3. 如果 modularity 差异显著 → 继续；否则 → 止损

### Phase 3: 多层图 + 环路检测（目标：1 周）

1. `graph_builder.py` 支持多层堆叠 + 残差连接
2. `analyzers.py` 加入 cycle detection（强连通分量 / Tarjan）
3. `exp02_full_model.py` — 全模型图构建
4. `exp04_cross_scale.py` — 0.5B vs 1.5B vs 3B 对比

### Phase 4: 剪枝应用（目标：1 周）

1. `analyzers.py` 加入 PageRank
2. `exp05_pruning.py` — PageRank pruning vs magnitude pruning
3. 对比 LLM-Rank 论文的结果

## 关键实现细节

### 稀疏化策略

全连接权重矩阵转图后边数 = d_in × d_out（一层 MLP 可能有数亿条边）。
必须稀疏化：

```python
# 方法 1：绝对阈值（简单但需要调参）
threshold = np.percentile(np.abs(W.flatten()), percentile)  # 如 top 5%

# 方法 2：每行 top-k（保持入度均匀）
for i in range(d_out):
    row = np.abs(W[i, :])
    top_k_idx = np.argsort(row)[-k:]
    # 只保留 top_k_idx 对应的边

# 方法 3：标准差过滤（保留统计显著的边）
mean_w = np.mean(np.abs(W))
std_w = np.std(np.abs(W))
threshold = mean_w + n_sigma * std_w  # 如 2σ
```

建议 Phase 1 用方法 1（percentile=95），Phase 2+ 用方法 2（k=32 或 64）。

### 规模控制

| 模型 | 参数量 | 单层 MLP 边数 | 全模型节点数 |
|------|--------|--------------|-------------|
| Qwen2.5-0.5B | 0.5B | ~3M (1536×2048×2) | ~100K |
| Qwen2.5-1.5B | 1.5B | ~10M | ~300K |
| Qwen2.5-3B | 3B | ~25M | ~600K |

Phase 1 只做单层，NetworkX 能处理 3M 边。
Phase 2+ 全模型需要 scipy.sparse + 自定义 Louvain，或用 graph-tool。

### 环路检测的注意事项

标准 Transformer 前向传播是 DAG——没有真正的环。
"环"来自两个地方：

1. **残差连接**：如果把 residual stream 的同一维度在不同层的出现视为"同一个节点"
   - 这是概念上的环：Layer k output → Layer k+1 processing → Layer k+1 output → 残差加回 → 和 Layer k input 同一维度
   - 实现时：给 residual stream 的每个维度一个全局节点 ID，跨层复用

2. **注意力的 cross-position 连接**：position i attend to position j，j 又 attend 回 i
   - 这在 token 级图里形成真正的环
   - 但这是 per-input 的，不是权重固有的

**建议先做方法 1（残差维度复用），因为这是权重层面固有的结构。**

### 和 Unified-SEL 现有代码的关系

`weight_graph/` 是独立模块，不依赖 `core/` 的任何代码。
但分析结果可以和 Unified-SEL 的概念做对照：

| weight_graph 概念 | Unified-SEL 对应 |
|---|---|
| 社区 (community) | Structure（一个功能模块） |
| modularity | 模块化程度 |
| PageRank | utility（重要性） |
| cycle | 无直接对应（新发现？） |
| degree distribution | surprise（连接模式的异常性） |

## 实验预期结果

### 最好情况
- H1 成立：trained modularity >> random modularity → 模型自组织模块化
- H2 成立：社区对应能力 → graph-guided pruning 优于 magnitude pruning
- 发现有意义的环路结构 → 和推理能力关联 → 新论文

### 最坏情况
- H1 不成立：modularity 无显著差异 → 权重分布本身就有伪社区结构
- 所有图指标都是 noise → 2-3 天沉没成本，止损

### 止损点
- **Phase 1 结束后**：如果单层 degree distribution 是纯幂律/纯高斯，没有任何有趣结构 → 止损
- **Phase 2 结束后**：如果 trained vs random 的 modularity 差异 p > 0.1 → 止损

## 运行方式

```bash
# Phase 1: 单层分析
cd F:\unified-sel
python -m weight_graph.experiments.exp01_single_layer

# Phase 2: 社区结构
python -m weight_graph.experiments.exp03_trained_vs_random

# 全部实验
python -m weight_graph.experiments.exp01_single_layer
python -m weight_graph.experiments.exp02_full_model
python -m weight_graph.experiments.exp03_trained_vs_random
python -m weight_graph.experiments.exp04_cross_scale
python -m weight_graph.experiments.exp05_pruning
```

---

## 长远愿景：Static Weight Topology as Metacognitive Signal for LLM Routing

### 核心命题

> **LLM 的静态权重图拓扑可以在不跑推理的情况下预测模型在特定任务上的能力边界，
> 从而作为零成本的路由决策信号。**

### 和 "Graph Probing" 论文 (arxiv 2506.01042, ICLR 2026) 的差异化

他们做的是 **运行时** 功能连接图（需要跑推理才能算共激活），我们做的是
**静态** 权重图（加载权重就够了，零推理成本）。如果静态拓扑就能预测能力，
那比运行时方法快几个数量级。

### 和现有 routing 方法的差异化

现有 routing（RouteLLM, BEST-Route, DLPO）都需要先跑一遍小模型，看
confidence / activations，然后决定是否转发。我们的方法是 **query-agnostic
的模型侧信号** ——不看 query，只看模型的"知识结构"，判断它在某个领域行不行。

### 研究路线图

```
Phase 0 (Week 1):  Ground Truth 建立
  → Qwen-0.5B 在 MMLU 57 subjects 上的 per-subject accuracy
  → 这是 ground truth："模型在不同领域的能力指纹"

Phase 1 (Week 2):  静态权重拓扑提取  ← 已完成 (exp01 + exp02)
  → 每层权重图 + 基础指标 + 社区检测 + PageRank

Phase 2 (Week 3-4): 逐层拓扑 × 任务性能关联  ← 核心实验
  → 每层提取 ~10 维拓扑向量
  → 和 MMLU subject accuracy 做回归
  → layer-wise pruning 验证因果性

Phase 3 (Week 5):  跨规模验证 (exp04)
  → 0.5B vs 1.5B vs 3B 的拓扑变化和能力提升的 correlation

Phase 4 (Week 6-7): 路由应用 (exp05 + 新实验)
  → Static Topo Routing vs Confidence Routing vs Surprise Routing
  → quality-cost Pareto curve

Phase 5 (Week 8):  论文写作
```

### 和 Capability Benchmark Track 的衔接

项目已经有了一套完整的 capability routing benchmark
（见 CAPABILITY_BENCHMARK_TRACK.md），包括：
- local_only / local_verify / local_escalate 三级 protocol
- confidence / diagnostic / external / counterfactual 四种 monitor
- surprise_gate routing 初步结果已优于 confidence routing

Weight Graph 的拓扑信号可以作为**第五种 monitor**接入现有 benchmark，
直接和 diagnostic / counterfactual 做对比。

### 和 Unified-SEL 的概念桥接

| Weight Graph 分析 | Unified-SEL 概念 | 元认知解读 |
|---|---|---|
| 社区 modularity | Structure 模块化 | "知识是否有序存放" |
| PageRank 分布 | utility 重要性 | "哪些路径是关键通道" |
| SCC / 环路 | 无直接对应 | "信息是否有反馈回路" |
| degree distribution | surprise | "连接模式的异常性 = 认知混乱度" |

## 参考文献

- [LLM-Rank](https://arxiv.org/abs/2410.13299) — Amazon, 2024. 权重→DAG→PageRank 做剪枝
- [Graph Probing Neural Topology of LLMs](https://arxiv.org/abs/2506.01042) — ICLR 2026. 运行时功能连接图, hub neurons, 拓扑剪枝
- [Circuit Tracing](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) — Anthropic, 2025. 归因图做可解释性
- [Neuronal Group Communication](https://arxiv.org/abs/2510.16851) — 2025. 通信拓扑做高效推理
- [Emergence of modular structure](https://www.science.org/doi/10.1126/sciadv.adm8430) — Science Advances, 2024. 训练中模块化结构涌现
- [RouteLLM](https://github.com/lm-sys/RouteLLM) — LMSYS, 2024. 成本感知 LLM 路由
- [BEST-Route](https://icml.cc/virtual/2025/poster/43788) — ICML 2025. Test-time optimal compute routing
- [DLPO](https://arxiv.org/abs/2603.07972) — AAAI 2026. 元认知策略优化，dual-loop RL routing
