# Weight Graph 项目目标重定向

**日期**: 2026-04-11 (第二次更新)
**状态**: 生效
**适用范围**: `F:/unified-sel/weight_graph/`

---

## 1. 决策

2026-04-10 的第一次重定向将项目目标从"全图 Louvain 跑通"转向"拓扑信号预测能力验证"。

本次（第二次）重定向将项目目标进一步拆分为两条独立主张，并明确优先级。

---

## 2. 两条主张及当前状态

### 主张 A：静态拓扑信号可作为路由先验（predictive prior）

**目标**：证明图的拓扑特征对模型能力或路由具有预测价值。

**当前状态**：本轮未证成（NOT YET VALIDATED）。

原因：
1. **模型错配**：exp06 用 1.5B 模型跑 MMLU，exp07 的 topo_matrix 从 0.5B 构建。两者不对齐，相关性分析无效。
2. **实验设计缺陷**：当前 exp08 实现是描述性深度轮廓分析（layer-depth profile），不是预测性回归/判定实验。
3. **功效不足**：大多数 topo feature 沿深度变化极小（slope ≈ 0），现有方法捕获不到信号。

**结论**：Task 4 降级为 descriptive depth-profile analysis，不支撑 predictive prior 主张。

**未来重启条件**（需同时满足）：
- 重跑 0.5B MMLU ground truth，或重建 1.5B topo matrix，使模型对齐
- 重新设计实验：用 per-layer activation 数据或 layer ablation 数据，而非粗糙的层段均值聚合
- 在此之前，不向 predictive prior 方向追加解释资源

---

### 主张 B：Hub Neuron 490 承担特异功能（mechanistic interpretation）

**目标**：验证 hub neuron #490 的因果重要性，并分析其功能特异性。

**当前状态**：待验证（TO BE TESTED）。

依据：
- PageRank 全模型第一（L23_mlp_in_490），L15-L18 也在 top 10
- 说明 #490 是跨层信息汇聚的关键维度
- 比主张 A 更可能给出因果证据

---

## 3. 当前任务优先级

### P0（第一优先级）：Task 5 — Neuron 490 Ablation + 分析

**目标**：获取 hub neuron 的因果证据，从"它很重要"推进到"它承担什么功能"。

**最小实验包**：

1. **Ablation 实验**（3 个 condition）：
   - baseline：完整模型
   - ablate #490：所有层的 hidden_dim[490] 置零
   - **同层 high PageRank 对照**：找一个非 #490 的高 PageRank neuron（如 L23 PageRank top5 中非 #490 的）置零
   - **随机对照**：随机选一个 neuron（如 #100）置零

2. **评估指标**：
   - 整体 perplexity 变化
   - MMLU 整体准确率变化
   - 分学科（STEM / Humanities / Social）准确率变化

3. **判断标准（Stop/Go）**：
   - **Go**：#490 ablation 的影响 > 同层 high PageRank 对照的影响（说明不只是"任何 hub 都重要"）
   - **Hold**：#490 有影响，但和 high PageRank 对照处于同一量级（重要性不特异）
   - **No-Go**：#490 与对照无显著差别 → mechanistic 线也降级

**配套分析**（ablation 之后的自然延伸）：
- #490 在各层、各输入类型、各学科上的激活强度分布
- 目标：把 narrative 从"它很重要"推进到"它可能承担什么功能"

**预计资源**：3 次模型加载 + ~5 个代表性 MMLU subject 评估 ≈ 3-4 小时

### P1（暂缓）：Predictive Prior 主张

- 必须先解决模型对齐问题
- 当前不占主线资源

### P2（观察性留存）：H1 + Hub Neuron 描述结论

- H1（trained vs random modularity）仍然成立，可以支撑观察性 claim
- Hub Neuron 490 的 PageRank 地位仍然成立
- 但均不升级为论文核心贡献，除非 Task 5 有强结果

---

## 4. Claim 重分级

### 4.1 成立（可保留）

- `C1`：训练后的静态权重图呈现强于随机初始化的模块化结构（trained=0.886 vs random=0.562）
- `C2`：静态拓扑分析可提取稳定的逐层和全局结构特征
- `C3`：Hub neuron #490 是跨层信息汇聚的关键维度（PageRank 全模型第一）

### 4.2 降级为观察性

- ~~"静态拓扑信号可预测模型能力差异"~~ → 本轮未证成，只能描述性记录"中层拓扑复杂度峰值"
- ~~"全图 modularity 的多 seed 显著性已严格确认"~~ → 单 seed，降级为观察性发现

### 4.3 新增待验证

- `C4`：Hub neuron #490 对模型能力有因果特异性贡献（Ablation 验证中）

---

## 5. 硬件评估

- Task 5（3 condition × 5 subject ablation）≈ 3-4 小时，无需 GPU，可行
- 瓶颈不在硬件，在实验设计和对照设置

---

## 6. 预计排程

| Day | 任务 |
|-----|------|
| Day 1 | 设计并实现 Neuron 490 ablation 脚本（含同层 high PageRank 对照） |
| Day 2 | 预跑验证脚本可行性，修复问题 |
| Day 3-4 | 正式跑 3 个 ablation condition |
| Day 5 | 出初步判断（Go/Hold/No-Go） |
| Day 6+ | 若有强结果，启动 activation/profile 分析 |

---

## 7. Stop/Go 决策表

| Day 5 结果 | 决策 | 后续 |
|------------|------|------|
| #490 影响 > high PageRank 对照 | **Go** | 继续 mechanistic line，Task 5 升级为 P0主线 |
| #490 影响 ≈ high PageRank 对照 | **Hold** | mechanistic 线降级，predictive prior 重开调研 |
| #490 影响 ≈ 随机对照 | **No-Go** | weight_graph 整体降为 observation sidecar |

---

## 8. 一句话版本

`weight_graph` 主线从"拓扑预测能力"转向"hub neuron 因果机制"——前者未证成暂挂，后者更可能给出硬结论。
