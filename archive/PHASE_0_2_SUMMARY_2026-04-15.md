# Phase 0-3 总结报告（2026-04-15）

## 核心发现

### 1. Toy Problem 的特殊性

**关键洞察**：两个任务的输入分布完全相同，仅标签函数相反。

```
任务 0: y = (x[0] + x[1] > 0.0)
任务 1: y = (x[0] + x[1] < 0.0)
```

**影响**：
- 基于输入统计的特征（input_nonnegative_ratio 等）无效
- 需要基于模型内部状态的特征（confidence, disagreement）
- x[0]+x[1] 符号不能区分任务，只能区分标签

### 2. Oracle 上界的真正含义

**误解**：Oracle 上界 = 0.7863 是可以用 x[0]+x[1] 符号路由达到的

**真相**：Oracle 上界假设已知任务标签，实际场景不可用

```
Oracle 上界 = (snapshot expert task_0 准确率 + current model task_1 准确率) / 2
           = (0.8438 + 0.7289) / 2
           = 0.7863
```

**关键**：这是在 checkpoint 时的准确率，不是最终准确率！

### 3. 专家质量诊断

**实验结果**（3 种子：7, 8, 9）：

| 指标 | Snapshot expert | Current model |
|---|---|---|
| task_0 准确率 | **0.6706** | 0.2878 |
| task_1 准确率 | 0.2982 | **0.7188** |

**Disagreement 分析**：
- task_0: 133 个 disagreement，snapshot 正确 108 次（**81%**）
- task_1: 156 个 disagreement，current 正确 132 次（**85%**）

**关键发现**：
1. **Disagreement 路由逻辑正确**：当 snapshot 和 current disagree 时，选择置信度高的专家是正确的策略
2. **核心瓶颈是专家质量**：snapshot expert task_0 准确率只有 0.6706，远低于 Oracle 上界的 0.8438
3. **准确率差异来源**：Oracle 上界使用 checkpoint 时的准确率，Phase 2 测量最终时的准确率

### 4. 根本原因发现（Phase 3）

**结构池在 task 1 训练时发生剧烈变化！**

以 seed 7 为例：
- Snapshot 保存时：12 个结构 [0-11]
- 最终时：11 个结构 [2, 10, 12-20]
- **被剪枝：10 个结构！**
- **新创建：9 个结构！**

**关键洞察**：
1. Snapshot expert 保存了结构权重，但这些结构在 task 1 训练时被剪枝了
2. Snapshot expert 的结构与当前结构池完全不匹配
3. `_predict_with_snapshot` 使用的是 snapshot 中保存的结构权重，但这些结构已经不在当前结构池中了

## 结论

**核心瓶颈不是路由质量，也不是专家质量，而是结构池剧烈变化。**

具体来说：
1. Disagreement 路由策略是正确的（81% 和 85% 的正确率）
2. 但 snapshot expert 的结构与当前结构池不匹配
3. 结构池在 task 1 训练时发生了剧烈变化（10 个结构被剪枝，9 个新结构创建）

## 下一步方向

### 方向 1：冻结结构池（最直接）

**问题**：结构池在 task 1 训练时发生剧烈变化

**解决方案**：
- 在 checkpoint 后冻结结构池，禁止创建新结构和剪枝旧结构
- 只更新现有结构的权重
- 这样 snapshot expert 的结构就不会被剪枝

### 方向 2：保存完整的模型状态

**问题**：Snapshot expert 只保存了结构权重，不包含结构池状态

**解决方案**：
- Snapshot expert 应该保存完整的模型状态，包括：
  - 结构池的所有结构
  - 每个结构的权重、反馈、local_readout
  - W_out
- 在使用 snapshot expert 时，恢复完整的模型状态

### 方向 3：实现结构级别的保护

**问题**：当前没有机制保护重要的结构不被剪枝

**解决方案**（来自 SEL-Lab）：
- 为每个结构设置锚点
- 实现双路径更新：memory_path_only + current_path_boost
- 当结构年龄超过阈值时，启用锚点正则化
- 标记重要结构为“不可剪枝”

### 方向 4：使用独立的专家网络

**问题**：Snapshot expert 和 current model 共享结构池

**解决方案**：
- 为每个任务创建独立的专家网络
- 专家网络之间不共享结构池
- 路由时选择对应的专家网络

## 推荐执行顺序

1. **立即执行**：实现结构池冻结（最简单，最直接）
2. **短期**：验证结构池冻结的效果
3. **中期**：实现结构级别的保护机制
4. **长期**：研究独立的专家网络架构

## 数据来源

所有结论基于真实验证，非模拟数据。

实验脚本：
- `experiments/phase0_theoretical_analysis.py`
- `experiments/phase1_oracle_routing_test.py`
- `experiments/phase2_expert_quality_analysis.py`
- `experiments/phase3_best_snapshot_test.py`
- `experiments/phase3_5_diagnose_snapshot.py`

结果文件：
- `results/phase2_expert_quality/20260415_113016.json`
- `results/phase3_best_snapshot/20260415_115334.json`
