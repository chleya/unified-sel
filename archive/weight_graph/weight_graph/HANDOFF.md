# Weight Graph 项目交接文档

**最后更新**: 2026-04-12

## 2026-04-11 第二次重定向

本次更新拆分了两条主张，明确了 Task 5 (Neuron 490 Ablation) 为当前 P0 优先级。

详见: `weight_graph/PROJECT_TARGET_REALIGNMENT_2026-04-11.md`

### 核心变化

| 项目 | 旧状态 | 新状态 |
|------|--------|--------|
| **Task 4** (Topo × MMLU) | 关键路径 | **降级**：描述性深度轮廓分析，非预测判定 |
| **Task 5** (Neuron 490) | P1 | **升级**：P0，mechanistic 因果证据 |
| Predictive Prior 主张 | 主线 | **暂挂**：模型错配，需重对齐 |
| Mechanistic 主张 | 辅助 | **主线**：ablation 更可能给出硬结论 |

### Stop/Go 标准

- **Go**：#490 ablation 影响 > 同层 high PageRank 对照
- **Hold**：#490 有影响，但不特异
- **No-Go**：#490 与随机对照无显著差别 → 整体降为 observation sidecar

---

## 2026-04-12 exp09 中间状态

### 已完成（1.5B 模型）

| Condition | PPL | acc | PPL Δ | acc Δ |
|-----------|-----|-----|-------|-------|
| baseline | 6.6265 | 0.5321 | — | — |
| hub_490 | 6.6704 | 0.5335 | **+0.66%** | +0.14% |
| high_pr_control | 6.6012 | 0.5246 | **-0.38%** | -0.75% |
| random_control | 🔄 进行中 | 🔄 | 🔄 | 🔄 |

### 初步 Stop/Go 分析

**条件1 (hub vs high_pr)**：
- PPL：+0.66% > -0.38% → ✅ True
- ACC：+0.14% < -0.75% → ❌ False（ACC 方向相反，不满足严格条件）

**结论**：严格 Stop/Go = **Hold**（ACC 条件不满足），但 PPL 信号极强。
#490 ablation 导致 PPL 上升 0.66%，high_pr ablation 反而让 PPL 略降。
从机制上高度支持 #490 的 hub 特异性。

需 random_control 完整判断条件2。

---

## 已知问题

1. **checkpoint 路径问题**：实验脚本运行在 `weight_graph/` 目录下时，
   checkpoint 写入 `weight_graph/results/weight_graph/exp09/`（相对路径），
   而历史记录可能在 `/f/unified-sel/results/weight_graph/exp09/`（绝对路径）。
   已在 exp09_neuron_ablation.py 中修复为检查多个可能路径。
2. **streaming checkpoint**：原来只在 condition 完成后写入，已改为每个 subject 评估完写入一次。
3. **模型路径**：优先检查 `models/`（相对于 weight_graph）和 `../models/`（父目录）。

---

## 当前 exp09 运行中

- PID: 16423
- 条件: random_control (#100)
- 预计: ~1.5 小时完成
- 日志: /tmp/exp09_final.log
- Checkpoint: weight_graph/results/weight_graph/exp09/neuron_ablation_checkpoint.json
