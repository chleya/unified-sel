# SPEC_HEALTH_ECU.md — 拓扑健康ECU技术规格

> **对应模块**：`topomem/health_controller.py`
> **对应阶段**：Phase N（ECU 扩展）
> **前置依赖**：memory.py（H1/H2 health）、topology.py
> **被依赖方**：`system.py`、`memory.py`

---

## 1. 问题与动机

### 1.1 旧架构的问题

在 ECU 引入之前，系统对 H1/H2 health 的使用是**散乱且硬编码的**：

```
process()
  → if h1_health < threshold: consolidation_pass()   # 只有这一招
```

问题：
- H1 和 H2 健康度各管各的，没有综合视角
- 只能"坏了再修"，无法预测
- 决策阈值是固定的，无法根据健康状态动态调整
- 没有历史数据，不知道健康分是在改善还是在恶化

### 1.2 ECU 的核心思想

ECU = Engine Control Unit（发动机控制单元）

类比：汽车的仪表盘 ECU 不只是看"引擎灯亮了没"，还看：
- 当前车速、转速、油温
- 趋势：油温是否在持续上升
- 预测：按当前消耗速度，燃油还能跑多少公里

**ECU 的目标：把"被动维修"变成"预测性维护"**

---

## 2. 核心抽象速查

| 类/枚举 | 职责 |
|---------|------|
| `HealthStatus` | 某时刻的完整健康快照 |
| `HealthTrend` | 健康趋势分析结果（含斜率、预测步数） |
| `TrendDirection` | 四级预警：GREEN / YELLOW / ORANGE / RED |
| `FaultCode` | OBD 故障码：C001~C005 |
| `HealthControllerConfig` | 所有可配置参数的默认值与调参指南 |
| `TopologyHealthController` | ECU 本身，单一信号源 |

---

## 3. 数据流全景图

```
输入（H1/H2 健康度，来自 memory._update_topology）
     │
     ▼
TopologyHealthController.compute_health_status()
     │
     ├──► HealthScore 计算（weighted_avg / min / geometric）
     │
     ├──► Trend 计算（线性回归 slope + 稳定性判断）
     │
     ├──► 行为参数推导
     │     ├── retrieval_gamma_mult → retrieval 权重
     │     ├── prune_aggressiveness → pruning 强度
     │     ├── consolidate_threshold → 动态阈值
     │     └── cluster_filter_enabled → 干扰过滤
     │
     ├──► 故障码检测（C001~C005）
     │
     └──► 输出：HealthStatus（含 trend + 行为参数）
              │
              ▼
         system.process() 使用
              │
              ├── should_early_intervene() → 趋势恶化预警
              ├── should_consolidate() → 正式触发维护
              ├── get_retrieval_gamma_multiplier() → retrieval 权重
              └── get_prune_aggressiveness() → pruning 强度
```

---

## 4. 健康分数计算

### 4.1 公式选择

```python
health_formula = "weighted_avg"  # 可选: "weighted_avg" | "min" | "geometric"

# weighted_avg（默认）：h1_weight * h1 + h2_weight * h2
# min：min(h1, h2) —— 保守主义，取最差指标
# geometric：sqrt(h1 * h2) —— 惩罚同时衰退
```

### 4.2 动态阈值

```
consolidate_threshold = 0.3 + 0.4 * (1 - health_score)
```

- 健康分 1.0 → 阈值 0.3（宽松）
- 健康分 0.5 → 阈值 0.5（中等）
- 健康分 0.2 → 阈值 0.7（严格）

---

## 5. 四级预警机制（TrendDirection）

| 级别 | 条件（slope 归一化） | 含义 | 系统行为 |
|------|---------------------|------|---------|
| GREEN | slope ≥ 0 | 健康分稳定或上升 | 正常行驶 |
| YELLOW | -0.01 ≤ slope < 0 | 轻微下降 | 降低 retrieval gamma |
| ORANGE | -0.02 ≤ slope < -0.01 | 明显下降 | **提前干预**（轻量 prune） |
| RED | slope < -0.02 | 快速衰退 | **立即 consolidation** |

### 5.1 斜率计算

对最近 `trend_window_size`（默认 10）步的健康分做线性回归：

```python
slope = polyfit(normalized_steps, scores, 1)[0]
# 归一化步距：最早=0，最晚=1
# slope 含义：每"单位步距"健康分变化量
```

### 5.2 稳定性判断

```python
is_stable = variance(scores) < stable_variance_threshold  # 0.005
```

- 波动大但斜率为零（如锯齿波）→ 仍算 GREEN，但不稳定
- 不稳定时，预测置信度降低 50%

---

## 6. 提前干预（Early Intervention）

### 6.1 触发条件

```
should_early_intervene() = True
  当 trend.direction ∈ {ORANGE, RED}
  且 NOT should_consolidate()
```

### 6.2 干预手段

当前实现：轻量级 prune（`aggressiveness=0.1`），不做完整 consolidation。

可扩展方向：
- retrieval 时加大 topology 权重
- 降低 engine 生成多样性（保守策略）
- 主动合并小簇（不触发完整拓扑重计算）

### 6.3 consolidation 后

调用 `health_controller.reset_history()` 清空趋势历史，避免用"修复前"的数据干扰新趋势计算。

---

## 7. OBD 故障码体系

### 7.1 故障码列表

| 码 | 含义 | 触发条件 |
|----|------|---------|
| `C001` | H1 快速衰退 | slope < -0.02 且 H1 明显弱于 H2，steps_until < 10 |
| `C002` | H2 快速衰退 | 同结构，H2 弱于 H1 |
| `C003` | 健康分触发阈值 | `should_consolidate() == True` |
| `C004` | 趋势预警 | `should_early_intervene() == True` |
| `C005` | 不稳定波动 | variance 突增，超过正常2倍阈值 |

### 7.2 去重机制

同一 `step` + 同一 `code` 不重复记录，保留最近 100 条。

### 7.3 日志格式

```python
[ECU] C004 TREND ALERT | direction=orange slope=-0.0150 score=0.580 est=12 steps | faults=['C004']
[ECU] C003 HEALTH THRESHOLD | score=0.280 threshold=0.460 | faults=['C004', 'C003']
```

---

## 8. 动态 Retrieval 策略

| 趋势 | 策略 | 理由 |
|------|------|------|
| GREEN | `vector` | 速度优先，向量检索最快 |
| YELLOW | `hybrid` | 平衡速度和抗噪 |
| ORANGE/RED | `topological` | 抗噪声优先，忽略语义模糊 |

```python
# system.py process() 中
if trend_dir in ("orange", "red"):
    strategy = "topological"
elif trend_dir == "yellow":
    strategy = "hybrid"
else:
    strategy = "vector"
```

---

## 9. Consolidation 效果评估

`consolidation_pass()` 现在返回增强报告：

```python
{
    "orphan_count": 3,
    "merge_count": 1,
    "cluster_count": 5,
    "node_count": 47,
    "topology_updated": True,
    "health_before": 0.280,   # 新增
    "health_after": 0.720,    # 新增
    "health_improvement": +0.440,  # 新增
}
```

`health_improvement > 0` 说明 consolidation 有效；长期可用来评估 ECU 决策质量。

---

## 10. 持久化策略

### 10.1 保存内容

`system.save()` 时写 `health_ecu.json`：

```json
{
  "health_history": [0.9, 0.85, 0.80, ...],
  "step_history": [1, 2, 3, ...],
  "step_counter": 42,
  "fault_log": [
    {"code": "C004", "step": 38, "health_score": 0.58, "details": {...}}
  ]
}
```

### 10.2 恢复

`system.load()` 时读 `health_ecu.json`，调用 `health_controller.load_state()` 恢复。

### 10.3 reset

`system.reset()` 重建 `_health_controller` 实例，清空所有历史。

---

## 11. 与 system/memory 的集成点

```
memory.py
  ├── _health_controller: TopologyHealthController  ← 构造时创建
  ├── _health_status: HealthStatus（缓存）          ← update_topology 时刷新
  ├── get_health_status() → 返回 HealthStatus
  ├── get_fault_log() → 返回故障日志
  └── update_topology() → 调用 compute_health_status() 刷新 _health_status

system.py
  ├── _health_controller: TopologyHealthController  ← __init__ 时创建
  ├── process()
  │     ├── get_health_status() → retrieval 策略决策
  │     ├── should_early_intervene() → 提前干预
  │     └── should_consolidate() → 触发 consolidation
  ├── consolidation_pass()
  │     ├── health_before / health_after → 效果评估
  │     └── reset_history() → 清空趋势
  ├── get_health_dashboard() → OBD 面板数据
  ├── save() → 保存 health_ecu.json
  ├── load() → 恢复 health_ecu.json
  └── reset() → 重建 _health_controller
```

---

## 12. 调参指南

| 参数 | 经验值 | 调大 | 调小 |
|------|--------|------|------|
| `consolidate_trigger_threshold` | 0.3 | 减少 consolidation 频率 | 增加 consolidation 频率 |
| `trend_threshold_slope` | -0.01 | 更敏感，黄区扩大 | 更迟钝，只在快速衰退时告警 |
| `stable_variance_threshold` | 0.005 | 波动大（噪声数据）选大 | 波动小（干净数据）选小 |
| `trend_window_size` | 10 | 反应慢但平滑 | 反应快但易抖 |
| `prune_aggressiveness_max` | 0.5 | 删得多 | 删得少 |
| `retrieval_gamma_min` | 0.0 | 健康差时更保守 | 健康差时仍保持一定语义权重 |

---

## 13. 架构约束与扩展

### 13.1 H3 扩展预留

`HealthControllerConfig` 已有 `h1_weight`、`h2_weight` 字段，加 H3 时：

1. 加 `h3_weight` 配置
2. `compute_health_status()` 加 `h3_health` 参数
3. `health_formula` 扩展支持三元组合

注意：当前数据集（文本嵌入）H3 特征极少且噪声大，H3 对最终效果提升不明显，暂不实现。

### 13.2 约束

- ECU 是**单一信号源**：所有健康相关决策必须经过 `health_controller`，禁止硬编码阈值判断
- Trend 计算在 `compute_health_status()` 中自动进行，无需外部调用
- `get_health_status()` 返回缓存值，`process()` 中每次 retrieval 前会先确保 `update_topology` 已刷新

---

## 14. 测试覆盖

| 测试类 | 覆盖内容 |
|--------|---------|
| `TestHealthStatus` | 数值裁剪、默认值 |
| `TestHealthControllerConfig` | 配置构造 |
| `TestComputeHealthStatus` | 健康分计算、gamma、prune、阈值 |
| `TestShouldConsolidate` | 阈值边界 |
| `TestShouldFilterClusters` | 簇过滤开关 |
| `TestTrendCalculation` | 斜率、稳定性、步数预测、窗口 |
| `TestShouldEarlyIntervene` | 提前干预触发/不触发边界 |
| `TestGetDiagnosticInfo` | 诊断信息完整性 |
| `TestRetrievalGammaMult` | gamma 范围边界 |

---

## 15. 待办与未来扩展

- [ ] **自适应阈值学习**：根据 consolidation 效果（health_improvement）反向调整阈值参数
- [ ] **多车况对比**：存储历史 run 的 health 分曲线，可视化长期趋势
- [ ] **干预策略库**：不只有 prune，可配置多种干预手段及其触发条件
- [ ] **Consolidation 日志**：记录每次 consolidation 的 health_improvement，评估决策质量
- [ ] **告警通道扩展**：当前只有日志，未来可加 webhook / 指标上报
