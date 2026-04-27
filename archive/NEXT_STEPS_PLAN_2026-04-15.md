# 下一步行动计划（2026-04-15）

**目标**：在 toy problem 上击败 EWC（avg_acc > 0.5005，p < 0.05）

**核心洞察**：Oracle 上界 = 0.7863，瓶颈是路由质量，不是学习质量

---

## 🎯 阶段 0：理论分析（今天完成，1-2 小时）

### 目标
在实现之前，先分析为什么当前方法失败，以及新方法为什么可能成功。

### 具体任务

#### 0.1 分析 toy problem 的特殊性
- [ ] 确认任务 0 和任务 1 的决策边界：`y = (x[0] + x[1] > 0.0)` vs `y = (x[0] + x[1] < 0.0)`
- [ ] 分析输入分布：x[0]+x[1] 的符号是否能作为路由信号？
- [ ] 计算理论上的完美路由准确率：如果用 x[0]+x[1] 符号路由，能达到多少？

#### 0.2 分析任务签名特征的适用性
- [ ] SEL-Lab 的 6 维特征在什么场景下有效？
- [ ] 在 toy problem 上，哪些特征最可能有区分能力？
  - `confidence`：可能有用（不同任务可能有不同的置信度分布）
  - `input_abs_mean`：可能有用（不同任务的输入分布可能不同）
  - `input_nonnegative_ratio`：**最可能有用**（任务 0 和任务 1 的输入符号分布可能不同）
  - `input_zero_ratio`：可能不太有用（toy problem 输入不太可能有零）
  - `conflict_score`：需要多个专家才有意义
  - `conflict_peak`：需要多个专家才有意义

#### 0.3 预测最小可行验证的效果
- [ ] 如果只用 `input_nonnegative_ratio` 做路由，预期准确率是多少？
- [ ] 如果用 `confidence + input_abs_mean + input_nonnegative_ratio`，预期准确率是多少？

**成功标准**：完成理论分析，明确哪些特征最可能有效

**时间限制**：今天完成（1-2 小时）

---

## 🚀 阶段 1：最小可行验证（明天完成，3-4 小时）

### 目标
用最简单的特征，快速验证方向是否正确。

### 具体任务

#### 1.1 实现简单的特征计算（1 小时）

在 `learner.py` 中添加：

```python
def _compute_simple_features(self, x: np.ndarray) -> Dict[str, float]:
    """计算 3 个最简单的特征"""
    output = self.predict_proba_single(x)
    
    return {
        "confidence": float(np.max(output)),
        "input_abs_mean": float(np.mean(np.abs(x))),
        "input_nonnegative_ratio": float(np.mean(x >= 0)),
    }
```

#### 1.2 实现简单的阈值路由（1 小时）

修改 `_ensemble_predict`：

```python
def _ensemble_predict(self, x: np.ndarray) -> np.ndarray:
    """基于简单特征的路由"""
    current_output = self.predict_proba_single(x)
    if not self._snapshot_experts:
        return current_output
    
    features = self._compute_simple_features(x)
    
    # 简单的阈值路由
    # 如果输入大部分是非负的，可能是任务 0，用快照专家
    if features["input_nonnegative_ratio"] > 0.5:
        snap_output = self._predict_with_snapshot(x, self._snapshot_experts[0])
        return snap_output
    else:
        return current_output
```

#### 1.3 快速测试（1 小时）

创建 `experiments/A1_fix7_simple_features_test.py`：

```python
# 只测试 3 个种子，快速验证
SEEDS = [7, 8, 9]

# 测试不同的路由策略
strategies = {
    "baseline": "无路由",
    "nonnegative_ratio": "只用 input_nonnegative_ratio",
    "all_3_features": "用全部 3 个特征",
}

# 运行并比较结果
```

**成功标准**：
- avg_acc 提升 ≥ 0.02（相比 baseline 0.4975）
- 或 task_0 准确率提升 ≥ 0.05

**失败应对**：
- 如果效果不明显（< 0.02），分析原因：
  - 特征是否有区分能力？
  - 阈值是否合适？
  - 是否需要更多特征？
- 如果特征无区分能力，转向其他方向（如 Wasserstein 漂移）

**时间限制**：明天完成（3-4 小时）

---

## 📊 阶段 2：完整特征实现（如果阶段 1 成功，2-3 天）

### 目标
实现 SEL-Lab 的完整 6 维任务签名特征。

### 具体任务

#### 2.1 实现完整的特征累积（1 天）

在 `learner.py` 中添加：

```python
def __init__(self, ...):
    # 添加任务统计累积
    self.task_stat_sums = {
        "confidence": 0.0,
        "conflict_score": 0.0,
        "conflict_peak": 0.0,
        "input_abs_mean": 0.0,
        "input_nonnegative_ratio": 0.0,
        "input_zero_ratio": 0.0,
        "steps": 0,
    }

def fit_one(self, x: np.ndarray, y: np.ndarray) -> float:
    # ... 现有代码 ...
    
    # 累积统计
    output = self.predict_proba_single(x)
    self.task_stat_sums["confidence"] += float(np.max(output))
    self.task_stat_sums["input_abs_mean"] += float(np.mean(np.abs(x)))
    self.task_stat_sums["input_nonnegative_ratio"] += float(np.mean(x >= 0))
    self.task_stat_sums["input_zero_ratio"] += float(np.mean(x == 0))
    self.task_stat_sums["steps"] += 1
    
    # 如果有多个专家，计算 conflict
    if len(self._snapshot_experts) > 0:
        snap_output = self._predict_with_snapshot(x, self._snapshot_experts[0])
        conflict = np.abs(output - snap_output)
        self.task_stat_sums["conflict_score"] += float(np.mean(conflict))
        self.task_stat_sums["conflict_peak"] += float(np.max(conflict))
```

#### 2.2 实现特征向量提取（半天）

```python
def _task_signature_feature_vector(self) -> np.ndarray:
    """返回 6 维任务签名特征向量"""
    steps = max(1, self.task_stat_sums["steps"])
    return np.array([
        self.task_stat_sums["conflict_score"] / steps,
        self.task_stat_sums["conflict_peak"] / steps,
        self.task_stat_sums["confidence"] / steps,
        self.task_stat_sums["input_abs_mean"] / steps,
        self.task_stat_sums["input_nonnegative_ratio"] / steps,
        self.task_stat_sums["input_zero_ratio"] / steps,
    ])
```

#### 2.3 实现基于特征向量的路由（半天）

```python
def _ensemble_predict_with_features(self, x: np.ndarray) -> np.ndarray:
    """基于任务签名特征的路由"""
    current_output = self.predict_proba_single(x)
    if not self._snapshot_experts:
        return current_output
    
    # 获取当前特征
    features = self._compute_simple_features(x)
    
    # 获取历史平均特征
    historical_features = self._task_signature_feature_vector()
    
    # 计算与历史特征的相似度
    # 如果当前特征与历史特征差异大，说明可能是新任务
    # ... 实现细节 ...
```

#### 2.4 15 种子正式测试（半天）

创建 `experiments/A1_fix8_task_signature_15seed.py`

**成功标准**：
- avg_acc 提升 ≥ 0.03（相比 baseline 0.4975）
- 或 task_0 准确率提升 ≥ 0.10
- 统计显著性 p < 0.10（可以放宽，因为样本量小）

**失败应对**：
- 如果效果仍不明显，分析：
  - 是否需要更多任务来训练特征？
  - 是否需要不同的路由策略？
- 考虑转向 Wasserstein 漂移或结构贝叶斯场

**时间限制**：2-3 天

---

## 🌊 阶段 3：Wasserstein 漂移（如果阶段 2 失败，3-5 天）

### 目标
用 TopoMem 的 Wasserstein 漂移作为更强的路由信号。

### 具体任务

#### 3.1 实现简单的分布漂移检测（2 天）

不需要完整的 TopoMem，只需要：

```python
class DistributionDriftDetector:
    def __init__(self, history_size: int = 100):
        self.input_history = []
        self.history_size = history_size
    
    def observe(self, x: np.ndarray):
        self.input_history.append(x.copy())
        if len(self.input_history) > self.history_size:
            self.input_history.pop(0)
    
    def compute_drift(self, x: np.ndarray) -> float:
        """计算当前输入与历史分布的漂移"""
        if len(self.input_history) < 10:
            return 0.0
        
        # 简单的漂移度量：均值和方差的变化
        historical_mean = np.mean(self.input_history, axis=0)
        historical_std = np.std(self.input_history, axis=0)
        
        current_mean = x
        drift = np.mean(np.abs(current_mean - historical_mean) / (historical_std + 1e-6))
        
        return float(drift)
```

#### 3.2 集成到路由策略（1 天）

```python
def _ensemble_predict_with_drift(self, x: np.ndarray) -> np.ndarray:
    """基于分布漂移的路由"""
    drift = self.drift_detector.compute_drift(x)
    
    # 如果漂移大，说明是新任务，用当前模型
    # 如果漂移小，说明是旧任务，用快照专家
    if drift > self.drift_threshold:
        return self.predict_proba_single(x)
    else:
        return self._predict_with_snapshot(x, self._snapshot_experts[0])
```

**成功标准**：
- avg_acc 提升 ≥ 0.03

**失败应对**：
- 如果效果不明显，说明 toy problem 太简单，分布漂移不明显
- 考虑转向结构贝叶斯场或扩展到更复杂的任务

**时间限制**：3-5 天

---

## 🧬 阶段 4：结构贝叶斯场（长期方向，1-2 周）

### 目标
实现你分析中的"结构贝叶斯场"思想。

### 具体方向

#### 4.1 强化 AdapterPool 的结构分布特性
- 让 effectiveness_score 更像"信念权重"
- 多 adapter 同时激活（加权组合，不是 Winner-Take-All）

#### 4.2 实现结构连续变形
- 类似 SDAS 的原型滑动平均
- Structure 权重的连续流动

#### 4.3 Inference 和 Learning 不分离
- 每一步都可以决策
- 边学习，边决策

**成功标准**：
- 遗忘率降低 ≥ 0.1
- 或 avg_acc 提升 ≥ 0.05

**时间限制**：1-2 周

---

## 📈 成功标准与时间限制总结

| 阶段 | 目标 | 成功标准 | 时间限制 | 失败后转向 |
|---|---|---|---|---|
| 0：理论分析 | 明确方向 | 完成分析 | 今天（1-2h） | - |
| 1：最小验证 | 快速验证 | avg_acc +0.02 | 明天（3-4h） | 分析原因 |
| 2：完整特征 | 完整实现 | avg_acc +0.03 | 2-3 天 | Wasserstein |
| 3：Wasserstein | 更强信号 | avg_acc +0.03 | 3-5 天 | 结构贝叶斯 |
| 4：结构贝叶斯 | 理论突破 | 遗忘率 -0.1 | 1-2 周 | 扩展任务 |

---

## ⚠️ 风险管理

### 风险 1：toy problem 太简单
- **表现**：所有方法效果都不明显
- **应对**：扩展到 5+ 任务、非线性决策边界

### 风险 2：特征无区分能力
- **表现**：任务签名特征在 toy problem 上无效
- **应对**：转向 Wasserstein 漂移或结构贝叶斯场

### 风险 3：计算成本过高
- **表现**：Wasserstein 距离计算太慢
- **应对**：使用简单的分布漂移度量

### 风险 4：过度工程化
- **表现**：参数越来越多，效果不明显
- **应对**：回归简单方法，减少参数

---

## 📝 每日检查清单

每天开始工作前，检查：

- [ ] 昨天的实验结果如何？
- [ ] 是否达到成功标准？
- [ ] 是否需要调整方向？
- [ ] 是否需要更新 STATUS.md 和 EXPERIMENT_LOG.md？

---

## 🎯 最终目标

**在 toy problem 上击败 EWC**：
- avg_acc > 0.5005
- p < 0.05（统计显著）
- 遗忘率 < EWC（可选）

**如果失败**：
- 分析原因，写清楚为什么失败
- 扩展到更复杂的任务，重新验证

---

**开始时间**：2026-04-15
**预计完成时间**：2026-04-22（1 周）
