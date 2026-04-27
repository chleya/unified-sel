# LeWorldModel → 三项目集成 Preflight Spec

**日期**: 2026-04-22
**Spec Status**: Draft / Not Implementation-Ready
**红线**: 不写大系统，不做代码实现，只做机制提取 + 映射设计 + 可测试不变量

> **审稿说明**: 本文档是 preflight 草案，不可直接进入实现。需要第三轮压成 implementation ticket 后再动代码。
> 第一阶段只做 unified-sel (P0)，CEE 和 CEP-CC 等 P0 信号验证过再接。

---

## 任务包 1: 论文机制精读 — 工程可迁移机制提取

### 1.1 核心训练目标

LeWM 的完整训练目标：

```
L_LeWM = L_pred + λ × SIGReg(Z)
```

| 组件 | 公式 | 作用 | 超参数 |
|------|------|------|--------|
| L_pred | MSE(ẑ_{t+1}, z_{t+1}) | 下一帧隐变量预测 | 无 |
| SIGReg | Σ EP(P(θᵀZ)) | 强制隐变量服从各向同性高斯 | λ (唯一) |

> **VERIFY AGAINST PDF**: SIGReg 的具体实现是否使用 Epps-Pulley 正态性检验统计量，
> 以及 M=1024 个随机方向等参数，均来自二手解读，未核对论文正文。
> 编码前必须对照 arXiv 2603.19312 PDF 确认。

**关键简化**: 从 PLDM 的 6 个可调超参数 → 1 个 (λ)，甚至可用二分搜索自动找。

### 1.2 架构组件

| 组件 | 规格 | 参数量 |
|------|------|--------|
| Encoder | ViT-tiny, patch=14, 12层, 3头, dim=192 | ~5M |
| Predictor | 6层 Transformer + AdaLN (动作注入) | ~10M |
| **总计** | | **~15M** |

- Encoder: o_t → z_t (像素→192维隐变量)
- Predictor: (z_t, a_t) → ẑ_{t+1} (当前状态+动作→预测下一状态)
- 端到端训练，无冻结层，无预训练编码器

> **VERIFY AGAINST PDF**: ViT-tiny 的具体配置 (patch size, 层数, 头数, 隐藏维度)
> 和 Predictor 的 AdaLN 动作注入方式，均来自二手解读。

### 1.3 SIGReg 机制详解

**理论基础**: Cramér-Wold 定理 — 如果一个高维分布在所有一维投影上都是高斯的，则该分布本身是高斯的。

> **VERIFY AGAINST PDF**: Cramér-Wold 定理是否是论文中 SIGReg 的理论依据，
> 还是来自 LeJEPA 论文。LeWM 可能直接引用 LeJEPA 的 SIGReg 而不重复推导。

**实现步骤** (VERIFY AGAINST PDF):
1. 生成 M=1024 个随机方向 θ₁, θ₂, ..., θ_M
2. 将隐变量 Z 投影到每个方向: P(θᵀZ)
3. 对每个投影计算 Epps-Pulley 正态性检验统计量 EP(P(θᵀZ))
4. 优化: 最小化 Σ EP(P(θᵀZ))，逼分布逼近各向同性高斯

> **VERIFY AGAINST PDF**: Epps-Pulley 是否是论文中使用的具体正态性检验。
> SIGReg 可能使用其他一维正态性检验 (如 Shapiro-Wilk, Anderson-Darling)。
> M=1024 的具体数值也需要确认。

**防崩塌原理**:
- 崩塌 = 退化分布 (所有输入映射到同一点) → 不是高斯 → SIGReg 惩罚
- 部分崩塌 = 各向异性 (只用少数维度编码) → 投影偏离高斯 → SIGReg 惩罚
- SIGReg 卡死了两条偷懒路径，编码器只能老老实实学出信息丰富、分布均匀的表示

**SIGReg 消除的工程依赖**:
| 旧方法 | SIGReg 替代 |
|--------|------------|
| EMA teacher (动量编码器) | 不需要 |
| Stop-gradient | 不需要 |
| 预训练 DINOv2 编码器 | 不需要 |
| VICReg/SimCLR 对比损失 | 不需要 |
| 多阶段训练 | 不需要 |
| 6个超参数调优 | 1个 (λ) |

### 1.4 Surprise 定义 — 预测残差

**LeWM 的 surprise = 预测残差，不是静态新颖度**

```
surprise(t) = ||ẑ_{t+1} - z_{t+1}||²
```

- ẑ_{t+1} = predictor(z_t, a_t) — 模型预测的下一状态
- z_{t+1} = encoder(o_{t+1}) — 实际观测的下一状态
- surprise 高 = 模型没预料到的事情发生了

**Violation-of-Expectation (VoE) 实验验证**:

| 场景 | Surprise 响应 | 含义 |
|------|-------------|------|
| 正常轨迹 | 低 | 模型预测准确 |
| 颜色改变 | 略高 | 视觉变化但物理合理 |
| 物体瞬移/穿墙 | **飙升** | 违反物理定律 |

**关键区分**: 模型区分了"看起来不同但物理合理"和"物理上不可能"——这不是在记忆视觉模式，而是捕捉了动态规律。

### 1.5 物理理解验证 (Probing)

线性探针从隐变量回归物理量:

| 物理量 | LeWM (r) | PLDM (r) | DINO-WM (r) |
|--------|----------|----------|-------------|
| 智能体位置 | 0.998 | 0.993 | 0.999 |
| 方块位置 | 0.999 | 0.994 | 0.999 |
| 方块角度 | 0.990 | 0.972 | 0.995 |

1500万参数的 LeWM 接近在 1.24 亿张图片上预训练的 DINOv2。

### 1.6 与替代方案对比

| 方法 | 防崩塌 | 预训练 | 超参数 | 规划速度 | 参数量 |
|------|--------|--------|--------|---------|--------|
| **LeWM** | SIGReg | 不需要 | 1 | ~1s | 15M |
| I-JEPA | EMA+stop-grad | DINOv2 | 6+ | N/A | 大 |
| DINO-WM | 冻结DINOv2 | DINOv2 | 多 | ~47s | 亿级 |
| PLDM | VICReg (不稳定) | 不需要 | 6 | ~1s | 15M |
| BYOL | EMA+stop-grad | 不需要 | 多 | N/A | 大 |

### 1.7 已知局限

| 局限 | 原因 | 对我们的影响 |
|------|------|------------|
| Two-Room 翻车 | 低内在维度 → 高斯先验过度约束 | ⚠️ CEP-CC 的 proto-symbol 簇可能也是低维结构 |
| OGBench-Cube 输给 DINO-WM | 视觉复杂 3D 场景需要更强视觉先验 | 对我们影响小（我们不处理像素） |
| SIGReg 可能损害低维环境 | 各向同性假设与低维流形冲突 | 🔴 CEP-CC 的核心风险 |

### 1.8 工程可迁移原则 (8 条)

1. **预测残差 > 静态新颖度**: surprise 应该是"我预测什么 vs 实际发生什么"的差距，不是"这个输入我见过没"的距离。TopoMem 的 embedding novelty 是后者，已被拒绝。

2. **两损失项足够**: 复杂系统不需要复杂损失。一个预测损失 + 一个正则化项可以替代 6+ 个超参数的调优地狱。

3. **高斯先验是强归纳偏置**: 它同时解决崩塌和各向异性，但可能在低维空间过度约束。

4. **端到端 > 分阶段**: 不冻结、不预训练、不 stop-gradient，所有组件同时更新。

5. **隐空间可探测**: 好的隐变量应该能线性回归出下游量——这是隐空间质量的客观度量。

6. **Surprise 区分结构违规 vs 表面干扰**: 真正的 surprise 应该对结构不可能事件飙升，对表面变化淡定。

7. **轻量级规划**: 192 维 token vs 数万个 token → 48× 加速。紧凑隐空间 = 高效规划。

8. **单一超参数可二分搜索**: 从"多维超空间搜索"退化为"简单二分查找"——工程可复现性的质变。

---

## 任务包 2: 映射到 unified-sel — PredictiveHealthMonitor 设计

### 2.1 核心映射逻辑

| LeWM 概念 | unified-sel 对应 | 说明 |
|-----------|-----------------|------|
| Encoder(o_t) → z_t | TaskEncoder(task) → z_t | 将任务编码为隐变量 |
| Predictor(z_t, a_t) → ẑ_{t+1} | TaskPredictor(z_t, control_context_t) → ẑ_{t+1} | 预测下一批任务的隐变量 |
| surprise = ‖ẑ - z‖² | predictive_residual = ‖ẑ - z‖² | 预测残差 = 健康信号 |
| SIGReg | 不引入 | 用 BatchHealthMonitor 的质心漂移替代 |

### 2.2 PredictiveHealthMonitor 设计

**定位**: 批量级健康监控，NOT 逐任务路由。

**与 BatchHealthMonitor 的关系**: 互补，不替代。

| 维度 | BatchHealthMonitor | PredictiveHealthMonitor |
|------|-------------------|------------------------|
| 信号类型 | 静态分布漂移 (质心余弦距离) | 动态预测残差 (预测误差) |
| 检测逻辑 | "新任务和旧任务长得不一样" | "新任务和我预测的不一样" |
| 域漂移检测 | ✅ 27.2x 分离比 | 待验证 |
| 渐进漂移检测 | ⚠️ 4.1x (弱) | 可能更强 (预测残差对渐进变化更敏感) |
| 结构违规检测 | ❌ 不适用 | ✅ 核心能力 |
| 计算成本 | 低 (EMA + 余弦距离) | 中 (需要训练 predictor) |

**架构草图**:

```
TaskEncoder (all-MiniLM-L6-v2, 已有)
    ↓
z_t = encode(task_t)
    ↓
control_context_t = encode_context(routing_decision_t)
    ↓
TaskPredictor (轻量 MLP, 新增)
    ↓
ẑ_{t+1} = predict(z_t, control_context_t)
    ↓
predictive_residual = ||ẑ_{t+1} - z_{t+1}||²
    ↓
health_signal = {
    residual_mean: float,
    residual_trend: str,        # "stable" | "rising" | "spiking"
    residual_z_score: float,
    status: str                 # "healthy" | "gradual_drift" | "domain_shift" | "anomaly"
}
```

**关键设计决策**:

1. **Control context 输入**: LeWM 的 predictor 接收 (z_t, a_t)。unified-sel 的任务流虽然没有显式"物理动作"，但有路由决策上下文。Predictor 接收:

   ```
   z_t + control_context_t → ẑ_{t+1}
   ```

   control_context 至少包含:
   - routing_decision: "accepted" | "verified" | "escalated"
   - solver_type: str (当前使用的 solver)
   - monitor_signal: float (当前 monitor 输出)
   - verifier_result: bool | None (验证结果，如可用)

   最小版 (preflight) 可以退化为纯时序预测 z_t → ẑ_{t+1}，但正式 spec 必须预留 control_context 接口。

2. **TaskEncoder 复用**: 直接用 BatchHealthMonitor 已有的 all-MiniLM-L6-v2，不引入新编码器。

3. **Predictor 是轻量 MLP**: 输入 384 + control_dim 维，隐藏层 128，输出 384 维。参数量 ~100K，训练成本极低。

4. **在线学习**: Predictor 用 EMA 方式在线更新，不需要离线训练。

5. **批量聚合**: 单任务残差噪声大，取滑动窗口 (window_size=10) 的均值和趋势。

### 2.3 无泄漏评估顺序

> **关键**: 在线更新 predictor 时必须防止"先看答案再预测"的泄漏。

**严格顺序**:

```
Step 1: PREDICT
    ẑ_{t+1} = predictor(z_t, control_context_t)

Step 2: OBSERVE
    z_{t+1} = encoder(task_{t+1})

Step 3: SCORE
    residual = ||ẑ_{t+1} - z_{t+1}||²

Step 4: UPDATE
    predictor.update(z_t, control_context_t, z_{t+1})
```

**评估时的窗口区分**:
- **Warmup window** (前 N_warm 个任务): predictor 尚未收敛，残差不可信，不纳入评估统计
- **Measured window** (N_warm 之后): 残差可信，纳入分离比、误报率等统计
- N_warm 的具体值需要实验确定，初步建议 N_warm = 20

**违反顺序的后果**:
- 如果先 observe 再 predict (即先看到 z_{t+1} 再预测)，predictor 会"偷看答案"，残差人为偏低
- 如果在 warmup window 内评估，残差会因 predictor 未收敛而人为偏高

### 2.4 与 BatchHealthMonitor 的融合

```python
class FusedHealthMonitor:
    def __init__(self):
        self.batch_monitor = BatchHealthMonitor()
        self.predictive_monitor = PredictiveHealthMonitor()

    def observe(self, task, routing_decision=None, solver_type=None,
                monitor_signal=None, verifier_result=None) -> Dict:
        control_context = {
            "routing_decision": routing_decision,
            "solver_type": solver_type,
            "monitor_signal": monitor_signal,
            "verifier_result": verifier_result,
        }
        batch_signal = self.batch_monitor.observe(task)
        predictive_signal = self.predictive_monitor.observe(task, control_context)
        return {
            "drift_signal": batch_signal["drift_signal"],
            "predictive_residual": predictive_signal["residual_mean"],
            "predictive_trend": predictive_signal["residual_trend"],
            "predictive_z_score": predictive_signal["residual_z_score"],
            "fused_status": self._fuse_status(batch_signal, predictive_signal),
        }
```

**融合逻辑**:
- drift_signal 高 + residual 低 → 域漂移但可预测 (新域但模型适应了)
- drift_signal 低 + residual 高 → 隐式异常 (分布没变但行为变了) → **最危险**
- 两者都高 → 明显域漂移
- 两者都低 → 健康

### 2.5 最小实验设计

**实验名**: predictive_health_preflight

**步骤**:
1. 用 code-20 / mixed-40 任务流模拟三种场景:
   - Control: 同域连续任务
   - Domain shift: code → reasoning 切换
   - Gradual shift: trivial → harder 渐进
2. 对每个场景记录:
   - BatchHealthMonitor 的 drift_signal
   - PredictiveHealthMonitor 的 residual_mean, residual_z_score
3. 计算分离比 (shift / control)
4. 与 BatchHealthMonitor 的 27.2x / 4.1x 基线对比

**评估指标** (所有阈值为 preflight targets, not validated acceptance criteria):

| 指标 | Provisional 阈值 | 说明 |
|------|-------------------|------|
| 域漂移分离比 | ≥ 10x (provisional) | 至少比 BatchHealthMonitor 的 27.2x 低，但仍有区分度 |
| 渐进漂移分离比 | ≥ 5x (provisional) | 目标超过 BatchHealthMonitor 的 4.1x |
| 误报率 (control 场景) | < 5% (provisional) | 健康场景下不应频繁报警 |
| 训练收敛步数 | < 100 tasks (provisional) | Predictor 应快速适应 |

> **注意**: 以上阈值是 preflight targets，不是验证结论。实际可接受阈值需要
> 在实验数据出来后根据 ROC 曲线和业务需求重新设定。

**失败条件** (provisional):
- 渐进漂移分离比 < 2x → PredictiveHealthMonitor 不比 BatchHealthMonitor 好
- 误报率 > 20% → 信号太噪声，不可用
- Predictor 不收敛 → 在线学习方案失败

### 2.6 红线确认

- ✅ 仅用于批量级健康监控
- ✅ 不用于逐任务路由 (TopoMem per-task routing 已被拒绝)
- ✅ 不声称 "predictive residual 可以预测单个任务的成功/失败"
- ✅ 成本数字基于假设成本模型

---

## 任务包 3: 映射到 CEE — PredictiveStateAdapter 设计

### 3.1 LeWM-like 层在 CEE 中的位置

```
CEE 管道:
  原始输入 → 任务编译 → 先例检索 → 推理 → [不确定性路由] → 规划 → 执行 → WorldState 回放
                                              ↑
                                    PredictiveStateAdapter
                                    在此处注入信号
```

**PredictiveStateAdapter 不是新管道步骤，而是不确定性路由的信号增强器。**

当前 UncertaintyRouter 的 5 维信号中，`evidence_coverage` 和 `model_self_confidence` 是硬编码占位值。PredictiveStateAdapter 为这两个维度提供真实测量。

### 3.2 PredictiveStateAdapter 读取什么

| 输入 | 来源 | 类型 | 说明 |
|------|------|------|------|
| WorldState.entities | world_state.py | Tuple[WorldEntity] | 当前世界实体 |
| WorldState.hypotheses | world_state.py | Tuple[WorldHypothesis] | 活跃假设 |
| CommitmentEvent | commitment.py | CommitmentEvent | 最近执行的承诺 |
| EventLog | event_log.py | EventLog | 不可变审计轨迹 |
| TaskSpec | tasks.py | TaskSpec | 当前任务规格 |

### 3.3 PredictiveStateAdapter 输出什么

| 信号 | 类型 | 范围 | 含义 |
|------|------|------|------|
| prediction_error | float | [0, ∞) | 世界状态预测残差 (核心信号) |
| latent_uncertainty | float | [0, 1] | 隐空间不确定性 (残差的归一化版) |
| residual_trend | str | enum | "stable" / "rising" / "spiking" |
| transition_plausibility | float | [0, 1] | 状态转移合理性评分 (高=合理) |

> **命名说明**: 使用 `transition_plausibility` 而非 `physical_plausibility`。
> CEE 不是像素物理世界，不直接接入物理环境或视觉状态。
> "transition" 强调这是关于 WorldState 转移的可预测性，不是物理定律的合规性。
> 备选名: `commitment_predictability`, `world_state_residual`。

### 3.4 信号如何影响 CEE

**严格权限: 信号只能影响不确定性路由，不能直接触发行动。**

| 信号值 | 对 CEE 的影响 | 权限级别 |
|--------|-------------|---------|
| prediction_error 低 | 降低 `needs_more_evidence` 概率 | 信号级 |
| prediction_error 中 | 提升 `needs_more_evidence` 概率 | 信号级 |
| prediction_error 高 (spiking) | 触发 `needs_human_review` | 信号级 |
| residual_trend = "spiking" | 提升 tool_risk_level 感知 | 信号级 |

**信号注入点**: UncertaintyRouter.evaluate() 的 RoutingSignals 构造

```python
# 当前 (硬编码):
signals = RoutingSignals(
    evidence_coverage=0.7,        # 占位
    model_self_confidence=0.75,   # 占位
    ...
)

# 注入后 (真实测量):
adapter_signals = predictive_adapter.evaluate(current_world_state, event_log)
signals = RoutingSignals(
    evidence_coverage=adapter_signals.transition_plausibility,
    model_self_confidence=1.0 - adapter_signals.latent_uncertainty,
    ...
)
```

### 3.5 PredictiveStateAdapter 不能做什么 (权限边界)

| 禁止行为 | 原因 |
|----------|------|
| ❌ 直接修改 WorldState | 只有 ModelRevisionEvent 可以修改状态 |
| ❌ 绕过 CommitmentPolicy | 策略评估不可绕过 |
| ❌ 直接调用工具 | 必须通过 ToolGateway |
| ❌ 跳过审批门控 | ApprovalGate 不可绕过 |
| ❌ 修改 EventLog | 事件日志不可变 |
| ❌ 自主决定 escalate | 只能提升信号，决定权在 UncertaintyRouter |
| ❌ 扩展自身权限 | CEE 红线规则 #2 |

**核心原则**: PredictiveStateAdapter 是**证据源**，不是**执行器**。它提供信号，不做出决定。

### 3.6 最小集成 Spec

**新增文件**: `src/cee_core/predictive_adapter.py`

```python
@dataclass(frozen=True)
class PredictiveAdapterSignals:
    prediction_error: float
    latent_uncertainty: float
    residual_trend: Literal["stable", "rising", "spiking"]
    transition_plausibility: float

class PredictiveStateAdapter:
    def __init__(self, latent_dim: int = 64, window_size: int = 10):
        self._encoder = StateEncoder(latent_dim)
        self._predictor = StatePredictor(latent_dim)
        self._history: List[np.ndarray] = []
        self._window_size = window_size

    def observe(self, world_state: WorldState, event_log: EventLog) -> PredictiveAdapterSignals:
        z_t = self._encoder.encode(world_state)
        if len(self._history) > 0:
            z_pred = self._predictor.predict(self._history[-1])
            residual = float(np.linalg.norm(z_pred - z_t))
        else:
            residual = 0.0
        self._history.append(z_t)
        if len(self._history) > self._window_size:
            self._history.pop(0)
        # ... compute trend, uncertainty, transition_plausibility
        return PredictiveAdapterSignals(...)
```

**修改文件**: `src/cee_core/uncertainty_router.py`
- RoutingSignals 构造时，如果 predictive_adapter 存在，用真实信号替换硬编码值
- 不改变路由逻辑本身

**修改文件**: `src/cee_core/runtime.py`
- execute_task_in_domain() 中，可选初始化 PredictiveStateAdapter
- 在推理步骤后、路由步骤前调用 adapter.observe()

### 3.7 三个可测试不变量

1. **信号只读不变量**: PredictiveStateAdapter.observe() 调用后，WorldState 的 state_id 不变，EventLog 长度不变。验证: `assert ws_before.state_id == ws_after.state_id`

2. **权限不扩展不变量**: 即使 prediction_error 持续 spiking，系统也不会绕过 CommitmentPolicy 或 ApprovalGate。验证: 注入极端信号后，所有 irreversible act 仍需人工审批。

3. **信号可忽略不变量**: 移除 PredictiveStateAdapter 后，CEE 管道仍可正常运行（退回硬编码信号）。验证: `predictive_adapter=None` 时所有测试通过。

---

## 任务包 4: 映射到 CEP-CC — SIGReg 适用性判断

### 4.1 SIGReg 在 CEP-CC 中的潜在作用

CEP-CC 的通信轨迹 `comm` 形状为 `[batch, 8, 6]`，经 tanh 压缩到 [-1, 1]。当前正则化体系:

| 现有正则化 | 作用 | 与 SIGReg 的关系 |
|-----------|------|-----------------|
| communication_energy (L2) | 惩罚幅度 | SIGReg 也约束幅度 (高斯→有限方差) |
| communication_sparsity (L1) | 促使稀疏 | SIGReg 不促稀疏，促各向同性 |
| communication_smoothness | 惩罚时间跳变 | SIGReg 无此维度 |
| effective_dimension | 惩罚高有效秩 | SIGReg 直接约束为固定维度 |
| communication_consistency | 同目标通信一致 | SIGReg 不关心目标对齐 |
| factor_bin_consistency | 同因子通信一致 | SIGReg 不关心因子结构 |

**SIGReg 的独特价值**: 强制通信隐变量服从各向同性高斯 → 防止通信崩塌到退化分布。

**SIGReg 的独特风险**: 各向同性假设可能摧毁 proto-symbol 簇的结构。

### 4.2 SIGReg 应该正则化哪个组件?

| 候选 | 可行性 | 风险 |
|------|--------|------|
| Speaker 内部状态 (state, dim=32) | ✅ 可行 | 低 — state 是中间表示，不是协议本身 |
| 通信轨迹 (comm, [8,6]) | ⚠️ 高风险 | 🔴 — 直接作用于协议载体，可能摧毁 proto-symbol |
| Listener 内部状态 (obs+comm, dim=64) | ⚠️ 中风险 | 中 — comm_state 包含协议信息 |
| 全部 | ❌ 过度约束 | 🔴🔴 — 多重约束可能互相冲突 |

**推荐**: 仅对 Speaker 内部状态 (state) 施加 SIGReg，不对通信轨迹施加。

**理由**:
1. Speaker state 是"意图空间"，不是"协议空间"。SIGReg 在意图空间防崩塌是安全的。
2. 通信轨迹是协议的物理载体，proto-symbol 簇存在于这个流形上。SIGReg 的各向同性假设会把这个流形"抹平"。
3. 如果 Speaker state 是各向同性的，通信轨迹仍然可以是非各向同性的（因为 comm_head 是非线性变换）。

### 4.3 SIGReg 可能损害 proto-symbol 簇的机制

**问题链**:
1. Proto-symbol 簇 = 通信轨迹在连续流形上的离散分区
2. 离散分区 = 非各向同性分布 (某些方向方差大，某些方向方差小)
3. SIGReg 强制各向同性 = 所有方向方差相等
4. → SIGReg 会"抹平"簇间差异，使 proto-symbol 难以形成

**定量预测**:
- 如果对 comm 施加 SIGReg，`prototype_reuse_rate` 应该下降 (簇变模糊)
- `cluster_compactness` 应该下降 (簇内/簇间距离比变小)
- `target_cluster_alignment` 应该下降 (簇不再与任务目标对齐)

**类比**: LeWM 在 Two-Room (低维) 上翻车，因为高斯先验过度约束了低维流形。CEP-CC 的 proto-symbol 簇也是低维结构 (6 维通信空间中的离散簇)，同样面临过度约束风险。

### 4.4 预测残差能否度量语义稳定性?

**LeWM 的 surprise = 预测残差**。在 CEP-CC 中:

```
surprise(t) = ||ẑ_{t+1} - z_{t+1}||²
```

其中 z_t = Speaker.state(t)，ẑ_{t+1} = Predictor(z_t)。

**语义稳定性定义**: 同一目标在不同干扰 (mirror_x, rotate90, velocity_scale) 下，通信协议是否保持语义一致。

**预测残差与语义稳定性的关系**:

| 场景 | 预测残差 | 语义稳定性 | 对应 |
|------|---------|-----------|------|
| 正常通信 (无干扰) | 低 | 高 | ✅ 一致 |
| 干扰变换 (mirror/rotate) | 中 | 应该高 (如果协议语义稳定) | ⚠️ 可能不一致 |
| 协议崩塌 | 高 | 低 | ✅ 一致 |
| 新规则模式 | 高 | 低 (需要适应) | ✅ 一致 |

**关键问题**: 预测残差高 ≠ 语义不稳定。干扰变换可能导致残差升高，但协议仍然语义稳定 (只是编码方式变了)。

**结论**: 预测残差可以作为语义稳定性的**必要条件** (残差低 → 可能稳定)，但不能作为**充分条件** (残差高 → 不一定不稳定)。需要结合 `target_cluster_alignment` 和 `audit_latent_probe_r2` 做综合判断。

### 4.5 实验设计: 四组对比 (含 negative-control)

| 组别 | SIGReg 目标 | λ 值 | 用途 |
|------|------------|------|------|
| Control (no-SIGReg) | 无 | 0 | 当前基线 |
| Weak-SIGReg (state) | Speaker.state | 0.01 | 轻微约束，proto-symbol 应保留 |
| Strong-SIGReg (state) | Speaker.state | 0.1 | 强约束，proto-symbol 可能受损 |
| **Negative-control (comm-SIGReg)** | **comm** | **0.01** | **验证 SIGReg 对 comm 的破坏性** |

> **默认策略**: 绝不对 comm 施加 SIGReg。comm-SIGReg 仅作为 negative-control ablation，
> 目的是实证证明 SIGReg 会破坏 proto-symbol clusters。实验结果应记录但不用于生产。

**测量指标**:

| 指标 | 来源 | 预期变化 (weak→strong) | Negative-control 预期 |
|------|------|----------------------|---------------------|
| prototype_reuse_rate | metrics.py | 下降 | **大幅下降** |
| cluster_compactness | metrics.py | 下降 | **大幅下降** |
| target_cluster_alignment | metrics.py | 下降 | **接近随机** |
| task_accuracy | run_experiment.py | 可能下降 | **显著下降** |
| communication_energy | losses.py | 可能上升 | 可能上升 |
| effective_dimension | losses.py | 趋向固定值 | 趋向固定值 |
| audit_latent_probe_r2 | metrics.py | 可能下降 | **显著下降** |

**环境**: compositional 规则 (最成熟的协议涌现环境)

**种子**: [7, 42, 123] (3 种子足够做初步判断)

**判断标准** (provisional):
- 如果 weak-SIGReg(state) 的 prototype_reuse_rate 下降 < 10% 且 task_accuracy 不变 → SIGReg(state) 可用
- 如果 weak-SIGReg(state) 的 prototype_reuse_rate 下降 > 20% → SIGReg(state) 有害，不推荐
- 如果 strong-SIGReg(state) 的 proto-symbol 完全消失 → 确认 SIGReg 对低维结构有害
- 如果 negative-control(comm-SIGReg) 的 proto-symbol 完全消失 → 实证确认 SIGReg 对 comm 有害

> **注意**: 以上判断标准是 provisional targets，不是验证结论。

### 4.6 明确推荐

**推荐: SIGReg 仅作为消融实验，不作为默认正则化。**

理由:
1. CEP-CC 的通信空间是低维的 (6 维)，LeWM 已证明 SIGReg 在低维环境 (Two-Room) 上可能过度约束。
2. Proto-symbol 簇是非各向同性的离散结构，SIGReg 的各向同性假设与协议涌现的目标冲突。
3. CEP-CC 已有 effective_dimension 正则化，功能与 SIGReg 部分重叠 (都约束隐空间维度)。
4. 如果需要防崩塌，communication_consistency + factor_bin_consistency 已提供目标导向的约束，比 SIGReg 的无目标约束更安全。

**例外**: 如果未来 CEP-CC 扩展到高维通信空间 (comm_dim >> 6)，SIGReg 的风险可能降低，值得重新评估。

---

## 跨任务包一致性检查

| 原则 | 任务包2 (unified-sel) | 任务包3 (CEE) | 任务包4 (CEP-CC) |
|------|---------------------|---------------|-----------------|
| Surprise = 预测残差 | ✅ predictive_residual | ✅ prediction_error | ✅ surprise(t) |
| LeWM 是基底不是元控制器 | ✅ 仅提供健康信号 | ✅ 仅提供路由信号 | ✅ 仅正则化中间状态 |
| 不用于逐任务路由 | ✅ 批量级 | ✅ 信号级 (不直接决定) | N/A |
| SIGReg 是消融不是默认 | N/A (不引入SIGReg) | N/A (不引入SIGReg) | ✅ 消融实验 |
| 权威边界不可绕过 | N/A | ✅ 三不变量 | N/A |
| 无泄漏评估顺序 | ✅ predict→observe→score→update | ✅ 同左 | N/A |
| 命名不使用 "physical" | N/A | ✅ transition_plausibility | N/A |

---

## 最小实现优先级

> **核心原则**: 第一阶段只做 unified-sel，不要同时动 CEE 和 CEP-CC。
> unified-sel 是机制验证库，最适合先做可证伪 preflight。
> CEE 和 CEP-CC 等 P0 信号验证过再接。

### P0: unified-sel PredictiveHealthMonitor preflight

**交付物**:
- `core/predictive_health.py` — PredictiveHealthMonitor class
- `experiments/capability/predictive_health_preflight.py` — preflight 实验
- `tests/test_predictive_health.py` — 单元测试

**验收标准**:
- predict → observe → score → update 顺序无泄漏
- warmup window / measured window 分离
- control_context 接口预留 (preflight 可退化为纯时序)
- 三场景 (control/shift/gradual) 分离比可计算
- smoke test 通过

**不做**:
- 不修改 STATUS.md
- 不实现 FusedHealthMonitor (P0 只做独立 PredictiveHealthMonitor)
- 不接入 CEE 或 CEP-CC

### P1: CEE PredictiveStateAdapter design

**前置条件**: P0 的 PredictiveHealthMonitor 在 unified-sel 上验证通过

**交付物**:
- `src/cee_core/predictive_adapter.py` — PredictiveStateAdapter class
- 三个不变量测试
- uncertainty_router.py 的信号注入修改

**验收标准**:
- 信号只读不变量通过
- 权限不扩展不变量通过
- 信号可忽略不变量通过
- 命名使用 transition_plausibility (不是 physical_plausibility)

### P2: CEP-CC SIGReg ablation only

**前置条件**: P0 验证通过 + P1 设计完成

**交付物**:
- CEP-CC 中的 SIGReg 消融实验脚本
- 四组对比结果 (no/weak(state)/strong(state)/negative-control(comm))

**验收标准**:
- negative-control(comm-SIGReg) 的 proto-symbol 破坏性被实证确认或否定
- weak-SIGReg(state) 的安全性被实证确认或否定
- SIGReg 不作为默认正则化

---

## P0 Decision (2026-04-22)

**Result: Predictive residual is detectable but not superior.**

### Evidence

| Metric | PredictiveHealthMonitor | BatchHealthMonitor |
|--------|------------------------|-------------------|
| Domain shift separation (10-seed) | 12.8x [10.4x, 15.6x] | 27.2x [18.4x, 37.3x] |
| Gradual shift separation (10-seed) | 2.1x [1.7x, 2.5x] | 4.1x [2.6x, 5.9x] |
| Mean first alert (5-seed) | task 29.0 (+9.0 from shift) | task 27.2 (+7.2 from shift) |
| False alarm rate | 0/5 | 0/5 |

### Decisions

1. **Do not promote PredictiveHealthMonitor as primary governance signal.**
   It detects domain shift (12.8x) but is weaker than BatchHealthMonitor (27.2x) and alerts later (+9.0 vs +7.2).

2. **Do not enter CEE P1 based on predictive residual alone.**
   P1 requires a reliable signal source. BatchHealthMonitor is more reliable.

3. **Use BatchHealthMonitor as the current health/drift signal baseline.**
   Not as "prediction_residual" — it is static distribution drift. The `prediction_residual` field in Governance spec remains auxiliary/empty.

4. **Keep PredictiveHealthMonitor as experimental sidecar.**
   Not connected to routing, not connected to CEE. Available for future fused health experiments or ablation studies.

5. **Do not pursue B (improve predictor) unless BatchHealthMonitor fails in a specific scenario.**
   Improving the predictor would expand P0 into "train a better predictor" — not the current mainline.

### Design Iteration Note

V1 (single-step embedding prediction z_t → ẑ_{t+1}) achieved only 1.2x separation — too noisy.
V2 (centroid-level prediction) achieved 12.8x — confirms that aggregation is essential for predictive residual.
But even with aggregation, it does not outperform static centroid drift.

### Known Issues

- Signal/alert misalignment: statistical separation is clear but some domain_shift scenarios end with status "healthy", indicating residual statistics and status policy are not fully aligned.
- Not suitable as governance trigger; only suitable as analysis signal.

### Mainline After P0

- **Short-term A**: Capability Router — real-LLM validation, no-revision triage, boundary policies
- **Short-term B**: Boundary-local amplification — paper writing, artifact audit
- **Health signal**: BatchHealthMonitor (static centroid drift)
- **Governance spec**: Architecture boundary only, not implementation task
- **PredictiveHealthMonitor**: Experimental sidecar, not promoted
