# STATUS.md - Current Progress

> **Project**: Granularity-Aligned Metacognition for LLMs (formerly Unified-SEL)
> **Last Updated**: 2026-04-27
> **Current Focus**: Academic + Methodology track — Boundary-local amplification paper + Capability Router research tool

---

**Previous Active Update (Archived)**: LSG Phase 26 (Rollback Protocol) + End-to-End Integration — moved to `archive/lsg_deprecated/`
**Previous Phase (Archived)**: Three-system integration (Router + OBD + LSG with rollback) — LSG direction deprecated

---

## 🔴 项目转向决定（2026-04-16）

**旧核心假设已降级**：
> Surprise-driven structural birth/death > methods requiring explicit task boundaries (e.g., EWC)

降级原因：
- Toy problem 上无法击败 EWC（avg_acc 0.5000 vs 0.5005, p=0.9484）
- Phase G-H 证据链崩塌（patch_size = bug_type 指纹）
- CLAIM_EVIDENCE_MAP 明确写"不能声称对 EWC 有统计优势"

**不是逻辑上不可验证，而是当前实验平台不支持验证。**

详见：`PROJECT_PIVOT_DECISION_2026-04-16.md`

---

## 🎯 新主线 1：Paper 线

> **Feedback retry is a boundary-local amplifier, not a universal enhancer.**

可安全声称的结论：
1. Boundary-local amplification 存在（Phase A, p=0.0008）— ✅ 安全
2. ABOVE-zone 过滤节省 54.4% feedback 调用（Phase E）— ✅ 安全
3. 倒 U 型模式真实存在 — ✅ 安全
4. Artifact audit：patch_size = bug_type 指纹 — ✅ 安全

不能声称：
- NEAR/BELOW 可区分（排除指纹后 ROC AUC=0.500）
- 跨 solver 泛化（同一任务池）
- 成本降低数字（假设成本模型）
- 对 EWC 有统计优势

详见：`papers/boundary_local_amplification_draft.md`

---

## 🎯 新主线 2：Tool 线

> **Capability Router** — primary product track.

当前最佳结果（code-20 / mixed-40，monitor_repair_triage）：
- `semantic`: success 1.0, cost 1.375 / 1.1875
- `counterfactual`: success 1.0, cost ~1.59 / ~1.30

TopoMem 路由实验结果（2026-04-16）：
- `topo_surprise`: ❌ FAIL — 成功率 0.7/0.85，结构性缺陷：embedding novelty ≠ answer correctness
- `topo_semantic_fusion`: ⚠️ 无优势 — 成功率恢复 1.0 但成本更高

**结论：TopoMem surprise 被拒绝作为 per-task routing monitor。TopoMem 降级为部署健康监控候选（OBD），不进入路由核心。**

产品结构：
- `capability-router/` = 主力产品（semantic/counterfactual/repair-triage）
- `topomem-obd/` = 次要，需独立验证（H1/H2/drift → batch-level health）
- `self-aware-llm` = 未来叙事，不作为当前构建目标

详见：`experiments/capability/README.md`

已完成：
- ✅ 标准化 result schema（capbench.result.v1 + machine-readable metadata）
- ✅ 统一 CLI（capbench run / compare / report / list-monitors / list-policies）
- ✅ README（experiments/capability/README.md）
- ✅ CapabilityBoundaryBench JSONL 导出（public/eval 双模式）
- ✅ Report Generator（capbench report）

待完成：
- Paper 投稿版（需真实 LLM 验证加强 Section 4.1）
- TopoMem OBD 真实 LLM 场景验证（当前是 synthetic tasks）
- 真实 LLM solver 替换（长期，GGUF 模型缺失）

---

## ✅ 当前真相摘要

| 结论 | 状态 | 证据 |
|------|------|------|
| Boundary-local amplification 存在 | ✅ 安全 | Phase A, p=0.0008 |
| ABOVE 过滤有效 | ✅ 安全 | Phase E, 54.4% 调用减少 |
| NEAR/BELOW 有排序信号 | ⚠️ 弱 | ROC AUC=0.769, 但阈值不稳定 |
| First-pass 信号可区分 NEAR/BELOW | ❌ 证伪 | 排除指纹后 ROC AUC=0.500 |
| 跨 solver 泛化 | ❌ 不成立 | 同一任务池 + bug_type 指纹 |
| 成本降低数字 | ⚠️ 假设 | 基于硬编码 cost_units |
| Unified-SEL 击败 EWC | ❌ 未验证 | p=0.9484 |
| TopoMem surprise 可作 routing monitor | ❌ 证伪 | success 0.7/0.85, embedding novelty ≠ answer correctness |
| TopoMem 可作部署健康监控 | ✅ 域漂移确认 | OBD 10-seed: centroid drift 27.2x [18.4x,37.3x], p≈0; 渐进漂移 4.1x [2.6x,5.9x]; BatchHealthMonitor 已集成 |
| LLM 置信度不可靠 (0.5B) | ✅ 确认 | Qwen2.5-0.5B: conf=0.95 但实际 15% 成功 |
| Real LLM NEAR 区间随规模出现 | ✅ 确认 | 严格验证: 1.5B=0%, 3B=15%; NEAR zone 真实存在且模型规模依赖 |
| Real LLM 完整倒U型 | ⚠️ 未完整 | 3B: ABOVE=25%, NEAR=15%, BELOW=60%; NEAR<ABOVE, 倒U型不完整 |
| Real LLM ABOVE-filtering | ✅ 确认 | Qwen2.5-1.5B=30%, 3B=25%; 可跳过对应 feedback 调用 |
| 1.5B 模型代码修复能力阈值 | ✅ 确认 | 仅能处理表面语法修改; 无法理解语义级修改; 低于 NEAR-zone 阈值 |
| 3B 模型跨过 NEAR-zone 阈值 | ✅ 确认 | 可处理语义级修改(max→min), feedback retry 有效(15%) |
| Prompt 工程对弱模型效果有限 | ✅ 确认 | 1.5B 在 temp 0.3-1.0 范围内对同一任务输出几乎相同; 能力上限决定表现 |
| semantic/counterfactual 检测 LLM 错误 | ✅ 确认 | 检测率 75%/88%，confidence 监控 0% |
| 路由策略提升 LLM 成功率 | ✅ 确认 | monitor_gate: 15%→75-95%; verifier_first: 100% |
| 域感知路由提升跨域性能 | ⚠️ 需修正 | V3 协议已重命名为 verify_escalate_no_revision；实际是 verify-then-escalate，不是域感知 |
| BatchHealthMonitor window=5 优于 window=10 | ✅ 确认 | 跨域: 65%→85% (semantic), 60%→80% (counterfactual) |
| Drift latch 防止尾部漂移消失 | ✅ 确认 | 85%→95% (semantic), 80%→92% (counterfactual) |
| 对弱模型跳过 revision 直接升级最优 | ✅ 确认 | 0 revision, ~50% API 调用量 vs verifier_first; 成功率受 oracle 假设影响 |
| Few-shot 提升 LLM 成功率 | ✅ 确认 | Zero-shot 15% → Few-shot 30% |
| LLM revision 无效 (0.5B) | ✅ 确认 | 0/17 修正成功 (0%) |
| LeWM 预测残差可作批量健康信号 | ⚠️ 可检测但非最优 | P0 完成：域漂移 12.8x (vs BHM 27.2x)，无时序优势（晚 1.8 tasks）；demoted to experimental sidecar |
| SIGReg 对 CEP-CC proto-symbol 有害 | ⚠️ 待验证 | 理论分析完成，待消融实验 |
| PredictiveStateAdapter 可增强 CEE 路由 | ⚠️ 待验证 | 设计完成，待实现 + 不变量测试 |

---

## ⚠️ 源码审核问题清单（P0-P2）

### P0（最高优先级）
- ❌ **Phase F 的 health signals 存在标签泄漏**：`simulate_health_signals()` 直接读取 `boundary_label`
- ✅ **处理方案**：降级为 oracle/synthetic health ablation，不声称是真实 TopoMem 信号
- 🔴🔴🔴 **`patch_size` 是 bug_type 的完美指纹**：每个 bug_type 的 first_patch_size 都是常量
  - count_abs_gt_two=59, count_adjacent_repeat_words=273, count_even=63, ...
  - 这意味着 `patch_size_to_message_len_ratio` 完全等价于 bug_type 的 one-hot 编码
  - Phase G 的 "first-pass only 分类器完美区分 NEAR/BELOW" 可能只是 bug_type 查表
  - **这是整个 Phase G-H 证据链的根本性缺陷**
- 🔴 **Phase G' 的 `extract_features()` 包含 blind 信号**：不是 first-pass only

### P1（高优先级）
- ❌ **Phase E 没有真正区分 NEAR 和 BELOW**：规则是 `first_error_type != "pass"` 就 feedback
- ❌ **Cross-solver/seed 泛化证据偏弱**：任务生成 seed 只是抽样，blind_success 硬编码
- ❌ **Phase G anti-overfit 不够**：普通 StratifiedKFold 可能有模板记忆
- ✅ **处理方案**：用 GroupKFold/leave-one-bug-type-out 重新验证

### P2（中优先级）
- ⚠️ **`patch_size_to_message_len_ratio` 可能是任务模板信号**
- ✅ **处理方案**：特征审计，禁用任务相关特征

---

## ✅ 可安全声称的结论（优化实验后）

> **优化实验结果（2026-04-15 v2 修复后）：**
>
> 使用随机化 solver + 多样化任务变体 + 严格特征审计 + GroupKFold + 训练集阈值优化：
>
> | 指标 | 结果 | 阈值 | 状态 |
> |------|------|------|------|
> | Near Recall | **100.0%** | ≥ 90% | ✅ PASS |
> | Below Filtered (train-thresh) | **46.2% ± 41.2%** | ≥ 50% | ❌ FAIL（差一点） |
> | Below Filtered (default=0.5) | **0.0%** | ≥ 50% | ❌ FAIL（概率校准差） |
> | ROC AUC | **0.769** | > 0.5 | ✅ 比随机好 |
>
> **可安全声称的结论：**
> 1. **Phase A**：Boundary-local amplification 存在（p=0.0008）— **✅ 安全**
> 2. **Phase E**：能过滤 ABOVE，减少 54.4% feedback 调用 — **✅ 安全**
> 3. **NEAR/BELOW 区分**：**⚠️ 弱证据** — ROC AUC=0.769 说明有排序信号，但无可部署调度策略
>
> **信号不稳定的原因**：
> - 只有 1/40 个 bug_type 是 MIXED（同时有 NEAR 和 BELOW）
> - 不同 bug_type 的 Below Filtered 差异巨大（0% - 100%）
> - `first_error_message_len_norm` 是最重要的特征，但可能仍和 bug_type 相关
> - 默认阈值=0.5 时 Below Filtered=0%，说明概率校准差
>
> **诚实实验 vs 优化实验对比**：
> - 诚实实验（排除所有 bug_type 相关特征）：ROC AUC = 0.500（随机）
> - 优化实验 v2（引入随机化 solver + 多样化任务）：ROC AUC = 0.769（有排序信号）
> - 优化实验 v2 修复后（移除 artifact 特征 + 训练集阈值）：ROC AUC = 0.769（基本不变）
> - 关键差异：随机化 solver 打破了 patch_size = bug_type 的确定性

---

## ✅ Phase D：核心发现

**250 traces 数据确认**：
- ABOVE (solved): 54.4%
- NEAR (hidden-gap): 38.8%
- BELOW (visible-fail): 6.8%

**Zone 命名（更严谨）**：
| Zone | 特征 | 映射 |
|------|------|------|
| solved | `first_visible_pass=True`, `first_error_type="pass"` | ≈ ABOVE |
| hidden-gap | `first_visible_pass=True`, `first_hidden_pass=False` | ≈ NEAR |
| visible-fail | `first_visible_pass=False` | ≈ BELOW（需跨 solver 验证）|

**核心发现**（最重要）：
> **blind_changed_code = False 但 feedback_success = True 是 near-boundary 的强特征。**

这解释了 boundary-local amplification 的机制：
- solver 已经接近正确解，但缺少特定错误信息
- blind retry 没有新信息，所以停滞
- feedback 注入新约束，搜索空间突然收缩

**Runtime Trace 边界规则**：
```
if not first_visible_pass:
    zone = "visible-fail"
    action = "skip_or_escalate"
elif first_error_type == "pass":
    zone = "solved"
    action = "accept"
else:
    zone = "hidden-gap"
    action = "feedback_retry"
```

---

## 🎯 Phase E：Runtime Boundary Scheduler Simulation 结果

**实验结果**：

| Policy | Success Rate | FB Calls | Efficiency |
|--------|-------------|-----------|------------|
| A: Always single-shot | 54.4% | 0 | - |
| B: Always feedback | 93.2% | 250 | 0.0037 |
| C: Oracle difficulty | 93.2% | 97 | 0.0096 |
| **D: Runtime trace** | **93.2%** | **114** | **0.0082** |

**关键发现**：
- ✅ Runtime scheduler 成功率与 always-feedback 相同（93.2%）
- ✅ Feedback 调用数节省 **54.4%**（250 → 114）
- ✅ Feedback 效率提升 **2.19x**

**Zone 级别分析（修正后）**：
| Zone | n | Runtime Scheduler FB Calls | Success |
|------|---|----------------------------|---------|
| ABOVE | 136 | 0 | 100% |
| NEAR | 97 | 97 | 100% |
| BELOW | 17 | 17 | 0% |

**重要修正说明**：
- Policy D 的 114 次 feedback = 97 NEAR + 17 BELOW
- **Policy D 目前只是一个 above-filter scheduler，不是完整的 near-boundary selector**
- 它成功过滤了 ABOVE 区的无效 feedback，但还不能区分 NEAR 和 BELOW

**Policy D 的学习能力**：
- ✅ 学会了：`first_error_type == "pass" → ABOVE → skip feedback`
- ❌ 没学会：`first_error_type == "other"` 中哪些是 NEAR，哪些是 BELOW

**结论**：runtime trace signal 可以在不降低成功率的前提下过滤掉 ABOVE 区的无效 feedback；但 BELOW 仍会被误送入 feedback。总体收益数字是真的——**同等成功率，更少反馈成本**——但表述要更严谨。

---

## 🎯 Phase F：NEAR vs BELOW Discriminator 结果

**关键发现 1：Visible test 太弱**
- ✅ `first_visible_pass == True` for **all 250 traces**
- ⚠️ 这意味着当前的 "hidden-gap" zone 实际上可能混入了 BELOW 样本
- 需要加强 visible tests 或重新定义 boundary zones

**关键发现 2：Classifier 可以完美区分 NEAR vs BELOW——但我们在作弊！**

| 分类器 | CV 准确率 | 训练准确率 |
|--------|----------|-----------|
| Random Forest | 1.000 ± 0.000 | 1.000 |
| Logistic Regression | 1.000 ± 0.000 | 1.000 |

**但是，重要警示**：
- 最强特征是 `feedback_changed_code`（重要性 0.400）
- **`feedback_changed_code` 是 feedback 之后才有的信号，不是 first-pass 信号！**
- 这是作弊！我们需要只用 first-pass 信号来预测。

**First-pass 信号中最有区分度的特征**：
| 特征 | NEAR 均值 | BELOW 均值 | Cohen's d |
|------|-----------|------------|-----------|
| patch_size_to_message_len_ratio | 4.511 | 1.870 | 1.040 |
| first_patch_size | 82.371 | 49.529 | 0.802 |
| expected_actual_distance | 0.231 | 0.200 | 0.605 |
| first_error_message_len | 23.701 | 27.824 | -0.375 |

**改进后的调度器模拟（使用作弊的 feedback_changed_code）**：
| 策略 | 成功率 | Feedback 调用数 | 节省 |
|------|--------|-----------------|------|
| Policy D (Original) | 93.2% | 114 | - |
| Improved (cheating) | 93.2% | 97 | 61.2% |
| Policy C (Oracle) | 93.2% | 233 | - |

**Phase F 结论收紧**：
1. ✅ NEAR 和 BELOW 在 first-pass 信号上确实有可区分的差异
2. ⚠️ 但我们当前的分类器使用了 `feedback_changed_code`，这是作弊
3. ⚠️ Visible test 太弱，导致 "hidden-gap" 定义有问题
4. 🔴 **下一步必须做：只用 first-pass 信号重新训练分类器**

---

## 🔴 Phase G：First-pass Only NEAR vs BELOW Discriminator — 结论已证伪

> ⚠️ **2026-04-15 源码审核结论：Phase G 的核心结论已被证伪。**
>
> `patch_size` 是 bug_type 的完美指纹（每个 bug_type 的 first_patch_size 都是常量），
> `patch_size_to_message_len_ratio` 完全等价于 bug_type 的 one-hot 编码。
> 分类器不是在学"能力边界"，而是在记忆 bug_type 指纹。
>
> 诚实实验（排除 bug_type 相关特征后）：ROC AUC = 0.500（随机）。
> 详见 EXPERIMENT_LOG.md "诚实实验" 条目。

### 原始结果（已被证伪，保留供参考）

原分类器使用了 `patch_size_to_message_len_ratio`（= bug_type one-hot）和 `first_patch_size`（= bug_type 常量），
因此达到了"完美分类"。这不是能力边界检测，而是 bug_type 查表。

| 原始指标 | 原始结果 | 实际含义 |
|----------|----------|----------|
| CV 准确率 1.000 | 完美分类 | = bug_type 查表，不是边界检测 |
| `patch_size_to_message_len_ratio` 重要性 0.443 | 最强信号 | = bug_type one-hot 编码 |
| `first_patch_size` 重要性 0.232 | 第二信号 | = bug_type 常量 |

### 证伪后的结论

- ❌ "只用 first-pass 信号就可以完美区分 NEAR 和 BELOW" → **证伪**（依赖 bug_type 指纹）
- ❌ "调度精度达到 Oracle 级别" → **证伪**（基于 bug_type 查表）
- ❌ "Feedback 节省 61.2%" → **不可信**（基于无效分类器）
- ✅ 排除 bug_type 指纹后，first-pass verifier 信号完全无法区分 NEAR/BELOW（ROC AUC=0.500）

---

## 🔴 Phase H：跨 solver 验证 — 结论已降级

> ⚠️ **2026-04-15 源码审核结论：Phase H 的泛化声称不成立。**
>
> 1. 两个 solver 使用同一任务池（seed=0），不是真正的泛化测试
> 2. `patch_size_to_message_len_ratio` 仍是特征重要性最高的特征（= bug_type 指纹）
> 3. "特征重要性跨 solver 一致"只是说明两个 solver 都产生了 bug_type 特征性的 patch_size
>
> 详见 EXPERIMENT_LOG.md "源码审核" 条目。

### 原始结果（需降级解读）

#### Solver 能力对比

| Boundary | SearchLocalSolver | HeuristicLocalSolver |
|----------|-------------------|---------------------|
| ABOVE | 136 (54.4%) | 16 (6.4%) |
| NEAR | 97 (38.8%) | 18 (7.2%) |
| BELOW | 17 (6.8%) | 216 (86.4%) |

#### 跨 solver 迁移学习

| 方向 | Accuracy | ROC AUC | 问题 |
|------|----------|---------|------|
| Search→Heuristic | 0.974 | 0.986 | 同一任务池，patch_size 仍是主特征 |
| Heuristic→Search | 0.518 | 1.000 | NEAR 漏判多 |

### 降级后的结论

- ❌ "跨 solver 泛化能力" → **不成立**（同一任务池 + bug_type 指纹）
- ❌ "特征重要性跨 solver 一致" → **无意义**（一致的是 bug_type 指纹，不是边界信号）
- ⚠️ 两个 solver 的 boundary 分布差异巨大 → 这一点是真的，但只说明 solver 能力不同

---

## ✅ 可安全声称的结论（2026-04-15 修订）

> **经源码审核和诚实实验验证后的安全结论：**
>
> 1. **Phase A**：Boundary-local amplification 存在——feedback retry 的收益集中在 NEAR 区（+49% gain, p=0.0008）— **✅ 安全**
> 2. **Phase D**：Runtime trace 可以记录 first-pass 信号 — **✅ 安全**（但信号质量有限）
> 3. **Phase E**：能过滤 ABOVE，减少 54.4% feedback 调用 — **✅ 安全**（但只能声称过滤 ABOVE）
>
> **⚠️ 部分证据**：
> - NEAR/BELOW 区分：优化实验 ROC AUC=0.785（有排序信号），但 Below Filtered 不稳定（0%-100%）
> - 只有 1/40 个 bug_type 是 MIXED，泛化性未验证
>
> **❌ 已证伪/降级的结论**：
> - Phase F "with health signals 完美分类" → **完全无效**（标签泄漏）
> - Phase G "first-pass only 完美区分 NEAR/BELOW" → **证伪**（bug_type 指纹）
> - Phase H "跨 solver 泛化能力" → **不成立**（同一任务池 + bug_type 指纹）
> - "31% 成本降低" → **基于假设成本模型**，非真实延迟测量
> - escalate 路径 100% 成功率 → **oracle 假设**

---

## ⚠️ 不能声称的边界

- ❌ 不能说 first-pass 信号可以区分 NEAR 和 BELOW（排除 bug_type 指纹后 ROC AUC=0.500）
- ❌ 不能说 `patch_size_to_message_len_ratio` 是能力边界信号（它是 bug_type 指纹）
- ❌ 不能说信号具有跨 solver 泛化能力（同一任务池，不是真泛化）
- ❌ 不能说 visible-fail zone = BELOW（需跨 solver 验证）
- ❌ 不能说倒 U 型已经跨 solver / 跨任务族普遍成立
- ⚠️ 成本降低数字基于假设成本模型，非真实延迟测量

---

## 📁 相关文件

- `double_helix/validate.py` - Phase A/B 实验
- `double_helix/boundary_awareness_analysis.py` - Phase B 分析
- `double_helix/no_leak_boundary_classifier.py` - Phase C 实验
- `double_helix/runtime_trace_boundary_experiment.py` - Phase D 实验
- `double_helix/runtime_scheduler_simulation.py` - Phase E 实验（已完成）
- `double_helix/near_below_discriminator.py` - Phase F 实验（已完成，发现作弊问题）
- `double_helix/first_pass_only_discriminator.py` - Phase G 实验（🔴 结论已证伪：patch_size = bug_type 指纹）
- `double_helix/phase_h_heuristic_solver_traces.py` - Phase H：收集 HeuristicLocalSolver traces（🔴 结论已降级：同一任务池）
- `double_helix/phase_h_analysis.py` - Phase H：跨 solver 分析（🔴 结论已降级）
- `double_helix/phase_g_anti_overfit_validation.py` - Phase G'：反过拟合验证（🔴 包含 blind 信号泄漏）
- `double_helix/honest_boundary_experiment.py` - 诚实实验：排除 bug_type 指纹后 ROC AUC=0.500
- `double_helix/optimized_boundary_v2.py` - 优化实验 v2：随机化 solver + 多样化任务（ROC AUC=0.769）

---

## 📜 历史记录

> 2026-04-15 之前的详细实验记录（Toy Problem、Capability-track 迭代、LSG Phase 16-26 等）已归档至 [`archive/STATUS_HISTORY.md`](archive/STATUS_HISTORY.md)。
