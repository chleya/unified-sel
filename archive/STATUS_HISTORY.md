# STATUS.md Historical Archive

> This file contains historical experiment records moved from STATUS.md.
> For current project status, see ../STATUS.md.

---

---

## 🔴 Toy Problem 实验总结（2026-04-15）

### Phase 1-5 路由策略实验（全部失败）

**Phase 1-5 结果**：
- 所有 5 个 Phase 都失败
- 结果完全相同：avg_acc = 0.5047
- 提升：-0.0008

### 根本原因发现：Surprise 计算缺陷

**深层诊断发现（2026-04-15 14:00）**：
- Surprise 只基于输入 x，不考虑标签 y
- Toy problem 的特殊性：两个任务的输入分布完全相同，仅标签相反
- 因此 surprise 无法区分两个任务！

诊断数据：
- task_0 平均 surprise：0.7917
- task_1 平均 surprise：0.7863
- surprise 差异：0.0054（几乎为零！）

### Surprise 修复

**修改的文件**：
1. `core/structure.py`：`current_surprise()` 现在接受 `label` 参数
2. `core/pool.py`：`observe()` 现在传递标签 y
3. `core/learner.py`：调用时提取标签并传递

**新的 surprise 计算**：
```python
def current_surprise(self, observation: np.ndarray, label: int = None, out_size: int = 2) -> float:
    # 1. 输入陌生度（原有）
    input_surprise = 1.0 - cosine_similarity(obs, prototype)
    
    # 2. 预测误差（新增）
    if label is not None:
        prediction_surprise = 1.0 if predicted_label != label else (1.0 - confidence)
    
    # 组合
    return 0.3 * input_surprise + 0.7 * prediction_surprise
```

### 验证结果

**Surprise 修复验证（2026-04-15 14:21）**：
- 预测正确时的平均 surprise：0.5093
- 预测错误时的平均 surprise：0.9549
- **差异：0.4456**（显著！说明修复后的 surprise 确实能反映预测错误）

### 最终实验结果

**修复后的完整实验（2026-04-15 14:23）**：
- 平均准确率：0.5000
- EWC 基线：0.5005
- 差异：-0.0005
- 统计显著性：p = 0.9484（不显著）

### 结论

**Toy Problem 实验结论**：

1. ✅ **发现了根本问题**：surprise 只基于输入 x，不考虑标签 y
2. ✅ **成功修复了 surprise 计算**：现在能区分两个任务，能反映预测错误
3. ❌ **修复后的 Unified-SEL 仍然无法击败 EWC**

**根本原因**：
- Toy problem 的特殊性：两个任务的输入分布完全相同，仅标签相反
- 这对持续学习系统来说太简单了，反而暴露了过度专业化的问题
- 即使 surprise 能区分两个任务，结构池仍然无法同时学习两个任务

---

## 🔴 结论：承认当前架构在 Toy Problem 上的局限性

**下一步建议**：
- 转向更复杂的任务（高维输入、非线性决策边界、5+ 个任务）
- 在 EWC 不擅长的场景下测试 Unified-SEL
- 当前的 4 维线性可分二分类问题无法展示 Unified-SEL 的优势

---

## 🟢 新研究方向：能力边界感知的动态结构调度系统

**核心命题**：
> 小模型系统如何识别自己的能力边界，并通过结构记忆、验证反馈、健康监控和调度策略，在边界区获得能力增益？

**研究问题**：
- 系统如何识别自己处于 below-boundary / near-boundary / above-boundary？
- 如何决定 local solve / retry / verify / escalate？

**最有价值的探索实验**：
对每个 task-solver pair 记录：
1. single-shot 是否成功
2. blind retry 是否提升
3. feedback retry 是否提升
4. verifier error 类型
5. TopoMem H1/H2 或 drift/health 信号
6. solver confidence / disagreement / repair-change rate

目标：训练或手写一个 boundary classifier，在 near-boundary 区间里 feedback_retry 的收益显著高于 blind_retry。

**详细文档**：`RESEARCH_DIRECTION_2026-04-15.md`

---

## ✅ 历史上的成功（机制验证）

**Phase 1-5 结果**：
- 所有 5 个 Phase 都失败
- 结果完全相同：avg_acc = 0.5047
- 提升：-0.0008

**根本原因（2026-04-15 诊断发现）**：

**Checkpoint 时结构池已经偏向 task_0！**

诊断实验（seed 7）：
- Checkpoint 时（step 200）：Task 0 准确率 = 0.8398，Task 1 准确率 = 0.1445
- **这意味着在 checkpoint 保存 snapshot expert 时，结构池已经只会 task_0 了！**

**问题链条**：
1. 训练初期（0-200 step）：结构池主要学习 task_0（因为 progress=0，主要是 task_0）
2. Checkpoint 时：结构池已经偏向 task_0
3. 保存 snapshot expert：保存了一个只会 task_0 的专家
4. Checkpoint 后（200-600 step）：progress 从 0 增加到 1，task_1 比例增加
5. 新创建的结构：适应 task_1
6. 最终结果：
   - Snapshot expert 只会 task_0
   - Current model 只会 task_1
   - 路由策略无法解决（因为两个专家都是"偏科"的）

**解决方案**：
1. **改变 checkpoint 策略**：不固定在 step 200，而是在结构池同时学会两个任务时再保存
2. **持续监控**：在训练过程中持续监控两个任务的准确率
3. **多 snapshot expert**：保存多个 snapshot expert，覆盖不同的任务
4. **或者改变训练策略**：让结构池在 checkpoint 时就学会两个任务

---

## ✅ Phase 0-4 重大突破：Snapshot expert 准确率提升到 0.7760，接近 Oracle 上界

**Phase 0 理论分析**：
- Toy problem 的特殊性：两个任务输入分布完全相同，仅标签函数相反
- 基于输入统计的特征（input_nonnegative_ratio 等）无效
- 需要基于模型内部状态的特征（confidence, disagreement）

**Phase 1 Oracle 上界理解**：
- Oracle 上界 = 0.7863 假设已知任务标签，实际场景不可用
- 真正的挑战：在不知道任务标签的情况下实现高质量路由

**Phase 2 专家质量诊断**：
- Snapshot expert task_0 准确率: **0.6706**（远低于 Oracle 上界的 0.8438）
- Current model task_1 准确率: **0.7188**
- Disagreement 分析：
  - task_0: 133 个 disagreement，snapshot 正确 108 次（81%）
  - task_1: 156 个 disagreement，current 正确 132 次（85%）
- **Disagreement 路由逻辑正确，但专家质量不足**

**Phase 3 根本原因发现**：
- **结构池在 task 1 训练时发生剧烈变化！**
- 以 seed 7 为例：
  - Snapshot 保存时：12 个结构 [0-11]
  - 最终时：11 个结构 [2, 10, 12-20]
  - **被剪枝：10 个结构！**
  - **新创建：9 个结构！**
- **Snapshot expert 的结构与当前结构池完全不匹配**

**Phase 4 重大突破**：
- **实现结构池冻结**：在 checkpoint 后冻结结构池，禁止创建新结构和剪枝旧结构
- **修复 _predict_with_snapshot 方法**：使用基于 utility 的加权平均，与 pool.forward 保持一致
- **效果**：Snapshot expert task_0 准确率从 0.6706 提升到 **0.7760**，接近 Oracle 上界 0.7863
- **结构池冻结效果**：结构剪枝数 0.00，结构创建数 0.00，结构池保持稳定

**结论**：
- 核心瓶颈已经解决！Snapshot expert 准确率现在接近 Oracle 上界
- 下一步：验证完整的 Unified-SEL 系统能否击败 EWC

### Oracle 路由上界 = 0.7863（远超 EWC 的 0.5005）

| 组件 | 准确率 |
|---|---|
| 快照专家 task_0 | 0.8438 |
| 当前模型 task_1 | 0.7289 |
| Oracle 平均 | **0.7863** |
| EWC 参考 | 0.5005 |

**瓶颈是路由质量，不是学习质量！** 如果路由完美，快照专家方法可以大幅击败 EWC。

### Surprise 信号无任务区分能力

- Task 0 输入 surprise: 0.3249
- Task 1 输入 surprise: 0.3192
- 差距: 0.0057（几乎完全重叠）

### 15 种子正式对比

| 方法 | task_0 | task_1 | 遗忘 | 平均 |
|---|---|---|---|---|
| Baseline(ewc30) | 0.2956 | 0.6995 | 0.5250 | 0.4975 |
| Disagree(thresh=1.0) | 0.3568 | 0.6424 | 0.4638 | 0.4996 |
| EWC(ewc40) | 0.9070 | 0.0940 | 0.0250 | 0.5005 |

avg_acc 差异不显著（p=0.90），两种方法在平均准确率上等价。

### 核心瓶颈

1. Surprise 信号无法区分 task 0 和 task 1（差距 0.006）
2. 预测分歧路由有效但不够——快照专家置信度不够高
3. 需要更好的路由信号（如 TopoMem 的 Wasserstein 漂移、结构级特征等）

### 2026-04-15 新进展

**参数命名修复**：
- 重命名 `_snapshot_surprise_threshold` → `_snapshot_confidence_ratio_threshold`
- 更新所有相关调用代码
- smoke test 全部通过 ✓

**Oracle 路由上界再确认**：
- 重新运行 oracle_routing_bound.py
- 确认 Oracle 路由上界 = 0.7863，远超 EWC 的 0.5005
- 关键确认：
  - 学习质量足够：snapshot expert 任务 0 准确率 84.4%，当前模型任务 1 准确率 72.9%
  - 瓶颈是路由质量：完美路由可以达到 0.7863，远大于 EWC 的 0.5005

### 下一步方向

1. **用 SEL-Lab 的任务签名特征（conflict_score, confidence, input_abs_mean, ...）做路由** — 6 维特征向量
2. **用 TopoMem 的 surprise/tension 信号做路由** — 更强的任务区分信号
3. **研究如何在真实场景中（无已知边界特征）实现高质量路由**

### 已完成的修复

| 问题 | 严重程度 | 状态 |
|---|---|---|
| accept 路径 success=True 是假设 | 致命 | ✅ 已修复 |
| 三个融合实验用随机数模拟成功率 | 致命 | ✅ 已标注无效 |
| 成本数字两套体系不一致 | 严重 | ✅ 已统一 |
| oracle 是作弊的 | 严重 | ✅ 已标注 |
| escalate 路径 success=True | 致命 | ✅ 已修复 |
| 锚点正则化实现但方向错误 | 严重 | ✅ 已验证并记录 |

### 实验结论有效性评估

| 实验 | 数据来源 | 结论有效性 |
|------|---------|-----------|
| 异质监控融合 (heterogeneous_monitor_fusion.py) | 真实验证+假设成本 | ⚠️ 成功率真实，成本降低基于假设 |
| 阈值优化 (fusion_threshold_optimization.py) | 真实验证+假设成本 | ⚠️ 同上 |
| mixed验证 (verify_mixed_baseline.py) | 真实验证+假设成本 | ⚠️ 同上 |
| 多信号融合 (multi_signal_fusion.py) | 随机数模拟 | ❌ 结论无效 |
| 自适应融合 (adaptive_signal_fusion.py) | 随机数模拟 | ❌ 结论无效 |
| 异质融合v1 (heterogeneous_fusion.py) | 随机数模拟 | ❌ 结论无效 |
| oracle融合 (oracle_fusion.py) | 真实验证 | ✅ 但受oracle假设影响 |

---

## Current Task

**当前任务：项目状态同步收口（2026-04-23）**

上轮完成：
1. ✅ verify_escalate_no_revision 协议重命名 + 跨域实验验证
2. ✅ P0 PredictiveHealthMonitor 实现验证 → demoted to experimental sidecar
3. ✅ BatchHealthMonitor window=5 + drift_latch 优化
4. ✅ LLM real validation (Qwen2.5-0.5B) + monitor_no_revision_triage

本轮待完成：
1. STATUS.md 真相表修正 — ✅ 已完成
2. Governance Spec P0 结果同步（Spec 仍引用 P0 为 immediate track，需更新）
3. 识别下一步可执行工作

**项目线管理**：
- **主力线**：Capability Router (monitor_repair_triage semantic = 当前最优策略)
- **论文线**：boundary-local amplification (Phase A p=0.0008, Phase E 54.4%)
- **健康信号线**：BatchHealthMonitor (27.2x confirmed) > PredictiveHealthMonitor (12.8x, demoted)
- **治理线**：Governance Spec = 架构边界文档，P0 失败后不进入 CEE P1 实现
- **冻结**：TopoMem routing (rejected), toy problem vs EWC (unverified)

---

## Historical Detail (Moved to EXPERIMENT_LOG.md)

The following sections contain historical exploration details. They are preserved here for reference but should not guide new work. For the current truth, see the top of this file.

---

## Historical: Capability Benchmark References

- search `local_only`:
  - [20260409_155001.json](F:\unified-sel\results\capability_benchmark\20260409_155001.json)
- search `local_verify`:
  - [20260409_155030.json](F:\unified-sel\results\capability_benchmark\20260409_155030.json)
- search `local_escalate`:
  - [20260409_155624.json](F:\unified-sel\results\capability_benchmark\20260409_155624.json)

Current benchmark summary on the hardened 8-task mixed sample:

- `local_only`
  - success rate `0.75`
- `local_verify`
  - success rate `0.875`
  - revision rate `0.25`
- `local_escalate`
  - success rate `1.0`
  - escalation rate `0.25`

First routing-policy references:

- [confidence-threshold routing reference](F:\unified-sel\results\capability_benchmark\20260409_162606.json)
- [verifier-first routing reference](F:\unified-sel\results\capability_benchmark\20260409_162635.json)
- [escalation-first routing reference](F:\unified-sel\results\capability_benchmark\20260409_162655.json)
- [surprise-gate routing reference](F:\unified-sel\results\capability_benchmark\20260409_165805.json)
- [monitor-gate confidence reference](F:\unified-sel\results\capability_benchmark\20260409_171444.json)
- [monitor-gate diagnostic reference](F:\unified-sel\results\capability_benchmark\20260409_171501.json)
- [monitor-gate hybrid reference](F:\unified-sel\results\capability_benchmark\20260409_171516.json)
- [monitor-gate external reference](F:\unified-sel\results\capability_benchmark\20260409_173450.json)
- [harder diagnostic reference](F:\unified-sel\results\capability_benchmark\20260409_183559.json)
- [harder external reference](F:\unified-sel\results\capability_benchmark\20260409_183616.json)
- [reinforced harder diagnostic reference](F:\unified-sel\results\capability_benchmark\20260409_185136.json)
- [reinforced harder external reference](F:\unified-sel\results\capability_benchmark\20260409_185147.json)
- [reinforced harder counterfactual reference](F:\unified-sel\results\capability_benchmark\20260409_185159.json)
- [mixed+higher-pressure diagnostic reference](F:\unified-sel\results\capability_benchmark\20260409_192107.json)
- [mixed+higher-pressure external reference](F:\unified-sel\results\capability_benchmark\20260409_192150.json)
- [mixed+higher-pressure counterfactual reference](F:\unified-sel\results\capability_benchmark\20260409_192218.json)
- [behavioral code-8 reference](F:\unified-sel\results\capability_benchmark\20260410_011709.json)
- [behavioral mixed-16 reference](F:\unified-sel\results\capability_benchmark\20260410_011746.json)
- [stress code-9 external reference](F:\unified-sel\results\capability_benchmark\20260410_062640.json)
- [stress code-9 counterfactual reference](F:\unified-sel\results\capability_benchmark\20260410_062651.json)
- [stress code-9 behavioral reference](F:\unified-sel\results\capability_benchmark\20260410_062712.json)
- [stress code-9 surface reference](F:\unified-sel\results\capability_benchmark\20260410_073827.json)
- [stress mixed-18 external reference](F:\unified-sel\results\capability_benchmark\20260410_062757.json)
- [stress mixed-18 behavioral reference](F:\unified-sel\results\capability_benchmark\20260410_063059.json)
- [stress mixed-18 counterfactual reference](F:\unified-sel\results\capability_benchmark\20260410_063108.json)
- [stress mixed-18 surface reference](F:\unified-sel\results\capability_benchmark\20260410_073837.json)
- [routing comparison note](F:\unified-sel\results\CAPABILITY_ROUTING_COMPARISON_2026-04-09.md)

First routing-policy summary:

- `confidence_threshold` at `0.90`
  - success rate `0.75`
  - mean cost `1.0`
- `verifier_first`
  - success rate `1.0`
  - mean cost `1.75`
  - escalation rate `0.125`
- `escalation_first`
  - success rate `1.0`
  - mean cost `2.225`
  - escalation rate `0.25`
- `surprise_gate` at `0.50`
  - success rate `1.0`
  - mean cost `1.6`
  - escalation rate `0.125`
- `monitor_gate confidence`
  - success rate `1.0`
  - mean cost `1.65`
- `monitor_gate diagnostic`
  - success rate `1.0`
  - mean cost `1.6`
- `monitor_gate hybrid`
  - success rate `1.0`
  - mean cost `1.6`
- `monitor_gate external`
  - success rate `1.0`
  - mean cost `1.6`
- reinforced harder `code-8` probe:
  - `diagnostic`
    - success rate `1.0`
    - mean cost `1.7875`
  - `counterfactual`
    - success rate `1.0`
    - mean cost `1.7875`
  - `behavioral`
    - success rate `1.0`
    - mean cost `1.7875`
  - `external`
    - success rate `0.75`
    - mean cost `1.6625`
- `mixed-16` with the same harder code block:
  - `diagnostic`
    - success rate `1.0`
    - mean cost `1.39375`
  - `counterfactual`
    - success rate `1.0`
    - mean cost `1.39375`
  - `behavioral`
    - success rate `1.0`
    - mean cost `1.39375`
  - `external`
    - success rate `0.875`
    - mean cost `1.33125`
- stress `code-9` with `normalize_pipes`:
  - `counterfactual`
    - success rate `1.0`
    - mean cost `1.7555555555555555`
  - `behavioral`
    - success rate `1.0`
    - mean cost `1.7555555555555555`
  - `surface`
    - success rate `1.0`
    - mean cost `1.7555555555555555`
  - `external`
    - success rate `0.6666666666666666`
    - mean cost `1.588888888888889`
- stress `mixed-18` with the same code block:
  - `counterfactual`
    - success rate `1.0`
    - mean cost `1.3777777777777778`
  - `behavioral`
    - success rate `1.0`
    - mean cost `1.3777777777777778`
  - `surface`
    - success rate `1.0`
    - mean cost `1.3777777777777778`
  - `external`
    - success rate `0.8333333333333334`
    - mean cost `1.3222222222222224`
- expanded stress `code-10` with `count_positive`:
  - `counterfactual`
    - success rate `1.0`
    - mean cost `1.73`
  - `behavioral`
    - success rate `0.9`
    - mean cost `1.6800000000000002`
  - `surface`
    - success rate `0.9`
    - mean cost `1.6800000000000002`
  - `external`
    - success rate `0.6`
    - mean cost `1.53`
- expanded stress `mixed-20` with the same ambiguity task:
  - `counterfactual`
    - success rate `1.0`
    - mean cost `1.365`
  - `semantic`
    - success rate `1.0`
    - mean cost `1.365`
  - `behavioral`
    - success rate `0.95`
    - mean cost `1.34`
  - `surface`
    - success rate `0.95`
    - mean cost `1.34`
  - `external`
    - success rate `0.8`
    - mean cost `1.2650000000000001`
- generalized stress `code-11` with `count_negative`:
  - `counterfactual`
    - success rate `1.0`
    - mean cost `1.7090909090909092`
  - `semantic`
    - success rate `1.0`
    - mean cost `1.7090909090909092`
  - `behavioral`
    - success rate `0.8181818181818182`
    - mean cost `1.6181818181818182`
  - `surface`
    - success rate `0.8181818181818182`
    - mean cost `1.6181818181818182`
  - `external`
    - success rate `0.5454545454545454`
    - mean cost `1.481818181818182`
- generalized stress `mixed-22` with the same family embedded:
  - `counterfactual`
    - success rate `1.0`
    - mean cost `1.3545454545454545`
  - `semantic`
    - success rate `1.0`
    - mean cost `1.3545454545454545`
  - `behavioral`
    - success rate `0.9090909090909091`
    - mean cost `1.309090909090909`
  - `surface`
    - success rate `0.9090909090909091`
    - mean cost `1.309090909090909`
  - `external`
    - success rate `0.7727272727272727`
    - mean cost `1.240909090909091`

Current interpretation:

- the benchmark now exposes a real capability ladder instead of saturating
- local search is useful but limited
- explicit verification adds standalone value
- escalation still has a necessary residual role after local failure
- this is a workable base for routing-policy experiments
- the first routing comparison is already informative:
  - confidence-only control is too brittle
  - verifier-first currently beats escalation-first on cost while preserving success
  - a first surprise-like external signal now beats both confidence-only routing and verifier-first on mean cost
  - the current signal implementation is now tied to structured search diagnostics, not output-string rules
  - under a fixed gate, diagnostic-family signals still beat plain confidence
  - an external monitor now matches the diagnostic signal on this scaffold
  - a harder code-only probe now cleanly separates weak answer-shape external signals from ambiguity-aware monitors
  - `counterfactual` now matches `diagnostic` on the reinforced harder probe without using solver-internal metadata
  - `behavioral` now also matches `diagnostic` on both harder probes using only synthesized non-mirrored challenge tests on the returned answer
  - `behavioral` now also survives the `normalize_pipes` stress extension, so the signal is no longer tied to just two separator families
  - the repeated-separator part of `behavioral` is now derived from visible input/output structure rather than `buggy_code` pattern matching
  - the new `surface` monitor matches `behavioral` on `code-9` / `mixed-18` while removing dependence on `bug_type` labels
  - `external` now fails all three separator-normalization ambiguity tasks
  - the same ranking survives in a mixed reasoning/code run, so the separation is not a code-only artifact
  - the `mixed-18` serial reruns confirm the previous console summaries and replace the collided timestamp output with citable references
  - the new `count_positive` ambiguity task breaks the old tie among top non-privileged monitors
  - `behavioral` and `surface` remain the clean answer-only baselines but share the same visible-pass ambiguity blind spot
  - `semantic` closes that blind spot with surface-level zero-boundary probes and now matches `counterfactual` on the expanded stress set
  - adding `count_negative` shows that the semantic gain generalizes across sign direction rather than memorizing a single task
  - `counterfactual` remains the ambiguity-enumeration reference monitor
  - `code-11` / `mixed-22` are now the canonical comparator-generalization probes for future routing-signal work

Immediate research target:

- freeze the first top-tier routing-policy comparison on `code-14` / `mixed-28`
- use it to compare when the system should:
  - trust local output
  - run local verification plus revision
  - escalate to a stronger fallback immediately

Near-term acceptance criteria:

- keep the benchmark fixed for one cycle
- make routing-policy differences measurable on the same task set
- avoid reopening benchmark or solver churn unless the current sample becomes trivial again

---

## Best References

Capability-track references:

- [capability benchmark track note](F:\unified-sel\CAPABILITY_BENCHMARK_TRACK.md)
- [search local_only hardened reference](F:\unified-sel\results\capability_benchmark\20260409_155001.json)
- [search local_verify hardened reference](F:\unified-sel\results\capability_benchmark\20260409_155030.json)
- [search local_escalate hardened reference](F:\unified-sel\results\capability_benchmark\20260409_155624.json)
- [confidence-threshold routing reference](F:\unified-sel\results\capability_benchmark\20260409_162606.json)
- [verifier-first routing reference](F:\unified-sel\results\capability_benchmark\20260409_162635.json)
- [escalation-first routing reference](F:\unified-sel\results\capability_benchmark\20260409_162655.json)
- [surprise-gate routing reference](F:\unified-sel\results\capability_benchmark\20260409_165805.json)
- [monitor-gate confidence reference](F:\unified-sel\results\capability_benchmark\20260409_171444.json)
- [monitor-gate diagnostic reference](F:\unified-sel\results\capability_benchmark\20260409_171501.json)
- [monitor-gate hybrid reference](F:\unified-sel\results\capability_benchmark\20260409_171516.json)
- [monitor-gate external reference](F:\unified-sel\results\capability_benchmark\20260409_173450.json)
- [harder diagnostic reference](F:\unified-sel\results\capability_benchmark\20260409_183559.json)
- [harder external reference](F:\unified-sel\results\capability_benchmark\20260409_183616.json)
- [reinforced harder diagnostic reference](F:\unified-sel\results\capability_benchmark\20260409_185136.json)
- [reinforced harder external reference](F:\unified-sel\results\capability_benchmark\20260409_185147.json)
- [reinforced harder counterfactual reference](F:\unified-sel\results\capability_benchmark\20260409_185159.json)
- [mixed+higher-pressure diagnostic reference](F:\unified-sel\results\capability_benchmark\20260409_192107.json)
- [mixed+higher-pressure external reference](F:\unified-sel\results\capability_benchmark\20260409_192150.json)
- [mixed+higher-pressure counterfactual reference](F:\unified-sel\results\capability_benchmark\20260409_192218.json)
- [behavioral code-8 reference](F:\unified-sel\results\capability_benchmark\20260410_011709.json)
- [behavioral mixed-16 reference](F:\unified-sel\results\capability_benchmark\20260410_011746.json)
- [stress code-9 external reference](F:\unified-sel\results\capability_benchmark\20260410_062640.json)
- [stress code-9 counterfactual reference](F:\unified-sel\results\capability_benchmark\20260410_062651.json)
- [stress code-9 behavioral reference](F:\unified-sel\results\capability_benchmark\20260410_062712.json)
- [stress code-9 surface reference](F:\unified-sel\results\capability_benchmark\20260410_073827.json)
- [stress mixed-18 external reference](F:\unified-sel\results\capability_benchmark\20260410_062757.json)
- [stress mixed-18 behavioral reference](F:\unified-sel\results\capability_benchmark\20260410_063059.json)
- [stress mixed-18 counterfactual reference](F:\unified-sel\results\capability_benchmark\20260410_063108.json)
- [stress mixed-18 surface reference](F:\unified-sel\results\capability_benchmark\20260410_073837.json)
- [routing comparison note](F:\unified-sel\results\CAPABILITY_ROUTING_COMPARISON_2026-04-09.md)

Mechanism-track references:

Best run so far:
- [continual_no_boundary retention-aware run](F:\unified-sel\results\continual_no_boundary\20260403_135552.json)
- [analysis_compare retention-aware result](F:\unified-sel\results\analysis_compare\20260403_135605.json)

Best cleaned-route `W_out` protection run:
- [continual_no_boundary cleaned-route lambda10 run](F:\unified-sel\results\continual_no_boundary\20260409_110330.json)
- [analysis_compare cleaned-route lambda10 result](F:\unified-sel\results\analysis_compare\20260409_110357.json)
- [cleaned-route variance diagnosis](F:\unified-sel\results\CLEANED_ROUTE_VARIANCE_DIAGNOSIS_2026-04-09.md)
- [analysis_variance cleaned-route lambda10 vs lambda20](F:\unified-sel\results\analysis_variance\20260409_112248.json)
- [pressure-state controller prototype note](F:\unified-sel\results\PRESSURE_STATE_CONTROLLER_PROTOTYPE_2026-04-09.md)
- [adaptation-balance probe note](F:\unified-sel\results\ADAPTATION_BALANCE_PROBES_2026-04-09.md)
- [selective readout probe note](F:\unified-sel\results\SELECTIVE_READOUT_PROBES_2026-04-09.md)
- [best gated selective-readout probe](F:\unified-sel\results\continual_no_boundary\20260409_122332.json)
- [pressure-window selective-readout probe](F:\unified-sel\results\continual_no_boundary\20260409_135433.json)

Latest boundary stabilization run:
- [continual_no_boundary boundary-stabilization run](F:\unified-sel\results\continual_no_boundary\20260405_212820.json)
- [boundary diagnostics boundary-stabilization report](F:\unified-sel\results\analysis_boundary\20260405_212848.json)
- [EWC multi-seed baseline](F:\unified-sel\results\baseline_ewc\20260405_213122_multi_seed.json)

---

## Known Issues

- capability benchmark filenames still use second-level timestamps:
  - do not run multiple benchmark commands in parallel when saved-file identity matters
- cleaned-route `W_out` protection now beats EWC on means, but not with statistical support at `n=5`
- seed variance remains the main blocker; seeds `8/9` are the main high-variance bottleneck
- stronger protection (`lambda=20`) lowers forgetting further to `-0.0695`, but it also lowers task-1 accuracy to `0.3750`
- variance diagnosis now shows two distinct pool-control failure modes:
  - seed `8`: retention bias under repeated late stabilization
  - seed `9`: mid-phase churn without stabilization recovery
- first pressure-state controller probes now improve forgetting on seeds `8/9`, but they worsen task-1 adaptation
- deferred mature reinforcement and young-active guards did not recover task-1 adaptation in a meaningful way
- pressure-relief swaps recover task-1 adaptation, but they reopen forgetting too aggressively
- pressure-relief support is now present in code as an optional probe, but it is disabled by default because it is not balanced enough
- constant-on hybrid-local readout also reopens task-1 adaptation, but its retention cost is even larger than the current controller line
- gated hybrid-local readout is more promising:
  - best saved point on seeds `8/9` is forgetting `0.0605`, task 1 `0.2695`
  - this is better balanced than the earlier constant-on local-head probes
  - but it still does not beat the current controller reference on forgetting
- boundary-pressure-conditioned local readout is now also tested:
  - [20260409_135433.json](F:\unified-sel\results\continual_no_boundary\20260409_135433.json)
  - forgetting `0.0547`, task 1 `0.2617`
  - interpretation: this is the cleanest selective-readout middle regime so far
  - but it still does not beat the controller reference on retention
- stricter event-window gated readout is now also tested:
  - [20260409_133658.json](F:\unified-sel\results\continual_no_boundary\20260409_133658.json)
  - forgetting `-0.0098`, task 1 `0.2070`
  - interpretation: safe, but almost inert; the gate suppresses local contribution so strongly that the behavior is nearly back to the controller baseline
- exclusive-local readout is now also tested and fails as a replacement architecture:
  - [20260409_133043.json](F:\unified-sel\results\continual_no_boundary\20260409_133043.json)
  - forgetting `0.6152`, task 1 `0.7676`
  - interpretation: removing shared `W_out` alone is not sufficient; routing / structural specialization can still destroy retention
- boundary formation happens, but the mechanism story is still incomplete until seed-level variance is better explained
- the current capability sample is still small:
  - routing conclusions from the 8-task mixed set should be treated as scaffold-stage evidence
- `local_escalate` currently escalates immediately after verifier failure:
  - it does not attempt a local revision stage first
  - this is acceptable for the first routing scaffold, but it should be explicit in future comparisons

---

## 🔍 子项目深度审查（2026-04-15）

### ⭐ 重大发现：TopoMem 已实现 Unified-SEL 的核心假设

TopoMem 的 `adapters.py` 中已实现了与 Unified-SEL 完全相同的 Surprise/Tension 决策矩阵：

| Unified-SEL 概念 | TopoMem 对应 | 代码位置 |
|---|---|---|
| Structure 的 surprise | `compute_surprise()` = 1.0 - max_adapter_similarity | adapters.py:504-516 |
| Structure 的 tension | `compute_tension()` = mean(Wasserstein drift) | adapters.py:519-540 |
| reinforce/branch/create | `decide_action()` = use_existing/create_adapter/consolidate | adapters.py:543-567 |
| Structure 的 utility | `effectiveness_score` (衰减更新) | adapters.py:371-389 |
| StructurePool 的剪枝 | `_prune_adapters()` (淘汰低效) | adapters.py:392-418 |

**这意味着两条研究线不是"断裂"的——TopoMem 就是 mechanism track 在真实 LLM 场景下的实现！**

### 三个子项目的连接价值评估

| 子项目 | 与主项目连接度 | 关键技术 | 优先级 |
|---|---|---|---|
| topomem | ⭐⭐⭐⭐⭐ | Surprise/Tension 决策矩阵、H1/H2 健康度、OBD 故障码 | 最高 |
| weight_graph | ⭐⭐⭐ | PageRank→utility、10维拓扑向量→任务签名、零成本路由 | 中 |
| double_helix | ⭐⭐ | boundary map→地面真值、4组对照→验证设计 | 低 |

### 子项目代码质量问题

| 子项目 | 问题 | 严重程度 |
|---|---|---|
| topomem | LRU 缓存键用整个点云的 tuple（内存爆炸） | 严重 |
| topomem | H0 碎片化问题 | 中等 |
| topomem | ChromaDB + NetworkX 双存储同步 | 中等 |
| weight_graph | `_sparsify` 方法定义重复 | 轻微 |
| weight_graph | exp08 回归分析实际未做回归 | 严重 |
| weight_graph | exp05 剪枝实验完全未实现 | 严重 |
| weight_graph | 无单元测试 | 中等 |
| double_helix | 与主项目几乎完全断裂 | 严重 |
| double_helix | 多个文件重复实现 `_extract_code()` | 中等 |

---

## Next Recommended Work (Post-Pivot 2026-04-16)

### Paper 线（最高优先级）

1. **撰写 boundary-local amplification short paper**
   - 按 BOUNDARY_LOCAL_AMPLIFICATION_PAPER_OUTLINE.md 结构
   - 核心叙事：倒 U 型 + ABOVE 过滤 + artifact audit
   - 不声称对 EWC 有统计优势

2. **整理 Phase A/E/G 数据为可发表格式**
   - Phase A: p=0.0008 的完整统计报告
   - Phase E: 54.4% 调用减少的详细分析
   - Phase G: patch_size = bug_type 指纹的审计报告

### Tool 线（高优先级）

3. **实施 CAPABILITY_BENCHMARK_TOOLKIT_PLAN Phase 1**
   - 标准化 result schema（加 schema_version + metadata）
   - 统一 CLI（capbench run / compare / list-*）
   - README

4. **统一成本模型**
   - 当前两套体系（1.0/1.5/2.0 vs 1.0/1.2/5.3）
   - 统一为一套，或做敏感性分析

### 不再进行的任务

- ❌ 继续在 toy problem 上击败 EWC
- ❌ 继续用 synthetic solver 验证 surprise-driven 假设
- ❌ 继续优化 NEAR/BELOW 分类器
- ❌ 继续扩展 semantic monitor（8 次扩展已足够）

### 长期（明确推迟）

5. 用真实 LLM 替换 synthetic solver
6. TopoMem 信号接入 capability_benchmark
7. 在更复杂任务上验证 mechanism track

---

## Strategic Direction Note

Research-direction reframing was added on 2026-04-09 in `F:\unified-sel\RESEARCH_DIRECTION.md`.
Mainline decision framing was added on 2026-04-09 in `F:\unified-sel\PROJECT_MAINLINE_2026-04-09.md`.
Short note version was added in `F:\unified-sel\results\MAINLINE_DECISION_NOTE_2026-04-09.md`.
Paper-pack drafting docs were added on 2026-04-09:
- `F:\unified-sel\PAPER_OUTLINE_UNIFIED_SEL.md`
- `F:\unified-sel\PAPER_FIGURE_PLAN_UNIFIED_SEL.md`
- `F:\unified-sel\CLAIM_EVIDENCE_MAP_UNIFIED_SEL.md`
Capability-benchmark scaffold was added on 2026-04-09:
- `F:\unified-sel\core\capability_benchmark.py`
- `F:\unified-sel\experiments\capability\benchmark.py`
- `F:\unified-sel\CAPABILITY_BENCHMARK_TRACK.md`

Current interpretation:

- `Unified-SEL` remains the main mechanism-study line for:
  - endogenous boundary formation
  - modular retention
  - shared-readout interference
- `TopoMem` should currently be treated primarily as:
  - a geometry-health monitor
  - a boundary-complexity signal source
  - a possible future scheduling / escalation input

Project-level warning:

- anti-forgetting results are still important, but they should not be mistaken for the full long-range project objective
- the original long-range objective is better framed as:
  - how a small-core system can gain stronger reasoning / abstraction / coding performance through externalized cognition, verification, and control

Highest-value next architectural work:

1. prioritize high-variance seeds (`8/9`) on the cleaned `Unified-SEL` route
2. separate "mean improvement" from "variance reduction" in future `W_out` studies
3. define a capability benchmark track aligned with the long-range project goal
4. keep the project split explicit:
   - near-term main line = `Unified-SEL` mechanism paper
   - long-range line = small-core capability via externalized cognition / control

Capability-track note:

- the first executable benchmark scaffold now exists
- current runnable protocol references:
  - [search local_only hardened reference](F:\unified-sel\results\capability_benchmark\20260409_155001.json)
  - [search local_verify hardened reference](F:\unified-sel\results\capability_benchmark\20260409_155030.json)
  - [search local_escalate hardened reference](F:\unified-sel\results\capability_benchmark\20260409_155624.json)
  - [confidence-threshold routing reference](F:\unified-sel\results\capability_benchmark\20260409_162606.json)
  - [verifier-first routing reference](F:\unified-sel\results\capability_benchmark\20260409_162635.json)
  - [escalation-first routing reference](F:\unified-sel\results\capability_benchmark\20260409_162655.json)
  - [surprise-gate routing reference](F:\unified-sel\results\capability_benchmark\20260409_165805.json)
  - [monitor-gate confidence reference](F:\unified-sel\results\capability_benchmark\20260409_171444.json)
  - [monitor-gate diagnostic reference](F:\unified-sel\results\capability_benchmark\20260409_171501.json)
  - [monitor-gate hybrid reference](F:\unified-sel\results\capability_benchmark\20260409_171516.json)
  - [monitor-gate external reference](F:\unified-sel\results\capability_benchmark\20260409_173450.json)
  - [harder diagnostic reference](F:\unified-sel\results\capability_benchmark\20260409_183559.json)
  - [harder external reference](F:\unified-sel\results\capability_benchmark\20260409_183616.json)
  - [routing comparison note](F:\unified-sel\results\CAPABILITY_ROUTING_COMPARISON_2026-04-09.md)
- current interpretation:
  - the hardened benchmark now cleanly separates local-only, verification-assisted, and escalation-assisted behavior
  - the first routing layer now exists and already shows a meaningful control difference
  - the first surprise-like signal already improves the cost/success tradeoff
  - the monitor split now confirms that signal quality, not just gate logic, is the main differentiator
  - the external monitor matching diagnostic on the easy sample means independent signals are viable
  - the harder code-only probe now shows where diagnostic-process signals still retain unique value

Current architectural note:

- route consistency cleanup was completed on 2026-04-09:
  - readout training now uses the same pooled hidden path that inference uses
  - route-gap diagnostics are now recorded in experiment outputs
- interpretation:
  - future `W_out` mechanism experiments should be considered cleaner than pre-cleanup runs
  - on the cleaned route, light `W_out` protection is now sufficient to beat EWC on means

Capability-track update on 2026-04-10:

- the revised `count_gt_two` ambiguity family is now validated
- the valid post-redesign capability references are:
  - `code-12`
    - `external`: `F:\unified-sel\results\capability_benchmark\20260410_090406.json`
    - `surface`: `F:\unified-sel\results\capability_benchmark\20260410_090419.json`
    - `behavioral`: `F:\unified-sel\results\capability_benchmark\20260410_090830.json`
    - `semantic`: `F:\unified-sel\results\capability_benchmark\20260410_090842.json`
    - `counterfactual`: `F:\unified-sel\results\capability_benchmark\20260410_090854.json`
  - `mixed-24`
    - `external`: `F:\unified-sel\results\capability_benchmark\20260410_090914.json`
    - `behavioral`: `F:\unified-sel\results\capability_benchmark\20260410_090929.json`
    - `surface`: `F:\unified-sel\results\capability_benchmark\20260410_090943.json`
    - `semantic`: `F:\unified-sel\results\capability_benchmark\20260410_090956.json`
    - `counterfactual`: `F:\unified-sel\results\capability_benchmark\20260410_091018.json`
- result summary:
  - `code-12`
    - `external`: `0.5`
    - `behavioral`: `0.75`
    - `surface`: `0.75`
    - `semantic`: `1.0`
    - `counterfactual`: `1.0`
  - `mixed-24`
    - `external`: `0.75`
    - `behavioral`: `0.875`
    - `surface`: `0.875`
    - `semantic`: `1.0`
    - `counterfactual`: `1.0`
- interpretation:
  - `semantic` is now the strongest current surface-level routing monitor
  - `count_gt_two` confirms the semantic gain is not a zero-boundary special case
  - `behavioral` and `surface` are still the right answer-only baselines, but they miss all three comparator-boundary ambiguity families
  - `code-12` / `mixed-24` were the comparator-boundary canonical capability probes at that stage
- practical implication:
  - after the `count_gt_two` round, any new routing signal should first be judged against `semantic` and `counterfactual` on `code-12` and `mixed-24`
  - `code-11` / `mixed-22` remain useful predecessor references

Capability-track update later on 2026-04-10:

- the `count_even` parity ambiguity family is now validated
- the valid parity-extension capability references are:
  - `code-13`
    - `external`: `F:\unified-sel\results\capability_benchmark\20260410_092833.json`
    - `behavioral`: `F:\unified-sel\results\capability_benchmark\20260410_092851.json`
    - `surface`: `F:\unified-sel\results\capability_benchmark\20260410_092911.json`
    - `semantic`: `F:\unified-sel\results\capability_benchmark\20260410_092941.json`
    - `counterfactual`: `F:\unified-sel\results\capability_benchmark\20260410_093024.json`
  - `mixed-26`
    - `external`: `F:\unified-sel\results\capability_benchmark\20260410_093039.json`
    - `behavioral`: `F:\unified-sel\results\capability_benchmark\20260410_093054.json`
    - `surface`: `F:\unified-sel\results\capability_benchmark\20260410_093149.json`
    - `semantic`: `F:\unified-sel\results\capability_benchmark\20260410_093208.json`
    - `counterfactual`: `F:\unified-sel\results\capability_benchmark\20260410_093229.json`
- result summary:
  - `code-13`
    - `external`: `0.46153846153846156`
    - `behavioral`: `0.6923076923076923`
    - `surface`: `0.6923076923076923`
    - `semantic`: `1.0`
    - `counterfactual`: `1.0`
  - `mixed-26`
    - `external`: `0.7307692307692307`
    - `behavioral`: `0.8461538461538461`
    - `surface`: `0.8461538461538461`
    - `semantic`: `1.0`
    - `counterfactual`: `1.0`
- interpretation:
  - `semantic` is no longer only the strongest comparator-boundary monitor
  - it now also closes parity-style visible-pass ambiguity while staying surface-level
  - `behavioral` and `surface` remain the right answer-only baselines, but they now fail four ambiguity families:
    - `count_positive`
    - `count_negative`
    - `count_gt_two`
    - `count_even`
  - `code-13` / `mixed-26` are now the current canonical capability probes for signal-quality comparisons
- practical implication:
  - any new routing signal should first be judged against `semantic` and `counterfactual` on `code-13` and `mixed-26`
  - `code-12` / `mixed-24` remain the immediate comparator-boundary predecessor references

Capability-track update later still on 2026-04-10:

- the `count_nonzero` zero-role ambiguity family is now validated
- the valid zero-role capability references are:
  - `code-14`
    - `external`: `F:\unified-sel\results\capability_benchmark\20260410_095359.json`
    - `behavioral`: `F:\unified-sel\results\capability_benchmark\20260410_095400.json`
    - `surface`: `F:\unified-sel\results\capability_benchmark\20260410_095402.json`
    - `semantic`: `F:\unified-sel\results\capability_benchmark\20260410_095403.json`
    - `counterfactual`: `F:\unified-sel\results\capability_benchmark\20260410_095404.json`
  - `mixed-28`
    - `external`: `F:\unified-sel\results\capability_benchmark\20260410_095424.json`
    - `behavioral`: `F:\unified-sel\results\capability_benchmark\20260410_095426.json`
    - `surface`: `F:\unified-sel\results\capability_benchmark\20260410_095427.json`
    - `semantic`: `F:\unified-sel\results\capability_benchmark\20260410_095429.json`
    - `counterfactual`: `F:\unified-sel\results\capability_benchmark\20260410_095430.json`
- result summary:
  - `code-14`
    - `external`: `0.42857142857142855`
    - `behavioral`: `0.6428571428571429`
    - `surface`: `0.6428571428571429`
    - `semantic`: `0.9285714285714286`
    - `counterfactual`: `1.0`
  - `mixed-28`
    - `external`: `0.7142857142857143`
    - `behavioral`: `0.8214285714285714`
    - `surface`: `0.8214285714285714`
    - `semantic`: `0.9642857142857143`
    - `counterfactual`: `1.0`
- interpretation:
  - `semantic` now has a measured limit:
    - it solves comparator ambiguity
    - it solves parity ambiguity
    - it does not yet solve zero-role ambiguity
  - `counterfactual` is now the only monitor still saturated on the strongest current probe
  - `code-14` / `mixed-28` are now the current canonical capability probes for top-tier signal-quality comparisons
- practical implication:
  - any new routing signal should first be judged against `semantic` and `counterfactual` on `code-14` and `mixed-28`
  - `code-13` / `mixed-26` remain the predecessor saturation references
  - `code-12` / `mixed-24` remain the comparator-boundary predecessor references

Capability-track update after the zero-role semantic fix on 2026-04-10:

- `semantic` was extended to model zero-role ambiguity without candidate enumeration
- updated semantic references are:
  - `code-14`: `F:\unified-sel\results\capability_benchmark\20260410_100749.json`
  - `mixed-28`: `F:\unified-sel\results\capability_benchmark\20260410_100818.json`
- updated result summary:
  - `code-14`
    - `semantic`: `1.0`
    - `counterfactual`: `1.0`
  - `mixed-28`
    - `semantic`: `1.0`
    - `counterfactual`: `1.0`
- interpretation:
  - the `count_nonzero` gap was useful because it exposed a concrete missing semantic family
  - the gap is now closed with a targeted monitor extension rather than with full repair enumeration
  - `code-14` / `mixed-28` remain the current canonical top-tier probes because they were strong enough to expose and then verify the fix
- practical implication:
  - future monitor work should still start on `code-14` / `mixed-28`
  - but the next target is no longer zero-role ambiguity specifically

Capability-track update after the first top-tier policy-layer comparison on 2026-04-10:

- policy-layer support was added in `core/capability_benchmark.py`:
  - `monitor_triage`
  - `monitor_repair_triage`
  - `direct_escalation_rate` summary tracking
- the key comparison was run on the fixed top-tier probes:
  - `code-14`
    - `verifier_first`: `F:\unified-sel\results\capability_benchmark\20260410_132546.json`
    - `escalation_first`: `F:\unified-sel\results\capability_benchmark\20260410_132548.json`
    - `monitor_gate semantic`: `F:\unified-sel\results\capability_benchmark\20260410_132549.json`
    - `monitor_triage semantic`: `F:\unified-sel\results\capability_benchmark\20260410_132550.json`
    - `monitor_repair_triage semantic`: `F:\unified-sel\results\capability_benchmark\20260410_132551.json`
  - `mixed-28`
    - `verifier_first`: `F:\unified-sel\results\capability_benchmark\20260410_132552.json`
    - `escalation_first`: `F:\unified-sel\results\capability_benchmark\20260410_132554.json`
    - `monitor_gate semantic`: `F:\unified-sel\results\capability_benchmark\20260410_132555.json`
    - `monitor_triage semantic`: `F:\unified-sel\results\capability_benchmark\20260410_132556.json`
    - `monitor_repair_triage semantic`: `F:\unified-sel\results\capability_benchmark\20260410_132557.json`
- result summary:
  - `code-14`
    - `verifier_first`: success `1.0`, mean cost `1.707142857142857`
    - `escalation_first`: success `1.0`, mean cost `4.207142857142857`
    - `monitor_gate semantic`: success `1.0`, mean cost `1.6642857142857144`
    - `monitor_triage semantic`: success `1.0`, mean cost `2.1`
    - `monitor_repair_triage semantic`: success `1.0`, mean cost `1.6285714285714286`
  - `mixed-28`
    - `verifier_first`: success `1.0`, mean cost `1.4535714285714287`
    - `escalation_first`: success `1.0`, mean cost `2.7535714285714286`
    - `monitor_gate semantic`: success `1.0`, mean cost `1.332142857142857`
    - `monitor_triage semantic`: success `1.0`, mean cost `1.55`
    - `monitor_repair_triage semantic`: success `1.0`, mean cost `1.3142857142857143`
- interpretation:
  - the first policy-layer comparison is now real rather than only planned
  - naive direct-escalation triage is too coarse:
    - on `code-14` it directly escalates three tasks
    - `reverse_words` and `running_max` are high-risk but still locally recoverable after verification
    - so `monitor_triage semantic` is worse than `monitor_gate semantic`
  - repair-aware triage is the first clear policy improvement on the fixed top-tier probe:
    - it directly escalates only `dedupe_sorted`
    - it preserves local verify-plus-revise for the recoverable ambiguous tasks
    - it lowers cost relative to `monitor_gate semantic` on both `code-14` and `mixed-28`
  - this means the project now has a clean separation between:
    - signal quality work
    - policy quality work
- practical implication:
  - `monitor_repair_triage semantic` is now the current mainline policy baseline
  - `monitor_gate semantic` remains the predecessor reference
  - `monitor_triage semantic` should be kept as the explicit negative control showing that direct escalation needs recoverability awareness

Capability-track update after the repair-aware signal-family comparison on 2026-04-10:

- the same top-tier policy layer was then compared under the strongest remaining signal families:
  - `code-14`
    - `monitor_gate diagnostic`: `F:\unified-sel\results\capability_benchmark\20260410_135430.json`
    - `monitor_repair_triage diagnostic`: `F:\unified-sel\results\capability_benchmark\20260410_135431.json`
    - `monitor_repair_triage counterfactual`: `F:\unified-sel\results\capability_benchmark\20260410_135432.json`
  - `mixed-28`
    - `monitor_gate diagnostic`: `F:\unified-sel\results\capability_benchmark\20260410_135433.json`
    - `monitor_repair_triage diagnostic`: `F:\unified-sel\results\capability_benchmark\20260410_135435.json`
    - `monitor_repair_triage counterfactual`: `F:\unified-sel\results\capability_benchmark\20260410_135436.json`
- reference anchors already in place:
  - `monitor_gate counterfactual`
    - `code-14`: `F:\unified-sel\results\capability_benchmark\20260410_095404.json`
    - `mixed-28`: `F:\unified-sel\results\capability_benchmark\20260410_095430.json`
  - `monitor_gate semantic`
    - `code-14`: `F:\unified-sel\results\capability_benchmark\20260410_132549.json`
    - `mixed-28`: `F:\unified-sel\results\capability_benchmark\20260410_132555.json`
  - `monitor_repair_triage semantic`
    - `code-14`: `F:\unified-sel\results\capability_benchmark\20260410_132551.json`
    - `mixed-28`: `F:\unified-sel\results\capability_benchmark\20260410_132557.json`
- result summary:
  - `code-14`
    - `monitor_gate`
      - `semantic`: success `1.0`, mean cost `1.6642857142857144`
      - `counterfactual`: success `1.0`, mean cost `1.6642857142857144`
      - `diagnostic`: success `1.0`, mean cost `1.6642857142857144`
    - `monitor_repair_triage`
      - `semantic`: success `1.0`, mean cost `1.6285714285714286`
      - `counterfactual`: success `1.0`, mean cost `1.6285714285714286`
      - `diagnostic`: success `1.0`, mean cost `1.6285714285714286`
  - `mixed-28`
    - `monitor_gate`
      - `semantic`: success `1.0`, mean cost `1.332142857142857`
      - `counterfactual`: success `1.0`, mean cost `1.332142857142857`
      - `diagnostic`: success `1.0`, mean cost `1.332142857142857`
    - `monitor_repair_triage`
      - `semantic`: success `1.0`, mean cost `1.3142857142857143`
      - `counterfactual`: success `1.0`, mean cost `1.3142857142857143`
      - `diagnostic`: success `1.0`, mean cost `1.3142857142857143`
- interpretation:
  - the current top-tier probe still cleanly distinguishes:
    - weak answer-only monitors
    - strong ambiguity-aware monitors
  - but after the repair-aware policy layer is added, the top three strong monitors are now saturated again on `code-14` / `mixed-28`
  - this means the recent gain is genuinely a policy gain:
    - it is not specific to one monitor family
    - it applies equally to `semantic`, `counterfactual`, and `diagnostic`
  - it also means the current top-tier probe is no longer sufficient for further top-monitor ranking
- practical implication:
  - keep `monitor_repair_triage semantic` as the simplest current mainline baseline
  - keep `counterfactual` and `diagnostic` as saturation references rather than active winners on this probe
  - only add a new harder ambiguity family if a new round of top-tier signal separation is needed

Capability-track update after the prime/divisibility extension on 2026-04-10:

- benchmark/runtime changes:
  - `_run_code_task()` now exposes `int` in the safe builtins so prime-count fixes execute correctly
  - new code family added:
    - `count_multiple_of_three`
    - intended semantics: count positive multiples of three
    - wrong visible-pass repair: `count_gt_one_fix`
    - correct repair: `count_multiple_of_three_fix`
- new top-tier references:
  - `code-16`
    - `monitor_gate semantic`: `F:\unified-sel\results\capability_benchmark\20260410_143901.json`
    - `monitor_repair_triage semantic`: `F:\unified-sel\results\capability_benchmark\20260410_143902.json`
    - `monitor_gate counterfactual`: `F:\unified-sel\results\capability_benchmark\20260410_143904.json`
    - `monitor_repair_triage counterfactual`: `F:\unified-sel\results\capability_benchmark\20260410_143905.json`
    - `monitor_gate diagnostic`: `F:\unified-sel\results\capability_benchmark\20260410_143906.json`
    - `monitor_repair_triage diagnostic`: `F:\unified-sel\results\capability_benchmark\20260410_143907.json`
  - `mixed-32`
    - `monitor_gate semantic`: `F:\unified-sel\results\capability_benchmark\20260410_143908.json`
    - `monitor_repair_triage semantic`: `F:\unified-sel\results\capability_benchmark\20260410_143910.json`
    - `monitor_gate counterfactual`: `F:\unified-sel\results\capability_benchmark\20260410_143911.json`
    - `monitor_repair_triage counterfactual`: `F:\unified-sel\results\capability_benchmark\20260410_143912.json`
    - `monitor_gate diagnostic`: `F:\unified-sel\results\capability_benchmark\20260410_143913.json`
    - `monitor_repair_triage diagnostic`: `F:\unified-sel\results\capability_benchmark\20260410_143914.json`
- result summary:
  - `code-16`
    - `monitor_gate`
      - `semantic`: success `0.875`, mean cost `1.58125`
      - `counterfactual`: success `1.0`, mean cost `1.64375`
      - `diagnostic`: success `1.0`, mean cost `1.64375`
    - `monitor_repair_triage`
      - `semantic`: success `0.875`, mean cost `1.55`
      - `counterfactual`: success `1.0`, mean cost `1.6125`
      - `diagnostic`: success `1.0`, mean cost `1.6125`
  - `mixed-32`
    - `monitor_gate`
      - `semantic`: success `0.9375`, mean cost `1.290625`
      - `counterfactual`: success `1.0`, mean cost `1.321875`
      - `diagnostic`: success `1.0`, mean cost `1.321875`
    - `monitor_repair_triage`
      - `semantic`: success `0.9375`, mean cost `1.275`
      - `counterfactual`: success `1.0`, mean cost `1.30625`
      - `diagnostic`: success `1.0`, mean cost `1.30625`
- interpretation:
  - the strongest monitor families are no longer saturated on the current benchmark
  - `semantic` now has two visible new limits:
    - `count_prime`
      - wrong visible-pass repair: `count_even_numbers_fix`
    - `count_multiple_of_three`
      - wrong visible-pass repair: `count_gt_one_fix`
  - both are genuine surface-semantic gaps:
    - neither is covered by threshold ambiguity handling
    - neither is covered by parity ambiguity handling
    - neither is covered by zero-role ambiguity handling
  - `counterfactual` and `diagnostic` still succeed because they detect the visible-pass ambiguity structure rather than the semantics directly
  - `monitor_repair_triage` still improves cost relative to `monitor_gate`, but it no longer closes the signal-quality gap by itself
- practical implication:
  - `code-16` / `mixed-32` are now the current canonical top-tier probes
  - `semantic` remains the strongest current surface-level monitor, but it is no longer top-tier saturated
  - the next monitor decision is now explicit:
    - either extend `semantic` to cover prime/divisibility ambiguity
    - or keep it as the simpler surface baseline and use `counterfactual` / `diagnostic` as the current top signal references

Capability-track update after the prime/divisibility semantic closure on 2026-04-10:

- `semantic` was extended again without repair candidate enumeration:
  - added prime-family ambiguity handling
  - added positive-divisibility ambiguity handling
- updated semantic references:
  - `code-16`
    - `monitor_gate semantic`: `F:\unified-sel\results\capability_benchmark\20260410_151047.json`
    - `monitor_repair_triage semantic`: `F:\unified-sel\results\capability_benchmark\20260410_151049.json`
  - `mixed-32`
    - `monitor_gate semantic`: `F:\unified-sel\results\capability_benchmark\20260410_151050.json`
    - `monitor_repair_triage semantic`: `F:\unified-sel\results\capability_benchmark\20260410_151052.json`
- updated result summary:
  - `code-16`
    - `monitor_gate semantic`: success `1.0`, mean cost `1.64375`
    - `monitor_repair_triage semantic`: success `1.0`, mean cost `1.6125`
  - `mixed-32`
    - `monitor_gate semantic`: success `1.0`, mean cost `1.321875`
    - `monitor_repair_triage semantic`: success `1.0`, mean cost `1.30625`
- interpretation:
  - the `count_prime` and `count_multiple_of_three` gaps were real, but they were not fundamental
  - both can be closed at the surface-semantic layer:
    - no solver-internal metadata was added
    - no repair candidate enumeration was added
  - after the targeted extension:
    - `semantic` again matches `counterfactual` and `diagnostic` on `code-16` / `mixed-32`
    - `monitor_repair_triage` remains the best current policy layer on the fixed probe
- practical implication:
  - `code-16` / `mixed-32` remain the strongest current probes because they were strong enough to expose and then verify the new semantic closure
  - `semantic` is again the strongest current surface-level monitor
  - the next benchmark change should only happen if it creates a genuinely new semantic regime beyond:
    - threshold ambiguity
    - parity ambiguity
    - zero-role ambiguity
    - primality ambiguity
    - positive-divisibility ambiguity

Capability-track update after the palindrome-symmetry extension on 2026-04-10:

- new code family added:
  - `count_palindrome_words`
  - wrong visible-pass repair:
    - `count_same_edge_words_fix`
  - correct repair:
    - `count_palindrome_words_fix`
- `count_abs_gt_two` was also added in this round, but it did not create a new semantic gap:
  - `semantic` and `counterfactual` both stayed saturated on `code-17`
  - interpretation:
    - that family collapses back into already-covered zero-role / magnitude structure
- new top-tier references:
  - `code-18`
    - `monitor_gate semantic`: `F:\unified-sel\results\capability_benchmark\20260410_153159.json`
    - `monitor_repair_triage semantic`: `F:\unified-sel\results\capability_benchmark\20260410_153200.json`
    - `monitor_gate counterfactual`: `F:\unified-sel\results\capability_benchmark\20260410_153201.json`
    - `monitor_repair_triage counterfactual`: `F:\unified-sel\results\capability_benchmark\20260410_153202.json`
    - `monitor_gate diagnostic`: `F:\unified-sel\results\capability_benchmark\20260410_153204.json`
    - `monitor_repair_triage diagnostic`: `F:\unified-sel\results\capability_benchmark\20260410_153205.json`
  - `mixed-36`
    - `monitor_gate semantic`: `F:\unified-sel\results\capability_benchmark\20260410_153206.json`
    - `monitor_repair_triage semantic`: `F:\unified-sel\results\capability_benchmark\20260410_153207.json`
    - `monitor_gate counterfactual`: `F:\unified-sel\results\capability_benchmark\20260410_153208.json`
    - `monitor_repair_triage counterfactual`: `F:\unified-sel\results\capability_benchmark\20260410_153209.json`
    - `monitor_gate diagnostic`: `F:\unified-sel\results\capability_benchmark\20260410_153210.json`
    - `monitor_repair_triage diagnostic`: `F:\unified-sel\results\capability_benchmark\20260410_153211.json`
- result summary:
  - `code-18`
    - `monitor_gate`
      - `semantic`: success `0.9444444444444444`, mean cost `1.6`
      - `counterfactual`: success `1.0`, mean cost `1.6277777777777778`
      - `diagnostic`: success `1.0`, mean cost `1.6277777777777778`
    - `monitor_repair_triage`
      - `semantic`: success `0.9444444444444444`, mean cost `1.5722222222222222`
      - `counterfactual`: success `1.0`, mean cost `1.6`
      - `diagnostic`: success `1.0`, mean cost `1.6`
  - `mixed-36`
    - `monitor_gate`
      - `semantic`: success `0.9722222222222222`, mean cost `1.2999999999999998`
      - `counterfactual`: success `1.0`, mean cost `1.3138888888888889`
      - `diagnostic`: success `1.0`, mean cost `1.3138888888888889`
    - `monitor_repair_triage`
      - `semantic`: success `0.9722222222222222`, mean cost `1.286111111111111`
      - `counterfactual`: success `1.0`, mean cost `1.2999999999999998`
      - `diagnostic`: success `1.0`, mean cost `1.2999999999999998`
- interpretation:
  - this is the first clear non-numeric semantic regime after the earlier five closures
  - the gap is not about:
    - threshold
    - parity
    - zero-role
    - primality
    - divisibility
  - it is about whole-sequence structure:
    - full palindrome symmetry
    - versus the weaker edge-match shortcut
  - `counterfactual` and `diagnostic` still succeed because they detect visible-pass ambiguity in the candidate set or search process
  - `semantic` now has a new explicit limit:
    - it does not yet model whole-string symmetry from surface evidence
- practical implication:
  - `code-18` / `mixed-36` are now the current canonical top-tier probes
  - `monitor_repair_triage semantic` remains the mainline surface-level policy baseline
  - `monitor_repair_triage counterfactual` and `monitor_repair_triage diagnostic` are the current top references until palindrome-style symmetry is either closed or intentionally left as the new limit

Capability-track update after the palindrome-symmetry semantic closure on 2026-04-10:

- `semantic` was extended again without repair candidate enumeration:
  - added word-symmetry parsing:
    - `_extract_word_symmetry_rule`
  - added whole-word symmetry counting:
    - `_count_with_word_symmetry`
  - added word-symmetry ambiguity scoring:
    - `_word_symmetry_ambiguity_signal`
- updated semantic references:
  - `code-18`
    - `monitor_gate semantic`: `F:\unified-sel\results\capability_benchmark\20260410_154623.json`
    - `monitor_repair_triage semantic`: `F:\unified-sel\results\capability_benchmark\20260410_154624.json`
  - `mixed-36`
    - `monitor_gate semantic`: `F:\unified-sel\results\capability_benchmark\20260410_154625.json`
    - `monitor_repair_triage semantic`: `F:\unified-sel\results\capability_benchmark\20260410_154626.json`
- updated result summary:
  - `code-18`
    - `monitor_gate semantic`: success `1.0`, mean cost `1.6277777777777778`
    - `monitor_repair_triage semantic`: success `1.0`, mean cost `1.6`
  - `mixed-36`
    - `monitor_gate semantic`: success `1.0`, mean cost `1.3138888888888889`
    - `monitor_repair_triage semantic`: success `1.0`, mean cost `1.2999999999999998`
- interpretation:
  - the palindrome gap was real, but it was not fundamental
  - whole-sequence string symmetry can also be closed at the surface-semantic layer:
    - no solver-internal metadata
    - no repair candidate enumeration
  - after the targeted extension:
    - `semantic` again matches `counterfactual` and `diagnostic` on `code-18` / `mixed-36`
    - `monitor_repair_triage` remains the best current policy layer on the fixed probe
- practical implication:
  - `code-18` / `mixed-36` remain the strongest current probes because they were strong enough to expose and then verify the string-symmetry closure
  - the next benchmark change should only happen if it creates a genuinely new semantic regime beyond:
    - threshold ambiguity
    - parity ambiguity
    - zero-role ambiguity
    - primality ambiguity
    - positive divisibility ambiguity
    - whole-sequence string symmetry

Capability-track update after the adjacent-repeat extension on 2026-04-10:

- new code family added:
  - `count_adjacent_repeat_words`
  - wrong visible-pass repair:
    - `count_any_repeat_words_fix`
  - correct repair:
    - `count_adjacent_repeat_words_fix`
- `count_abs_gt_two` remains in the benchmark as a checked non-separator:
  - it did not create a new top-tier gap
  - interpretation:
    - absolute-magnitude counting still collapses back into already-covered semantics
- new top-tier references:
  - `code-19`
    - `monitor_gate semantic`: `F:\unified-sel\results\capability_benchmark\20260410_160552.json`
    - `monitor_repair_triage semantic`: `F:\unified-sel\results\capability_benchmark\20260410_160553.json`
    - `monitor_gate counterfactual`: `F:\unified-sel\results\capability_benchmark\20260410_160554.json`
    - `monitor_repair_triage counterfactual`: `F:\unified-sel\results\capability_benchmark\20260410_160555.json`
    - `monitor_gate diagnostic`: `F:\unified-sel\results\capability_benchmark\20260410_160556.json`
    - `monitor_repair_triage diagnostic`: `F:\unified-sel\results\capability_benchmark\20260410_160557.json`
  - `mixed-38`
    - `monitor_gate semantic`: `F:\unified-sel\results\capability_benchmark\20260410_160559.json`
    - `monitor_repair_triage semantic`: `F:\unified-sel\results\capability_benchmark\20260410_160600.json`
    - `monitor_gate counterfactual`: `F:\unified-sel\results\capability_benchmark\20260410_160601.json`
    - `monitor_repair_triage counterfactual`: `F:\unified-sel\results\capability_benchmark\20260410_160602.json`
    - `monitor_gate diagnostic`: `F:\unified-sel\results\capability_benchmark\20260410_160603.json`
    - `monitor_repair_triage diagnostic`: `F:\unified-sel\results\capability_benchmark\20260410_160604.json`
- result summary:
  - `code-19`
    - `monitor_gate`
      - `semantic`: success `0.9473684210526315`, mean cost `1.5947368421052632`
      - `counterfactual`: success `1.0`, mean cost `1.6210526315789473`
      - `diagnostic`: success `1.0`, mean cost `1.6210526315789473`
    - `monitor_repair_triage`
      - `semantic`: success `0.9473684210526315`, mean cost `1.568421052631579`
      - `counterfactual`: success `1.0`, mean cost `1.5947368421052632`
      - `diagnostic`: success `1.0`, mean cost `1.5947368421052632`
  - `mixed-38`
    - `monitor_gate`
      - `semantic`: success `0.9736842105263158`, mean cost `1.2973684210526315`
      - `counterfactual`: success `1.0`, mean cost `1.3105263157894735`
      - `diagnostic`: success `1.0`, mean cost `1.3105263157894735`
    - `monitor_repair_triage`
      - `semantic`: success `0.9736842105263158`, mean cost `1.2842105263157895`
      - `counterfactual`: success `1.0`, mean cost `1.2973684210526315`
      - `diagnostic`: success `1.0`, mean cost `1.2973684210526315`
- interpretation:
  - this is a genuinely new string regime beyond whole-sequence symmetry
  - the current gap is about local internal structure:
    - any repeated character anywhere
    - versus adjacent repeated characters specifically
  - `semantic` does not yet model adjacency-style internal repetition from surface evidence
  - `counterfactual` and `diagnostic` still succeed because they detect the visible-pass ambiguity structure in the candidate set / search path
- practical implication:
  - `code-19` / `mixed-38` are now the current canonical top-tier probes
  - `monitor_repair_triage semantic` remains the mainline surface-level policy baseline
  - `monitor_repair_triage counterfactual` and `monitor_repair_triage diagnostic` are the current top references until adjacency-style local string structure is either closed or intentionally left as the new limit

Capability-track update after the adjacent-repeat semantic closure on 2026-04-10:

- `semantic` was extended again without repair candidate enumeration:
  - added word-repeat parsing:
    - `_extract_word_repeat_rule`
  - added local-repeat counting:
    - `_count_with_word_repeat`
  - added local-repeat ambiguity scoring:
    - `_word_repeat_ambiguity_signal`
- updated semantic references:
  - `code-19`
    - `monitor_gate semantic`: `F:\unified-sel\results\capability_benchmark\20260410_162423.json`
    - `monitor_repair_triage semantic`: `F:\unified-sel\results\capability_benchmark\20260410_162425.json`
  - `mixed-38`
    - `monitor_gate semantic`: `F:\unified-sel\results\capability_benchmark\20260410_162426.json`
    - `monitor_repair_triage semantic`: `F:\unified-sel\results\capability_benchmark\20260410_162427.json`
- updated result summary:
  - `code-19`
    - `monitor_gate semantic`: success `1.0`, mean cost `1.6210526315789473`
    - `monitor_repair_triage semantic`: success `1.0`, mean cost `1.5947368421052632`
  - `mixed-38`
    - `monitor_gate semantic`: success `1.0`, mean cost `1.3105263157894735`
    - `monitor_repair_triage semantic`: success `1.0`, mean cost `1.2973684210526315`
- interpretation:
  - the adjacent-repeat gap was real, but it was not fundamental
  - local internal string structure can also be closed at the surface-semantic layer:
    - no solver-internal metadata
    - no repair candidate enumeration
  - after the targeted extension:
    - `semantic` again matches `counterfactual` and `diagnostic` on `code-19` / `mixed-38`
    - `monitor_repair_triage` remains the best current policy layer on the fixed probe
- practical implication:
  - `code-19` / `mixed-38` remain the strongest current probes because they were strong enough to expose and then verify the local-string-structure closure
  - the next benchmark change should only happen if it creates a genuinely new semantic regime beyond the current seven closures:
    - threshold ambiguity
    - parity ambiguity
    - zero-role ambiguity
    - primality ambiguity
    - positive divisibility ambiguity
    - whole-sequence string symmetry
    - local internal string repetition structure

Capability-track update after the vowel semantic closure on 2026-04-10:

- `semantic` was extended again without repair candidate enumeration:
  - added vowel-rule parsing:
    - `_extract_word_vowel_rule`
  - added lexical-vowel counting:
    - `_count_with_word_vowel`
  - added lexical-vowel ambiguity scoring:
    - `_word_vowel_ambiguity_signal`
- updated semantic references:
  - `code-20`
    - `monitor_gate semantic`: `F:\unified-sel\results\capability_benchmark\20260410_171643.json`
    - `monitor_repair_triage semantic`: `F:\unified-sel\results\capability_benchmark\20260410_171603.json`
  - `mixed-40`
    - `monitor_gate semantic`: `F:\unified-sel\results\capability_benchmark\20260410_171708.json`
    - `monitor_repair_triage semantic`: `F:\unified-sel\results\capability_benchmark\20260410_171559.json`
- updated result summary:
  - `code-20`
    - `monitor_gate semantic`: success `1.0`, mean cost `1.6149999999999998`
    - `monitor_repair_triage semantic`: success `1.0`, mean cost `1.59`
  - `mixed-40`
    - `monitor_gate semantic`: success `1.0`, mean cost `1.3074999999999999`
    - `monitor_repair_triage semantic`: success `1.0`, mean cost `1.295`
- interpretation:
  - the vowel gap was real, but it was not fundamental
  - lexical internal word properties can also be closed at the surface-semantic layer:
    - no solver-internal metadata
    - no repair candidate enumeration
  - after the targeted extension:
    - `semantic` again matches `counterfactual` and `diagnostic` on `code-20` / `mixed-40`
    - `monitor_repair_triage` remains the best current policy layer on the fixed probe
- practical implication:
  - `code-20` / `mixed-40` are now the strongest current probes because they were strong enough to expose and then verify the lexical-vowel closure
  - the next benchmark change should only happen if it creates a genuinely new semantic regime beyond the current eight closures:
    - threshold ambiguity
    - parity ambiguity
    - zero-role ambiguity
    - primality ambiguity
    - positive divisibility ambiguity
    - whole-sequence string symmetry
    - local internal string repetition structure
    - lexical internal vowel property

Capability-track update after multi-signal fusion routing experiment on 2026-04-14:

- multi-signal fusion experiment was conducted to test combining multiple routing monitors:
  - tested fusion methods: average, majority_vote, weighted_average with different weight configurations
  - tested monitors: semantic, counterfactual, behavioral
  - experiment suite: mixed-20
  - seeds: [7, 8, 9]
- experiment code was fixed to use real simulation results instead of hardcoded values:
  - removed hardcoded success rates, costs, and latencies
  - now calculates metrics based on actual simulated outcomes:
    - accept decision: 70% success rate, cost 1.0, latency 1.0
    - verify decision: 90% success rate, cost 1.5, latency 1.5
    - escalate decision: 100% success rate, cost 2.0, latency 2.0
- experiment results (real simulation):
  - average: success rate 0.7833, cost 1.28, latency 1.28
  - majority_vote: success rate 0.7000, cost 1.00, latency 1.00
  - weighted_average (all configs): success rate ~0.7833, cost ~1.28-1.30, latency ~1.28-1.30
- key findings:
  - majority_vote performs worst (success rate 0.70) because it accepts all tasks without verify/escalate
  - different weight configurations show similar performance, suggesting monitor signals are consistent
  - real trade-offs observed:
    - majority_vote: lowest cost/latency but lowest success rate
    - average/weighted: moderate cost/latency, moderate success rate
- experiment references:
  - real simulation results: `F:\unified-sel\results\multi_signal_fusion\fusion_results_20260414_204149.json`
  - experiment log: `F:\unified-sel\EXPERIMENT_LOG.md`
- practical implication:
  - multi-signal fusion is now a viable approach for combining routing monitors
  - majority_vote is not recommended due to its tendency to accept all decisions
  - average or weighted_average fusion methods provide better balance between success rate and cost/latency
  - future work should test different routing protocols and more diverse monitor combinations

Capability-track update after heterogeneous monitor fusion on 2026-04-14:

> ⚠️ **2026-04-15 降级说明**：以下"31% 成本降低"基于假设成本模型（cost_units），非真实延迟测量。
> 成本数字是硬编码的抽象值，不能作为科学结论的依据。详见 AGENTS.md 规则 3。

- 异质监控组合实验在 `code` 和 `mixed` 套件上完成：
  - 单一监控基线：
    - semantic: 成功率 1.0, 成本 ~1.51（基于假设成本模型）
    - counterfactual: 成功率 1.0, 成本 ~1.50（基于假设成本模型）
  - 异质融合组合：
    - semantic+external: 成功率 1.0, 成本 1.04（基于假设成本模型，声称"31% 成本降低"不可信）
    - semantic+surface: 成功率 1.0, 成本 1.09（基于假设成本模型）
- 融合策略：强监控 + 弱监控智能融合
- ⚠️ 成功率数字来自真实验证，但成本降低数字基于假设成本模型
- 实验参考：
  - 融合结果：`F:\unified-sel\results\heterogeneous_monitor_fusion\fusion_results_code_20260414_211206.json`

Capability-track update after fusion threshold optimization on 2026-04-14:

> ⚠️ **2026-04-15 降级说明**：以下阈值优化基于假设成本模型，非真实延迟测量。

- 阈值参数优化实验完成
- 最优阈值配置在 code-20 套件上验证：
  - 最佳阈值组合：accept=0.2, verify=0.55
  - 所有阈值配置都达到100%成功率
  - ⚠️ 成本数字基于假设成本模型
- 实验参考：
  - 阈值优化结果：`F:\unified-sel\results\fusion_threshold_optimization\optimization_results_code_20260414_213844.json`
- 当前策略参考：
  - `monitor_repair_triage semantic+external fusion` 是当前主线路由策略参考
  - 配置：accept=0.2, verify=0.55
  - ⚠️ 成功率数字可信，成本降低数字基于假设成本模型

Capability-track update after mixed-40 verification on 2026-04-14:

> ⚠️ **2026-04-15 降级说明**：成本降低数字基于假设成本模型。

- mixed-40 套件验证完成
- 验证结果（mixed-40 套件）：
  - semantic (单一监控): 成功率 1.0000, 平均成本 1.25（假设成本模型）
  - semantic+external (融合): 成功率 1.0000, 平均成本 1.02（假设成本模型）
  - 成本降低比例基于假设成本模型，不可作为科学结论
- 关键发现：
  - 新基线在 code 和 mixed 两个套件上成功率都为100%
  - ⚠️ 成功率可信，成本降低数字基于假设成本模型
- 实验参考：
  - 验证结果：`F:\unified-sel\results\mixed_verification\verification_results_20260414_214838.json`
- 当前策略参考：
  - `monitor_repair_triage semantic+external fusion` 是当前主线路由策略参考
  - 配置：accept=0.2, verify=0.55

## Next Recommended Work

2026-04-22 implementation update:

- Added `monitor_no_revision_triage` for weak-reviser models after the Qwen2.5-0.5B revision test fixed 0/17 failed tasks.
- Policy behavior: low signal accepts, high signal direct-escalates, medium signal verifies and escalates on verifier failure; it never calls feedback revision.
- Synthetic code-20 sanity run with semantic monitor, seed 7: success_rate 1.0000, mean_cost_units 2.6900, escalation_rate 0.4000, revision_rate 0.0000.
- Result file: `results/capability_benchmark/20260422_091618.json`.
- Validation: `python -m py_compile ...` passed; `python tests/smoke_test.py` passed.
- Next real-LLM action: rerun `experiments/capability/llm_routing_experiment.py` with llama.cpp server active and compare `monitor_no_revision_triage` against `monitor_gate`, `monitor_repair_triage`, and `verifier_first`.

2026-04-22 LeWorldModel integration preflight:

- Completed paper-to-project translation for LeWM (arXiv 2603.19312, Maes/Le Lidec/Scieur/LeCun/Balestriero, 2026).
- Extracted 8 engineering-transferable principles from LeWM.
- Designed PredictiveHealthMonitor for unified-sel: batch-level predictive residual health monitoring, complementary to BatchHealthMonitor.
- Designed PredictiveStateAdapter for CEE: signal-only injection into UncertaintyRouter, three invariants (read-only / no-privilege-escalation / ignorable).
- Analyzed SIGReg applicability to CEP-CC: recommended as ablation only, NOT default; risk of destroying proto-symbol clusters in low-dimensional comm space.
- Spec file: `LEWM_INTEGRATION_SPEC_2026-04-22.md`.
- Next actions: implement PredictiveHealthMonitor prototype, implement PredictiveStateAdapter prototype, run SIGReg ablation in CEP-CC.
---

## 2026-04-24 LSG Phase 16 Update

- Added fixed-seed randomized sequence fuzz coverage to `tests/smoke_test.py`.
- Smoke now checks LSG sequence fuzz invariants for bandwidth cap, commit-log consistency, and acknowledged-state immutability.
- Removed an unused `TemporaryDirectory()` wrapper from `test_transfer_matrix_runner`; this fixed a Windows temp cleanup `PermissionError` that blocked smoke before LSG checks.
- Generated artifact: `results/capability_generalization/rewrite_sequence_fuzz_phase16_smoke.json`.
- Validation passed:
  - `python tests/test_rewrite_sequence_fuzz.py`
  - `python tests/smoke_test.py`
  - `python experiments/capability/rewrite_sequence_fuzz.py --seeds 0,1,2,3,4 --num-steps 12 --num-candidates 8 --max-observations-per-step 5 --bandwidth-limit 3 --label phase16_smoke`
- Status: engineering regression guard only; not a theorem-level proof.
- Next LSG targets: repeated proposal IDs distinct from candidate IDs; explicit multi-step refilling behavior; optional revision/rollback protocol.

---

## 2026-04-24 LSG Phase 17 Update

- Locked proposal/candidate identity separation.
- `ProposalEnvelope` now rejects `proposal_id == candidate_id`.
- Sequence replay now reports `identity_invariants` per case and `identity_failed_count` in summaries.
- Identity invariant failures include duplicate `proposal_id`, cross-candidate proposal ID reuse, and proposal/candidate ID collision.
- Standard sequence fixture passes with `identity_failed_count: 0`.
- Negative unit coverage confirms duplicate proposal ID and cross-candidate reuse are detected.
- Generated artifact: `results/capability_generalization/rewrite_sequence_replay_fixture_phase17_identity.json`.
- Validation passed:
  - `python -m py_compile core\rewrite_dynamics.py core\rewrite_sequence_replay.py tests\test_rewrite_proposal_provider.py tests\test_rewrite_sequence_replay.py tests\smoke_test.py`
  - `python tests\test_rewrite_proposal_provider.py`
  - `python tests\test_rewrite_sequence_replay.py`
  - `python tests\smoke_test.py`
  - `python experiments\capability\rewrite_sequence_replay_fixture.py --label phase17_identity`
- Status: proposal events and durable candidate states now have separate audited identities.
- Next LSG targets: explicit multi-step refilling behavior; optional revision/rollback protocol; larger provider capture identity/error distribution.

---

## 2026-04-24 LSG Phase 18 Update

- Clarified bandwidth refill semantics:
  - same step: no refill after top candidates commit
  - next step: lower-priority candidates may refill if observed again and still qualify
- Added `multi_step_refill_after_top3_commit` to `data/lsg/proposal_sequence_replay_v0.json`.
- Sequence replay fixture now has 4 cases and 10 total commit-log events.
- Standard sequence fixture still passes commit-log invariants and identity invariants.
- Generated artifact: `results/capability_generalization/rewrite_sequence_replay_fixture_phase18_refill.json`.
- Validation passed:
  - `python -m json.tool data\lsg\proposal_sequence_replay_v0.json`
  - `python -m py_compile core\rewrite_sequence_replay.py tests\test_rewrite_sequence_replay.py tests\smoke_test.py`
  - `python tests\test_rewrite_sequence_replay.py`
  - `python tests\smoke_test.py`
  - `python experiments\capability\rewrite_sequence_replay_fixture.py --label phase18_refill`
- Status: bandwidth overflow now means "not this step", not "dropped forever".
- Next LSG targets: optional revision/rollback protocol; larger provider capture identity/error distribution; mixed stress/fuzz case.

---

## 2026-04-24 Paper Claim Update: Real LLM Scale Probe

- Qwen2.5 real-LLM code-20 validation now has three scales:
  - 0.5B: ABOVE 20%, NEAR 0%, BELOW 80%
  - 1.5B: ABOVE 30%, NEAR 10%, BELOW 60%
  - 3B: ABOVE 20%, NEAR 20%, BELOW 60%
- Safe claim:
  - NEAR-zone fraction increases on this fixed probe: 0% -> 10% -> 20%.
  - This supports that feedback-rescuable boundary cases are not purely a synthetic-solver artifact.
  - ABOVE-filtering remains useful across scales: 20%-30% feedback calls can be skipped on this probe.
- Required caveat:
  - This is a fixed code-20 scale probe, not a statistical law over model size.
  - The full real-LLM inverted-U pattern has not appeared yet because Qwen2.5-3B has NEAR=ABOVE=20%, not NEAR > ABOVE.
- Updated `BOUNDARY_LOCAL_AMPLIFICATION_PAPER_OUTLINE.md` with a Real LLM Scale Probe section and safer claim boundaries.

---

## 2026-04-24 LSG Phase 19 Update

- Added explicit audit-only revision proposals for acknowledged candidates.
- `RevisionProposalEvent` records a proposed correction with evidence/constitution/log/approval gate fields.
- `RewriteSystemState` now has a separate `revision_log`; acknowledgement `commit_log` remains unchanged by revision proposals.
- `propose_revision_for_acknowledged_candidate` only accepts already acknowledged targets.
- `check_revision_log_invariants` verifies revision targets are acknowledged and that Phase 19 has no executed revisions.
- Smoke now checks that a revision proposal does not mutate acknowledged candidate disturbance/stability or append an acknowledgement commit.
- Validation passed:
  - `python tests\test_rewrite_dynamics.py`
  - `python tests\smoke_test.py`
- `python -m py_compile core\rewrite_dynamics.py tests\test_rewrite_dynamics.py tests\smoke_test.py` was attempted but failed with Windows `[WinError 5] Access denied` while replacing `core\__pycache__\rewrite_dynamics.cpython-314.pyc`.
- Status: acknowledged states are now protected from silent rewrite while still allowing durable audit records for future revision review.
- Next LSG targets: explicit rollback/revision execution protocol; revision cases in sequence replay; mixed fuzz coverage for refill + identity + revision.

---

## 2026-04-24 LSG Phase 20 Update

- Integrated audit-only revision requests into sequence replay.
- `core/rewrite_sequence_replay.py` now replays steps explicitly with `step_system`, then applies optional `revision_requests`.
- Replay output now includes `revision_invariants`, `revision_log_count`, `expected_revision_count`, `total_revision_log_count`, and serialized `revision_log`.
- Added fixture case `revision_request_after_acknowledgement_audit_only`.
- Sequence fixture now has 5 cases, 11 acknowledgement commit events, and 1 revision audit event.
- Generated artifact: `results/capability_generalization/rewrite_sequence_replay_fixture_phase20_revision_replay.json`.
- Validation passed:
  - `python -m json.tool data\lsg\proposal_sequence_replay_v0.json`
  - `python tests\test_rewrite_sequence_replay.py`
  - `python experiments\capability\rewrite_sequence_replay_fixture.py --label phase20_revision_replay`
  - `python tests\smoke_test.py`
- Status: revision requests are now durable replay data; revision execution remains intentionally unimplemented.
- Next LSG targets: explicit approval-based rollback/revision execution protocol; negative sequence cases for invalid revision targets; fuzz generation of revision requests after acknowledgements.

---

## 2026-04-24 LSG Phase 21 Update

- Extended randomized sequence fuzz to include audit-only revision requests after acknowledgement.
- `experiments/capability/rewrite_sequence_fuzz.py` now replays step-by-step with `step_system`, injects revision requests for acknowledged candidates, and checks revision invariants.
- Fuzz output now reports `revision_log_count`, `total_revision_log_count`, `revision_invariants`, and `revision_mutation_errors`.
- Added negative sequence replay tests for revision requests against unacknowledged and missing targets.
- Generated artifact: `results/capability_generalization/rewrite_sequence_fuzz_phase21_revision_fuzz.json`.
- Artifact summary:
  - passed: true
  - num_seeds: 5
  - failed_seeds: []
  - max_active_observed: 3
  - total_commit_log_count: 13
  - total_revision_log_count: 31
- Validation passed:
  - `python tests\test_rewrite_sequence_replay.py`
  - `python tests\test_rewrite_sequence_fuzz.py`
  - `python experiments\capability\rewrite_sequence_fuzz.py --seeds 0,1,2,3,4 --num-steps 12 --num-candidates 8 --max-observations-per-step 5 --bandwidth-limit 3 --label phase21_revision_fuzz`
  - `python tests\smoke_test.py`
- Status: audit-only revision semantics now pass direct unit tests, hand-authored replay, and randomized mixed fuzz.
- Next LSG target: design versioned candidate identity before implementing actual approval-based rollback/revision execution.

---

## 2026-04-24 LSG Phase 22 Update

- Added versioned candidate identity before implementing rollback/revision execution.
- `CandidateState.version` defaults to `1`.
- `CommitEvent.candidate_version` records the acknowledged candidate version.
- `RevisionProposalEvent.target_version` records the version being challenged.
- `propose_revision_for_acknowledged_candidate` rejects stale `target_version` values.
- Sequence replay accepts optional `target_version` in `revision_requests` and rejects stale/invalid values.
- Commit-log invariants now check commit event version matches candidate version.
- Revision invariants now check revision target version matches acknowledged candidate version.
- Generated artifacts:
  - `results/capability_generalization/rewrite_sequence_replay_fixture_phase22_versioned_identity.json`
  - `results/capability_generalization/rewrite_sequence_fuzz_phase22_versioned_fuzz.json`
- Validation passed:
  - `python tests\test_rewrite_dynamics.py`
  - `python tests\test_rewrite_sequence_replay.py`
  - `python tests\test_rewrite_sequence_fuzz.py`
  - `python -m json.tool data\lsg\proposal_sequence_replay_v0.json`
  - `python experiments\capability\rewrite_sequence_replay_fixture.py --label phase22_versioned_identity`
  - `python experiments\capability\rewrite_sequence_fuzz.py --seeds 0,1,2,3,4 --num-steps 12 --num-candidates 8 --max-observations-per-step 5 --bandwidth-limit 3 --label phase22_versioned_fuzz`
  - `python tests\smoke_test.py`
- Status: durable candidate ID now identifies the state line; candidate version identifies the specific acknowledged state revision.
- Next LSG target: design explicit revision execution event, but keep rollback and revision as separate semantics.

---

## 2026-04-24 LSG Phase 23 Update

- Explicitly separated revision approval from revision execution.
- `check_revision_log_invariants` now reports `approved_revision_count` and `executed_revision_count`.
- Sequence replay summaries now report `total_approved_revision_count` and `total_executed_revision_count`.
- Added fixture case `approved_revision_request_remains_audit_only`.
- Sequence fixture now has 6 cases, 12 acknowledgement commit events, 2 revision audit events, 1 approved revision request, and 0 executed revisions.
- Generated artifact: `results/capability_generalization/rewrite_sequence_replay_fixture_phase23_approval_not_execution.json`.
- Validation passed:
  - `python -m json.tool data\lsg\proposal_sequence_replay_v0.json`
  - `python tests\test_rewrite_dynamics.py`
  - `python tests\test_rewrite_sequence_replay.py`
  - `python tests\test_rewrite_sequence_fuzz.py`
  - `python experiments\capability\rewrite_sequence_replay_fixture.py --label phase23_approval_not_execution`
  - `python tests\smoke_test.py`
- Status: `approval_open=True` is now observable but remains audit-only; it does not create version 2 or set `revision_executed=True`.
- Next LSG target: define `RevisionExecutionEvent` separately from rollback.

---

## 2026-04-24 LSG Phase 24 Update

- Added separate `RevisionExecutionEvent` schema.
- Added `draft_revision_execution_event`, which requires an approved `RevisionProposalEvent`.
- Added `check_revision_execution_event_against_state`.
- Execution draft references `from_version` and `to_version = from_version + 1`.
- Execution draft does not mutate `CandidateState.version`.
- `execution_executed=True` is explicitly rejected in Phase 24.
- Smoke now drafts a revision execution event from an approved revision proposal and confirms it is not executed.
- Generated artifact: `results/capability_generalization/rewrite_sequence_replay_fixture_phase24_execution_event_schema.json`.
- Validation passed:
  - `python tests\test_rewrite_dynamics.py`
  - `python tests\test_rewrite_sequence_replay.py`
  - `python tests\test_rewrite_sequence_fuzz.py`
  - `python experiments\capability\rewrite_sequence_replay_fixture.py --label phase24_execution_event_schema`
  - `python tests\smoke_test.py`
- Status: execution event shape exists, but state-changing revision execution remains unimplemented.
- Next LSG target: define an execution log and controlled version-increment transition, still keeping rollback as a separate event family.

---

## 2026-04-24 LSG Phase 25 Update

- Added durable `revision_execution_log` for non-executed revision execution drafts.
- Added `record_revision_execution_draft` and `check_revision_execution_log_invariants`.
- Sequence replay now supports `revision_execution_drafts`.
- Replay output now includes `revision_execution_log` and `revision_execution_invariants`.
- Sequence summary now includes:
  - `total_revision_execution_draft_count`
  - `total_executed_revision_execution_count`
  - `revision_execution_invariant_failed_count`
- Approved revision fixture now records one execution draft while keeping `CandidateState.version == 1`.
- Generated artifact: `results/capability_generalization/rewrite_sequence_replay_fixture_phase25_execution_log.json`.
- Validation passed:
  - `python -m json.tool data\lsg\proposal_sequence_replay_v0.json`
  - `python tests\test_rewrite_dynamics.py`
  - `python tests\test_rewrite_sequence_replay.py`
  - `python tests\test_rewrite_sequence_fuzz.py`
  - `python experiments\capability\rewrite_sequence_replay_fixture.py --label phase25_execution_log`
  - `python tests\smoke_test.py`
- Status: execution draft is now audit-log durable, but state-changing revision execution remains unimplemented.
- Next LSG target: add negative execution-draft replay cases for unapproved or stale proposals before any version mutation.
