# LSG_ENGINEERING_SPEC_v0

Date: 2026-04-23
Status: 中文可开工 v0 规格
Scope: 最小 learned state governance kernel

---

## 0. 目标

本项目不研究“更强的 agent 行为”，而研究一类**状态治理机制**：

> 系统不是直接根据动作列表反应，而是先估计“候选内容是否有资格改写当前正式状态”。

该资格由一个连续比值控制：

```text
R_t = U_t N_t / (A_t P_t + epsilon)
```

其中：

- `U_t`：候选内容与当前正式状态的不相容度
- `N_t`：忽略该候选内容的代价
- `A_t`：当前正式状态的承认深度
- `P_t`：当前正式状态的传播负荷
- `epsilon > 0`：数值稳定项

本项目的工程目标不是构造“大而全持续认知体”，而是实现一个**最小 learned state governance kernel**，并验证它是否满足以下局部性质：

1. 不越权 commit
2. 不抖振
3. 小扰动不改正式状态
4. 持续高扰动在证据充分时可触发正式承认

---

## 1. 范围与非目标

### 1.1 本期范围

本期只做：

- 明确的状态机
- 显式的 `CommitLog`
- 8 个代理量接口
- `R_t` 的计算与迟滞切换
- 两个最小命题的实验验证：
  - 命题 1：不能越权 commit
  - 命题 2：不抖振

### 1.2 本期不做

本期不做：

- 通用持续认知体
- 全自动 memory design search
- 内生 world model 训练
- 大模型直接修改制度事实
- 复杂多智能体协作
- CEE 运行时集成
- LLM API 调用
- learned proxy head 训练

---

## 2. 最小系统对象

系统在离散时间 `t = 0, 1, 2, ...` 上运行。

总状态定义为：

```text
z_t = (x_t, ell_t, m_t, q_t)
```

其中：

### 2.1 工作状态 `x_t`

表示当前工作层内容：

```text
x_t = {
  working_summary,        // 当前前台工作摘要
  candidate_summary,      // 当前候选内容摘要
  evidence_buffer,        // 当前证据缓冲
  mismatch_buffer         // 当前冲突/残差缓冲
}
```

### 2.2 制度状态 `ell_t`

表示正式层级和治理边界：

```text
ell_t = {
  persistence_class,       // draft/episodic/durable/rule/commitment
  commit_stage,            // draft/pending/verified/committed
  verification_stage,      // none/in_progress/passed/failed
  pending_commitments,     // 未决承诺集合
  permission_state,        // 权限矩阵快照
  rollback_availability,   // 是否可回滚
  log_ready                // 是否已满足写日志条件
}
```

### 2.3 模式状态 `m_t`

最小模式集：

```text
m_t in {M, F, V, C}
```

对应：

- `M`: maintain
- `F`: foreground
- `V`: verify
- `C`: commit-candidate

### 2.4 治理标量 `q_t`

```text
q_t = (U_t, N_t, A_t, P_t, R_t)
```

---

## 3. 8 个最小代理量接口

### 3.1 `u1`：语义/事实冲突

Input:

```text
- candidate_summary
- committed_fact_summary
- active_rule_summary
- recent_tool_results
- recent_verifier_results
```

Output:

```text
- contradiction_score in [0, 1]
- conflict_type
```

来源：

- 小模型 `conflict_head`
- 显式 rule/fact/tool 冲突检查

### 3.2 `u2`：预测失配

Input:

```text
- predicted_next_state
- observed_next_state
- candidate_action_or_claim
- world_model_residual_stats
- tool_outcome_stats
```

Output:

```text
- mismatch_score in [0, 1]
- residual_type
```

来源：

- 显式 residual 统计
- 小模型 `residual_head`

### 3.3 `n1`：忽略后的目标损失

Input:

```text
- current_goal_state
- candidate_summary
- current_plan_state
- recent_failures
- active_controller
```

Output:

```text
- ignore_loss_score in [0, 1]
- affected_goal_axis
```

来源：

- 小模型 `goal_loss_head`

### 3.4 `n2`：忽略后的承诺悬挂代价

Input:

```text
- pending_commitments
- unresolved_verifications
- deadline_state
- irreversibility_state
- candidate_summary
```

Output:

```text
- carry_cost_score in [0, 1]
- commitment_axis
```

来源：

- 显式 pending/deadline 统计
- 小模型 `commitment_pressure_head`

### 3.5 `a1`：制度层级量

Input:

```text
- persistence_class
- commit_stage
- verification_stage
```

Output:

```text
- level_score in [0, 1]
```

来源：

- 显式映射，不能学习替代

建议初始映射：

```text
draft      -> 0.10
pending    -> 0.25
verified   -> 0.55
durable    -> 0.65
rule       -> 0.80
committed  -> 1.00
```

### 3.6 `a2`：当前正式状态的证据锚定强度

Input:

```text
- current_state_evidence_count
- current_state_independent_source_count
- current_state_verifier_pass_rate
- current_state_tool_confirmation_rate
- current_state_source_confidence_stats
```

Output:

```text
- current_anchor_score in [0, 1]
- weak_spot_flag
```

来源：

- 显式统计
- 小模型 `current_evidence_anchor_head`

说明：

- `a2` 指当前正式状态的锚定强度，放在分母。
- 候选内容的证据锚定强度不放入 `A_t`，而进入证据门 `E_t`。
- 这样避免“候选证据越强，越难 commit”的悖论。

### 3.7 `p1`：依赖扇出量

Input:

```text
- reference_count
- downstream_rule_links
- dependent_commitment_count
- reuse_count
```

Output:

```text
- fanout_score in [0, 1]
```

来源：

- 显式图统计，不能由大模型直接决定

### 3.8 `p2`：回滚代价量

Input:

```text
- repair_step_count
- external_side_effect_count
- recomputation_cost
- affected_commitments
- human_review_needed
```

Output:

```text
- rollback_cost_score in [0, 1]
```

来源：

- 显式成本统计
- 小模型 `rollback_cost_head`

---

## 4. 治理标量计算

先定义：

```text
U_t = 1 - (1 - u1) * (1 - u2)
```

```text
N_t = 1 - (1 - n1) * (1 - n2)
```

```text
A_t = a1 * a2
```

```text
P_t = 1 - (1 - p1) * (1 - p2)
```

再定义：

```text
R_t_raw = U_t * N_t / (A_t * P_t + epsilon)
```

考虑工程稳定性，实际切换不直接使用 `R_t_raw`，而使用平滑后的 `Rbar_t`：

```text
Rbar_{t+1} = (1 - alpha) * Rbar_t + alpha * R_t_raw
```

再做斜率裁剪：

```text
R_{t+1} = clip(Rbar_{t+1}, R_t - Delta_max, R_t + Delta_max)
```

其中：

- `alpha` 控制平滑强度
- `Delta_max` 控制单步最大改变量

---

## 5. 阈值与迟滞

定义两个主阈值：

```text
0 < theta_F < theta_C
```

以及迟滞带：

```text
theta_F^- < theta_F^+ < theta_C^- < theta_C^+
```

建议初始只调 4 个数字：

```text
theta_F^-
theta_F^+
theta_C^-
theta_C^+
```

约束：

```text
theta_F^- < theta_F^+ < theta_C^- < theta_C^+
```

含义：

- `theta_F^+`：从 maintain 进入 foreground 的阈值
- `theta_F^-`：从 foreground/verify 回到 maintain 的阈值
- `theta_C^+`：进入 commit-candidate 的阈值
- `theta_C^-`：从 commit-candidate 退回 foreground 的阈值

---

## 6. 证据门与宪法门

定义两个布尔量。

### 6.1 证据门 `E_t`

```text
E_t = 1 iff
- candidate_anchor_score >= candidate_anchor_threshold
- candidate_evidence_count >= min_evidence_count
- verifier_pass_rate >= min_verifier_pass
- no hard contradiction remains
```

说明：

- `E_t` 评估候选内容是否有足够证据。
- `E_t` 不是自由 head 输出。
- `E_t` 可以使用显式统计和小模型聚合，但最终必须可审计。

### 6.2 宪法门 `K_t`

```text
K_t = 1 iff
- permission_state allows requested transition
- irreversible action requires verified state
- log_ready == true
- no bypass of constitutional constraints
```

说明：

- `K_t` 不是自由 head 输出。
- 大模型不能直接打开 `K_t`。
- 候选内容不能修改 `K_t` 的规则。

---

## 7. 模式切换律

### 7.1 从 Maintain

```text
m_{t+1} =
  F, if R_t >= theta_F^+
  M, if R_t <  theta_F^+
```

### 7.2 从 Foreground

```text
m_{t+1} =
  M, if R_t <= theta_F^-
  V, if R_t >= theta_C^- and E_t = 0
  C, if R_t >= theta_C^+ and E_t = 1 and K_t = 1
  F, otherwise
```

### 7.3 从 Verify

```text
m_{t+1} =
  M, if R_t <= theta_F^-
  C, if R_t >= theta_C^+ and E_t = 1 and K_t = 1
  V, otherwise
```

### 7.4 从 Commit-Candidate

```text
m_{t+1} =
  M, if formal_commit_done = 1
  V, if E_t = 0 or K_t = 0
  F, if R_t <= theta_C^-
  C, otherwise
```

---

## 8. 最小状态更新方程

### 8.1 工作状态更新

```text
x_{t+1} = f_{m_t}(x_t, c_t)
```

定义：

- `Phi(c_t)`：候选内容编码
- `mismatch(c_t, x_t)`：候选与当前状态的冲突/残差函数
- `verify(c_t, x_t, ell_t)`：验证产生的新证据量

为简化，写：

```text
x_t = (w_t, e_t, r_t)
```

其中：

- `w_t`：工作摘要
- `e_t`：证据缓冲
- `r_t`：冲突/残差缓冲

#### 在 `M` 中

```text
w_{t+1} = (1 - alpha_M) * w_t + alpha_M * Phi(c_t)
e_{t+1} = e_t
r_{t+1} = (1 - beta_M) * r_t + beta_M * mismatch(c_t, x_t)
```

#### 在 `F` 中

```text
w_{t+1} = (1 - alpha_F) * w_t + alpha_F * Phi(c_t)
e_{t+1} = e_t + gamma_F * local_evidence(c_t)
r_{t+1} = (1 - beta_F) * r_t + beta_F * mismatch(c_t, x_t)
```

#### 在 `V` 中

```text
w_{t+1} = w_t
e_{t+1} = e_t + gamma_V * verify(c_t, x_t, ell_t)
r_{t+1} = (1 - beta_V) * r_t + beta_V * postverify_mismatch(c_t, x_t)
```

#### 在 `C` 中

```text
w_{t+1} = Psi(w_t, Phi(c_t))
e_{t+1} = e_t
r_{t+1} = r_t
```

注意：

- `C` 仍然不是正式 commit。
- 正式 commit 只由 `formal_commit_done` 和 `CommitLog` 共同确认。

### 8.2 制度状态更新

```text
ell_{t+1} = g(ell_t, m_t, E_t, K_t, log_t)
```

简化制度状态：

```text
ell_t = {
  commit_stage,
  persistence_class,
  pending_count,
  boundary_flags
}
```

#### `commit_stage`

```text
s_{t+1} =
  draft,     if m_t = M
  pending,   if m_t = F and s_t = draft
  verified,  if m_t = V and E_t = 1
  committed, if m_t = C and E_t = 1 and K_t = 1 and log_t = 1
  s_t,       otherwise
```

#### `persistence_class`

```text
p_{t+1} =
  p_t,          if s_{t+1} in {draft, pending}
  promote(p_t), if s_{t+1} = verified
  commitment,   if s_{t+1} = committed
  p_t,          otherwise
```

#### `pending_count`

```text
h_{t+1} =
  h_t
  + 1[m_t = F and s_t = draft]
  - 1[s_{t+1} in {verified, committed}]
```

#### `boundary_flags`

由显式规则更新：

- committed 后某些回滚权限收紧
- log 缺失禁止进入 committed
- human review required 时保持部分 gate 关闭

---

## 9. CommitLog 规格

所有正式跃迁都必须记录。

最小事件结构：

```text
CommitEvent {
  event_id
  timestamp
  from_mode
  to_mode
  from_stage
  to_stage
  candidate_id
  U
  N
  A
  P
  R
  E_t
  K_t
  evidence_snapshot_id
  verifier_snapshot_id
  proposal_origin      // model | human | tool | system
  commit_executed      // bool
}
```

不可省略字段：

- `from_stage`
- `to_stage`
- `R`
- `E_t`
- `K_t`
- `proposal_origin`
- `commit_executed`

硬约束：

```text
commit_executed = true
=> from_mode = C
=> E_t = 1
=> K_t = 1
=> log_t = 1
```

---

## 10. 三类来源分工

### 10.1 必须显式统计

这些不能交给大模型直接决定：

- `a1` 制度层级
- `p1` 依赖扇出
- `pending_commitments`
- `commit_stage`
- `verification_stage`
- `permission_state`
- `rollback_availability`
- `formal_commit_done`

### 10.2 适合小模型 head 预测

- `u1` conflict
- `u2` residual 映射
- `n1` goal-loss-if-ignore
- `n2` commitment pressure 映射
- `a2` current evidence anchor 聚合
- `p2` rollback cost 聚合

### 10.3 大模型只能 proposal

大模型可以建议：

- 新 feature 候选
- 新阈值建议
- 新聚合方式建议
- 局部规则补丁建议

大模型不能直接决定：

- `a1`
- `p1`
- `commit_stage`
- `formal_commit_done`
- `K_t`
- `log_ready`
- 阈值即时生效

---

## 11. 四个可证伪命题

### 命题 1：不能越权 commit

若 formal commit 只能从 `C` 且必须满足 `E_t = 1, K_t = 1, log_t = 1`，则不存在无证据、无权限、无日志的正式承认跃迁。

### 命题 2：不抖振

若 `R_t` 单步变化有界且阈值带有正宽迟滞带，则模式切换不会在一步内往返抖振，有限时间窗口内切换次数有上界。

### 命题 3：小扰动不改正式状态

若连续一段时间 `R_t < theta_C^+` 或 `E_t = 0`，则系统不可能进入 `committed`。

### 命题 4：持续高扰动可触发正式承认

若连续一段时间 `R_t >= theta_C^+`，且验证过程最终使 `E_t = 1`，同时 `K_t = 1`，则系统可达 `committed`。

---

## 12. 最小实验

### 实验 A：越权 commit 对抗

输入：

- 高 `R_t`
- 但 `E_t = 0` 或 `K_t = 0`

要求：

- 系统不能进入 `formal_commit_done = 1`

检验：

- 是否存在任何旁路 commit
- 是否出现“语义 commit”但制度未记账

### 实验 B：阈值噪声下的非抖振

在 `theta_F`、`theta_C` 附近加入有界噪声，比较：

- 单阈值 router
- 带迟滞治理核

指标：

- 切换次数
- 最小驻留时间
- 误触发 commit 次数

### 实验 C：小扰动流

持续注入小冲突、低锚定候选。

指标：

- `M/F/V/C` 访问分布
- 是否出现 `committed`
- pending 是否过度累积

### 实验 D：持续高扰动 + 补证据

先让 `R_t` 持续高，再逐步补 verifier 证据。

指标：

- `F -> V -> C -> committed` 路径是否出现
- 承认延迟
- commit precision

---

## 13. 失败条件

项目不以“性能没涨”判死，而以结构性失败判死。

### 失败条件 1

存在任何旁路 formal commit。

这意味着整个治理核制度层无效。

### 失败条件 2

迟滞后仍高频抖振。

说明 `R_t` 动力学或阈值设计不可用。

### 失败条件 3

小扰动持续触发 `committed`。

说明 `R_t` 不是改写资格比，只是重要性分数。

### 失败条件 4

持续高扰动 + 足够证据时仍不能 commit。

说明系统塌成“永久悬挂器”。

### 失败条件 5

`E_t` 可以轻易被伪一致性骗过。

说明证据门没有现实锚定意义。

---

## 14. 与现有工作的本质差异

### 相比 world model 路线

现有 world model 路线主要回答：

- 如何形成稳定 latent
- 如何从 residual/surprise 中得到异常信号

本项目不把 world model 当终点，而只把它当作 `u2` 的信号源之一。

本项目新增的是：

> 状态改写资格的制度化治理。

### 相比读写记忆控制

现有 read/write decoupling 工作主要控制：

- 何时读
- 何时写
- 写入如何强化或筛选

本项目更进一步，把写入和正式承认放进同一个受约束切换系统中。

重点不是 memory operation，而是：

> 状态是否有资格被制度化改写。

### 相比 memory design search

现有 design search 强调：

- 记忆结构与更新策略可以被元学习

本项目不先做开放搜索，而先固定治理骨架，研究：

> 在最小制度结构内，改写资格是否可以稳定估计并控制。

### 相比单一 reward / 普通 router

本项目不依赖单一 reward，也不把治理看成普通动作分类。

它的核心是：

- viability constraints
- constitutional constraints
- contextual optimization

先守边界，再优化上下文。

---

## 15. 第一阶段建议

不要一上来全做。

### Phase 0

先只实现：

- `commit_stage`
- `CommitLog`
- `a1`
- `p1`
- `E_t`
- `K_t`
- 最小 transition table

目标：

```text
证明“不能越权 commit”
```

### Phase 1

再接：

- `u2`
- `a2`
- `R_t`
- 迟滞阈值

目标：

```text
证明“非抖振”
```

### Phase 2

最后再接：

- `u1`
- `n1`
- `n2`
- `p2`

目标：

```text
开始验证“小扰动不 commit / 持续高扰动可 commit”
```

---

## 16. 与二变量动力学的关系

本工程规格保留 `U/N/A/P/R`，因为它适合工程接口。

但底层解释仍以二变量为主：

```text
D_t ≈ f(U_t, N_t)
S_t ≈ g(A_t, P_t)
R_t ≈ D_t / (S_t + epsilon)
```

因此：

- `R_t` 是边界统计
- `D_t/S_t` 是更底层动力学
- `CommitLog/E_t/K_t` 是正式承认边界

---

## 17. 最后一句收束

这份 v0 规格真正要造的，不是一个“会做很多动作的 agent”，而是一个：

> 把正式状态改写从普通推理中剥离出来，单独治理的最小内核。

