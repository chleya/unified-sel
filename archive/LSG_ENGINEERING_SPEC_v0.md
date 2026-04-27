# LSG Engineering Spec v0

Date: 2026-04-23
Status: executable project kickoff spec
Scope: minimal learned state governance kernel

---

## 0. Goal

This project does not study "stronger agent behavior".
It studies a state governance mechanism:

> The system should not react directly from an action list. It should first estimate whether a candidate content has qualification to rewrite the current formal state.

The qualification is controlled by a continuous ratio:

```text
R_t = (U_t * N_t) / (A_t * P_t + epsilon)
```

where:

- `U_t`: incompatibility between candidate content and current formal state
- `N_t`: cost of ignoring the candidate content
- `A_t`: acknowledgement depth of the current formal state
- `P_t`: propagation load of the current formal state
- `epsilon > 0`: numerical stability term

The engineering goal is not to build a broad continuous cognitive system.
The engineering goal is to implement a minimal learned state governance kernel and verify these local properties:

1. no unauthorized commit
2. no chattering
3. small disturbances do not rewrite formal state
4. sustained high disturbance can trigger formal acknowledgement when evidence is sufficient

---

## 1. Scope And Non-Goals

### 1.1 In Scope

This phase implements only:

- explicit state machine
- explicit `CommitLog`
- eight proxy interfaces
- `R_t` computation and hysteretic switching
- two minimal proposition tests:
  - Proposition 1: no unauthorized commit
  - Proposition 2: no chattering

### 1.2 Out Of Scope

This phase does not implement:

- general continuous cognition
- fully automatic memory design search
- endogenous world model training
- large-model direct mutation of formal facts
- complex multi-agent collaboration

---

## 2. Minimal System Objects

The system runs in discrete time:

```text
t = 0, 1, 2, ...
```

Total state:

```text
z_t = (x_t, ell_t, m_t, q_t)
```

### 2.1 Working State `x_t`

Current work-layer content:

```text
x_t = {
  working_summary,       // current foreground work summary
  candidate_summary,     // current candidate content summary
  evidence_buffer,       // current evidence buffer
  mismatch_buffer        // current conflict/residual buffer
}
```

### 2.2 Institutional State `ell_t`

Formal layer and governance boundary:

```text
ell_t = {
  persistence_class,       // draft/episodic/durable/rule/commitment
  commit_stage,            // draft/pending/verified/committed
  verification_stage,      // none/in_progress/passed/failed
  pending_commitments,     // unresolved commitments
  permission_state,        // permission matrix snapshot
  rollback_availability,   // whether rollback is possible
  log_ready                // whether logging conditions are satisfied
}
```

### 2.3 Mode State `m_t`

Minimal mode set:

```text
m_t in {M, F, V, C}
```

Modes:

- `M`: maintain
- `F`: foreground
- `V`: verify
- `C`: commit-candidate

### 2.4 Governance Scalars `q_t`

```text
q_t = (U_t, N_t, A_t, P_t, R_t)
```

---

## 3. Eight Minimal Proxy Interfaces

### 3.1 `u1`: Semantic / Factual Conflict

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

Sources:

- small model `conflict_head`
- explicit rule/fact/tool conflict checks

### 3.2 `u2`: Predictive Mismatch

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

Sources:

- explicit residual statistics
- small model `residual_head`

### 3.3 `n1`: Goal Loss If Ignored

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

Source:

- small model `goal_loss_head`

### 3.4 `n2`: Commitment Carry Cost If Ignored

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

Sources:

- explicit pending/deadline statistics
- small model `commitment_pressure_head`

### 3.5 `a1`: Institutional Level

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

Source:

- explicit mapping only
- must not be replaced by learned output

Initial mapping:

```text
draft      -> 0.10
pending    -> 0.25
verified   -> 0.55
durable    -> 0.65
rule       -> 0.80
committed  -> 1.00
```

### 3.6 `a2`: Evidence Anchor Strength

Input:

```text
- evidence_count
- independent_source_count
- verifier_pass_rate
- tool_confirmation_rate
- source_confidence_stats
```

Output:

```text
- anchor_score in [0, 1]
- weak_spot_flag
```

Sources:

- explicit statistics
- small model `evidence_anchor_head`

### 3.7 `p1`: Dependency Fanout

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

Source:

- explicit graph statistics only
- must not be directly decided by a large model

### 3.8 `p2`: Rollback Cost

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

Sources:

- explicit cost statistics
- small model `rollback_cost_head`

---

## 4. Governance Scalar Computation

Define:

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

Then:

```text
R_t_raw = (U_t * N_t) / (A_t * P_t + epsilon)
```

For engineering stability, switching does not use `R_t_raw` directly.
It uses a smoothed ratio:

```text
Rbar_{t+1} = (1 - alpha) * Rbar_t + alpha * R_t_raw
```

Then slope clipping:

```text
R_{t+1} = clip(Rbar_{t+1}, R_t - Delta_max, R_t + Delta_max)
```

where:

- `alpha` controls smoothing strength
- `Delta_max` controls the maximum single-step ratio change

---

## 5. Thresholds And Hysteresis

Define two main thresholds:

```text
0 < theta_F < theta_C
```

and hysteresis bands:

```text
theta_F^- < theta_F^+ < theta_C^- < theta_C^+
```

Initial implementation should tune only four numbers:

```text
theta_F^-
theta_F^+
theta_C^-
theta_C^+
```

Required ordering:

```text
theta_F^- < theta_F^+ < theta_C^- < theta_C^+
```

---

## 6. Evidence Gate And Constitutional Gate

### 6.1 Evidence Gate `E_t`

```text
E_t = 1 iff
- anchor_score >= anchor_threshold
- evidence_count >= min_evidence_count
- verifier_pass_rate >= min_verifier_pass
- no hard contradiction remains
```

### 6.2 Constitutional Gate `K_t`

```text
K_t = 1 iff
- permission_state allows requested transition
- irreversible action requires verified state
- log_ready == true
- no bypass of constitutional constraints
```

Neither gate is a free head output.
Both are explicit conditions plus controlled estimator aggregation.

---

## 7. Mode Switching Law

### 7.1 From Maintain

```text
m_{t+1} =
  F, if R_t >= theta_F^+
  M, if R_t <  theta_F^+
```

### 7.2 From Foreground

```text
m_{t+1} =
  M, if R_t <= theta_F^-
  V, if R_t >= theta_C^- and E_t = 0
  C, if R_t >= theta_C^+ and E_t = 1 and K_t = 1
  F, otherwise
```

### 7.3 From Verify

```text
m_{t+1} =
  M, if R_t <= theta_F^-
  C, if R_t >= theta_C^+ and E_t = 1 and K_t = 1
  V, otherwise
```

### 7.4 From Commit-Candidate

```text
m_{t+1} =
  M, if formal_commit_done = 1
  V, if E_t = 0 or K_t = 0
  F, if R_t <= theta_C^-
  C, otherwise
```

---

## 8. Minimal State Update Equations

### 8.1 Working State Update

```text
x_{t+1} = f_{m_t}(x_t, c_t)
```

Definitions:

- `Phi(c_t)`: candidate content encoding
- `mismatch(c_t, x_t)`: conflict/residual function between candidate and current state
- `verify(c_t, x_t, ell_t)`: verification-produced evidence

For compactness, write the work state as:

```text
x_t = (w_t, e_t, r_t)
```

where:

- `w_t`: working summary
- `e_t`: evidence buffer
- `r_t`: mismatch buffer

#### In `M`

```text
w_{t+1} = (1 - alpha_M) * w_t + alpha_M * Phi(c_t)
e_{t+1} = e_t
r_{t+1} = (1 - beta_M) * r_t + beta_M * mismatch(c_t, x_t)
```

#### In `F`

```text
w_{t+1} = (1 - alpha_F) * w_t + alpha_F * Phi(c_t)
e_{t+1} = e_t + gamma_F * local_evidence(c_t)
r_{t+1} = (1 - beta_F) * r_t + beta_F * mismatch(c_t, x_t)
```

#### In `V`

```text
w_{t+1} = w_t
e_{t+1} = e_t + gamma_V * verify(c_t, x_t, ell_t)
r_{t+1} = (1 - beta_V) * r_t + beta_V * postverify_mismatch(c_t, x_t)
```

#### In `C`

```text
w_{t+1} = Psi(w_t, Phi(c_t))
e_{t+1} = e_t
r_{t+1} = r_t
```

### 8.2 Institutional State Update

```text
ell_{t+1} = g(ell_t, m_t, E_t, K_t, log_t)
```

Reduced institutional state:

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
  p_t,        if s_{t+1} in {draft, pending}
  promote(p_t), if s_{t+1} = verified
  commitment, if s_{t+1} = committed
  p_t,        otherwise
```

#### Pending Count

```text
h_{t+1} =
  h_t
  + 1[m_t = F and s_t = draft]
  - 1[s_{t+1} in {verified, committed}]
```

#### Boundary Flags

Updated by explicit rules:

- committed state may tighten rollback permissions
- missing log forbids `committed`
- human review required keeps selected gates closed

---

## 9. CommitLog Spec

Every formal transition must be recorded.

Minimum event structure:

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

Non-optional fields:

- `from_stage`
- `to_stage`
- `R`
- `E_t`
- `K_t`
- `proposal_origin`
- `commit_executed`

---

## 10. Three Source Classes

### 10.1 Must Be Explicit Statistics

These must not be directly decided by a large model:

- `a1` institutional level
- `p1` dependency fanout
- `pending_commitments`
- `commit_stage`
- `verification_stage`
- `permission_state`
- `rollback_availability`

### 10.2 Suitable For Small-Model Heads

- `u1` conflict
- `u2` residual mapping
- `n1` goal loss if ignored
- `n2` commitment pressure mapping
- `a2` evidence anchor aggregation
- `p2` rollback cost aggregation

### 10.3 Large Model Can Only Propose

Allowed proposal areas:

- new feature candidates
- new threshold suggestions
- new aggregation suggestions
- local rule patch suggestions

Large model must not directly decide:

- `a1`
- `p1`
- `commit_stage`
- `formal_commit_done`

---

## 11. Four Falsifiable Propositions

### Proposition 1: No Unauthorized Commit

If formal commit can only occur from `C` and must satisfy `E_t = 1`, `K_t = 1`, and `log_t = 1`, then no formal acknowledgement transition can occur without evidence, permission, and log.

### Proposition 2: No Chattering

If `R_t` has bounded single-step change and thresholds have positive-width hysteresis bands, then mode switching cannot oscillate back and forth in one step, and switching count is bounded in any finite time window.

### Proposition 3: Small Disturbances Do Not Rewrite Formal State

If `R_t < theta_C^+` for a continuous interval, or `E_t = 0`, then the system cannot enter `committed`.

### Proposition 4: Sustained High Disturbance Can Trigger Formal Acknowledgement

If `R_t >= theta_C^+` for a continuous interval, verification eventually makes `E_t = 1`, and `K_t = 1`, then `committed` is reachable.

---

## 12. Minimal Experiments

### Experiment A: Unauthorized Commit Adversary

Input:

- high `R_t`
- but `E_t = 0` or `K_t = 0`

Requirement:

- system must not enter `formal_commit_done = 1`

Checks:

- any bypass commit
- any semantic commit without institutional logging

### Experiment B: Non-Chattering Under Threshold Noise

Add bounded noise around `theta_F` and `theta_C`.

Compare:

- single-threshold router
- hysteretic governance kernel

Metrics:

- switch count
- minimum dwell time
- false commit count

### Experiment C: Small Disturbance Stream

Continuously inject small-conflict, low-anchor candidates.

Metrics:

- `M/F/V/C` visit distribution
- whether `committed` appears
- whether pending over-accumulates

### Experiment D: Sustained High Disturbance + Evidence Accumulation

Keep `R_t` high, then gradually add verifier evidence.

Metrics:

- whether `F -> V -> C -> committed` appears
- acknowledgement latency
- commit precision

---

## 13. Failure Conditions

This project is not killed by "benchmark performance did not improve".
It is killed by structural failure.

### Failure 1

Any bypass formal commit exists.
This means the institutional layer is invalid.

### Failure 2

High-frequency chattering persists after hysteresis.
This means `R_t` dynamics or thresholds are unusable.

### Failure 3

Small disturbance streams repeatedly trigger `committed`.
This means `R_t` is just an importance score, not a rewrite qualification ratio.

### Failure 4

Sustained high disturbance plus sufficient evidence still cannot commit.
This means the system collapsed into a permanent suspender.

### Failure 5

`E_t` is easily fooled by pseudo-consistency.
This means the evidence gate has no real anchoring value.

---

## 14. Essential Difference From Existing Work

### Compared With World Model Routes

Existing world model routes mainly answer:

- how to form stable latent state
- how to derive anomaly signals from residual or surprise

This project does not treat a world model as the endpoint.
It treats world-model residual as one possible source for `u2`.

The additional object is:

> institutional governance of state rewrite qualification

### Compared With Read/Write Memory Control

Read/write decoupling work mainly controls:

- when to read
- when to write
- how writes are reinforced or filtered

This project goes further by placing writing and formal acknowledgement inside the same constrained switching system.
The focus is not memory operation.
The focus is:

> whether state has qualification to be institutionally rewritten

### Compared With Memory Design Search

Memory design search emphasizes:

- memory structures and update policies can be meta-learned

This project does not begin with open-ended search.
It first fixes a governance skeleton and studies:

> whether rewrite qualification can be estimated and controlled stably inside a minimal institutional structure

### Compared With Single Reward Or Ordinary Router

This project does not rely on a single reward and does not treat governance as ordinary action classification.

It uses:

- viability constraints
- constitutional constraints
- contextual optimization

First preserve boundaries, then optimize context.

---

## 15. Phase Plan

Do not implement everything at once.

### Phase 0

Implement only:

- `commit_stage`
- `CommitLog`
- `a1`
- `p1`
- `E_t`
- `K_t`
- minimal transition table

Goal:

```text
prove no unauthorized commit
```

### Phase 1

Add:

- `u2`
- `a2`
- `R_t`
- hysteresis thresholds

Goal:

```text
prove non-chattering
```

### Phase 2

Add:

- `u1`
- `n1`
- `n2`
- `p2`

Goal:

```text
start testing small disturbance does not commit / sustained high disturbance can commit
```

---

## 16. Closing Definition

This v0 spec does not build an agent that can do many actions.
It builds:

> a minimal kernel that separates formal state rewrite from ordinary inference and governs it independently.

