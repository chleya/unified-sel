# Rewrite Qualification Dynamics v0

Date: 2026-04-23
Status: pre-implementation theory spec
Scope: continuous structure beneath attention/governance actions

This note compresses the attentional governance idea below the action level.

The surface actions are:

```text
foreground, background, preempt, sustain, inhibit, acknowledge
```

But these are not primitive.
They are threshold effects of a deeper competition:

```text
order self-stability vs disturbance rewrite pressure
```

The target of this document is a computable minimal structure for that competition.

---

## 1. Root Question

Do not ask first:

```text
What action should the controller take?
```

Ask:

```text
How much pressure does this candidate exert against the current state order,
and how much self-stability does the current order still have?
```

Actions are downstream threshold crossings.

---

## 2. Two Continuous Variables

The minimum structure needs two continuous quantities, not one.

### 2.1 Disturbance Rewrite Pressure `D_t`

`D_t` measures how strongly a candidate content pushes against the currently maintained order.

Definition:

```text
D_t = effective pressure for changing current foreground allocation,
      control mode, or durable state.
```

It is not salience.
It is not risk.
It is not confidence.
It is the estimated cost of continuing to ignore or suppress the candidate.

Suggested range:

```text
D_t in [0, 1]
```

High `D_t` means:

- current order is becoming inconsistent if unchanged
- the candidate is goal-relevant
- the candidate has unresolved future consequences
- external or tool reality is pulling against the current state

### 2.2 Order Self-Stability `S_t`

`S_t` measures how strongly the current order should resist being changed.

Definition:

```text
S_t = effective stability of the currently maintained identity/order/mode.
```

It is not conservatism.
It is not refusal.
It is the estimated cost of allowing this candidate to restructure the system.

Suggested range:

```text
S_t in [0, 1]
```

High `S_t` means:

- current goal or commitment is still valid
- candidate evidence is weak, ambiguous, or unauthorized
- durable write would have high blast radius
- current mode is protecting an invariant or irreversible boundary

### 2.3 Why Not One Variable?

A single "rewrite pressure" score collapses two different cases:

| Case | `D_t` | `S_t` | Correct behavior |
|---|---:|---:|---|
| strong real drift against weak current order | high | low | switch / commit candidate after checks |
| strong anomaly against strong invariant | high | high | verify / defer / human handoff, not direct commit |
| weak noise against weak order | low | low | background or ignore |
| weak signal against strong order | low | high | inhibit |

One scalar cannot distinguish:

```text
high-pressure valid update
from
high-pressure attack on a protected boundary
```

Therefore v0 keeps two axes.

---

## 3. Candidate-Level State

Every candidate that may affect control or memory receives a dynamic record.

```python
@dataclass
class RewriteDynamics:
    candidate_id: str
    disturbance_pressure: float
    order_stability: float
    pressure_velocity: float
    stability_velocity: float
    pressure_evidence_count: int
    stability_evidence_count: int
    last_updated_at: int
    phase: RewritePhase
```

Minimum phase enum:

```python
class RewritePhase(Enum):
    SUPPRESSED = "suppressed"
    BACKGROUND = "background"
    FOREGROUND = "foreground"
    MODE_SWITCH = "mode_switch"
    COMMIT_REVIEW = "commit_review"
    ACKNOWLEDGED = "acknowledged"
    REJECTED = "rejected"
```

---

## 4. How To Estimate `D_t`

`D_t` should aggregate four pressure sources.

```python
@dataclass
class DisturbanceComponents:
    incompatibility: float
    goal_relevance: float
    consequence_suspense: float
    external_traction: float
```

### 4.1 `incompatibility`

How incompatible the candidate is with the current maintained order.

Examples:

- verifier disagreement
- memory contradiction
- prediction residual
- tool result contradicts plan

### 4.2 `goal_relevance`

How much the current goal depends on resolving this candidate.

Examples:

- blocker for current task
- needed for answer correctness
- affects route choice
- affects verifier result

### 4.3 `consequence_suspense`

How much future consequence remains unresolved if ignored.

Examples:

- pending commitment
- unfinished verification
- delayed external effect
- possible rollback obligation

### 4.4 `external_traction`

How much the candidate is anchored in external reality rather than internal speculation.

Examples:

- user approval or correction
- tool output
- environment feedback
- reproducible test result

### 4.5 v0 Aggregation

Use a conservative max-plus aggregation:

```python
def disturbance_pressure(c: DisturbanceComponents) -> float:
    primary = max(c.incompatibility, c.goal_relevance)
    secondary = 0.5 * max(c.consequence_suspense, c.external_traction)
    return clamp01(primary + secondary)
```

Reason:

- incompatibility and goal relevance can independently create immediate pressure
- consequence and external traction amplify pressure but should not alone force commit

---

## 5. How To Estimate `S_t`

`S_t` should aggregate four stabilizing sources.

```python
@dataclass
class StabilityComponents:
    continuity_value: float
    authority_barrier: float
    ambiguity_load: float
    perturbation_cost: float
```

### 5.1 `continuity_value`

How valuable it is to preserve the current order.

Examples:

- active goal still valid
- current plan is passing verifier
- existing memory remains predictive
- no regression evidence

### 5.2 `authority_barrier`

How much authority is missing for change.

Examples:

- no user approval
- source lacks permission
- system/policy boundary touched
- protected invariant involved

### 5.3 `ambiguity_load`

How unclear the proposed change is.

Examples:

- temporary exception vs durable rule unclear
- conflicting sources
- under-specified candidate
- low evidence count

### 5.4 `perturbation_cost`

How costly it would be to let this candidate restructure state.

Examples:

- large blast radius
- irreversible write
- many downstream memories affected
- high rollback cost

### 5.5 v0 Aggregation

Use max aggregation:

```python
def order_stability(c: StabilityComponents) -> float:
    return max(
        c.continuity_value,
        c.authority_barrier,
        c.ambiguity_load,
        c.perturbation_cost,
    )
```

Reason:

- one strong stabilizing factor is enough to block direct formalization
- high disturbance against high stability should route to verification/defer, not blind commit

---

## 6. Boundary Geometry

The dynamics are governed by the relation between `D_t` and `S_t`.

Define:

```text
margin_t = D_t - S_t
ratio_t = D_t / (S_t + epsilon)
```

Use both:

- `margin_t` captures absolute crossing pressure
- `ratio_t` captures relative destabilization

### 6.1 Phase Regions

| Region | Condition | Phase |
|---|---|---|
| stable suppression | `D_t < theta_bg` and `S_t >= D_t` | `SUPPRESSED` |
| weak background | `D_t >= theta_bg` and `margin_t < theta_fg` | `BACKGROUND` |
| foreground tension | `margin_t >= theta_fg` and `ratio_t < theta_switch` | `FOREGROUND` |
| mode switch | `ratio_t >= theta_switch` and `S_t < theta_protected` | `MODE_SWITCH` |
| protected conflict | `D_t high` and `S_t high` | `FOREGROUND` or `COMMIT_REVIEW` with verifier/handoff |
| formalization eligible | `ratio_t >= theta_commit` and hard constraints pass | `COMMIT_REVIEW` |
| acknowledged | commit gate passes | `ACKNOWLEDGED` |

The exact thresholds are empirical.
v0 should start with:

```text
theta_bg = 0.20
theta_fg = 0.15
theta_switch = 1.50
theta_commit = 2.00
theta_protected = 0.70
```

---

## 7. Time Dynamics

The core signal is not only level but accumulation.

### 7.1 Update Rule

```python
def update_dynamics(prev, d_new, s_new, alpha=0.3):
    D = (1 - alpha) * prev.disturbance_pressure + alpha * d_new
    S = (1 - alpha) * prev.order_stability + alpha * s_new
    vD = D - prev.disturbance_pressure
    vS = S - prev.order_stability
    return D, S, vD, vS
```

### 7.2 Persistent Drift vs Temporary Exception

Persistent drift:

```text
D_t stays elevated or rises across repeated evidence
S_t decays as ambiguity drops
```

Temporary exception:

```text
D_t spikes once
S_t remains high or recovers
```

This is the first empirical split the project must demonstrate.

### 7.3 Chattering Guard

Mode changes should require hysteresis:

```text
enter threshold > exit threshold
```

Example:

```text
enter foreground: margin_t >= 0.15
exit foreground: margin_t <= 0.05
```

Without hysteresis, the system is only a noisy router.

---

## 8. Relation To `R_t`

The older qualification ratio:

```text
R_t = (U_t * N_t) / (A_t * P_t + epsilon)
```

can be reinterpreted:

```text
D_t ~ f(U_t, N_t)
S_t ~ g(A_t, P_t)
R_t ~ D_t / (S_t + epsilon)
```

This is cleaner than treating `R_t` as the root variable.

`R_t` is a boundary statistic.
`D_t` and `S_t` are the underlying dynamics.

---

## 9. Minimum Experiments

### 9.1 Temporary Spike Test

Setup:

- one high disturbance observation
- no repeated evidence
- high ambiguity and authority barrier

Expected:

```text
D_t spikes
S_t stays high
phase does not reach ACKNOWLEDGED
```

Pass:

```text
temporary false acknowledgement rate <= 0.05
```

### 9.2 Persistent Drift Test

Setup:

- repeated compatible evidence against current order
- ambiguity decreases over time
- external traction remains nonzero

Expected:

```text
D_t remains high
S_t decreases
phase reaches COMMIT_REVIEW or ACKNOWLEDGED after repeated evidence
```

Pass:

```text
persistent drift commit-review recall >= 0.8
```

### 9.3 Protected Boundary Test

Setup:

- high disturbance candidate touching protected authority boundary
- high authority barrier or perturbation cost

Expected:

```text
D_t high
S_t high
phase routes to FOREGROUND/DEFER/HANDOFF, not direct ACKNOWLEDGED
```

Pass:

```text
protected direct acknowledgement count == 0
```

### 9.4 Chattering Test

Setup:

- small perturbations around a phase boundary

Expected:

```text
hysteresis prevents frequent phase flips
```

Pass:

```text
phase flip rate < 0.2 under bounded irrelevant perturbation
```

---

## 10. Implementation Target

If implemented, this file should precede action-level attention tests.

Suggested files:

```text
core/rewrite_dynamics.py
tests/test_rewrite_dynamics.py
experiments/capability/rewrite_dynamics_sanity.py
```

Do not add LLM calls in v0.

Expected artifact:

```json
{
  "num_cases": 0,
  "temporary_false_acknowledgement_rate": 0.0,
  "persistent_drift_commit_review_recall": 0.0,
  "protected_direct_acknowledgement_count": 0,
  "phase_flip_rate": 0.0,
  "mean_peak_disturbance": 0.0,
  "mean_final_stability": 0.0
}
```

---

## 11. Narrow Claim

This spec claims only:

```text
The apparent actions of attention/governance can be modeled as threshold
effects over two continuous variables: disturbance rewrite pressure and
order self-stability.
```

It does not claim:

- biological realism
- mathematical optimality
- that two variables are sufficient for all agents
- that formal acknowledgement is safe without hard constraints
- that persistent drift should always be committed

The first thing to earn is simpler:

```text
temporary spikes, persistent drift, and protected-boundary conflicts occupy
different regions of the D/S phase space.
```

