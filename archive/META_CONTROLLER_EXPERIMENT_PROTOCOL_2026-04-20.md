# Meta-Controller Experiment Protocol - 2026-04-20

## Status

This document is a draft experimental protocol, not a validated project result.

It should be treated as future work adjacent to the current Capability Router and TopoMem-OBD tracks. It must not be used to claim that Unified-SEL has already learned a unified cognitive-control mechanism.

## Core Claim To Test

A learned unification mechanism is not a hand-written global controller. It is a meta-policy that learns, over long-horizon interaction, how to assign temporary system control to the right subsystem under uncertainty, cost, memory, and stability constraints.

The target claim is:

> A learnable meta-controller can form a stable but revisable control law for when to continue habit, when to deliberate, when to read memory, when to write memory, and when to broadcast a state update.

The experiment succeeds only if this learned control law remains causally useful under ablation, replacement, randomization, and transfer tests.

## Experimental Question

Can a meta-controller trained in partially observable, regime-shifting environments reduce:

- arbitration regret
- switch latency
- unnecessary memory access
- unnecessary deliberation cost
- long-horizon state drift

compared with fixed rules, random arbitration, and signal-ablated variants?

## Minimal System

The experiment should start with five components.

| Component | Role | Notes |
|---|---|---|
| Perception / Predictor | Predict next observation, reward, and regime stability | Produces surprise and prediction error |
| Habit Policy | Cheap default policy for stable regimes | Fast, low compute, brittle under shifts |
| Planner | Expensive deliberative policy | Stronger under shifts, high compute cost |
| Episodic Memory | Stores sparse event facts and regime-relevant cases | Read/write controlled by meta-controller |
| Meta-Controller | Learns arbitration and memory-control policy | The experimental object |

The shared workspace should be deliberately small. It should not contain full history.

## Shared Workspace State

At each step, the system exposes a bounded state vector `g_t`.

Recommended fields:

| Field | Meaning | Source |
|---|---|---|
| `task_progress` | Distance to current objective or subgoal completion | Environment / evaluator |
| `surprise` | Prediction error under current predictor | Predictor |
| `uncertainty` | Confidence spread or entropy over next action/outcome | Predictor / policy |
| `control_cost_estimate` | Expected cost of planner, memory read, and switch | Cost model |
| `memory_relevance` | Estimated value of memory retrieval | Memory index / learned scorer |
| `conflict_score` | Disagreement among habit, planner, predictor, and memory | Policy comparison |
| `invariant_violation` | Drift from long-term constraints or identity/state invariants | State auditor |
| `recent_failure_source` | Encoded source of recent failure | Evaluator / diagnostics |

This state is the compression target. If `g_t` is unstable, non-comparable across time, or unbounded, the meta-controller cannot be meaningfully evaluated.

## Meta-Controller Actions

At each decision point, the meta-controller outputs a macro-action tuple:

```text
a_t = (
  dominant_module,
  deliberation_precision,
  memory_read_gate,
  memory_write_gate,
  broadcast_gate
)
```

Recommended action values:

| Action | Values |
|---|---|
| `dominant_module` | `habit`, `planner`, `memory_assisted_planner`, `recover`, `hold` |
| `deliberation_precision` | Continuous `[0, 1]` or discrete `{low, medium, high}` |
| `memory_read_gate` | `none`, `top_k`, `targeted_case`, `state_summary` |
| `memory_write_gate` | `none`, `working_only`, `episodic`, `schema_update` |
| `broadcast_gate` | `none`, `workspace_update`, `interrupt`, `regime_shift_alert` |

The first implementation can use discrete actions to simplify attribution and regret calculation.

## Environment Design

Use a partially observable long-horizon environment with explicit regime shifts.

The environment must force all four behaviors:

- habit formation in stable phases
- de-habitualization after unexpected shifts
- selective memory retrieval for sparse historical facts
- long-term constraint preservation

### Episode Structure

Each episode contains `T` steps split into latent regimes.

```text
episode:
  regime_1: stable mapping, habit works
  regime_2: shifted mapping, old habit fails
  regime_3: memory-dependent phase, current observation is insufficient
  regime_4: long-term constraint phase, short-term reward can harm later success
```

The agent observes only partial state. Regime identity is hidden.

### Minimal Environment Variables

| Variable | Description |
|---|---|
| `latent_regime` | Hidden rule set controlling transition and reward |
| `visible_observation` | Partial cue available to all policies |
| `memory_fact_id` | Sparse historical fact required later |
| `constraint_budget` | Long-term resource or invariant that can be damaged |
| `shift_schedule` | When and how rules change |
| `compute_budget` | Cost pressure for planner and memory access |

### Example Task Template

Use a grid, symbolic workflow, or text-based POMDP where:

- In stable phases, a cheap habit policy reaches local goals.
- After a hidden rule switch, the same habit policy becomes systematically wrong.
- A key instruction appears early, disappears, and is needed later.
- The planner can infer the new rule but costs more.
- Reading memory can recover the hidden instruction but also costs more.
- A tempting short-term action increases immediate reward but violates a long-term invariant.

The exact domain can be synthetic at first, but it must not leak regime labels directly into observations.

## Reward Function

Use a composite objective:

```text
J =
  R_task
  - lambda_compute * C_compute
  - lambda_switch * C_switch
  - lambda_read * C_memory_read
  - lambda_write * C_memory_write
  - lambda_drift * D_drift
  + lambda_recovery * R_recovery
```

Recommended components:

| Term | Meaning |
|---|---|
| `R_task` | Long-horizon task success, not single-step accuracy |
| `C_compute` | Planner and simulation cost |
| `C_switch` | Cost for changing dominant module |
| `C_memory_read` | Cost for retrieval calls |
| `C_memory_write` | Cost for durable memory writes |
| `D_drift` | Divergence of bounded state from verified task/invariant state |
| `R_recovery` | Reward for rapid restoration after surprise or regime shift |

Important: cost numbers must be labeled as assumed cost models unless measured from real runtime or API cost.

## Training Plan

### Stage A - State Compression

Train or fit the bounded workspace state before training the full controller.

Targets:

- predict near-term failure
- detect prediction error and regime instability
- estimate memory relevance
- estimate planner value
- summarize unresolved conflicts

Acceptance criteria:

- `g_t` has fixed size
- `g_t` is comparable across episodes
- `g_t` predicts failure better than raw superficial features
- removing history beyond `g_t` does not destroy basic arbitration signal

### Stage B - Switch Learning

Train the meta-controller as a contextual bandit or top-level HRL policy.

Candidate macro-actions:

- continue habit
- call planner
- read memory then plan
- broadcast interrupt
- write/update memory
- enter recovery mode

The first implementation should log all candidate action values when possible, so arbitration regret can be estimated.

### Stage C - Writeback Learning

Train selective memory write policy after switch behavior is stable.

Write policy should answer:

- Did this event causally contribute to success or recovery?
- Is this content episode-specific or reusable?
- Does this update reduce future uncertainty?
- Does this update increase drift or contradiction risk?

Do not write every observation. The experiment should penalize uncontrolled memory growth and false durable writes.

## Baselines

Use at least these baselines.

| Baseline | Description | Purpose |
|---|---|---|
| Habit only | Never plans, never reads memory | Lower-bound cheap policy |
| Planner always | Always deliberates | Tests cost-aware advantage |
| Memory always | Always retrieves before acting | Tests selective retrieval |
| Fixed priority rules | Hand-written uncertainty/surprise/cost thresholds | Tests whether learning is necessary |
| Random arbitration | Same modules, randomized dominant module | Tests causal value of arbitration |
| Oracle arbitration | Uses hidden labels or hindsight best action | Upper bound only, never a deployable claim |

Oracle results must be labeled as upper bounds, not system performance.

## Required Metrics

### 1. Switch Latency

Number of steps from true regime shift to effective dominant-module change.

```text
switch_latency = t_first_effective_switch - t_regime_shift
```

Measure separately for:

- habit to planner
- habit to memory-assisted planner
- planner back to habit

### 2. Arbitration Regret

Difference between chosen macro-action value and hindsight best macro-action value.

```text
regret_t = V(best_macro_action_t) - V(chosen_macro_action_t)
```

Report cumulative regret and post-shift regret.

### 3. Memory Read Precision / Recall

Treat memory reads as retrieval decisions.

```text
read_precision = useful_reads / total_reads
read_recall = useful_reads / required_memory_events
```

Also report false read rate in stable phases.

### 4. Memory Write Precision

Treat durable writes as update decisions.

```text
write_precision = future_useful_writes / total_durable_writes
```

Report memory growth per episode and contradiction rate.

### 5. Drift Under Horizon

Measure divergence between bounded internal state and verified environment/invariant state as horizon increases.

```text
drift_h = distance(g_h, audited_state_h)
```

Report slope of drift versus episode length.

### 6. Recovery After Surprise

Performance recovery after regime shift.

```text
recovery_k = performance_at_shift_plus_k / pre_shift_performance
```

Report area under recovery curve for fixed windows.

### 7. Cost-Adjusted Success

Task success after subtracting compute, memory, and switch costs.

This is secondary to arbitration metrics. It should not be the only headline.

## Falsification Tests

### Test 1 - Fixed Rule Replacement

Replace the learned meta-controller with hand-written rules:

```text
if surprise > threshold: call planner
elif uncertainty > threshold: read memory
elif cost_budget_low: use habit
else: continue current module
```

Failure condition:

- If fixed rules match learned controller under transfer, the experiment has not shown a learned unification mechanism.

### Test 2 - Randomized Dominance

Keep all modules intact but randomize or delay the dominant-module decision.

Failure condition:

- If performance barely degrades, module arbitration is not causally central.

### Test 3 - Signal Masking

Train or evaluate variants with one control signal removed:

- no uncertainty
- no surprise
- no cost
- no memory relevance
- no invariant violation
- no conflict score

Expected degradation:

| Removed Signal | Expected Failure |
|---|---|
| uncertainty | memory reads become poorly timed |
| surprise | slow de-habitualization after shifts |
| cost | overuse of planner and memory |
| memory relevance | low read precision |
| invariant violation | long-horizon drift increases |
| conflict score | delayed switching under policy disagreement |

If removing a signal has no measurable effect, either the signal is redundant or it is not being used.

### Test 4 - Cross-Environment Transfer

Train on one family of shifts and test on held-out families:

- different shift frequency
- different memory horizon
- different planner cost
- different observation noise
- different long-term constraint

Failure condition:

- If the controller only works in the original environment, it learned a script rather than a transferable control law.

## Minimum Experimental Matrix

| Variant | Train Env | Test Env | Controller | Signals | Purpose |
|---|---|---|---|---|---|
| V0 | A | A | habit only | none | cheap lower bound |
| V1 | A | A | planner always | none | expensive upper-ish baseline |
| V2 | A | A | fixed rules | all | rule baseline |
| V3 | A | A | learned | all | in-distribution result |
| V4 | A | A | learned | no uncertainty | signal ablation |
| V5 | A | A | learned | no surprise | signal ablation |
| V6 | A | A | learned | no cost | signal ablation |
| V7 | A | A | learned | no invariant | signal ablation |
| V8 | A | B | fixed rules | all | transfer baseline |
| V9 | A | B | learned | all | transfer test |
| V10 | A | B | random arbitration | all | causal arbitration test |
| V11 | A | B | oracle arbitration | hidden labels | upper bound only |

Run at least 20 seeds for the synthetic version. Report confidence intervals, not only mean scores.

## Logging Schema

Every step should log:

```json
{
  "episode_id": "string",
  "seed": 0,
  "t": 0,
  "latent_regime": "hidden_in_train_logged_for_eval",
  "observation_id": "string",
  "g_t": {
    "task_progress": 0.0,
    "surprise": 0.0,
    "uncertainty": 0.0,
    "control_cost_estimate": 0.0,
    "memory_relevance": 0.0,
    "conflict_score": 0.0,
    "invariant_violation": 0.0
  },
  "meta_action": {
    "dominant_module": "habit",
    "deliberation_precision": "low",
    "memory_read_gate": "none",
    "memory_write_gate": "none",
    "broadcast_gate": "none"
  },
  "module_outputs": {
    "habit_action": "string",
    "planner_action": "string",
    "memory_result_ids": []
  },
  "costs": {
    "compute": 0.0,
    "switch": 0.0,
    "memory_read": 0.0,
    "memory_write": 0.0
  },
  "reward": 0.0,
  "task_success_delta": 0.0,
  "drift_score": 0.0,
  "regime_shift_event": false
}
```

For training, hidden regime labels must not be exposed to the controller. They may be logged for evaluation and oracle upper-bound calculations.

## Acceptance Criteria

The learned meta-controller is worth treating as a candidate unified-control mechanism only if all conditions hold:

1. It beats fixed rules on arbitration regret under held-out environments.
2. It has lower switch latency than fixed rules after unexpected shifts.
3. It uses less compute than planner-always at comparable task success.
4. It has higher memory read precision than memory-always without losing required recall.
5. It reduces long-horizon drift compared with variants lacking invariant signals.
6. Randomizing dominant-module decisions causes significant degradation.
7. Signal masking produces interpretable, localized failures.
8. Transfer performance remains above fixed-rule and random-arbitration baselines.

If these do not hold, the correct conclusion is negative:

> The system has modules and routing behavior, but has not shown a learned unified control law.

## First Implementation Slice

Build the smallest version in this order:

1. Implement a symbolic POMDP with hidden regime shifts, memory-dependent delayed facts, and explicit compute cost.
2. Implement habit, planner, episodic memory, and fixed-rule controller.
3. Add full per-step logging before adding learning.
4. Compute switch latency, read precision, cost-adjusted success, and recovery metrics.
5. Add learned contextual-bandit controller over discrete macro-actions.
6. Add fixed-rule replacement and randomized dominance tests.
7. Add signal masking.
8. Add held-out shift schedules and held-out cost structures.

Do not start with neural memory design search. That belongs after the minimal arbitration result is established.

## Relation To Current Repository Tracks

This protocol should stay separate from the current validated tracks:

- Capability Router: practical accept / verify / escalate routing toolkit.
- TopoMem-OBD: deployment health and drift monitoring.
- Boundary-local amplification: paper track with current empirical support.

This protocol is a future mechanism experiment. It can reuse benchmark infrastructure and logging conventions, but it should not inherit claims from earlier Unified-SEL mechanism experiments.

## One-Line Summary

The experiment does not ask whether the system can call many modules. It asks whether a learned meta-controller forms a stable, transferable, and causally necessary policy for when to switch, who should dominate, what should enter shared state, and what should be written back.
