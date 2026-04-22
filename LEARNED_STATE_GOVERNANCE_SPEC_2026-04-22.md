# Learned State Governance Spec

Date: 2026-04-22
Spec Status: Design Draft / Not Implementation-Ready

This document turns the current "existence model" discussion into an engineering design frame that can be mapped onto `unified-sel`, `cognitive-execution-engine`, and related F-drive projects.

Core principle:

> Define the viable state space first, define authority boundaries second, allow policy learning last.

In practical terms:

1. Hard constraints keep the system inside legal and auditable boundaries.
2. Learned governance chooses authority, write, revision, and commit actions inside those boundaries.
3. Phase-diagram and threshold scans identify useful variables and transition regions; they do not replace governance.

This is not a code implementation ticket. The only immediate implementation track remains `unified-sel` P0 `PredictiveHealthMonitor` as described in `LEWM_INTEGRATION_SPEC_2026-04-22.md`.

---

## 0. System Objects

### SelfState

The system's computable state of itself.

It records capability, memory, control, uncertainty, commitment, resource, and invariant state in a structured form that can be audited and replayed.

### GovernancePolicy

The policy layer that decides, within hard constraints:

- who controls the next step
- whether to retrieve memory
- whether to verify
- whether to write
- whether to defer
- whether to commit or hand off to a human

### CommitLog

The append-only record of which internal changes were officially accepted and which remained provisional.

CommitLog is part of the governance boundary, not a learned memory bucket.

---

## 1. Three-Layer Design

### Layer 1: Hard Constraint Layer

This layer is not "intelligence". It is constitution plus guard.

It answers only one question:

> What must never happen?

#### State Variables

Minimum reliable state:

| Variable | Meaning |
|---|---|
| `permission_state` | which module may request which action or tool |
| `reversibility_state` | whether the current action or commit can be rolled back |
| `evidence_state` | whether minimum evidence exists for the requested state promotion |
| `risk_state` | tool, external side-effect, and commitment risk |
| `boundary_state` | whether the action touches identity, permission, or real-world commit boundaries |

#### Learnable Parameters

None by default.

Auxiliary estimators may be learned, but estimator outputs cannot override hard constraints.

Example:

- A risk scorer can be learned.
- The rule "high-risk irreversible commits require verification" cannot be learned away.

#### Non-Negotiable Constraints

Viability constraints:

- no unauthorized durable / rule / commitment writes
- no direct commit in high-risk irreversible conditions
- no promotion to anchored fact or commitment without minimum evidence
- no `GovernancePolicy` bypass around `CommitLog`

Constitutional constraints:

- every durable-or-higher write must include `evidence`, `source`, and `revision_policy`
- every real-world commit must include `why_committed`
- every irreversible action must be auditable
- every policy-gate decision must be replayable

#### Optimization Objective

This layer does not maximize reward. It minimizes illegal trajectories:

- zero illegal state transitions
- zero unauthorized writes
- minimal commit precheck misses
- maximal audit and rollback availability

#### Minimal Falsification Tests

H1 bypass test:

Construct a path where a planner tries to write durable / commitment state without evidence. If the system allows it, the hard constraint layer fails.

H2 irreversible miscommit test:

Construct a high commitment-risk scenario where verifier approval is missing. If the system commits externally, the hard constraint layer fails.

H3 missing log test:

Trigger a legal commit while suppressing `CommitLog` writing. If the system accepts the commit without a log event, the hard constraint layer fails.

---

### Layer 2: Learned Governance Layer

This layer is the governance core.

It does not decide what is legal. It decides what to do inside the legal region:

- who should control the next step
- whether memory should be read
- whether the result should be verified
- whether state should be written
- whether commit should be delayed or handed off

#### SelfState Fields

Recommended groups:

| Group | Fields |
|---|---|
| `capability_state` | task-family capability estimate, recent failure modes, recent verifier results, planner success estimate |
| `memory_state` | working / episodic / durable / rule / commitment occupancy, memory conflict level, retrieval sufficiency, write pressure |
| `control_state` | current owner: habit / planner / verifier / memory / human / policy; deliberation level; recent switch history |
| `uncertainty_state` | self-confidence, source-confidence, prediction residual, semantic drift, verifier disagreement |
| `commitment_state` | anchored facts, pending facts, reversible hypotheses, irreversible commits, pending commitments |
| `resource_state` | time budget, cost budget, tool risk, rollback availability |
| `identity_invariants` | permission boundaries, non-destructible goal constraints, long-term consistency boundaries |

#### Learnable Policy Families

AuthoritySwitchingLaw:

Inputs:

- prediction residual
- self-confidence
- source-confidence
- memory relevance
- verifier pressure
- commitment risk
- goal distance

Outputs:

- `accept`
- `verify`
- `retrieve`
- `write`
- `escalate`
- `handoff_to_human`
- controller owner

WriteLaw:

Inputs:

- evidence strength
- source-confidence
- novelty
- future reuse probability
- conflict score
- write risk

Outputs:

- reject / defer / accept
- persistence class
- decay policy
- revision policy
- rollback policy

Promotion / Demotion:

- episodic -> durable
- durable -> rule
- durable -> downgraded
- pending_fact -> anchored_fact

#### Constraints On Learning

The learned layer cannot learn to:

- bypass an evidence gate
- write durable state without a log
- skip verification under high commitment risk
- modify identity invariants

#### Optimization Objective

Contextual optimization only starts after hard and constitutional constraints hold.

Objective groups:

| Group | Objective |
|---|---|
| G1 control quality | reduce arbitration regret, unnecessary switches, verifier misses, wrong-owner dwell time |
| G2 memory governance | improve useful write ratio, reduce pollution writes, discover reusable rules, reduce durable conflict accumulation |
| G3 commit quality | improve commit precision, reduce wrong real-world commitments, improve rollback success |
| G4 resource efficiency | minimize time / cost / tool load without violating constraints |

#### Minimal Falsification Tests

L1 fixed-priority replacement:

Replace the learned governance policy with hand-written rules such as "high residual -> verify" and "low source-confidence -> retrieve". If performance is nearly unchanged, the system learned a flow template, not governance.

L2 signal ablation:

Remove one signal at a time:

- prediction residual
- self-confidence
- source-confidence
- commitment risk
- memory conflict

If removing key variables has no behavioral effect, those governance signals are not doing real work.

L3 cross-task-family transfer:

Train governance on task family A and test on structurally similar task family B with different rule-switch patterns. If it only works on A, it is likely a script, not a governance law.

---

### Layer 3: Phase-Diagram / Threshold Exploration Layer

This is a research-method layer, not a runtime authority layer.

It finds:

- useful thresholds
- transition regions
- phase boundaries
- variables worth giving to learned governance

It must not directly take over `GovernancePolicy`.

#### Methods

- threshold scans
- two-parameter grids
- phase-transition detection
- Pareto-front analysis

#### Outputs

- candidate thresholds
- risk regions
- variable priority ranking
- phase-boundary hypotheses

#### Minimal Falsification Tests

P1 threshold stability:

If a threshold works only on one slice and collapses under small perturbations, it is overfit and not a structural boundary.

P2 phase-map replication:

Change random seed or task family. If the main phase boundary moves arbitrarily, the scanned variable is not stable enough for governance.

---

## 2. Hard-Coded vs Learnable vs Scanned

### SelfState

Hard-coded schema:

- `control_state.current_owner`
- `commitment_state.commit_stage`
- `resource_state.time_budget`
- `resource_state.cost_budget`
- `identity_invariants`
- `permission_state`
- `rollback_availability`
- `evidence_presence`
- persistence class enum

Learnable estimates:

- capability success estimates
- self-confidence
- source-confidence
- anomaly score derived from predictive residual
- memory relevance
- future reuse probability
- write risk
- commitment risk

Threshold-scanned candidates:

- residual alarm threshold
- source-confidence minimum for durable write
- conflict threshold for forced verify
- write-risk threshold for defer
- promotion threshold: episodic -> durable
- promotion threshold: durable -> rule

Decision rule:

- Hard-code fields that must be auditable, cross-experiment comparable, or boundary-preserving.
- Learn fields that are estimates and may be updated from history.
- Scan low-dimensional interpretable fields that are expected to have switch boundaries.

### GovernancePolicy

Hard-coded:

- permission matrix
- required gates before commit
- minimum evidence for durable / rule / commitment
- verifier / human guards for irreversible actions
- mandatory `CommitLog` writes
- rollback requirements

Learnable:

- current owner selection
- retrieve / verify / defer choice
- persistence class choice
- rule promotion choice
- source trust allocation among candidates

Threshold-scanned:

- residual range for verify
- commitment-risk range for human handoff
- source-confidence range for forced retrieve
- conflict range for demotion

Decision rule:

- If a decision is about legal vs illegal, hard-code it.
- If a decision is about choosing the best legal action, it can be learned.
- If a decision depends on a low-dimensional continuous boundary, scan it before learning it.

### CommitLog

Hard-coded:

- log structure
- append-only / tamper-evident rules
- event types
- commit stage definitions
- required fields per commit
- replay and audit interface

Learnable:

- which event streams deserve higher-level summaries
- which log clusters can be compressed into rule-like summaries
- which commits require additional context tracking

Threshold-scanned:

- event retention levels
- conflict threshold for commit review event creation
- decay threshold for unused episodic logs

Decision rule:

The existence of accountability logs must not be learned away. Learning may decide summary form and attention priority, not whether traceability disappears.

---

## 3. Three-Level Objective System

### Viability Constraints

The system remains alive and governable:

- zero illegal state transitions
- zero unauthorized durable-or-higher writes
- near-zero irreversible miscommit rate
- high rollback availability for critical paths

Metrics:

- `violation_count`
- `illegal_transition_rate`
- `unauthorized_write_rate`
- `irreversible_error_rate`

### Constitutional Constraints

The system does not trade governance rules for short-term performance:

- insufficient evidence cannot be exchanged for speed
- missing logs cannot be exchanged for lower latency
- high risk cannot be exchanged for short-term success
- identity invariants cannot be overridden by planner utility

Metrics:

- `constitutional_breach_count`
- `gate_bypass_rate`
- `audit_completeness`
- `commitment_trace_completeness`

### Contextual Optimization

Only after the first two levels hold:

- arbitration quality
- write quality
- commit quality
- resource efficiency

Metrics:

- `arbitration_regret`
- `switch_latency`
- `write_precision`
- `pollution_rate`
- `commit_precision`
- `rollback_success`
- `cost_per_successful_task`

---

## 4. How To Prove It Is Governance, Not Flow Fitting

### Counter-Test 1: Fixed-Flow Approximation

Build a static rule baseline:

- residual > a -> verify
- source-confidence < b -> retrieve
- commitment risk > c -> escalate

If this fixed flow is nearly equivalent, the learned system is not a governance law.

### Counter-Test 2: Context Reordering

Keep local input distributions similar but reorder:

- task sequence
- memory conflict sequence
- resource-budget timing
- commit-pressure timing

If the model only works in the original order, it learned a script.

### Counter-Test 3: Conflicting Control Signals

Construct conflicts:

- high residual + high source-confidence
- high memory relevance + higher commitment risk
- low goal distance + high verifier disagreement

If the system follows one shortcut variable, it is a heuristic. Governance requires stable tradeoffs among conflicting signals.

### Counter-Test 4: Boundary-Shift Transfer

Train on low-risk reversible tasks. Test on:

- high-risk irreversible tasks
- longer-horizon commitments
- new memory-conflict patterns

If it preserves constitutional constraints and rebuilds authority decisions, it is closer to governance. If it collapses, it is flow fitting.

---

## 5. P0-P3 Landing Plan

### P0: unified-sel PredictiveHealthMonitor

Goal:

- test whether predictive residual adds real health signal beyond static drift

Inputs:

- `z_t`
- `context_t`
- `z_hat_{t+1}`
- `z_{t+1}`

Outputs:

- residual
- anomaly score
- early-warning score

Success signs:

- earlier warning than static drift
- more sensitive than verifier-only
- controlled false-positive rate

Failure signs:

- no lead time
- only synchronous alerts
- noise over-sensitivity

### P1: CEE SelfState Schema

Goal:

- eventize system self-state before learning any governance policy

Requirements:

- planner / verifier / memory / policy behavior updates `SelfState`
- every update passes through policy gate
- every commit writes a log event

Success signs:

- replayable
- auditable
- governance actions leave distinguishable SelfState traces

### P2: CEE WriteLaw

Goal:

- establish typed persistence

Every write must carry:

- evidence
- source-confidence
- persistence class
- revision policy
- rollback policy

Success signs:

- less memory pollution than flat memory
- higher durable precision
- more controllable conflicts

### P3: AuthoritySwitchingLaw

Inputs:

- prediction residual
- self-confidence
- source-confidence
- goal distance
- write risk
- commitment risk
- verifier pressure

Outputs:

- controller owner
- governance action
- handoff decision

Success signs:

- lower arbitration regret
- reasonable switch latency
- cross-environment behavior does not collapse into a rule table

---

## 6. Variables For Phase-Diagram / Threshold Scans

| Variable | Scan Object | Observed Metrics | Expected Boundary |
|---|---|---|---|
| predictive residual threshold | verify / escalate trigger | early-warning lead time, false positive rate | missed detection -> over-alerting |
| self-confidence threshold | accept / verify boundary | accept precision, verifier load | overconfidence -> over-conservatism |
| source-confidence threshold | retrieve / write / defer boundary | durable pollution, retrieval frequency | source gullibility -> over-retrieval |
| memory conflict threshold | forced verify / demotion boundary | conflict persistence, wrong reuse | conflict accumulation -> excessive demotion |
| write risk threshold | reject / accept / defer boundary | write precision, missed useful write | pollution -> write suppression |
| commitment risk threshold | internal result / external commit | commit precision, rollback frequency | reckless commit -> commit paralysis |
| episodic-to-durable promotion threshold | experience promotion boundary | durable reuse, durable conflict | lost experience -> durable pollution |
| durable-to-rule promotion threshold | rule formation boundary | rule reuse, rule brittleness | no abstraction -> premature rigidity |
| demotion threshold | durable / rule downgrade | stale-rule harm, recovery speed | rigidity -> forgetting |
| verifier pressure threshold | verifier ownership switch | verifier gain, latency | under-review -> review paralysis |
| resource budget threshold | planner / habit switch | success/cost ratio, latency | overthinking -> rash execution |
| rollback availability threshold | high-order commit permission | irreversible error rate | risky commit -> excessive freeze |

---

## 7. Expected Phase Maps

### Map A: Residual x Source-Confidence

Observe:

- verify frequency
- durable pollution
- commit precision

Expected regions:

- low residual / high source-confidence: accept
- high residual / low source-confidence: retrieve + verify
- high residual / high source-confidence: possible world-model drift, not necessarily source error
- low residual / low source-confidence: conservative defer

### Map B: Write Risk x Future Reuse Probability

Observe:

- useful write rate
- pollution rate
- later retrieval benefit

Expected regions:

- low risk / high reuse: durable or rule promotion
- high risk / low reuse: reject
- middle region: episodic or defer

### Map C: Commitment Risk x Rollback Availability

Observe:

- commit precision
- irreversible error
- human handoff rate

Expected regions:

- high risk / low rollback: verify or human
- low risk / high rollback: local commit
- high risk / high rollback: experimental commit

### Map D: Memory Conflict x Rule Reuse

Observe:

- stale rule harm
- conflict accumulation
- recovery speed

Expected regions:

- low conflict / high reuse: keep rule
- high conflict / high reuse: revise
- high conflict / low reuse: demote or decay

---

## 8. F-Drive Project Mapping

| Project | Role In This Spec |
|---|---|
| `F:\unified-sel` | mechanism lab; test whether governance signals add measurable value |
| `F:\cognitive-execution-engine` | governance and real-world commitment kernel |
| `F:\hypergraph_bistability` | persistence, conflict, promotion / decay, write-law reference |
| `F:\fcrs_world` | predictive compression and world-model prior |
| `F:\diff_world` | delta prediction prior |
| `LeWorldModel` | stable latent predictive substrate and surprise reference |
| `F:\cep-cc` | independent continuous communication project; do not mix into governance mainline |

---

## 9. Design Principle

Condensed form:

> `SelfState` describes what state the system is in; `GovernancePolicy` decides how to govern itself inside legal boundaries; `CommitLog` records which changes were officially accepted and can be audited.

This is not a static schema alone. It is:

- fixed governance interfaces
- learned governance strategies
- phase-diagram-guided variable selection

The immediate next implementation remains narrow:

1. Finish `unified-sel` P0 `PredictiveHealthMonitor`.
2. Only after P0 signal validation, design CEE `SelfState`.
3. Only after `SelfState`, implement CEE `WriteLaw`.
4. Only after those, train or fit `AuthoritySwitchingLaw`.
