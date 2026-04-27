# Attentional Governance Kernel v0

Date: 2026-04-23
Status: Concept-to-engineering bridge
Scope: finite foreground control for state rewrite systems

This note rewrites Learned State Governance in attention-control language.
It intentionally avoids treating "attention" as Transformer token weighting.

The target is a human-like attention control kernel:

```text
finite foreground bandwidth + background suppression + mode switching + formal acknowledgement
```

The kernel does not decide what is true by itself.
It decides what can occupy foreground control bandwidth, what must remain suspended, what is suppressed, when the foreground mode switches, and what foreground result is formally acknowledged into durable state.

Core claim:

```text
State governance can be modeled as foreground control under limited bandwidth.
```

---

## 1. Six Actions

The v0 kernel has exactly six control actions.

| Action | Meaning | Durable state effect |
|---|---|---|
| `foreground` | move an item into the active control workspace | none |
| `background` | keep an item available but not control-dominant | none |
| `preempt` | interrupt current focus because a signal has override priority | none |
| `sustain` | keep an item active despite weak immediate stimulus | none |
| `inhibit` | suppress an item, source, path, or candidate from re-entering foreground | none |
| `acknowledge` | formally accept a foreground result into durable state | yes, through commit gate only |

The important distinction is:

```text
foreground is not commit
preempt is not authority
sustain is not truth
acknowledge is the only durable state promotion
```

---

## 2. Minimum State

```python
@dataclass
class AttentionalState:
    current_focus: FocusSlot | None
    foreground_slots: list[FocusSlot]
    background_pool: list[AttentionItem]
    salience_vector: SalienceVector
    sustain_set: list[SustainItem]
    inhibition_mask: list[InhibitionRule]
    mode: AttentionMode
    bandwidth_limit: int
    acknowledgement_log: list[AcknowledgementRecord]
```

### 2.1 `current_focus`

The item currently controlling processing.

Examples:

- a pending verifier
- a high-risk commitment
- a memory conflict
- a user goal
- a tool-result anomaly
- a write candidate under review

```python
@dataclass
class FocusSlot:
    item_id: str
    item_type: Literal[
        "goal",
        "commitment",
        "verification",
        "memory_conflict",
        "tool_anomaly",
        "rewrite_candidate",
        "human_handoff",
    ]
    source: str
    entered_at: int
    priority: float
    sustain_until: int | None
    mode_owner: str
```

### 2.2 `foreground_slots`

Limited-capacity active workspace.

Rule:

```text
len(foreground_slots) <= bandwidth_limit
```

Default v0:

```text
bandwidth_limit = 3
```

If a new item preempts while slots are full, the kernel must demote, inhibit, or reject an existing foreground item.

### 2.3 `background_pool`

Items available for later attention but not currently controlling behavior.

```python
@dataclass
class AttentionItem:
    item_id: str
    content: str
    item_type: str
    source: str
    created_at: int
    last_foregrounded_at: int | None
    salience: float
    inhibited: bool
```

### 2.4 `salience_vector`

Signals competing for foreground entry.

```python
@dataclass
class SalienceVector:
    residual: float
    source_anomaly: float
    verifier_disagreement: float
    memory_conflict: float
    commitment_risk: float
    goal_urgency: float
    budget_pressure: float
    repeated_noise: float
```

Interpretation:

- `residual`: prediction or expectation failure
- `source_anomaly`: source trust or authorization abnormality
- `verifier_disagreement`: generator/verifier conflict
- `memory_conflict`: contradiction with durable memory
- `commitment_risk`: irreversible or pending obligation risk
- `goal_urgency`: current task pressure
- `budget_pressure`: compute/time/context pressure
- `repeated_noise`: recurring low-value interruption signal

### 2.5 `sustain_set`

Items that remain foreground-eligible even when their immediate salience is low.

Examples:

- pending commitment
- unfinished verification
- active long goal
- irreversible warning
- unresolved human approval request

```python
@dataclass
class SustainItem:
    item_id: str
    reason: str
    min_priority: float
    expires_at: int | None
    required_mode: str | None
```

### 2.6 `inhibition_mask`

Temporary suppression over items, sources, routes, or actions.

Examples:

- suppress repeated low-value retrieval
- suppress noisy source
- suppress recently failed planner path
- suppress unsafe direct commit path
- suppress re-proposal of rejected candidate

```python
@dataclass
class InhibitionRule:
    rule_id: str
    target_type: Literal["item", "source", "route", "action", "candidate"]
    target_id: str
    strength: float
    created_at: int
    expires_at: int
    reason: str
```

### 2.7 `mode`

The current foreground processing regime.

```python
class AttentionMode(Enum):
    HABIT = "habit"
    RETRIEVE = "retrieve"
    VERIFY = "verify"
    DEFER = "defer"
    COMMIT_REVIEW = "commit_review"
    HUMAN_HANDOFF = "human_handoff"
```

Mode meanings:

- `HABIT`: normal low-friction operation
- `RETRIEVE`: memory/context acquisition dominates
- `VERIFY`: contradiction/error checking dominates
- `DEFER`: preserve issue without acting
- `COMMIT_REVIEW`: durable acknowledgement is being considered
- `HUMAN_HANDOFF`: external approval dominates

### 2.8 `acknowledgement_log`

Append-only record of what foreground processing formally accepted.

This is equivalent to the commit log boundary in governance terms, but the attention-language meaning is:

```text
the item has moved from foreground content to recognized durable state.
```

---

## 3. Six-Action Mechanics

### 3.1 `foreground`

Move an item into active bandwidth.

Entry score:

```text
foreground_score =
    stimulus_salience
    + goal_bias
    + sustain_boost
    - inhibition_penalty
    - bandwidth_cost
```

Rules:

- high score permits entry into `foreground_slots`
- foreground entry does not imply correctness
- foreground entry does not imply durable write

### 3.2 `background`

Keep an item available without active control.

Use when:

- salience is real but not urgent
- item may become relevant later
- bandwidth is full
- hard constraints block immediate action

Rules:

- background items may decay
- background items may later preempt if salience rises
- background items cannot mutate durable state

### 3.3 `preempt`

Interrupt current focus because a signal has override priority.

Preemption candidates:

- irreversible action warning
- source authorization anomaly
- verifier disagreement above threshold
- memory conflict with high blast radius
- human approval required

Rules:

- preemption must name the displaced focus
- preemption must log the triggering salience component
- repeated preemption by the same failed signal creates inhibition

### 3.4 `sustain`

Keep an item active despite weak immediate stimulus.

Sustain candidates:

- pending commitment
- unfinished verifier
- unresolved conflict
- long goal
- rollback obligation

Rules:

- sustain consumes bandwidth
- sustain must expire or resolve
- expired sustain items are demoted unless renewed by evidence

### 3.5 `inhibit`

Suppress an item, source, route, action, or candidate.

Inhibition candidates:

- repeated low-value residual
- noisy source
- stale candidate
- rejected write proposal
- planner path with repeated verifier failure

Rules:

- inhibition is time-limited by default
- inhibition strength decays unless reinforced
- inhibition cannot hide hard safety signals

### 3.6 `acknowledge`

Promote foreground result into recognized durable state.

Rules:

- only allowed from `COMMIT_REVIEW`
- must pass the LSG commit gate
- must append an acknowledgement record
- must preserve source, proxy values, hard-constraint result, and verifier result

Acknowledgement is the only action with durable state effect.

---

## 4. Mode Switching

### 4.1 Mode Transition Table

| From | Trigger | To |
|---|---|---|
| `HABIT` | retrieval insufficiency | `RETRIEVE` |
| `HABIT` | verifier disagreement or conflict | `VERIFY` |
| `HABIT` | candidate durable write | `COMMIT_REVIEW` |
| `RETRIEVE` | sufficient context found | `HABIT` |
| `RETRIEVE` | retrieval conflict found | `VERIFY` |
| `VERIFY` | unresolved ambiguity | `DEFER` |
| `VERIFY` | durable write candidate survives | `COMMIT_REVIEW` |
| `COMMIT_REVIEW` | hard constraint fails | `DEFER` |
| `COMMIT_REVIEW` | approval required | `HUMAN_HANDOFF` |
| `COMMIT_REVIEW` | acknowledgement succeeds | `HABIT` |
| `DEFER` | new evidence arrives | `VERIFY` |
| `HUMAN_HANDOFF` | approval granted | `COMMIT_REVIEW` |
| `HUMAN_HANDOFF` | approval denied or timeout | `DEFER` |

### 4.2 Switching Rules

- switching must preserve unresolved sustain items
- switching may inhibit the displaced route if it repeatedly failed
- switching cannot acknowledge durable state by itself
- switching into `COMMIT_REVIEW` only creates eligibility for acknowledgement

---

## 5. Bandwidth Rules

The attention kernel is invalid if it allows unbounded foreground.

Minimum v0 bandwidth rules:

```text
foreground_slots <= 3
sustain_set <= 5
active verifier tasks <= 2
active rewrite candidates <= 3
human handoff requests <= 1
```

Overflow policy:

1. hard safety signals remain eligible
2. unresolved commitments outrank ordinary retrieval
3. repeated noise is inhibited
4. low-utility candidates return to background
5. if no demotion is safe, switch to `DEFER`

---

## 6. Connection To Rewrite Qualification

The attention kernel decides whether a candidate reaches `COMMIT_REVIEW`.
The LSG gate decides whether it can be acknowledged.

Mapping:

| Attention term | LSG term |
|---|---|
| foreground result | rewrite candidate |
| acknowledge | commit |
| acknowledgement log | commit log |
| inhibition mask | hard/soft suppression |
| sustain set | pending commitment state |
| mode switch | authority switch |
| salience vector | proxy inputs |

The ratio

```text
R_t = (U_t * N_t) / (A_t * P_t + epsilon)
```

belongs at the `acknowledge` boundary, not at every foreground decision.

Foreground can be driven by salience.
Acknowledgement requires qualification.

---

## 7. v0 Sanity Experiments

### 7.1 Foreground Capacity Test

Claim:

```text
The kernel must choose under limited foreground bandwidth.
```

Pass:

```text
foreground_slots never exceeds bandwidth_limit
```

Fail:

```text
any step has unbounded active foreground growth
```

### 7.2 Preemption Test

Claim:

```text
High-risk anomalies can interrupt habit mode.
```

Setup:

- start in `HABIT`
- inject high verifier disagreement or irreversible warning

Expected:

```text
HABIT -> VERIFY or COMMIT_REVIEW
```

Pass:

```text
preemption occurs and displaced focus is logged
```

### 7.3 Sustain Test

Claim:

```text
Pending commitments remain foreground-eligible even when immediate salience drops.
```

Expected:

```text
pending commitment remains in sustain_set until resolved or expired
```

### 7.4 Inhibition Test

Claim:

```text
Repeated low-value interruptions lose foreground access.
```

Setup:

- repeated residual spikes that verifier marks as noise

Expected:

```text
inhibition rule added for source/item/path
foreground re-entry rate decreases
```

### 7.5 Acknowledgement Boundary Test

Claim:

```text
Foreground processing cannot mutate durable state without acknowledgement.
```

Pass:

```text
zero durable mutations outside acknowledge/commit path
```

---

## 8. First Implementation Target

If implemented, start with a deterministic simulator:

```text
core/attentional_governance.py
tests/test_attentional_governance.py
experiments/capability/attention_kernel_sanity.py
```

Do not start with LLM calls.
Do not train salience heads.
Do not add biological claims.

Minimum artifact:

```json
{
  "num_steps": 0,
  "max_foreground_slots": 0,
  "preemption_count": 0,
  "sustain_expired_count": 0,
  "inhibition_count": 0,
  "acknowledgement_count": 0,
  "durable_mutation_without_acknowledgement": 0
}
```

---

## 9. What This Spec Claims

Narrow claim:

```text
A governance kernel can be expressed as finite foreground control:
items compete for foreground, some signals preempt, some items persist,
some routes are inhibited, modes switch, and only acknowledged results
enter durable state.
```

What it does not claim:

- it is a model of biological consciousness
- it reproduces human attention in full
- salience ranking is enough
- Transformer attention is equivalent to this mechanism
- foreground entry implies truth
- acknowledgement implies safety without the LSG gate

