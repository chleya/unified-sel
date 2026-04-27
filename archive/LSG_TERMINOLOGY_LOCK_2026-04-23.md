# LSG Terminology Lock

Date: 2026-04-23
Status: vocabulary lock for v0

This file freezes terminology for the next LSG project stage.
Use these meanings unless explicitly revised in a later spec.

---

## 1. Root Terms

### `D_t`: Disturbance Rewrite Pressure

Meaning:

```text
The pressure a candidate content exerts against the current maintained order.
```

It answers:

```text
What is the cost or instability of continuing to ignore this candidate?
```

It is not:

- salience
- importance
- risk
- confidence
- novelty alone
- attention score

### `S_t`: Order Self-Stability

Meaning:

```text
The strength with which the current maintained order should resist being changed.
```

It answers:

```text
What is the cost of allowing this candidate to restructure the system?
```

It is not:

- conservatism
- refusal to learn
- safety score alone
- old-state confidence alone

### `R_t`

Meaning:

```text
A boundary statistic derived from D_t and S_t, or from proxy variables.
```

Preferred v0 interpretation:

```text
R_t = D_t / (S_t + epsilon)
```

Compatibility interpretation:

```text
R_t = (U_t * N_t) / (A_t * P_t + epsilon)
```

`R_t` is not the root variable.
`D_t` and `S_t` are the root variables.

---

## 2. Candidate And State Terms

### Candidate Content

Any proposed content that might affect the current order.

Examples:

- new fact
- tool result
- verifier contradiction
- user correction
- memory conflict
- proposed rule change
- pending commitment update

### Current Order

The currently maintained state structure.

Includes:

- active working interpretation
- formal memory/facts
- current goal context
- commitments
- permission boundary
- rollback boundary

### Formal State

State that can affect future behavior after the current step.

Formal state is not the same as:

- model output
- temporary context
- foreground content
- candidate proposal

### Formal Acknowledgement

The act of accepting a candidate into formal state.

Formal acknowledgement requires:

```text
phase == commit_review
evidence gate open
constitutional gate open
log ready
commit event written
```

---

## 3. Phase Terms

### `suppressed`

The candidate lacks enough pressure to challenge current order, or current order strongly resists it.

### `background`

The candidate is retained as available context but does not control foreground processing.

### `foreground`

The candidate enters active processing but has no durable state effect.

Foreground is not commit.

### `verify`

The candidate requires evidence gathering or conflict checking.

Verify is not commit.

### `commit_review`

The candidate is eligible for formal acknowledgement review.

Commit review is not yet formal acknowledgement.

### `acknowledged`

The candidate has crossed the formal boundary and has a matching commit event.

This is the only durable state promotion phase in v0.

### `rejected`

The candidate is explicitly denied or expired.

Rejected content must not mutate durable state.

---

## 4. Attention Terms

Use attention language only as interpretation.

### Foreground

Means limited active control bandwidth.
It does not mean truth or commitment.

### Suppression / Inhibition

Means the current order or learned history prevents a candidate from repeatedly entering active processing.

It must not hide hard safety signals.

### Sustainment

Means unresolved consequence keeps an item active or eligible.

It does not mean the item is true.

### Switching

Means a phase or mode boundary was crossed.

It is not a primitive action.
It is a threshold effect.

### Acknowledgement

Means formal state recognition.

It is the attention-language equivalent of commit.

---

## 5. Gate Terms

### Evidence Gate `E_t`

Open only when the candidate has sufficient evidence.

Candidate evidence belongs here, not in `S_t`.

### Constitutional Gate `K_t`

Open only when the requested state transition is allowed by explicit constraints.

This gate is not a learned free head.

### Log Gate

Open only when a commit event can be written.

No log, no formal acknowledgement.

---

## 6. Forbidden Conflations

Do not conflate:

| Wrong conflation | Correct distinction |
|---|---|
| salience = disturbance pressure | salience may feed `D_t`, but `D_t` is rewrite pressure |
| foreground = commit | foreground is active processing only |
| high `D_t` = allow commit | high `D_t` plus high `S_t` means protected conflict |
| evidence strength = old-state stability | candidate evidence feeds `E_t`; old-state anchoring may feed `S_t` |
| LLM confidence = evidence | confidence is at most an input, not a gate |
| memory write = formal acknowledgement | memory write must pass commit boundary |
| attention mechanism = Transformer attention | this project studies finite foreground control |
| phase rule = biological claim | phase rule is engineering abstraction |

---

## 7. One-Sentence Lock

Use this wording as the project anchor:

```text
LSG models formal state rewrite as a two-variable qualification process:
candidate disturbance pressure D_t challenges current order self-stability S_t,
and only candidates that cross the correct phase boundary and pass evidence,
constitutional, and logging gates can be formally acknowledged.
```

