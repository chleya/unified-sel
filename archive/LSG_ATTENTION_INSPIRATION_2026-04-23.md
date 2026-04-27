# LSG / Attentional Governance Inspiration Note

Date: 2026-04-23
Status: project inspiration and synthesis note

This note records the reasoning path that led to the next project direction.
It is not an implementation spec and not a paper draft.
Its purpose is to preserve the conceptual compression so future work does not drift back into vague "agent governance" or generic "attention" language.

---

## 1. Starting Point

The original question was how to study a learned state governance mechanism for LLM or small-model systems.

The first formulation was:

```text
R_t = (U_t * N_t) / (A_t * P_t + epsilon)
```

where `R_t` measures whether a candidate content has qualification to rewrite current formal state.

The early interpretation was governance-like:

- who can write
- what can commit
- when a candidate becomes formal state
- how to prevent unauthorized state mutation

This led to the first engineering form:

- explicit state machine
- evidence gate
- constitutional gate
- commit log
- no-bypass invariant
- hysteresis around rewrite thresholds

The key insight was that the project is not about stronger agent action.
It is about separating ordinary inference from formal state rewrite.

---

## 2. Shift Toward Human Attention

The next intuition was that this is not only a governance problem.
It resembles a human-like attention control problem.

But the relevant form of attention is not Transformer token attention.
It is not a saliency map.
It is not "which token attends to which token".

The relevant human-like function is:

```text
what can occupy the current foreground control bandwidth
```

The useful attention functions were identified as:

1. selection
2. amplification
3. suppression
4. switching
5. sustainment

In system terms:

- selection decides what gets processed first
- amplification boosts selected content
- suppression prevents irrelevant or unsafe content from dominating
- switching changes the active mode when conflict or anomaly appears
- sustainment keeps pending commitments or unresolved risks active

This reframed governance from a rule table into a foreground-control problem.

---

## 3. Six Surface Actions

The attention-style design first produced six apparent actions:

```text
foreground
background
preempt
sustain
inhibit
acknowledge
```

Their meanings were:

- `foreground`: move content into active processing
- `background`: keep content available without active control
- `preempt`: interrupt current focus
- `sustain`: keep something active despite weak current stimulus
- `inhibit`: suppress repeated or unsafe re-entry
- `acknowledge`: formally accept a foreground result into durable state

This was a useful intermediate language.
But it was still too late in the causal chain.

Those actions describe what has already happened at the surface.
They do not explain what accumulated before the action occurred.

---

## 4. Deeper Compression

The discussion then compressed the surface actions into a deeper opposition:

```text
order self-stability vs disturbance rewrite pressure
```

The key question stopped being:

```text
what action should be taken?
```

and became:

```text
how strongly does a candidate content pressure the current order,
and how strongly does the current order resist being changed?
```

This produced two continuous variables:

```text
D_t = disturbance rewrite pressure
S_t = order self-stability
```

This is the current root abstraction.

---

## 5. `D_t`: Disturbance Rewrite Pressure

`D_t` measures how strongly a candidate pushes against the currently maintained order.

It is not salience.
It is not importance.
It is not risk.
It is the estimated cost or instability of continuing to ignore or suppress the candidate.

Candidate sources for `D_t`:

- incompatibility with current order
- goal relevance
- consequence suspense
- external traction

Interpretation:

- high incompatibility means the old state is becoming self-inconsistent
- high goal relevance means the current goal cannot proceed without resolving it
- high consequence suspense means unresolved future obligations are accumulating
- high external traction means the signal is anchored in tool/user/environment feedback rather than internal speculation

---

## 6. `S_t`: Order Self-Stability

`S_t` measures how strongly the current order should resist being changed.

It is not simple conservatism.
It is the cost of allowing this candidate to restructure the system.

Candidate sources for `S_t`:

- continuity value
- authority barrier
- ambiguity load
- perturbation cost

Interpretation:

- high continuity value means the current order still works
- high authority barrier means the candidate lacks permission or touches protected boundaries
- high ambiguity load means the candidate may be a temporary exception or unclear change
- high perturbation cost means a formal rewrite would affect many downstream commitments or be hard to roll back

---

## 7. Why Two Variables, Not One

A single score cannot separate these cases:

| Case | `D_t` | `S_t` | Correct behavior |
|---|---:|---:|---|
| persistent real drift | high | low | verify then commit-review |
| protected-boundary conflict | high | high | verify/defer/handoff, not direct commit |
| weak noise | low | high | suppress |
| weak harmless novelty | low | low | background or ignore |

This distinction is central.

High pressure alone is not enough.
Sometimes high pressure is exactly why the system should slow down and protect the current order.

---

## 8. Surface Actions As Phase Effects

Once `D_t` and `S_t` are primary, the earlier actions become phase effects:

- suppression means `S_t` dominates `D_t`
- foreground means `D_t` is close enough to challenge `S_t`
- preemption/switching means `D_t/S_t` crosses a control boundary
- sustainment means `S_t` or consequence suspense keeps an item active
- acknowledgement means a candidate crosses a higher formalization boundary and passes gates

The project should therefore not start by training an action classifier.
It should first model the two-variable dynamics and phase boundaries.

---

## 9. Relationship To `R_t`

The original ratio can be reinterpreted as a boundary statistic:

```text
R_t = (U_t * N_t) / (A_t * P_t + epsilon)
```

The cleaner interpretation is:

```text
D_t ~ f(U_t, N_t)
S_t ~ g(A_t, P_t)
R_t ~ D_t / (S_t + epsilon)
```

`R_t` is not the root mechanism.
`D_t` and `S_t` are the root dynamics.

Important correction:

- evidence for the candidate should feed the evidence gate `E_t`
- evidence anchoring of the current formal state can contribute to `S_t`

Otherwise the model risks the paradox that stronger candidate evidence lowers rewrite qualification.

---

## 10. CEE As Prior Work

We found a closely related earlier project:

```text
F:\cognitive-execution-engine
```

Relevant CEE components:

- `WorldState`
- `CommitmentEvent`
- `ModelRevisionEvent`
- `EventLog`
- `MemoryPromotion`
- red-line invariant tests

CEE already implements a hard commitment kernel:

- no approval, no committed state mutation
- policy/meta patches are hard-blocked
- every committed fact has provenance
- denied/pending events do not appear in replayed `WorldState`
- `requires_approval` without approval never enters `WorldState`

This means the new project does not start from nothing.

CEE gives the hard institutional layer.
The new project adds continuous rewrite qualification dynamics above it:

```text
D_t / S_t dynamics
    -> phase boundaries
        -> CEE-style acknowledgement / commit boundary
```

---

## 11. Hypergraph / Stability Prior Work

We also found a second relevant prior line:

```text
F:\hypergraph_bistability
```

That project is less directly about commit governance, but more directly about:

- bistability
- multi-stability
- control
- inhibition
- order parameters
- phase-like switching

It is relevant to the language of:

- stable basins
- perturbation
- threshold crossing
- chattering
- control surfaces

But it should not replace CEE as the commit layer.

The likely split is:

- CEE contributes the hard commit/event/replay/provenance layer
- hypergraph bistability contributes intuition for stability and switching dynamics

---

## 12. Project Identity

The next project should not be described as:

- human attention simulator
- consciousness model
- general self-governing agent
- memory system
- world model

The sharper description is:

```text
Two-variable rewrite qualification model for formal state governance.
```

Or:

```text
Attentional State Governance through disturbance/stability dynamics.
```

Shortest formulation:

> Formal state rewrite should occur only when disturbance pressure overcomes order self-stability and the candidate passes evidence, constitutional, and logging gates.

---

## 13. What Must Not Be Lost

The following points are the core of the discussion:

1. Attention is not the root object. Rewrite qualification is deeper.
2. Governance actions are surface phase effects, not primitives.
3. Two variables are necessary: pressure and stability.
4. High disturbance plus high stability means protected conflict, not direct commit.
5. Formal acknowledgement is separate from foreground processing.
6. CEE already solved much of the hard commitment boundary.
7. The new contribution is continuous qualification dynamics plus hysteretic phase control.

---

## 14. First Research Claim To Earn

Do not claim broad intelligence or human-like cognition.

The first claim to earn is:

```text
Temporary spikes, sustained drift, and protected-boundary conflicts occupy
different regions of the D/S phase space and produce different formal-state
outcomes under the same commit boundary.
```

This is narrow enough to test.
It is also meaningful enough to justify the project.

