# Weight Graph Fragility Prior Realignment

**Date**: 2026-04-12
**Status**: Active
**Scope**: `F:/unified-sel/weight_graph/`

---

## 1. Decision

`weight_graph` is no longer framed primarily as:

> static topology predicts capability or routing quality

It is now framed as:

> static topology may provide a low-cost fragility prior for intervention, monitoring, and triage

This is a narrower claim, but it matches the evidence currently available.

---

## 2. Why The Reframe Happened

The previous `Topo x capability` line is not validated yet.

The current blockers are:

- `exp06` MMLU ground truth was produced with local `1.5B`
- `exp07` topo matrix was produced from the `0.5B` cache path
- `exp08` is a depth-profile description pass, not a decisive predictive test
- most topo features change only weakly across depth, so the current aggregate method has low power

Therefore:

- Task 4 should be treated as descriptive, not decisive
- the capability-prediction claim is downgraded for this cycle

---

## 3. What Still Holds

These points remain valid:

- trained vs random modularity gap is strong
- static weight topology contains non-trivial organization
- cross-layer hub structure is real enough to motivate intervention tests
- `#490` appears more sensitive on perplexity than generic controls in current ablation evidence

These support a fragility-oriented question better than a capability-prediction question.

---

## 4. New Core Question

The working question for this branch becomes:

> can static topology identify model locations that are unusually fragile, intervention-worthy, or worth monitoring?

This is intentionally different from:

- capability prediction
- routing prior as a mainline claim
- single-neuron "ability core" storytelling

---

## 5. Branch Split

### Branch A: Predictive Prior

Status:

- downgraded

Meaning:

- do not spend mainline effort trying to rescue `Topo x capability` with the current evidence stack
- only reopen if model alignment is fixed and a stronger evaluation design exists

Reopen conditions:

- align model identity between topo features and task metrics
- use per-layer activation or intervention-aware data, not only depth-segment averages
- define predictive uplift against a non-topology baseline

### Branch B: Fragility Prior

Status:

- active sidecar

Meaning:

- use topology to rank where intervention is likely to matter
- target pruning, ablation, health monitoring, and intervention selection

---

## 6. What `exp09` Means Now

Current interpretation of `exp09`:

- `NO-GO` for promoting a mechanistic mainline claim
- not enough evidence that `#490` is a task-accuracy-specific capability core
- still useful evidence that `#490` may be language-modeling sensitive or intervention-sensitive

So the correct downgrade is:

- not "the neuron is irrelevant"
- but "the neuron does not justify a large mechanistic promotion yet"

---

## 7. New Near-Term Goal

The next bounded follow-up is:

1. run an activation/profile analysis for `#490`
2. compare it with:
- same-layer high-PageRank control
- random control
3. check whether activation patterns are stable enough to support a fragility story

This is a close-out task, not a new mainline.

---

## 8. Stop/Go For The New Frame

### Go

- topology-derived candidates consistently expose unusually fragile sites
- or `#490` shows a stable, interpretable activation profile distinct from controls

### Hold

- there is some intervention sensitivity, but not enough separation from controls

### Stop

- no stable profile
- no useful intervention ranking signal
- no practical monitoring value

If `Stop`, the branch should be written up as observation-only side evidence.

---

## 9. Practical Use Cases Under This Frame

If this line survives, likely use cases are:

- ablation candidate ranking
- pruning-risk ranking
- monitoring-risk prior
- health or instability triage

Not:

- direct capability prediction
- direct routing superiority claims

---

## 10. One-Sentence Summary

`weight_graph` should now be judged by whether static topology helps find fragile or intervention-sensitive structure, not by whether it directly predicts benchmark ability.
