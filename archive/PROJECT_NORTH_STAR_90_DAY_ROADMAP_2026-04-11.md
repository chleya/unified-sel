# Project North Star and 90-Day Exploration Roadmap

**Date**: 2026-04-11
**Status**: Active project-level direction note
**Scope**: Unified-SEL, capability routing, weight_graph

---

## 1. North star

The project is **not** primarily about writing papers.
The real target is:

> build a usable model-boundary awareness system that can reliably change system decisions.

In practical terms, the long-term goal is:

> detect when a model is near its capability boundary, decide whether local repair is likely to work, and use low-cost structural priors plus runtime evidence to control accept / revise / escalate decisions.

This is the project-level objective that sits above any individual benchmark or paper.

---

## 2. What counts as a high-value target

A target is worth pursuing only if it satisfies most of the following:

- it changes system behavior, not just interpretation
- it can be falsified within weeks, not months
- it can transfer beyond a single benchmark instance
- it is difficult to replace by simply scaling compute or data
- it moves the project toward a reusable decision mechanism

By this standard, the project should prioritize mechanism discovery and control improvement over benchmark accumulation.

---

## 3. Three-layer project frame

### Layer A. Boundary sensing

Core question:

> does a reusable, non-enumerative signal exist that detects when the local model is near a real capability boundary?

Current evidence:

- `semantic` now matches `counterfactual` on the current hardened capability-routing benchmark
- this is strong evidence inside the current probe set
- it is not yet evidence of broad generalization

What matters next:

- held-out family validation
- paraphrase / perturbation robustness
- cross-task transfer of the same signal idea

Success condition:

- a signal family that still works when the benchmark stops rewarding family-by-family closure

Failure condition:

- performance collapses once the exact closed family inventory is perturbed or withheld

### Layer B. Recoverability-aware control

Core question:

> can the system distinguish between risk that is locally recoverable and risk that should immediately escalate?

Current evidence:

- the project already has an `accept / verify+revise / escalate` control structure
- `monitor_repair_triage` is already the strongest practical policy layer on the current benchmark

What matters next:

- identify recoverable vs non-recoverable failure families
- learn whether recoverability has its own predictive signal
- improve the decision boundary between revise and escalate

Success condition:

- lower cost at fixed success, or higher success at fixed cost, by making better recoverability decisions

Failure condition:

- the system only detects danger but cannot predict whether local repair is viable

### Layer C. Structural priors

Core question:

> can static structural signals provide a low-cost prior that improves runtime control?

Current evidence:

- `weight_graph` already shows non-trivial static structure
- but it has not yet shown predictive value for task capability or routing

What matters next:

- task-side ground truth
- topology-to-capability association
- hybrid use with runtime signals, not topology-only replacement claims

Success condition:

- static topology provides a useful prior that improves routing or calibration when combined with runtime evidence

Failure condition:

- topology remains descriptive but not decision-relevant

---

## 4. What the project should stop treating as primary goals

The following are no longer top-level objectives:

- chasing additional benchmark closure by default
- full-graph Louvain as a gating milestone
- purely descriptive topology findings without decision impact
- philosophical claims about self-awareness or self-reflection

These may still matter as supporting evidence, but they are not the project's highest-leverage targets.

---

## 5. 90-day roadmap

### Days 1-30: prove or break boundary sensing generalization

Primary objective:

- determine whether current `semantic` success is a reusable mechanism or benchmark-local closure

Tasks:

1. freeze `code-20` / `mixed-40`
2. design held-out family evaluation
3. design lexical / probe paraphrase validation
4. run policy invariance checks across existing policy families
5. rewrite the capability claim only after those results are in

Decision at day 30:

- if `semantic` generalizes, boundary sensing becomes the core engine of the project
- if it fails, redesign the signal family rather than adding more closures blindly

### Days 31-60: push from sensing to recoverability-aware control

Primary objective:

- determine whether the system can predict local recoverability, not just risk

Tasks:

1. partition current failures into recoverable vs non-recoverable regimes
2. test whether existing signals separate those regimes
3. design a stronger revise-vs-escalate controller
4. measure cost-success improvements relative to current `monitor_repair_triage`

Decision at day 60:

- if recoverability can be predicted, the project moves from routing benchmark work to real adaptive control work
- if not, keep the project framed as boundary detection rather than autonomous repair control

### Days 61-90: test structural priors as hybrid decision support

Primary objective:

- determine whether static topology can strengthen runtime control as a prior

Tasks:

1. complete task-side ground truth for `weight_graph`
2. run topology-to-capability analysis
3. if positive, convert topology into a coarse prior signal
4. test hybrid control:
   - structural prior
   - runtime semantic signal
   - combined controller

Decision at day 90:

- if topology improves hybrid control, the project gains a second major engine
- if topology fails to predict capability, downgrade it to side evidence and keep the mainline on runtime boundary sensing

---

## 6. Operational priority order

Current project-wide priority should be:

1. `Capability routing generalization`
2. `Recoverability-aware control`
3. `Topology-to-capability validation`
4. `Hybrid structural-prior routing`
5. `Paper packaging`

This does not mean papers are unimportant.
It means papers should summarize progress on the real mechanism, not drive the mechanism choice.

---

## 7. Concrete near-term decisions

### Keep investing in

- held-out / paraphrase validation for capability routing
- repair-vs-escalate decision quality
- `Task 2 -> Task 4` on `weight_graph`
- simplified, decision-relevant benchmarks

### Reduce investment in

- additional closure families without a new regime
- full-graph community detection engineering
- descriptive analysis that does not affect control decisions
- broad narrative framing before mechanism validation

---

## 8. Project-level success definition

The project should be considered genuinely successful if, within this roadmap, it achieves the following:

1. a boundary signal that generalizes beyond a closed benchmark family set
2. a controller that uses that signal to improve accept / revise / escalate decisions
3. at least one low-cost prior or auxiliary signal that further improves robustness or calibration

If only item 1 succeeds, the project still has strong scientific value.
If items 1 and 2 succeed, the project has strong systems value.
If all three succeed, the project becomes a credible new approach to adaptive inference control.

---

## 9. One-sentence summary

The project north star is not “publish a routing paper” or “analyze weight graphs,” but:

> discover and validate a reusable boundary-awareness mechanism that lets a system decide when to trust, repair, or escalate model outputs.
