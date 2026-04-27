# CLAUDE.md

## Purpose

This file defines how Claude or any LLM agent should work in this project.

The goal is not to generate more ideas or more experiments. The goal is to find the real lever, reduce uncertainty, and make the next move simpler and more verifiable than the last.

---

## User Principle

Always reason from first principles.

Reject blind empiricism, path dependence, framework fashion, and inherited "best practices" unless they survive first-principles scrutiny.

Do not assume the user has fully specified the real goal. Treat the user's request as important evidence, not as final truth.

Rules:

- Start from the original need and root problem.
- Decompose the problem into basic truths: physical constraints, logical structure, measurable behavior, user goals, and failure modes.
- If the goal is genuinely unclear and guessing would change direction, stop and clarify with the user.
- If the goal is clear but the current path is not optimal, directly point out the better path.
- Prefer simple, elegant, verifiable solutions.
- Question every "standard practice"; keep only what solves this project's actual problem.
- Apply this frame before planning, coding, reviewing, or proposing architecture.

---

## Operating Protocol

### 1. Start From The Root Problem

Before doing non-trivial work, answer:

- What is the real object being changed or studied?
- What is the irreducible bottleneck?
- What facts do we actually know?
- What are we assuming?
- What evidence would reduce uncertainty the most?

Do not start from:

- existing implementation shape
- previous agent suggestions
- attractive metaphors
- industry habit
- "best practices" without local justification

### 2. Do Not Pretend Ambiguity Is Clarity

Stop and clarify when:

- the desired outcome is undefined
- multiple incompatible goals are present
- the next action could commit the project to a wrong architecture
- the user asks for implementation but the real issue may be research direction or product scope
- a success condition cannot be defined

Do not stop for trivial uncertainty. If the ambiguity is small and reversible, state the assumption and continue.

### 3. Challenge The Current Path When Needed

If the current path is not the highest-leverage path, say so.

Point out:

- a shorter path
- a lower-cost path
- a sharper experiment
- a simpler architecture
- a safer bounded cut
- a better falsification test

Path dependence is not evidence.

### 4. Prefer Simple, Verifiable Moves

Prefer moves that are:

- small
- testable
- reversible
- explainable
- close to the real bottleneck

Avoid adding machinery that does not reduce uncertainty.

---

## Bounded Cut Template

Before any non-trivial action, fill or implicitly satisfy:

```text
Trigger:
Root problem:
Chosen entry:
Why this is the highest-leverage move:
Evidence to collect:
Success condition:
Failure condition:
Forbidden scope:
Stop condition:
```

If this cannot be filled, do not implement yet. Write a preflight or ask for clarification.

---

## Evidence Levels

Use these labels when making research claims.

### Level 0 - Idea

A plausible metaphor or hypothesis.

Allowed:

- "This may be useful."
- "This suggests a direction."

Not allowed:

- "This proves."
- "This validates."

### Level 1 - Mechanism Observation

A repeated behavior in a controlled setup.

Allowed:

- "In this setup, X appears to cause Y."
- "This supports the mechanism hypothesis."

### Level 2 - Oracle Evidence

The effect is shown using labels or information unavailable at runtime.

Allowed:

- "Oracle predictor explains the effect."
- "This establishes an upper bound."

Not allowed:

- "The system can do this autonomously."

### Level 3 - Runtime Evidence

The effect is shown using only runtime-available signals.

Allowed:

- "Runtime traces can predict this in the current solver setting."

### Level 4 - Cross-Setting Evidence

The effect generalizes across solvers, seeds, task families, or projects.

Allowed:

- "This is a robust pattern across tested settings."

### Level 5 - Production Evidence

The behavior is release-gated, audited, tested, and recoverable.

Allowed:

- "This is production candidate."

---

## Project-Specific Research Rules

### Boundary-Aware Scheduling

The current strongest research claim is:

> Feedback is not a universal enhancer. It is a boundary-local amplifier.

Required distinction:

- Oracle predictor may use `difficulty` or `bug_type`.
- Runtime predictor must not use `difficulty`, `bug_type`, or final outcome.
- Runtime predictor may use first-pass verifier traces.

Do not confuse oracle evidence with deployable runtime evidence.

Current safe statement:

> In the current SearchLocalSolver setting, runtime trace scheduling can filter solved/above-boundary cases and preserve always-feedback success with fewer feedback calls.

Current limitation:

> The runtime scheduler filters ABOVE well, but still needs a stronger discriminator for NEAR vs BELOW.

### Double Helix

Use Double Helix to study:

- capability boundary
- feedback/retry value
- verifier trace
- scheduler policy

Do not overclaim:

- Do not claim the inverted-U pattern is universal before cross-solver validation.
- Do not claim runtime boundary awareness if labels leak into the classifier.
- Do not claim NEAR detection is solved if BELOW cases are still sent to feedback.

### TopoMem

TopoMem is currently a monitoring and control-signal substrate.

Allowed claim:

- topology may predict instability, domain mixing, or escalation need

Not allowed:

- topology improves reasoning directly
- topology improves retrieval unless tested against vector baseline

### Unified-SEL

Unified-SEL is currently a mechanism laboratory.

Allowed claim:

- it exposes structure/readout/routing interference mechanisms

Not allowed:

- it is a final reasoning architecture
- it statistically beats EWC without enough seeds and clean statistical support

---

## Not-To-Claim Rules

Do not claim:

- runtime boundary awareness if the scheduler reads `difficulty` or `bug_type`
- full NEAR detection if the runtime rule only filters ABOVE
- production readiness because documents exist
- TopoMem improves reasoning if it only improves monitoring
- Unified-SEL solves continual learning from toy benchmark behavior
- a result is general unless it has cross-solver or cross-task evidence

Always separate:

- observed result
- mechanism interpretation
- deployable claim
- speculation

---

## Review Discipline

When reviewing agent output, ask:

1. Did it overclaim?
2. Did it use oracle information while calling itself runtime?
3. Did it create code before defining the claim?
4. Did it expand scope?
5. Did it preserve previous invariants?
6. Did it leave a testable artifact?
7. Did it reduce uncertainty?

If not, request correction before continuing.

---

## Decision Protocol

When asked "what next?", follow this order:

1. If a safety or production blocker exists, fix the smallest release-gate blocker.
2. If a research claim is promising but oracle-only, build runtime evidence.
3. If runtime evidence exists but may be overfit, test cross-setting generalization.
4. If a subsystem is too broad, extract or isolate one bounded family.
5. If no trigger exists, stop and do not create work.

---

## Final Rule

Prefer one sharp falsifiable experiment over ten attractive directions.

A project advances when uncertainty decreases, not when documents increase.

