# MAINLINE_EXECUTION_PLAN_2026-04-11

**Date**: 2026-04-11  
**Status**: Active execution plan (planning-only)  
**Mode**: You lead strategy; minimax agents implement; this document defines scope, cadence, and acceptance gates.

---

## 1. Project North Star (30-day focus)

Primary question for the next 30 days:

> Can a non-enumerative semantic surface signal, without candidate enumeration, stably drive a cheaper, interpretable, and transferable routing policy?

This plan enforces a shift from internal closure wins to external validity and policy value.

---

## 2. Priority and Branching

Priority order for execution:

1. Capability routing generalization (mainline A-track)
2. Cost-quality Pareto and policy upgrade (mainline A-track)
3. Minimal deployable shell validation (mainline A-track)
4. `weight_graph` fragility-prior branch (W-track sidecar)

Resource split guideline:

- A-track: 80%
- W-track: 20%

Rule:

- W-track must not block A-track milestones.

---

## 3. Non-Goals (frozen for this 30-day cycle)

- No new ambiguity family by default
- No repeated manual patching for synthetic benchmark point wins (e.g. 0.95 -> 1.0)
- No `weight_graph` promotion to mainline before practical fragility or monitoring value
- No paper-packaging-first workflow

Exception gate for adding a new family:

- Creates a real failure on frozen `code-20`/`mixed-40`
- Represents a genuinely new semantic regime
- Not a lexical restatement of an existing closed regime

---

## 4. Execution Calendar (Day 1-30)

### Week 1 (2026-04-12 to 2026-04-18): External Validity

Objective:

- Break or validate the current claim outside the in-loop benchmark comfort zone.

Work packages (A-track):

1. A1 Protocol hygiene
2. A2 Stronger paraphrase variant
3. A3 Naturalized code-repair set
4. A4 Transfer matrix runner and summarizer

Expected output:

- Transfer matrix over:
  - Domains: `standard`, `paraphrase`, `stronger_paraphrase`, `naturalized`
  - Monitors: `semantic`, `semantic+guard`, `counterfactual`, `confidence`
  - Solvers: `search`, `heuristic`

Stop/Go:

- Go if `semantic+guard` remains near `counterfactual` on external domains.
- Hold if large degradation appears only outside in-distribution domains.
- No-Go if performance collapses broadly once domain shifts are introduced.

### Week 2 (2026-04-19 to 2026-04-25): Pareto Curve

Objective:

- Convert “good or bad” into “worth or not worth”.

Work packages (A-track):

1. A5 Sweep `low_signal_guard_band`
2. A6 Pareto report generator
3. A7 Multi-seed stability run

Required sweep:

- `low_signal_guard_band`: 0.00 to 0.30
- Step: 0.02

Stop/Go:

- Promote if `semantic+guard` has near-frontier or frontier behavior against `counterfactual`.
- Hold if quality gain requires cost close to `counterfactual` with no clear advantage.

### Week 3 (2026-04-26 to 2026-05-02): Minimal Hybrid Router

Objective:

- Move from monitor comparison to policy composition.

Work packages (A-track):

1. A8 Deterministic hybrid policy
2. A9 Correction-path concentration analysis
3. A10 Ablation of hybrid components

Hybrid v1 design scope:

- Prior: semantic surface signal
- Correction: confidence + verifier outcome + routing signal tier logic
- No learned policy in this cycle; deterministic only

Stop/Go:

- Go if hybrid outperforms any single signal on success-cost tradeoff.
- Hold if gains are unstable across domains or seeds.

### Week 4 (2026-05-03 to 2026-05-09): Minimal Real Shell

Objective:

- Validate that trend survives outside experiment scripts.

Work packages (A-track):

1. A11 Integrate a minimal real shell
2. A12 Collect routing logs and cost-quality stats
3. A13 Compare shell vs benchmark trend consistency

Candidate shell forms:

- Local code-fix gateway
- Small agent router
- Local-vs-cloud repair demo

Stop/Go:

- Go if benchmark trend direction remains consistent in shell traffic.
- No-Go if shell behavior disconnects from benchmark behavior.

### Day 29-30 (2026-05-10 to 2026-05-11): Program Decision

Deliverable:

- 30-day decision note with explicit Go/Hold/No-Go and next-cycle scope.

---

## 5. `weight_graph` Sidecar Branch (W-track)

Purpose:

- Test whether static topology can act as a low-cost fragility prior for intervention, monitoring, and triage.

Scope rule:

- Sidecar only in this cycle; no mainline promotion without practical fragility value.

### W-Week 1: Reframe and Activation Close-Out

1. W1 Freeze the downgrade of `Topo x capability` as descriptive-only
2. W2 Activation-profile task design for `#490`
3. W3 Control selection and reporting contract

### W-Week 2: Bounded Mechanistic Follow-Up

1. W4 Run `#490` activation/profile analysis
2. W5 Compare `#490` vs same-layer high-PageRank control
3. W6 Compare `#490` vs random control

### W-Week 3: Fragility Utility Check

1. W7 Rank intervention-sensitive candidates using topology
2. W8 Test whether ranking beats random intervention choice
3. W9 Summarize practical monitoring/intervention value

### W-Week 4: Branch Close-Out

1. W10 Write observation-only fallback if no fragility signal survives
2. W11 Write fragility-prior note if signal is usable
3. W12 Final branch decision memo

W-track decision gate:

- Promote only if topology helps find intervention-sensitive or monitoring-relevant structure.
- Otherwise downgrade to descriptive side evidence.

---

## 6. Hard Acceptance Gates (must be explicit)

### A-track gates

1. External validity gate
- On `stronger_paraphrase + naturalized`, `semantic+guard` success gap vs `counterfactual` <= 3pp.

2. Pareto gate
- `semantic+guard` must show lower or comparable mean cost at near-equal success on at least 2 shifted domains.

3. Stability gate
- Multi-seed (>=3) results preserve ordering direction; no single-seed-only claim.

4. Policy gate
- Hybrid must beat best single-signal baseline on at least one locked evaluation setting without regressions on major domains.

### W-track gates

1. Fragility value gate
- Topology-based ranking must beat random or naive intervention choice.

2. Robustness gate
- Signal must remain distinct from same-layer high-PageRank and random controls.

3. Utility gate
- The branch must show practical intervention or monitoring value; otherwise no promotion.

---

## 7. Required Weekly Artifacts

Each week must produce:

1. Raw result JSON files
2. One normalized summary table (same schema every week)
3. One decision note with Go/Hold/No-Go
4. One next-week execution delta list

Output folders:

- `results/capability_generalization/`
- `results/capability_pareto/`
- `results/capability_hybrid/`
- `results/weight_graph_prior/`
- `results/decision_notes/`

---

## 8. Minimax Work Allocation Template

Use this template per task package:

- Owner: minimax-agent-id
- Track: A or W
- Input: exact scripts/files/params
- Output: exact JSON paths + summary markdown path
- Acceptance: numeric threshold(s)
- Risk: top 1-2 known failure modes
- ETA: expected completion date

No task is complete without all of:

- reproducible command
- saved result artifact
- threshold-based pass/fail statement

---

## 9. Immediate Assignment List (Day 1)

A-track first:

1. A1 Protocol hygiene patch
2. A2 `stronger_paraphrase` variant implementation
3. A3 `naturalized` benchmark slice implementation
4. A4 transfer-matrix summarizer

Then W-track sidecar:

5. W1 depth-profile precompute
6. W2 Task4 eval harness with fixed thresholds

---

## 10. Decision Rule at Day 30

Ask only these 4 questions:

1. Does the signal hold outside the original benchmark loop?
2. Does it have a clear Pareto advantage?
3. Does hybrid policy add stable value over single-signal routing?
4. Does the sidecar branch show practical fragility or monitoring value?

Program decision:

- 3/4 yes: continue and scale this line
- 2/4 yes: continue narrowly with strict scope reduction
- <=1 yes: downgrade claim, reframe as benchmark-specific or descriptive result

---

## 11. One-Sentence Program Summary

For this cycle, success means proving that semantic boundary sensing is externally valid, policy-useful, and deployable enough to survive beyond a synthetic benchmark loop, while `weight_graph` is judged strictly by fragility and intervention value rather than descriptive novelty.

