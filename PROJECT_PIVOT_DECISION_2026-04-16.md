# PROJECT PIVOT DECISION — 2026-04-16

## Decision

The project's original core hypothesis is **downgraded from "main result" to "unverified mechanism hypothesis"**.
Two new main lines replace it.

---

## Old Core Hypothesis (DOWNGRADED)

> Surprise-driven structural birth/death > methods requiring explicit task boundaries (e.g., EWC).

### Why It Must Be Downgraded

1. **Toy problem failure**: After fixing surprise to incorporate prediction error, Unified-SEL still cannot beat EWC (avg_acc 0.5000 vs EWC 0.5005, p=0.9484). The 4-dim linear-separable binary classification problem is too simple to expose structural pool advantages.

2. **No statistical advantage**: CLAIM_EVIDENCE_MAP already states "Do not claim statistical superiority over EWC yet." This has been true since 2026-04-09 and remains true.

3. **Platform limitation, not logical impossibility**: The hypothesis is not logically unfalsifiable. It is unverifiable under the current experimental paradigm — synthetic deterministic solver + synthetic task generator. Verifying it would require real LLM solvers, multi-task families, many seeds, strict holdout, and statistical testing, at significantly higher cost and uncertain effect size.

4. **Phase G-H evidence chain collapsed**: `patch_size` is a perfect fingerprint of `bug_type`, making Phase G's "first-pass only perfect classification" a lookup table, not boundary detection. Phase H's "cross-solver generalization" used the same task pool.

### What Remains Valid

- Surprise computation fix (now incorporates prediction error) — a genuine engineering improvement
- Structure pool architecture — functional, but not proven superior
- The mechanism itself (reinforce/branch/create) — not disproven, just not proven better than EWC

---

## New Main Line 1: Paper Line

> **Feedback retry is a boundary-local amplifier, not a universal enhancer.**

### What This Means

External verification and feedback chains concentrate their benefit in the NEAR-boundary zone. ABOVE-zone tasks waste feedback calls. BELOW-zone tasks cannot use feedback at all. This inverted-U pattern has real empirical support.

### Evidence Status

| Claim | Evidence | Status |
|-------|----------|--------|
| Boundary-local amplification exists | Phase A: p=0.0008 | SAFE |
| ABOVE-zone filtering saves feedback calls | Phase E: 54.4% reduction | SAFE |
| NEAR/BELOW discrimination has signal | ROC AUC=0.769 | WEAK |
| NEAR/BELOW discrimination is deployable | Below Filtered=46.2% ± 41.2% | FAIL |

### Paper Scope

A short paper, not a systems paper. Core contributions:
1. The inverted-U finding itself (negative result + positive characterization)
2. ABOVE-zone filtering as a practical scheduling insight
3. Artifact audit showing why NEAR/BELOW discrimination failed (patch_size = bug_type fingerprint)
4. Honest reporting of what does NOT work

### What This Paper Does NOT Claim

- Does not claim surprise-driven routing is superior
- Does not claim NEAR/BELOW discrimination is solved
- Does not claim cross-solver generalization
- Does not claim cost reduction from assumed cost models

---

## New Main Line 2: Tool Line

> **Capability benchmark + routing policy toolkit.**

### What This Means

The semantic monitor / counterfactual monitor / repair-aware triage system has genuine engineering value as a "small model capability boundary detection and scheduling benchmark." It does not prove Unified-SEL's core mechanism, but it is a reproducible, extensible tool.

### Deliverables

1. **Benchmark scaffold**: code-20 / mixed-40 task suite with canonical references
2. **Monitor library**: semantic, counterfactual, behavioral, surface, diagnostic, external
3. **Routing policies**: monitor_gate, monitor_triage, monitor_repair_triage
4. **Result schema**: standardized JSON output for all experiments
5. **CLI**: reproducible experiment execution
6. **Documentation**: README, usage guide, extension guide

### Why This Is Valuable Independently

- The benchmark cleanly separates local-only, verification-assisted, and escalation-assisted behavior
- The monitor ranking (counterfactual > semantic > behavioral > surface > external) is reproducible
- The repair-aware triage policy is a genuine policy improvement over naive gate or naive triage
- The benchmark has been stress-tested across 8+ ambiguity families

### What This Toolkit Does NOT Claim

- Does not claim the monitors generalize to real LLMs (currently synthetic solver)
- Does not claim routing policies are optimal
- Does not claim cost numbers are real (they use assumed cost models)

---

## TopoMem Role Downgrade (2026-04-16)

**TopoMem is downgraded from routing-core candidate to deployment-health candidate.**

### Evidence

TopoMem surprise signal was tested as a per-task routing monitor on code-20 / mixed-40:
- **topo_surprise standalone**: FAIL — success_rate 0.7/0.85 vs baseline 1.0
- **topo_semantic_fusion**: No advantage — success recovered but cost increased

### Structural Reason

```
TopoMem surprise measures:  "How novel is this task in embedding space?"
Routing monitor needs:      "How likely is the solver's answer to be wrong?"
These are orthogonal.
```

- Similar tasks can be answered incorrectly (low surprise ≠ correct)
- Novel tasks can be answered correctly (high surprise ≠ wrong)
- In code tasks, answer correctness is determined by semantic traps, visible-pass ambiguity, test coverage gaps — exactly what semantic monitor detects

This is a structural mismatch, not a threshold problem. No parametric adjustment can fix it.

### Implications

- **Capability Router remains primary product track** — its core is semantic/counterfactual/repair-triage
- **TopoMem does not enter the routing core**
- **TopoMem's appropriate role is deployment health monitoring (OBD)**: "Is this deployment environment degrading?" not "Is this answer correct?"
- **Self-Aware LLM remains future narrative**, not current build target — it requires both capability-router and topomem-obd to independently accumulate stable evidence first

### Product Structure

```
capability-router/              ← primary product
├── monitors/
│   ├── semantic_monitor.py     # primary
│   ├── counterfactual_monitor.py  # reference
│   └── repair_triage.py        # policy layer
├── router.py
├── benchmark.py
└── reports.py

topomem-obd/                    ← secondary, separate validation needed
├── health_monitor.py           # H1/H2/drift
├── drift_detector.py
├── fault_codes.py
└── batch_dashboard.py
```

Do not merge these into "self-aware-llm" until both independently have stable evidence.

See: `TOPOMEM_ROUTING_MONITOR_RESULT_2026-04-16.md`

---

## What Changes Concretely

| Before Pivot | After Pivot |
|---|---|
| Main goal: beat EWC | Main goal: publish boundary-local amplification finding |
| Surprise-driven routing as core narrative | Boundary-local amplifier as core narrative |
| Unified-SEL mechanism as main result | Mechanism as unverified hypothesis (future work) |
| Capability benchmark as supporting evidence | Capability benchmark as independent tool product |
| Continue synthetic solver experiments | Stop synthetic solver experiments for hypothesis testing |
| "31% cost reduction" as headline | Cost numbers clearly marked as assumed models |
| TopoMem as routing-core candidate | TopoMem as deployment-health candidate (OBD) |
| Self-Aware LLM as near-term build target | Self-Aware LLM as future narrative (after both tracks independently validate) |

---

## What Does NOT Change

- AGENTS.md red-line rules remain in force
- Smoke test requirement remains in force
- EXPERIMENT_LOG.md updates remain mandatory
- Do not modify F:\sel-lab, F:\SDAS, F:\fcrs_mis
- Do not manually edit results/ JSON files

---

## Future Work (Explicitly Deferred)

These are NOT current tasks. They are recorded here so future agents know the landscape.

1. **Verify core hypothesis with real LLMs**: Replace synthetic solver with Qwen2.5-0.5B/1.5B, test on multi-task sequences, 10+ seeds
2. **Extend to complex tasks**: 5+ task sequences, nonlinear decision boundaries, high-dimensional input
3. **Bayesian surprise**: Replace heuristic surprise with KL-divergence of posterior updates
4. **Information bottleneck**: Use I(X;Z) and I(Z;Y) to evaluate Structure quality
5. **Causal inference**: Move from correlation-based routing to do-calculus-based routing
6. **TopoMem signal integration**: ~~Connect TopoMem surprise/tension to capability_benchmark routing~~ → TopoMem surprise rejected as routing monitor. H1/H2 may be validated as deployment health (OBD) signal in separate experiment.

---

## Decision Authority

This pivot was decided on 2026-04-16 based on:
- Toy problem failure (avg_acc 0.5000 vs EWC 0.5005, p=0.9484)
- Phase G-H evidence chain collapse (patch_size = bug_type fingerprint)
- CLAIM_EVIDENCE_MAP's own "Do not claim statistical superiority over EWC yet"
- The genuine, reproducible boundary-local amplification finding (p=0.0008)
- The functional capability benchmark toolkit with 8+ ambiguity families

This is not a failure. It is discovering that the original hypothesis cannot serve as the main result, while simultaneously producing a more reliable research finding and a usable tool direction.
