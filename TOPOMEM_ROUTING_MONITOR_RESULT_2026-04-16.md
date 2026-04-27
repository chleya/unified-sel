# TopoMem Routing Monitor Result — 2026-04-16

## Verdict

**TopoMem surprise is rejected as a standalone per-task routing monitor on the frozen code-20 / mixed-40 probe.**

**TopoMem is not useless.** It cannot serve as a per-task answer reliability signal. It remains a candidate for deployment health monitoring (OBD), not routing core.

---

## Experiment Summary

### Experiment 1: TopoSurpriseRoutingMonitor (standalone)

| Suite | success_rate | mean_cost | revision_rate | accept_wo_verify |
|-------|-------------|-----------|---------------|-----------------|
| code-20 | 0.7 | 1.06 | 0.10 | 0.85 |
| mixed-40 | 0.85 | 1.035 | 0.05 | 0.90 |
| **semantic baseline** | **1.0** | **1.375** | **0.65** | **0.10** |
| **semantic baseline** | **1.0** | **1.1875** | **0.325** | **0.55** |

**Result: ❌ FAIL** — success_rate dropped by 0.15-0.30 below baseline.

### Experiment 2: TopoSemanticFusionMonitor (30% topo + 70% semantic)

| Suite | success_rate | mean_cost | revision_rate | accept_wo_verify |
|-------|-------------|-----------|---------------|-----------------|
| code-20 | 1.0 | 1.435 | 0.85 | 0.10 |
| mixed-40 | 1.0 | 1.225 | 0.45 | 0.55 |
| **semantic baseline** | **1.0** | **1.375** | **0.65** | **0.10** |
| **semantic baseline** | **1.0** | **1.1875** | **0.325** | **0.55** |

**Result: ⚠️ No advantage** — success recovered (carried by semantic), but cost increased by 0.04-0.06.

---

## Structural Explanation

The failure is structural, not parametric. No threshold tuning or weight adjustment can fix it.

### The Fundamental Mismatch

```
TopoMem surprise measures:  "How novel is this task's prompt in embedding space?"
Routing monitor needs:      "How likely is the solver's answer to be wrong?"
```

These two quantities are **orthogonal**:

- **Similar tasks can be answered incorrectly.** A task whose prompt is nearly identical to previously seen tasks (low surprise) may still have a wrong answer due to semantic traps, visible-pass ambiguity, or test coverage gaps.
- **Novel tasks can be answered correctly.** A task whose prompt is very different from anything seen before (high surprise) may be trivially solvable.
- **In code tasks, answer correctness is primarily determined by:** semantic ambiguity, visible-pass ambiguity, test coverage gaps — exactly what the semantic monitor detects.

### Why Threshold Tuning Cannot Fix This

The surprise signal has mean ~0.23-0.31 across code-20/mixed-40. This is because most code tasks in the benchmark share structural similarities (function definitions, parameter lists, return statements). The signal has very low variance and does not discriminate between correct and incorrect answers.

Raising the threshold would send more tasks to verification, but this would be indiscriminate — it would verify novel-but-correct tasks while still missing similar-but-wrong tasks. Lowering the threshold would accept more without verification, increasing false accepts.

The problem is not where the threshold is set. The problem is that the signal does not carry the information needed for the decision.

### Why Fusion Does Not Help

Adding 30% topo signal to 70% semantic signal injects noise. The topo component pushes some tasks' signals up or down based on their embedding novelty, which is uncorrelated with answer correctness. This causes:
- Some correct tasks to get unnecessarily verified (cost increase)
- Some incorrect tasks to get their signal diluted (risk, though partially masked by semantic)

The fusion result confirms this: success_rate recovers to 1.0 (semantic carries), but cost increases.

---

## Leakage Check

| Check | Result | Pass? |
|-------|--------|-------|
| surprise ↔ bug_type (Cramér's V) | ≈ 0 | ✅ No leakage |
| surprise ↔ boundary_label (point-biserial r) | ≈ 0 | ✅ No leakage |

The signal is clean but uninformative. It does not cheat, but it also does not help.

---

## Product Conclusion

### What This Means for Capability Router

**Capability Router remains primary product track.** Its core is:

```
capability-router/
├── monitors/
│   ├── semantic_monitor.py        # primary
│   ├── counterfactual_monitor.py  # reference
│   └── repair_triage.py           # policy layer
├── router.py                      # accept / verify / escalate
├── benchmark.py                   # code-20 / mixed-40 / future probes
└── reports.py
```

TopoMem does **not** enter the routing core.

### What This Means for TopoMem

**TopoMem is downgraded from routing-core candidate to deployment-health candidate.**

Its appropriate role is not "should this answer be verified?" but "is this deployment environment degrading?"

```
topomem-obd/
├── health_monitor.py              # H1/H2/drift
├── drift_detector.py
├── fault_codes.py
└── batch_dashboard.py
```

The right question for TopoMem is not "is this answer correct?" but:
- Is the current input distribution drifting from historical distribution?
- Is the current embedding manifold fragmenting?
- Should the system as a whole enter conservative mode?

### What This Means for Self-Aware LLM

**Self-Aware LLM remains future narrative, not current build target.**

The original vision was: Self-Aware LLM = TopoMem + semantic + router. This cannot be directly claimed because TopoMem's per-task signal does not contribute to routing.

If both capability-router and topomem-obd independently accumulate stable evidence, they can be merged into self-aware-llm later. But not now.

---

## Forbidden Actions

- ❌ Do not tune topo_surprise threshold
- ❌ Do not adjust fusion weights
- ❌ Do not run new routing experiments with surprise signal
- ❌ Do not claim TopoMem is useless
- ❌ Do not claim SelfAwareAgent is validated
- ❌ Do not claim H1/H2 will work as routing signals (untested)

---

## Next Step (If TopoMem Validation Continues)

The next validation target should be **TopoMem OBD Drift Validation**, not routing monitor.

**Goal**: Verify whether H1/H2/drift_score can predict batch-level failure rate or verifier/escalation pressure.

**NOT**: Whether H1/H2 can predict individual task answer correctness.

**Design sketch**:
1. Run code-20 / mixed-40 with a distribution shift injected mid-batch
2. Track H1/H2/drift_score across the shift boundary
3. Check whether H1/H2 changes correlate with batch-level success_rate changes
4. If yes: TopoMem OBD has a validated use case
5. If no: TopoMem remains research artifact without product path

This is a separate experiment with a separate evaluation framework. It should not be mixed with routing monitor experiments.

---

## Data Source

All numbers from real experimental verification (SearchLocalSolver, code-20/mixed-40, seed=7).

Cost units based on assumed cost model (hardcoded 1.0/1.2/1.5/2.2). NOT real latency measurements.

Result files:
- `results/capability_benchmark/20260416_043017.json` (topo_surprise, code-20, monitor_gate)
- `results/capability_benchmark/20260416_043026.json` (semantic, code-20, monitor_gate)
- `results/capability_benchmark/20260416_043237.json` (topo_semantic_fusion, code-20, monitor_gate)
- `results/capability_benchmark/20260416_043313.json` (topo_surprise, code-20, monitor_repair_triage)
- `results/capability_benchmark/20260416_043322.json` (semantic, code-20, monitor_repair_triage)
- `results/capability_benchmark/20260416_043401.json` (topo_semantic_fusion, code-20, monitor_repair_triage)
- `results/capability_benchmark/20260416_043437.json` (topo_surprise, mixed-40, monitor_repair_triage)
- `results/capability_benchmark/20260416_043447.json` (semantic, mixed-40, monitor_repair_triage)
- `results/capability_benchmark/20260416_043550.json` (topo_semantic_fusion, mixed-40, monitor_repair_triage)
