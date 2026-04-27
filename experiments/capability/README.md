# Capability Router — Research Benchmark

> **Purpose**: A controlled benchmark for studying accept/verify/escalate routing decisions in LLM systems.
>
> **Scope**: Research tool using synthetic deterministic solvers. Not a production system.

---

## What This Is

Capability Router evaluates whether a model's output is reliable enough to accept directly, or whether it needs verification or escalation to a stronger model. It provides:

- **Benchmark suites**: code-20 (code-only) and mixed-40 (code + reasoning) with controlled difficulty
- **Routing monitors**: Multiple monitor families that compute routing signals from task properties and solver output
- **Routing policies**: Protocols that map routing signals to accept/verify/escalate decisions
- **Result schema**: Versioned JSON output with self-documenting metadata

### Relationship to Main Research

This benchmark is the **tool line** of the project, supporting the **paper line** (Boundary-Local Amplification) by providing:

1. A controlled environment to test routing signals
2. Reproducible baselines for accept/verify/escalate policies
3. Infrastructure for the inverted-U pattern validation

The benchmark itself is not the contribution — the contribution is the **Granularity Alignment Principle** and the **Boundary-Local Amplification** finding that this benchmark helped validate.

---

## What This Is NOT

- ❌ **Not a proof of surprise-driven structural birth/death** — that hypothesis was archived (p = 0.9484)
- ❌ **Not validated on real LLMs** — currently uses synthetic deterministic solvers
- ❌ **Not a production routing service** — cost numbers use assumed abstract models, not real latency
- ❌ **Not a claim of 100% escalation success** — escalation path uses OracleSolver (returns expected_answer directly), so success is by assumption

---

## Quick Start

```bash
cd F:\unified-sel

# List available monitors and policies
python experiments/capability/capbench.py list-monitors
python experiments/capability/capbench.py list-policies

# Run a benchmark
python experiments/capability/capbench.py run \
  --suite mixed \
  --protocol monitor_repair_triage \
  --routing-monitor semantic \
  --num-tasks 40 \
  --seed 7

# Compare two results
python experiments/capability/capbench.py compare \
  results/capability_benchmark/baseline.json \
  results/capability_benchmark/experiment.json
```

---

## Current Best Results

On frozen benchmarks with `monitor_repair_triage` protocol:

| Suite | Monitor | success_rate | mean_cost |
|-------|---------|-------------|-----------|
| code-20 | semantic | 1.0 | 1.375 |
| mixed-40 | semantic | 1.0 | 1.1875 |

---

## Monitor Guide

| Monitor | When to Use | Notes |
|---------|------------|-------|
| semantic | Primary choice for code tasks | Combines surface heuristics + ambiguity detection + probe tests |
| counterfactual | Reference/baseline | Strongest signal but requires candidate enumeration |
| diagnostic | When solver-internal metadata available | Matches counterfactual quality |
| external | Quick baseline, no code execution | Surface heuristics only |
| behavioral | Answer-only scenarios | Misses ambiguity without visible-pass failure |
| surface | Simplest, string-pattern only | Fails all ambiguity families |
| confidence | When only solver confidence available | Weakest standalone |
| topo_surprise | **DO NOT USE for routing** | Rejected: embedding novelty != answer correctness |
| topo_semantic_fusion | **DO NOT USE for routing** | No advantage over semantic-only |

### Why topo_surprise Was Rejected

TopoMem's embedding novelty signal was tested as a per-task routing monitor and **rejected** (success 0.7/0.85 vs baseline 1.0). The structural reason: **embedding novelty does not predict answer correctness**. This is not a threshold problem — it is a structural mismatch that no parametric adjustment can fix.

This is a key example of the **Granularity Alignment Principle**: embedding novelty measures *input distribution shift* (batch-level), not *answer correctness* (per-task). It works for batch-level deployment health monitoring (confirmed: centroid drift 27.2× separation, p ≈ 0), but not for per-task routing.

---

## Batch Health (OBD)

Every `capbench run` result includes a `batch_health` field from the BatchHealthMonitor. This is a **batch-level** drift detector (not per-task), based on TopoMem's embedding infrastructure.

| Signal | Meaning |
|--------|---------|
| `status` | `healthy` or `drift_detected` |
| `half_split_drift` | Cosine distance between first-half and second-half task centroids |
| `mean_pairwise_similarity` | Average similarity between consecutive tasks |

Drift thresholds (based on 10-seed validation):
- Domain shift (code→reasoning): centroid drift ~0.71, 27.2× vs control
- Gradual shift (trivial→harder): centroid drift ~0.09, 4.1× vs control
- Control (same domain): centroid drift ~0.04

This monitor answers **"has the task distribution shifted?"** not **"is this specific task surprising?"** — the latter was rejected as a routing signal.

---

## Policy Guide

| Policy | When to Use |
|--------|------------|
| monitor_repair_triage | Best current policy. 3-tier with repairability check. |
| monitor_no_revision_triage | For weak revisers: 3-tier policy that verifies but escalates instead of revising. |
| monitor_triage | 3-tier without repairability check. |
| monitor_gate | Simple 2-tier: accept or verify. |
| local_only | Baseline: always accept. |
| local_verify | Baseline: always verify. |

---

## Result Schema

Every result JSON includes:

```json
{
  "schema_version": "capbench.result.v1",
  "metadata": {
    "data_source": "verified_execution",
    "cost_model": "abstract_units_v1",
    "oracle_assumption": true,
    "verifier_policy": "monitor_repair_triage",
    "benchmark_suite": "mixed-40",
    "task_count": 40,
    "seeds": [7],
    "generated_at": "2026-04-16T12:00:00"
  },
  "suite": "mixed",
  "protocol": "monitor_repair_triage",
  "routing_monitor_name": "semantic",
  "summary": {
    "success_rate": 1.0,
    "mean_cost_units": 1.1875,
    "escalation_rate": 0.0,
    "revision_rate": 0.325,
    "verifier_rate": 0.45,
    "direct_escalation_rate": 0.0,
    "accepted_without_verifier_rate": 0.55
  }
}
```

---

## Adding a New Monitor

1. Subclass `RoutingMonitor` in `core/capability_benchmark.py`:

```python
class MyRoutingMonitor(RoutingMonitor):
    name = "my_monitor"
    def score(self, task: BenchmarkTask, attempt: SolverAttempt) -> float:
        return 0.5  # your signal in [0.0, 1.0]
```

2. Register in `build_routing_monitor()`:

```python
if name == "my_monitor":
    return MyRoutingMonitor()
```

3. Add to `capbench.py` MONITORS dict and argparse choices.

4. Test on code-20 / mixed-40 and compare with semantic baseline.

---

## Limitations

1. **Synthetic solver**: SearchLocalSolver is deterministic and rule-based. Results may not transfer to real LLMs.
2. **Assumed cost model**: Cost numbers use hardcoded abstract units. NOT real latency.
3. **Single task domain**: All tasks are code repair with controlled bug types.
4. **Oracle escalation**: Escalation success rate is 100% by assumption.
5. **Small sample sizes**: Most experiments use single runs, not multi-seed with confidence intervals.
6. **Monitor overfitting risk**: Semantic monitor has been extended multiple times on the same probe set.

---

## Citation

If you use this benchmark, please cite the main project:

```bibtex
@misc{unified_sel_2026,
  title={Granularity-Aligned Metacognition: Why Embedding Novelty Predicts Distribution Shift but Not Answer Correctness},
  author={[Authors]},
  year={2026},
  note={Research codebase for boundary-local amplification and capability routing}
}
```
