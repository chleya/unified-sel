# Capability Benchmark Toolkit Plan

## Vision

Transform the capability benchmark from a research artifact into a standalone, reproducible, extensible toolkit for "small model capability boundary detection and scheduling."

This toolkit does NOT prove Unified-SEL's core mechanism. It IS a usable benchmark + routing policy system with genuine engineering value.

---

## Current State

### What Exists

| Component | Status | Location |
|-----------|--------|----------|
| Task generator | Functional | `core/capability_benchmark.py` |
| Solver (SearchLocalSolver) | Functional | `core/capability_benchmark.py` |
| Verifier | Functional | `core/capability_benchmark.py` |
| Monitor library (10 families) | Functional | `core/capability_benchmark.py` |
| Routing policies (3 levels) | Functional | `core/capability_benchmark.py` |
| Result schema (capbench.result.v1) | Functional | Per-experiment JSON files |
| Benchmark suite (code-20 / mixed-40) | Functional | Generated at runtime |
| CLI (capbench) | Functional | `experiments/capability/capbench.py` |
| README | Functional | `experiments/capability/README.md` |
| Benchmark export (JSONL) | Functional | `experiments/capability/export_bench.py` |
| Report generator | Functional | `capbench report` subcommand |
| Public/eval benchmark split | Functional | `data/capability_boundary_bench/` |

### What Is Missing

| Component | Priority | Effort |
|-----------|----------|--------|
| Monitor extension guide | Medium | Low |
| Real LLM solver adapter | Medium | High |
| Regression test suite | Medium | Medium |
| Unified cost model | Medium | Low |
| PyPI packaging | Low | High |

---

## Deliverables

### D1: Standardized Result Schema

**Status: COMPLETE** — schema_version is now `capbench.result.v1` with machine-readable metadata.

Current JSON output includes:

```json
{
  "schema_version": "capbench.result.v1",
  "metadata": {
    "data_source": "verified_execution",
    "cost_model": "abstract_units_v1",
    "oracle_assumption": true,
    "verifier_policy": "monitor_repair_triage",
    "benchmark_suite": "code-20",
    "task_count": 20,
    "seeds": [7],
    "generated_at": "2026-04-16T12:00:00"
  },
  "experiment_id": "string (timestamp-based)",
  "benchmark_config": { ... },
  "solver_config": { ... },
  "routing_config": { ... },
  "results": { ... }
}
```

**Key principle**: Every result file must carry its own `data_source` and `cost_model` tags so that downstream consumers know what can and cannot be claimed.

### D2: Unified CLI

**Status: COMPLETE** — `capbench` CLI with 5 subcommands.

Current CLI:
```
capbench run --suite code --num-tasks 20 --seed 7 --policy monitor_repair_triage --monitor semantic
capbench compare --baseline results/...json --experiment results/...json
capbench report --result results/...json [--output report.md]
capbench list-monitors
capbench list-policies
```

Implementation: `experiments/capability/capbench.py`

### D3: Monitor Library Documentation

Current monitors and their properties:

| Monitor | Signal Source | Strengths | Known Gaps |
|---------|--------------|-----------|------------|
| counterfactual | Repair candidate enumeration | Saturated on all current probes | Requires candidate generation |
| diagnostic | Solver search process | Matches counterfactual | Requires solver-internal metadata |
| semantic | Surface-level code analysis | No solver internals needed | Fails on novel ambiguity families until extended |
| behavioral | Synthesized challenge tests | Answer-only, no bug_type labels | Misses ambiguity without visible-pass failure |
| surface | Output string patterns | No bug_type labels | Same blind spots as behavioral |
| external | Output string rules | Simplest | Fails all ambiguity families |

**Extension guide**: How to add a new monitor family.
1. Implement `monitor_fn(task, solver_output) -> float` in `core/capability_benchmark.py`
2. Register in `MONITOR_REGISTRY`
3. Test on existing canonical probes (code-20 / mixed-40)
4. Report success rate and cost

### D4: Routing Policy Documentation

Current policies:

| Policy | Logic | When to Use |
|--------|-------|-------------|
| monitor_gate | If monitor_signal > threshold: verify, else accept | Simplest baseline |
| monitor_triage | Gate + direct escalation for high-risk tasks | When some tasks are known unrecoverable |
| monitor_repair_triage | Gate + verify-revise for recoverable, escalate for unrecoverable | Best current policy |

**Key insight from benchmark**: Repair-aware triage is the first clear policy improvement. It directly escalates only genuinely unrecoverable tasks (e.g., `dedupe_sorted`) while preserving local verify-plus-revise for recoverable ambiguous tasks.

### D5: Benchmark Suite Versioning

Current suites are generated at runtime from seed + task count. This is reproducible but not named/versioned.

**Target**:
- `code-20`: 20 code-only tasks, canonical reference
- `mixed-40`: 40 mixed tasks (code + reasoning), canonical reference
- Version pinning: suite definition includes task list hash
- Future suites: `code-25`, `mixed-50`, etc. only when genuinely new ambiguity families are added

### D6: README

Content:
1. What this toolkit is (capability boundary detection + routing benchmark)
2. What it is NOT (not a proof of surprise-driven structural birth/death)
3. Quick start: install, run first experiment, compare results
4. Monitor guide: which monitor to use when
5. Policy guide: which policy to use when
6. Extension guide: how to add monitors, policies, task families
7. Limitations: synthetic solver, assumed cost model, no real LLM validation

---

## Implementation Order

### Phase 1 (COMPLETE): Schema + CLI + README

1. ✅ Add `schema_version` and `metadata` fields to result JSON output
2. ✅ Implement `capbench` CLI wrapper
3. ✅ Write README.md
4. ✅ Implement `capbench compare` subcommand
5. ✅ Implement `capbench report` subcommand
6. ✅ Export benchmark to JSONL (public/eval modes)
7. ✅ Public/private benchmark split

### Phase 2 (Short-term): Documentation + Testing

4. Write monitor extension guide
5. Write policy documentation
6. Add regression tests for canonical probe results
7. Implement `capbench compare` subcommand

### Phase 3 (Medium-term): Real LLM Integration

8. Implement LLMAdapter for Qwen2.5-0.5B
9. Re-run canonical probes with real LLM
10. Compare synthetic vs real LLM results

### Phase 4 (Long-term): Packaging

11. PyPI packaging
12. Continuous integration
13. Community contribution guide

---

## Canonical References (Current)

These are the result files that define the current state of the art:

| Probe | Policy | Monitor | Success | Cost | Reference |
|-------|--------|---------|---------|------|-----------|
| code-20 | monitor_repair_triage | semantic | 1.0 | 1.59 | 20260410_171603.json |
| mixed-40 | monitor_repair_triage | semantic | 1.0 | 1.295 | 20260410_171559.json |
| code-20 | monitor_repair_triage | counterfactual | 1.0 | 1.59 | (from code-19 round) |
| mixed-40 | monitor_repair_triage | counterfactual | 1.0 | 1.295 | (from code-19 round) |

---

## Limitations (Must Be Documented)

1. **Synthetic solver**: SearchLocalSolver is deterministic and rule-based. Results may not transfer to real LLMs.
2. **Assumed cost model**: Cost numbers use hardcoded abstract units (1.0/1.2/5.3 or 1.0/1.5/2.0). They are NOT real latency measurements.
3. **Single task domain**: All tasks are code repair with controlled bug types. Generalization to other domains is untested.
4. **Monitor overfitting risk**: Semantic monitor has been extended 8 times to close gaps. Each extension is tested on the same probe that exposed the gap. This creates a risk of benchmark-driven development.
5. **No real escalation**: Escalation path uses OracleSolver (returns expected_answer). Real escalation success rate would be <100%.
6. **Small sample sizes**: Most experiments use single runs, not multi-seed with confidence intervals.

---

## Success Criteria

The toolkit is "done enough" when:

1. A new user can `pip install capbench && capbench run --suite code-20 --policy monitor_repair_triage --monitor semantic` and get reproducible results
2. Every result file carries `data_source` and `cost_model` metadata
3. The README clearly states what can and cannot be claimed from the results
4. A researcher can add a new monitor family in <1 hour by following the extension guide
