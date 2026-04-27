# Granularity-Aligned Metacognition for LLMs

> **Research Question**: When should an LLM trust its own output, verify it, or escalate to a stronger model?
>
> **Core Finding**: Feedback retry is a boundary-local amplifier — it produces an inverted-U benefit pattern concentrated in the "near-boundary" zone (p = 0.0008).
>
> **Methodological Contribution**: The Granularity Alignment Principle — a signal's predictive power depends on whether its measurement granularity matches the decision granularity.

---

## What This Project Is

A research codebase investigating **metacognitive monitoring** for LLMs — specifically, how a system can evaluate its own capability boundaries and make routing decisions (accept / verify / escalate) based on internal signals.

### Two Independent Contributions

| Track | Status | Description |
|-------|--------|-------------|
| **Boundary-Local Amplification** | ✅ Published-quality finding | Feedback retry benefit follows inverted-U curve; ABOVE-filtering saves 54.4% of feedback calls |
| **Capability Router** | ✅ Research tool | Benchmark + routing system with 10 monitors and 3 policy layers for studying accept/verify/escalate decisions |
| TopoMem OBD | ✅ Confirmed (separate) | Batch-level deployment health monitoring via embedding drift detection |

### What This Project Is NOT

- ❌ **Not a "self-aware" or "conscious" AI** — we study monitoring signals, not consciousness
- ❌ **Not a continual learning system that beats EWC** — toy-problem comparison was inconclusive (p = 0.9484) and archived
- ❌ **Not a production routing service** — the Capability Router is a research benchmark using synthetic tasks

---

## Repository Structure

```
unified-sel/
  core/
    capability_benchmark.py    # Main benchmark engine (monitors + policies)
    runtime.py                 # Path management, save/load
    structure.py               # Archived: DFA structure unit (kept for smoke test compatibility)
    pool.py                    # Archived: StructurePool lifecycle (kept for smoke test compatibility)
    learner.py                 # Archived: DFA learner (kept for smoke test compatibility)
  experiments/
    capability/
      capbench.py              # CLI: run / compare / report / list-monitors / list-policies
      export_bench.py          # Export benchmark to JSONL (public/eval modes)
      benchmark.py             # Benchmark runner
      validate_above_filtering.py  # Validation experiments
      README.md                # Capability Router user guide
  data/
    capability_boundary_bench/ # JSONL benchmark files
  topomem/                   # TopoMem subsystem (batch-level health monitoring only)
  archive/                   # Archived experiments and documents
    experiments_continual/   # Old mechanism-track experiments (EWC comparison, etc.)
    meta_controller/         # Deprecated meta-controller experiments
    cep_cc/                  # Deprecated CEP-CC experiments
    weight_graph/            # Deprecated weight analysis experiments
    lsg_deprecated/          # Deprecated learned-state-governance experiments
  tests/
    smoke_test.py            # Quick validation
  papers/
    boundary_local_amplification_draft.md  # Paper draft
```

---

## Quick Start

### Run the Capability Benchmark

```bash
# List available monitors and policies
python experiments/capability/capbench.py list-monitors
python experiments/capability/capbench.py list-policies

# Run a benchmark
python experiments/capability/capbench.py run \
  --suite code-20 \
  --protocol monitor_repair_triage \
  --monitor semantic \
  --seed 7

# Compare two configurations
python experiments/capability/capbench.py compare \
  results/capability_benchmark/run_a.json \
  results/capability_benchmark/run_b.json
```

### Run Smoke Tests

```bash
python tests/smoke_test.py
```

---

## Core Concepts

### The Granularity Alignment Principle

> **A signal's usefulness depends on whether its measurement granularity matches the decision granularity.**

| Signal | Measures | Granularity | Good For | Bad For |
|--------|----------|-------------|----------|---------|
| Semantic monitor | Answer surface structure | Per-task | Detecting visible-pass ambiguity | Predicting cross-task generalization |
| Embedding novelty (TopoMem surprise) | Input distribution shift | Batch-level | Detecting deployment environment drift | Per-task routing decisions |

**Key insight**: TopoMem's embedding novelty signal was *rejected* as a per-task routing monitor (success 0.7/0.85 vs baseline 1.0) because embedding novelty ≠ answer correctness. However, it was *confirmed* as a batch-level deployment health monitor (centroid drift 27.2× separation, p ≈ 0). This is not a threshold problem — it is a structural mismatch that no parametric adjustment can fix.

### Boundary-Local Amplification

When a solver is "near" its capability boundary (close to correct but missing a specific constraint), feedback retry produces outsized benefit:

- **ABOVE zone** (already correct): Feedback is wasted → filter out
- **NEAR zone** (close to correct): Feedback is highly effective → +49% gain, p = 0.0008
- **BELOW zone** (far from correct): Feedback is useless → escalate instead

This produces an **inverted-U pattern** that generalizes across model scales (Qwen2.5 0.5B → 1.5B → 3B shows increasing NEAR-zone fraction: 0% → 10% → 20%).

---

## Scientific Integrity

### Red-Line Rules

1. **No oracle overclaim**: Escalation path success rates assume oracle (expected_answer). Any "100% success" claim must be labeled "based on oracle assumption".
2. **No simulated cost as real cost**: `cost_units` and `latency_units` are hardcoded abstract values. Any "cost reduction X%" conclusion must be labeled "based on assumed cost model".
3. **No hidden-test leakage**: Public benchmark files (`.public.jsonl`) must not contain `hidden_tests`, `fixed_code`, or `expected_route`.
4. **No self-awareness validated claim**: The self-aware LLM direction is a future narrative. Do not claim it is validated or operational.

### Truth Table

| Conclusion | Status | Evidence |
|------------|--------|----------|
| Boundary-local amplification exists | ✅ SAFE | Phase A, p = 0.0008 |
| ABOVE filtering saves feedback calls | ✅ SAFE | Phase E, 54.4% reduction |
| NEAR/BELOW has ranking signal | ⚠️ WEAK | ROC AUC = 0.769, threshold unstable |
| Unified-SEL beats EWC | ❌ UNVERIFIED | p = 0.9484 |
| TopoMem surprise works as routing monitor | ❌ REJECTED | success 0.7/0.85, structural mismatch |
| TopoMem works as deployment health monitor | ✅ CONFIRMED | OBD 10-seed: 27.2× [18.4×, 37.3×], p ≈ 0 |

---

## Citation

If you use this codebase, please cite:

```bibtex
@misc{unified_sel_2026,
  title={Granularity-Aligned Metacognition: Why Embedding Novelty Predicts Distribution Shift but Not Answer Correctness},
  author={[Authors]},
  year={2026},
  note={Research codebase for boundary-local amplification and capability routing}
}
```

---

## License

MIT License — see individual files for details.

---

*This project was originally named "Unified-SEL" and explored surprise-driven structural evolution for continual learning. After rigorous experimental evaluation, the core hypothesis (surprise-driven structural birth/death > EWC) was found to be unverifiable on the toy problem (p = 0.9484). The project pivoted to its current focus on metacognitive monitoring and capability routing. Archived materials are preserved in `archive/` for reference.*
