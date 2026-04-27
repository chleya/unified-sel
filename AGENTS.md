# AGENTS.md — Agent Entry Point

Read this file before starting any work on this project.

---

## Current Direction (2026-04-27)

**This project is a research codebase for "Granularity-Aligned Metacognition for LLMs."**

The old "Unified-SEL" core hypothesis (surprise-driven structural birth/death > EWC) is **archived and unverified**. Do not promote it.

### Research Tracks

| Track | Status | Description |
|-------|--------|-------------|
| Boundary-local amplification | ✅ SAFE | Inverted-U feedback benefit pattern (p = 0.0008); ABOVE-filtering saves 54.4% calls |
| Capability Router | ✅ Research tool | Benchmark for accept/verify/escalate routing decisions |
| TopoMem OBD | ✅ CONFIRMED | Batch-level deployment health monitoring (drift detection) |
| surprise > EWC | ❌ ARCHIVED | Unverified (p = 0.9484), do not claim |
| self-aware-llm | ❌ FUTURE NARRATIVE | Not a build target, do not claim validated |

### Key Documents

| Document | Purpose |
|----------|---------|
| README.md | Project overview, quick start, core concepts |
| STATUS.md | Current progress, truth table, known issues |
| EXPERIMENT_LOG.md | All experiment records |
| experiments/capability/README.md | Capability Router user guide |
| data/capability_boundary_bench/README.md | Benchmark leakage rules |
| papers/boundary_local_amplification_draft.md | Paper draft |

---

## What Is This Project?

A research codebase investigating **metacognitive monitoring** for LLMs:

1. **Paper line**: Feedback retry is a boundary-local amplifier (inverted-U pattern, p = 0.0008). ABOVE-zone filtering saves 54.4% feedback calls. This is the publishable finding.

2. **Tool line**: Capability Router — a benchmark + routing system that evaluates when an LLM should accept its own output, verify it, or escalate. Works on synthetic code-repair and reasoning tasks with multiple monitors and policy layers.

3. **Methodology line**: The Granularity Alignment Principle — a signal's predictive power depends on whether its measurement granularity matches the decision granularity.

---

## What TopoMem Is and Is Not

- TopoMem surprise was tested as a per-task routing monitor and **rejected** (success 0.7/0.85 vs baseline 1.0).
- Structural reason: embedding novelty does not predict answer correctness.
- TopoMem batch-level drift detection is **confirmed** for domain shifts (centroid drift 27.2× separation, p ≈ 0).
- **Do not re-promote TopoMem as a routing signal.**

---

## Red-Line Rules

1. **No oracle overclaim**: Escalation path success rates assume oracle (expected_answer). Do not claim 100% escalation success as a real result.

2. **No simulated cost as real cost**: `cost_units` and `latency_units` are hardcoded abstract values. Any "cost reduction X%" conclusion must be labeled "based on assumed cost model".

3. **No hidden-test leakage**: Public benchmark files (`.public.jsonl`) must not contain `hidden_tests`, `fixed_code`, or `expected_route`. Eval files (`.eval.jsonl`) are internal only.

4. **No self-awareness validated claim**: The self-aware LLM direction is a future narrative. Do not claim it is validated or operational.

5. **No new synthetic solver experiments without preflight**: Before running any new experiment with synthetic solvers, check EXPERIMENT_LOG.md for prior results and STATUS.md for current truth.

6. **No modifying source projects**: Do not edit files in F:\sel-lab, F:\SDAS, or F:\fcrs_mis. Read-only reference.

---

## Working Protocol

1. Read README.md — understand the project's current framing
2. Read STATUS.md — understand where the project is
3. Read EXPERIMENT_LOG.md — avoid repeating experiments
4. Do one thing at a time
5. Run `python tests/smoke_test.py` after any code change
6. Update EXPERIMENT_LOG.md and STATUS.md after completing work

---

## Project Structure

```
unified-sel/
  core/
    capability_benchmark.py    # Routing benchmark engine (monitors + policies)
    runtime.py                 # Path management, save/load
    structure.py               # Archived: DFA structure unit (kept for compatibility)
    pool.py                    # Archived: StructurePool lifecycle (kept for compatibility)
    learner.py                 # Archived: DFA learner (kept for compatibility)
  experiments/
    capability/
      capbench.py              # CLI: run / compare / report / list-monitors / list-policies
      export_bench.py          # Export benchmark to JSONL (public/eval modes)
      benchmark.py             # Benchmark runner
      validate_above_filtering.py  # Validation experiments
      README.md                # User guide
  data/
    capability_boundary_bench/ # JSONL benchmark files + README
  topomem/                   # TopoMem subsystem (batch-level health monitoring only)
  archive/                   # Archived experiments and documents
    experiments_continual/   # Old mechanism-track experiments
    meta_controller/         # Deprecated meta-controller experiments
    cep_cc/                  # Deprecated CEP-CC experiments
    weight_graph/            # Deprecated weight analysis experiments
    lsg_deprecated/          # Deprecated learned-state-governance experiments
  papers/
    boundary_local_amplification_draft.md  # Paper draft
  results/                   # All experiment outputs (do not edit manually)
  tests/
    smoke_test.py            # Quick validation
```

---

## Truth Table (Summary)

| Conclusion | Status | Evidence |
|------------|--------|----------|
| Boundary-local amplification exists | ✅ SAFE | Phase A, p = 0.0008 |
| ABOVE filtering saves feedback calls | ✅ SAFE | Phase E, 54.4% reduction |
| NEAR/BELOW has ranking signal | ⚠️ WEAK | ROC AUC = 0.769, threshold unstable |
| Real-LLM NEAR zone emerges at scale | ✅ CONFIRMED | Strict validation: 1.5B=0%, 3B=15%; scale-dependent |
| Real-LLM inverted-U complete | ⚠️ INCOMPLETE | 3B: NEAR(15%) < ABOVE(25%), not full inverted-U |
| Real-LLM ABOVE-filtering works | ✅ CONFIRMED | 1.5B=30%, 3B=25% ABOVE; feedback calls skippable |
| Unified-SEL beats EWC | ❌ UNVERIFIED | p = 0.9484 |
| TopoMem surprise works as routing monitor | ❌ REJECTED | success 0.7/0.85, structural mismatch |
| TopoMem works as deployment health monitor | ✅ CONFIRMED | OBD 10-seed: 27.2× [18.4×, 37.3×], p ≈ 0 |
