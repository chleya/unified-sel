# TopoMem Routing Monitor Preflight — 2026-04-16

## Purpose

Formal specification for integrating TopoMem signals into capability_benchmark's routing monitor interface. This document defines what signals are available, what signals are prohibited, how to adapt the interface, what benchmarks to use, and what constitutes success or failure.

**Status**: First experiment (surprise signal) already run. This document captures the preflight spec retroactively and defines the framework for future signal attempts.

---

## Available TopoMem Signals

| Signal | Source | Computation | Range | Semantic Meaning |
|--------|--------|-------------|-------|-----------------|
| surprise | `adapters.py:504-516` | 1.0 - max_cosine_similarity(query_emb, adapter_pool) | [0, 1] | How novel is this task vs previously seen tasks |
| tension | `adapters.py:519-540` | mean(Wasserstein_distance(fp[i], fp[i-1]) for last N) | [0, +inf) | How rapidly is the system's topology changing |
| h1_health | `self_awareness.py:722-745` | min(mean_persistence / (1 + fragmentation) / 0.1, 1.0) | [0, 1] | Quality of H1 topological structure |
| h1_drift | `self_awareness.py:755-764` | Wasserstein_distance(current, baseline, dim=1) | [0, +inf) | How far H1 structure has drifted from baseline |
| h2_to_h1_ratio | `self_awareness.py:655-720` | betti_2 / max(betti_1, 1) | [0, +inf) | Domain mixing sensitivity |
| h2_drift | `self_awareness.py:785-790` | Wasserstein_distance(current, baseline, dim=2) | [0, +inf) | How far H2 structure has drifted from baseline |
| drift_status | `self_awareness.py:295-303` | stable/evolving/drifting/restructured | categorical | Overall drift classification |

### Signal-to-Routing Semantics Mapping

| Signal | Hypothesized Routing Meaning | Verified? |
|--------|------------------------------|-----------|
| high surprise | Task is novel → may need verification | ❌ FAIL — novelty ≠ difficulty |
| high tension | System unstable → be conservative | Not yet tested |
| low h1_health | Embedding space degraded → distrust routing | Not yet tested |
| high h1_drift | System has shifted → recalibrate thresholds | Not yet tested |
| high h2_to_h1_ratio | Domain mixing → may need different adapter | Not yet tested |
| drift_status=drifting | Active regime change → escalate more | Not yet tested |

---

## Prohibited Signals

These signals MUST NOT be used as routing monitor inputs. They constitute label leakage or oracle access.

| Prohibited Signal | Why Prohibited | Where It Appears |
|-------------------|---------------|-----------------|
| `boundary_label` | Direct ground truth (ABOVE/NEAR/BELOW) | Phase F health signals |
| `bug_type` | Perfect task fingerprint | Phase G patch_size |
| `hidden_test_results` | Oracle access to verification outcome | Verifier internals |
| `expected_answer` | Oracle solution | OracleSolver |
| `patch_size` | Perfect bug_type fingerprint | Phase G features |
| `first_patch_size` | Same as patch_size | Phase G features |
| `patch_size_to_message_len_ratio` | Same as bug_type one-hot | Phase G features |
| `verifier_passed` (before decision) | Post-hoc ground truth | Verifier output |

### Leakage Test

Any new TopoMem routing monitor must pass this test before results are accepted:

1. Compute signal for each task in code-20
2. Check correlation between signal and `bug_type`: if Cramér's V > 0.3, the signal is a bug_type proxy
3. Check correlation between signal and `boundary_label`: if point-biserial r > 0.3, the signal is a boundary label proxy
4. If either correlation exceeds threshold, the signal is contaminated and results are invalid

---

## Monitor Interface Adaptation

### Existing Interface

```python
class RoutingMonitor:
    name = "base"
    def score(self, task: BenchmarkTask, attempt: SolverAttempt) -> float:
        raise NotImplementedError
```

- Input: `BenchmarkTask` (has `.prompt`, `.family`, `.metadata`) + `SolverAttempt` (has `.answer`, `.confidence`, `.notes`)
- Output: `float` in [0.0, 1.0] — routing signal. Higher = more scrutiny needed.

### TopoMem Signal Adaptation Challenge

TopoMem signals operate on **embeddings** (384-dim vectors from all-MiniLM-L6-v2). The `RoutingMonitor.score()` interface does not provide embeddings. Two adaptation strategies:

**Strategy A: Embed task prompt on-the-fly**
- Call `EmbeddingManager.encode(task.prompt)` inside `score()`
- Compute surprise against accumulated embedding pool
- Pro: No interface changes needed
- Con: Embedding computation adds latency; surprise is cumulative (first task always = 1.0)

**Strategy B: Pre-compute embeddings, pass via metadata**
- Compute embeddings once at benchmark setup
- Store in `task.metadata["embedding"]`
- Read from metadata in `score()`
- Pro: No redundant computation; deterministic across runs
- Con: Requires benchmark runner modification

**Chosen**: Strategy A (implemented as `TopoSurpriseRoutingMonitor`). No interface changes needed.

### Implemented Monitors

| Monitor Name | Class | Strategy | Status |
|---|---|---|---|
| `topo_surprise` | `TopoSurpriseRoutingMonitor` | A: on-the-fly embedding | ❌ FAIL (experiment run) |
| `topo_semantic_fusion` | `TopoSemanticFusionMonitor` | A: 30% topo + 70% semantic | ⚠️ No advantage (experiment run) |

---

## Benchmark Protocol

### Frozen Baseline

- **code-20**: 20 code-only tasks, seed=7, SearchLocalSolver
- **mixed-40**: 40 mixed tasks (20 code + 20 reasoning), seed=7, SearchLocalSolver
- **Protocol**: `monitor_repair_triage` (current best policy)
- **Baseline monitor**: `semantic` (current strongest)

### Baseline Numbers (seed=7, monitor_repair_triage)

| Suite | Monitor | success_rate | mean_cost | revision_rate | accept_wo_verify |
|-------|---------|-------------|-----------|---------------|-----------------|
| code-20 | semantic | 1.0 | 1.375 | 0.65 | 0.10 |
| mixed-40 | semantic | 1.0 | 1.1875 | 0.325 | 0.55 |

### Metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| success_rate | Fraction of tasks with correct final answer | ≥ baseline |
| mean_cost_units | Average cost per task (assumed model) | ≤ baseline or justified increase |
| revision_rate | Fraction of tasks where revision was attempted | Informative only |
| accept_wo_verify_rate | Fraction accepted without verifier | Informative only |
| false_accept_rate | Fraction accepted without verify that failed | ≤ 0.05 |

---

## Success / Failure Criteria

### Pass Condition (any one suffices)

1. **Same success, lower cost**: success_rate ≥ baseline AND mean_cost < baseline
2. **Same cost, better generalization**: success_rate and cost ≈ baseline on code-20/mixed-40, but significantly better on held-out or paraphrased tasks
3. **Fusion improvement**: topo + semantic fusion > semantic-only on success_rate OR cost, with statistical significance

### Fail Condition (any one triggers)

1. **Success drop**: success_rate < baseline by more than 0.05
2. **Signal is oracle/leakage**: signal correlates with boundary_label or bug_type (Cramér's V > 0.3)
3. **Signal is noise**: fusion with semantic does not improve over semantic-only
4. **Cost increase without justification**: mean_cost > baseline AND success_rate ≤ baseline

---

## Experiment Results (2026-04-16)

### Experiment 1: TopoSurpriseRoutingMonitor (surprise signal)

| Suite | success_rate | mean_cost | revision_rate | accept_wo_verify | Verdict |
|-------|-------------|-----------|---------------|-----------------|---------|
| code-20 | 0.7 | 1.06 | 0.10 | 0.85 | ❌ FAIL |
| mixed-40 | 0.85 | 1.035 | 0.05 | 0.90 | ❌ FAIL |

**Failure reason**: surprise signal too low (mean ~0.23-0.31), causing 85-90% tasks to be accepted without verification. Structural defect: embedding similarity ≠ answer correctness. Surprise measures task novelty, not answer quality.

**Leakage check**: surprise does NOT correlate with bug_type or boundary_label (Cramér's V ≈ 0). Signal is clean but uninformative.

### Experiment 2: TopoSemanticFusionMonitor (30% topo + 70% semantic)

| Suite | success_rate | mean_cost | revision_rate | accept_wo_verify | Verdict |
|-------|-------------|-----------|---------------|-----------------|---------|
| code-20 | 1.0 | 1.435 | 0.85 | 0.10 | ⚠️ No advantage |
| mixed-40 | 1.0 | 1.225 | 0.45 | 0.55 | ⚠️ No advantage |

**Failure reason**: success recovered to 1.0 (carried by semantic), but cost increased. Topo signal added noise, not information. Fusion failed criterion #3.

### Overall Verdict

**TopoMem surprise signal: ❌ FAILED as routing monitor.**

The fundamental issue is that surprise (1.0 - max_cosine_similarity) measures how novel a task's prompt is relative to previously seen prompts. This is orthogonal to whether the solver's answer is correct. A task can be highly similar to previous tasks (low surprise) yet have an incorrect answer, or highly novel (high surprise) yet be trivially solvable.

---

## Remaining Signal Candidates

| Signal | Hypothesis | Feasibility | Risk |
|--------|-----------|-------------|------|
| tension | High tension → system unstable → verify more | Requires persistent topology tracking across tasks | Medium — tension may be too slow-moving for per-task routing |
| h1_health | Low health → embedding space degraded → distrust local answer | Requires persistent SelfAwareness instance | Low — health reflects system state, not task difficulty |
| h1_drift | High drift → recent regime change → be conservative | Requires baseline diagram | Medium — drift is cumulative, may not help per-task |
| h2_to_h1_ratio | High ratio → domain mixing → may need different strategy | Requires sufficient nodes (MIN_NODES_FOR_H2=20) | High — may not fire on small task sets |
| drift_status | "drifting" → escalate more | Requires history of diagrams | Low — categorical signal, easy to integrate |

### Recommended Next Attempt

If a second TopoMem signal attempt is made, the best candidate is **h1_health**:
- It reflects system state (is the embedding space healthy?) rather than task novelty
- Low h1_health could legitimately mean "the system's internal model is degraded, so verify more"
- It does not suffer from the novelty ≠ correctness structural defect
- Implementation: maintain a persistent SelfAwareness instance across tasks, use `get_h1_health()` as a modifier to the routing signal

However, this requires significant engineering (persistent SelfAwareness, feeding task embeddings into it across the benchmark run). The expected effect size is uncertain.

---

## Decision

**Current state**: TopoMem surprise signal failed as routing monitor. The preflight experiment is complete with a negative result.

**Options**:
1. **Attempt h1_health signal** — different semantics (system state vs task novelty), but uncertain payoff
2. **Abandon TopoMem as routing monitor** — reposition as deployment health monitoring (OBD for LLMs)
3. **Focus on tool line** — standardize capability benchmark toolkit without TopoMem integration

**Recommendation**: Option 3 is the safest next step. The toolkit has genuine engineering value without TopoMem. Option 2 (OBD) is a valid long-term direction but requires a different evaluation framework. Option 1 is a research bet with uncertain returns.
