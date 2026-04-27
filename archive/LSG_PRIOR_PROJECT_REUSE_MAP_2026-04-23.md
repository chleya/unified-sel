# LSG Prior Project Reuse Map

Date: 2026-04-23
Status: reference map for v0 implementation

This map identifies what to reuse from prior F-drive projects for the LSG / rewrite qualification project.

The goal is not to merge projects.
The goal is to reuse proven boundaries, tests, and concepts while keeping Phase 0 small.

---

## 1. Primary Prior: Cognitive Execution Engine

Path:

```text
F:\cognitive-execution-engine
```

Why it matters:

CEE already implements a hard state-commitment kernel.
It is the closest prior project to LSG Phase 0.

Relevant files:

```text
F:\cognitive-execution-engine\docs\ARCHITECTURE_STATE_FIRST_2026-04-16.md
F:\cognitive-execution-engine\src\cee_core\commitment.py
F:\cognitive-execution-engine\src\cee_core\revision.py
F:\cognitive-execution-engine\src\cee_core\event_log.py
F:\cognitive-execution-engine\src\cee_core\memory_promotion.py
F:\cognitive-execution-engine\src\cee_core\planner.py
F:\cognitive-execution-engine\tests\test_red_line_invariants.py
F:\cognitive-execution-engine\tests\test_memory_promotion.py
```

### 1.1 Reusable Concepts

Reuse conceptually:

- state-first architecture
- reducer defines valid state evolution
- model output cannot directly mutate formal state
- append-only event log
- commitment event before formal revision
- revision event as actual state mutation record
- replayed state excludes denied or pending changes
- provenance on every committed fact
- `requires_approval` blocks revision
- policy/meta patches are hard-blocked

### 1.2 Reusable Test Ideas

Directly adapt these red-line invariants:

| CEE invariant | LSG Phase 0 equivalent |
|---|---|
| no approval, no committed mutation | no `acknowledged` without `constitution_open` |
| policy/meta patches hard-blocked | protected boundary never direct-acknowledges |
| every committed fact has provenance | every acknowledgement has `CommitEvent` |
| denied/pending not in replay | background/verify candidates do not mutate durable state |
| `requires_approval` never enters `WorldState` | `constitution_open = false` blocks acknowledgement |

### 1.3 What Not To Reuse Yet

Do not import CEE code in Phase 0.

Reasons:

- Phase 0 tests the two-variable dynamics independently.
- CEE has a larger architecture than needed for the simulator.
- Direct integration would hide whether `D_t/S_t` has useful behavior.

Phase 0 should mirror CEE discipline, not depend on CEE modules.

### 1.4 Future Integration Point

If Phase 0 passes, CEE can become the formal acknowledgement layer:

```text
LSG commit_review
  -> CEE CommitmentEvent
  -> CEE ModelRevisionEvent
  -> CEE EventLog replay
```

---

## 2. Secondary Prior: Hypergraph Bistability

Path:

```text
F:\hypergraph_bistability
```

Why it matters:

This project contains earlier work on multi-stability, control, inhibition, and order parameters.
It is conceptually close to `D_t/S_t` dynamics.

Relevant files:

```text
F:\hypergraph_bistability\experiments\research\multi_stability_core.py
F:\hypergraph_bistability\experiments\research\control_inhibition.py
F:\hypergraph_bistability\experiments\research\control_experiment.py
F:\hypergraph_bistability\experiments\research\control_precision.py
F:\hypergraph_bistability\experiments\research\optimal_control_grid.py
F:\hypergraph_bistability\experiments\verification\experiment_durable_memory.py
F:\hypergraph_bistability\experiments\verification\experiment_integrated_memory.py
```

### 2.1 Reusable Concepts

Reuse conceptually:

- stable basin
- perturbation threshold
- inhibition under repeated low-value signal
- control boundary
- order parameter
- transition under sustained pressure

### 2.2 What Not To Reuse Yet

Do not use hypergraph implementation in Phase 0.

Reasons:

- LSG v0 needs a small deterministic simulator.
- Hypergraph dynamics may add unnecessary complexity.
- The first claim is not about hypergraphs; it is about phase separation in `D/S` space.

### 2.3 Future Use

Use later for:

- visualization of phase regions
- analogy to stable basins
- extended experiments on multi-candidate competition
- inhibition/control ablations

---

## 3. Current Project: Unified-SEL

Path:

```text
F:\unified-sel
```

Role:

`unified-sel` remains the active research workspace.
LSG v0 should live here as a sidecar mechanism experiment.

Current LSG documents:

```text
LSG_ATTENTION_INSPIRATION_2026-04-23.md
LSG_ACTION_GUIDE_2026-04-23.md
REWRITE_QUALIFICATION_DYNAMICS_v0.md
ATTENTIONAL_GOVERNANCE_KERNEL_v0.md
LSG_ENGINEERING_SPEC_v0.md
LSG_THEOREM_NOTE_v0.md
```

### 3.1 Files To Create For Phase 0

```text
core/rewrite_dynamics.py
tests/test_rewrite_dynamics.py
experiments/capability/rewrite_dynamics_sanity.py
```

### 3.2 Files To Avoid Touching In Phase 0

Do not modify:

```text
core/capability_benchmark.py
core/llm_solver.py
experiments/capability/llm_routing_experiment.py
topomem/data/chromadb/*
```

Do not modify CEE from this repo.
Do not archive or delete old project files.

---

## 4. Reuse Summary

| Source | Reuse Now | Reuse Later | Do Not Do |
|---|---|---|---|
| CEE | invariants, commit-boundary concepts | event/revision integration | import whole runtime in Phase 0 |
| hypergraph_bistability | stability/control vocabulary | phase visualization, inhibition ablations | depend on hypergraph model in Phase 0 |
| unified-sel | workspace, tests, result format | integration with capability experiments | touch existing router mainline |

---

## 5. Phase 0 Reuse Rule

Phase 0 may copy ideas.
Phase 0 should not copy architecture.

The immediate target is still:

```text
small deterministic D/S simulator
```

not:

```text
new CEE runtime
new memory system
new hypergraph model
```

