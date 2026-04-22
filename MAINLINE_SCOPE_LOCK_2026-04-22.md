# Unified-SEL Mainline Scope Lock

Date: 2026-04-22

Purpose: keep `F:\unified-sel` focused after the LeWorldModel / governance review and prevent sidecar projects from being re-promoted without evidence.

## Current Project Shape

`unified-sel` is the active mechanism and evaluation lab.

The project currently has three allowed active lanes:

| Lane | Status | Allowed Work |
|---|---|---|
| Capability Router | Primary tool lane | real-LLM validation, policy comparison, benchmark/report tooling |
| Boundary-local amplification | Primary paper lane | paper draft, artifact audit, group/solver validation |
| Batch-level health monitoring | Secondary evidence lane | BatchHealthMonitor validation, OBD-style drift monitoring |

Everything else is sidecar, archived, or design-only until it earns its way back through a scoped preflight.

## Decisions From 2026-04-22

### PredictiveHealthMonitor

P0 result:

- Domain drift is detectable through predictive residual.
- Gradual drift is detectable but weak.
- It does not beat `BatchHealthMonitor`.
- It has no temporal advantage in the tested stream.

Decision:

- Do not promote `PredictiveHealthMonitor` to the primary governance or routing signal.
- Keep it as an experimental sidecar / ablation signal.
- Do not connect it to CEE.
- Do not use it as justification for implementing learned governance.

Current baseline:

- `BatchHealthMonitor` remains the active health-monitoring baseline.

### Learned State Governance

`LEARNED_STATE_GOVERNANCE_SPEC_2026-04-22.md` is an architecture boundary document, not an implementation ticket.

Blocked until further evidence:

- no `SelfState` implementation in `unified-sel`
- no `WriteLaw` implementation in `unified-sel`
- no `AuthoritySwitchingLaw` implementation in `unified-sel`
- no CEE governance integration based on this spec

The spec may be used to audit boundaries and terminology, but not to add new learned control code.

### CEE

`F:\cognitive-execution-engine` is not the current research lab.

Its role is narrowed to:

- commitment kernel
- approval boundary
- event / revision log
- hard policy gates

Critical invariant:

`requires_approval` means pending approval, not advisory execution. It must not create a `ModelRevisionEvent` or mutate `WorldState` before approval.

### CEP-CC

CEP-CC is a separate project at `F:\cep-cc`.

Rules:

- do not mix CEP-CC result docs back into the active `unified-sel` mainline
- do not use CEP-CC results to justify Capability Router claims
- do not run CEP-CC experiments from `unified-sel` unless explicitly doing archival migration

### TopoMem

Accepted:

- TopoMem / topology signals may be used as batch-level deployment health / OBD candidates.

Rejected:

- TopoMem surprise is not a per-task routing monitor.
- Embedding novelty is not answer correctness.

## Active Claim Boundaries

Safe claims:

- Boundary-local amplification exists in the audited setup.
- ABOVE-zone filtering can reduce unnecessary feedback calls in the current benchmark.
- Capability Router policies are useful in the current benchmark and real-LLM sanity tests.
- Batch-level drift monitoring is a valid health-monitoring lane.
- Qwen2.5-0.5B self-confidence is unreliable in the tested setting.

Unsafe claims:

- Unified-SEL beats EWC.
- TopoMem surprise predicts answer correctness.
- PredictiveHealthMonitor is better than BatchHealthMonitor.
- Learned governance has been validated.
- CEE has learned self-governance.
- CEP-CC is evidence for the unified-sel mainline.

## GitHub Upload Batches

Do not commit the whole dirty worktree.

### Batch 1: Mainline Documentation

Stage only:

- `README.md`
- `PROJECT_OVERVIEW_AND_INDEX_2026-04-22.md`
- `ARCHIVE_AND_CLEANUP_PLAN_2026-04-22.md`
- `MAINLINE_SCOPE_LOCK_2026-04-22.md`
- `LEWM_INTEGRATION_SPEC_2026-04-22.md`
- `LEARNED_STATE_GOVERNANCE_SPEC_2026-04-22.md`
- `EXPERIMENT_LOG.md` only if the P0 result section is intentionally included

Do not stage:

- `STATUS.md` unless explicitly reviewed for encoding and content drift
- TopoMem Chroma files
- generated DB/vector/cache artifacts
- CEP-CC result documents
- Meta-controller result documents

### Batch 2: P0 PredictiveHealthMonitor Sidecar

Stage only if keeping the sidecar implementation:

- `core/predictive_health.py`
- `experiments/capability/predictive_health_preflight.py`
- `experiments/capability/temporal_advantage_test.py`
- `tests/test_predictive_health.py`
- generated result JSONs only if they are small and intentionally part of evidence

Commit message should say it is a sidecar / negative-control result, not a promoted mainline feature.

### Batch 3: Capability Router Mainline

Stage only files needed for the current Capability Router tool lane, such as:

- `core/capability_benchmark.py`
- `core/llm_solver.py`
- `experiments/capability/`
- `data/capability_boundary_bench/`
- relevant tests

This batch should be separated from P0 health-monitor sidecar work.

### Batch 4: Archive Migration

Only after explicit approval:

- move CEP-CC docs/scripts to archive or rely on `F:\cep-cc`
- move Meta-controller docs/scripts to archive
- move old mechanism-track experiments to archive
- remove generated TopoMem temp DBs
- externalize large artifacts

No destructive cleanup should happen in this batch without explicit approval.

## Task Gate For Future LLMs

Before accepting any new task, require it to identify one lane:

1. Capability Router
2. Boundary-local amplification
3. Batch-level health monitoring
4. Archive / cleanup
5. External project: CEE
6. External project: CEP-CC
7. Design-only governance

If a task cannot name its lane, it should not modify code.

Every active implementation task must state:

- baseline
- metric
- failure condition
- reproducible command
- expected artifact

## Immediate Next Step

The next engineering step is not more abstraction.

Recommended sequence:

1. Commit scoped documentation batch.
2. Commit or discard P0 PredictiveHealthMonitor sidecar as a separate batch.
3. Return to Capability Router real-LLM validation.
4. Keep CEE limited to hard-gate fixes.
5. Keep CEP-CC separate.

