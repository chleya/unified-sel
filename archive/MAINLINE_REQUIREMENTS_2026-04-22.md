# Mainline Requirements

**Date**: 2026-04-22
**Rule**: Every active line must have baseline, metric, failure condition, reproducible command, and result artifact.

---

## Active Lines

### 1. Capability Router

| Item | Value |
|------|-------|
| **Baseline** | SearchLocalSolver (100% on code-20) |
| **Metric** | success_rate, escalation_rate, cost_units, revision_rate |
| **Failure condition** | No monitor+protocol combination achieves success_rate > 0.5 on real LLM |
| **Reproducible command** | `python experiments/capability/llm_routing_experiment.py` (requires llama.cpp server on port 8081) |
| **Result artifact** | `results/capability_benchmark/*.json` |

Current status:
- Semantic monitor: 75% detection, counterfactual: 88%, confidence: 0%
- monitor_gate: 15%→75-95% (synthetic), verifier_first: 100% (oracle)
- monitor_no_revision_triage: success_rate 1.0, revision_rate 0.0 (synthetic)
- Real-LLM (Qwen2.5-0.5B): 8-15% base success, few-shot 30%
- Next: real-LLM validation with no-revision triage

### 2. Boundary-local Amplification

| Item | Value |
|------|-------|
| **Baseline** | Uniform feedback benefit (no zone dependence) |
| **Metric** | p-value for inverted-U pattern, ABOVE-zone filtering savings |
| **Failure condition** | p > 0.05 for zone-dependent feedback benefit |
| **Reproducible command** | `python experiments/continual/` (Phase A/E experiments) |
| **Result artifact** | `results/double_helix/` |

Current status:
- Phase A: p=0.0008 (inverted-U confirmed)
- Phase E: 54.4% ABOVE-zone filtering savings
- NEAR/BELOW ranking: ROC AUC=0.769, threshold unstable (WEAK)
- Next: paper writing, artifact audit

### 3. Health/OBD Monitoring

| Item | Value |
|------|-------|
| **Baseline** | Random separation (1.0x) |
| **Metric** | Centroid drift separation ratio, bootstrap CI, paired t-test p-value |
| **Failure condition** | Domain shift separation CI lower bound < 2.0x |
| **Reproducible command** | `python experiments/capability/topomem_obd_multiseed.py` |
| **Result artifact** | `results/topomem_obd_preflight/*.json` |

Current status:
- BatchHealthMonitor: 27.2x [18.4x, 37.3x] domain shift, 4.1x [2.6x, 5.9x] gradual (10-seed, CONFIRMED)
- PredictiveHealthMonitor: 12.8x [10.4x, 15.6x] domain shift, 2.1x [1.7x, 2.5x] gradual (10-seed, detectable but not superior)
- PredictiveHealthMonitor: no temporal advantage over BatchHealthMonitor (late by 1.8 tasks)
- Decision: BatchHealthMonitor is primary health signal; PredictiveHealthMonitor is experimental sidecar

---

## Non-Active (Architecture Boundary Only)

These are not implementation tasks. They define boundaries for future work.

- **Learned State Governance**: SelfState / GovernancePolicy / CommitLog schema only
- **CEE SelfState / WriteLaw**: Requires reliable governance signal source first
- **LeWorldModel beyond P0**: SIGReg applicability, PredictiveStateAdapter — all pending P0 signal validation
- **CEP-CC integration**: Independent project at F:\cep-cc, do not mix into unified-sel
- **PredictiveHealthMonitor as primary signal**: Rejected — detectable but not superior

---

## Rejected / Not Promoted

| Claim | Status | Evidence |
|-------|--------|----------|
| TopoMem per-task routing | REJECTED | success 0.7/0.85 vs baseline 1.0; embedding novelty ≠ answer correctness |
| surprise > EWC | UNVERIFIED | p=0.9484 |
| PredictiveHealthMonitor as primary governance signal | NOT PROMOTED | 12.8x vs 27.2x BatchHealthMonitor; no temporal advantage |

---

## Governance Rule

> Before P0 residual signal passes, Governance spec is architecture boundary only, not implementation task.

Current primary health signal: **BatchHealthMonitor** (static centroid drift).

`prediction_residual` field in Governance spec remains auxiliary/empty. Do not substitute BatchHealthMonitor drift as prediction_residual.
