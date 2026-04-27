# Weak Model Deployment Policy

**Status**: Active  
**Scope**: Qwen2.5-0.5B to 3B, code repair tasks  
**Last Updated**: 2026-04-24

---

## Executive Summary

For small LLMs (0.5B-3B parameters), self-revision is ineffective. The optimal strategy is:

> **Do not let weak models revise. Let them be verified, routed, and escalated.**

This document defines the policy, evidence, and implementation for deploying weak models in production routing systems.

---

## Evidence Summary

### Revision Effectiveness by Model Size

| Model | Params | Revision Success | NEAR Zone | Policy |
|-------|--------|-----------------|-----------|--------|
| Qwen2.5-0.5B | 0.5B | **0/17 (0%)** | 0% | No-revision triage |
| Qwen2.5-1.5B | 1.5B | **~2/20 (10%)** | 10% | No-revision triage |
| Qwen2.5-3B | 3.4B | **~4/20 (20%)** | 20% | Monitor-based triage |

**Key finding**: At 0.5B, revision is completely wasted. At 1.5B, it occasionally helps but is not worth the API cost. At 3B, it starts to become viable but still marginal.

### Optimal Policies

| Policy | 0.5B Success | 1.5B Success | 3B Success | Revision Calls |
|--------|-------------|-------------|-----------|---------------|
| local_only | 10-25% | 20-30% | 20-30% | 0 |
| monitor_no_revision_triage | **100%** | **100%** | **100%** | **0** |
| monitor_repair_triage | 100% | 100% | 100% | 17-20 |
| verifier_first | 100% | 100% | 100% | 17-20 |

**Conclusion**: `monitor_no_revision_triage` (semantic) achieves 100% success with zero revision calls, making it optimal for cost-sensitive deployments.

---

## Confidence Signal Status

### Problem

llama.cpp API does not return token logprobs. The `_estimate_confidence` method in `LlamaCppSolver` returns a hardcoded 0.95 for all outputs, making it useless for routing.

### Solution

Use `weak_model_confidence.estimate_confidence_weak_model()` which provides:

| Component | Weight (no tests) | Weight (with tests) | Description |
|-----------|------------------|---------------------|-------------|
| structure | 0.25 | 0.15 | Has def, return, proper indentation |
| syntax | 0.30 | 0.15 | AST parseable? |
| semantic | 0.45 | 0.20 | Function name match, bug pattern fix |
| test | 0.00 | 0.50 | Visible test pass rate |

**Calibration**: Output range [0.05, 0.95] with sigmoid-like compression to avoid overconfidence.

### Uncertainty Bands

| Band | Confidence Range | Action |
|------|-----------------|--------|
| LOW_UNCERTAINTY | >= 0.75 | Accept output |
| MEDIUM_UNCERTAINTY | 0.50 - 0.75 | Verify then accept |
| HIGH_UNCERTAINTY | 0.25 - 0.50 | Escalate on hard tasks |
| VERY_HIGH_UNCERTAINTY | < 0.25 | Always escalate |

---

## Deployment Policy

### For 0.5B Models

```
Strategy: verify_escalate_no_revision
- Single-shot solve
- Verify output
- If correct: accept
- If wrong: escalate (no revision attempt)
- Confidence: use weak_model_confidence heuristic
```

**Rationale**: 0% revision success means every revision call is wasted.

### For 1.5B Models

```
Strategy: monitor_no_revision_triage (semantic)
- Single-shot solve
- Semantic monitor check
- If signal >= threshold: accept
- If signal < threshold: verify
- If verify passes: accept
- If verify fails: escalate (no revision)
```

**Rationale**: 10% NEAR zone means revision occasionally helps but not enough to justify the cost.

### For 3B Models

```
Strategy: monitor_repair_triage (conditional)
- Single-shot solve
- Monitor check
- If signal >= threshold: accept
- If signal in mid-range: verify + revision
- If signal < threshold: escalate
```

**Rationale**: 20% NEAR zone makes revision viable for cost-insensitive deployments.

---

## Monitor Reliability

| Monitor | 0.5B Detection | 1.5B Detection | 3B Detection | Notes |
|---------|---------------|---------------|-------------|-------|
| confidence | **0%** | **0%** | **0%** | Completely unreliable |
| semantic | 76-89% | ~80% | ~80% | Direction inverted (higher for correct) |
| counterfactual | 81-88% | ~85% | ~85% | Weak positive separation |

**Recommendation**: Use semantic monitor as primary signal. Do not use confidence.

---

## Cost Model

| Action | Cost Units | When to Use |
|--------|-----------|-------------|
| local_only | 1.0 | Baseline, no routing |
| verify | +0.5 | Always for uncertain outputs |
| revision | +1.5 | Only for 3B+ models |
| escalate | +3.0 | When weak model fails |

**Savings with no-revision triage**: ~50% vs verifier_first (eliminates all revision calls).

---

## Integration with LSG

Capability Router decisions should be treated as **proposals** to LSG, not direct state changes:

1. Router decides "escalate" → LSG records proposal
2. LSG gate checks: is escalation threshold valid?
3. LSG commit log records the decision
4. System state (routing policy) only changes after LSG approval

**Constraint**: Router failures should not automatically trigger policy changes. Batch-level drift (TopoMem OBD) can propose policy review, but execution requires LSG approval.

---

## Open Questions

1. **Does semantic monitor generalize to non-code tasks?** Untested on reasoning/QA.
2. **What is the optimal threshold for 7B models?** NEAR zone may exceed ABOVE at larger scales.
3. **Can weak_model_confidence be calibrated per task type?** Current calibration is uniform.

---

## Files

- `core/weak_model_confidence.py` — Alternative confidence estimation
- `tests/test_weak_model_confidence.py` — Unit tests
- `core/capability_benchmark.py` — Policy implementations
- `experiments/capability/llm_routing_experiment.py` — Experiment framework
