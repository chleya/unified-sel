# LSG Phase 4 Result - Proxy Mediation Boundary

Date: 2026-04-23

## Decision

Phase 4 adds a mediation layer between model proposals and effective rewrite qualification inputs.

The model may propose:

- candidate content
- suggested proxy scores
- optional authority requests for audit

The model may not decide:

- institutional level `a1`
- dependency fanout `p1`
- evidence gate
- constitutional gate
- log readiness
- threshold updates

## Implemented Artifacts

- `core/rewrite_proxy_mediator.py`
  - `ExplicitProxyState`
  - `MediatedProposal`
  - `mediate_proposal`
  - `mediated_audit_record`
  - `observation_from_mediated_proposal`

- `tests/test_rewrite_proxy_mediator.py`
  - verifies system-owned `a1` and `p1` override model proxy values
  - verifies requested gate and threshold authority is ignored
  - verifies high-pressure mediated proposals cannot commit when system gates are closed
  - verifies mediated proposals can commit when system gates are open and stability is low

- `experiments/capability/rewrite_proxy_mediation_sanity.py`
  - produces a JSON sanity artifact comparing closed-gate and open-gate cases

- `tests/smoke_test.py`
  - includes a minimal mediation regression in the LSG smoke path

## Result

The mediation boundary passed:

```text
closed_gates_do_not_commit: true
model_authority_requests_ignored: true
open_system_gates_can_commit: true
```

Generated artifact:

```text
results/capability_generalization/rewrite_proxy_mediation_sanity_phase4_smoke.json
```

## Validation

Commands run:

```text
python -m py_compile F:\unified-sel\core\rewrite_proxy_mediator.py F:\unified-sel\tests\test_rewrite_proxy_mediator.py F:\unified-sel\experiments\capability\rewrite_proxy_mediation_sanity.py F:\unified-sel\tests\smoke_test.py
python F:\unified-sel\tests\test_rewrite_proxy_mediator.py
python F:\unified-sel\experiments\capability\rewrite_proxy_mediation_sanity.py --label phase4_smoke
python F:\unified-sel\tests\test_rewrite_dynamics.py
python F:\unified-sel\tests\test_rewrite_proposal_provider.py
python F:\unified-sel\experiments\capability\rewrite_dynamics_sanity.py
python F:\unified-sel\experiments\capability\rewrite_dynamics_sweep.py --label phase4_regression
python F:\unified-sel\experiments\capability\rewrite_dynamics_proposal_boundary.py
python F:\unified-sel\experiments\capability\rewrite_proposal_schema_validation.py
python F:\unified-sel\experiments\capability\rewrite_dynamics_cee_roundtrip.py
python F:\unified-sel\tests\smoke_test.py
```

Observed status:

```text
All rewrite proxy mediator tests passed
All rewrite dynamics tests passed
All rewrite proposal provider tests passed
All smoke tests passed
```

The smoke run emitted an unrelated Hugging Face unauthenticated-request warning while loading `sentence-transformers/all-MiniLM-L6-v2`; it did not fail the run.

## Interpretation

This phase closes the main loophole introduced by model-generated proposal JSON:

> a model can suggest a high rewrite pressure, but it cannot convert that suggestion into evidence, permission, log readiness, institutional shallowness, or low dependency fanout.

That keeps the project aligned with the central LSG boundary:

> formal state rewrite is separated from ordinary inference and model proposal.

## Next Phase

Phase 5 should move from scripted mediation to proxy calibration:

- build a small table-driven fixture set for `u1/u2/n1/n2/a2/p2`
- compare explicit proxy values against model-suggested proxy values
- measure disagreement and override rates
- keep `a1`, `p1`, gates, and log readiness system-owned

