# LSG Phase 11 Result - Boundary Audit Hardening

Date: 2026-04-23

## Decision

Phase 11 performs a boundary audit instead of adding new functionality.

The audit found a concrete bypass risk:

```text
explicit_state_from_json() used bool(value) for gate fields
```

That meant malformed replay/capture data such as:

```json
{"evidence_open": "false"}
```

would evaluate as `True` in Python.

This is a boundary bug because replay/capture explicit gates are part of the trusted path.

## Fix

Updated:

- `core/rewrite_proposal_replay.py`
- `core/rewrite_proxy_mediator.py`

### Replay JSON hardening

`explicit_state_from_json()` now requires:

- `evidence_open`, `constitution_open`, `log_ready` are real booleans
- `a1_institutional_level`, `p1_dependency_fanout` are numeric scores in `[0, 1]`
- boolean values are rejected for numeric fields

### Direct Python construction hardening

`ExplicitProxyState.__post_init__()` now validates all direct construction paths:

- required and optional proxy values must be numeric scores in `[0, 1]`
- boolean values are rejected as proxy scores
- gates must be real booleans

This closes both:

```text
JSON replay/capture bypass
direct ExplicitProxyState construction bypass
```

## Added Regression Tests

Updated:

- `tests/test_rewrite_proposal_replay.py`
- `tests/test_rewrite_proxy_mediator.py`

New checks:

```text
explicit state rejects gate string bypass
explicit state rejects bool numeric bypass
explicit proxy state rejects invalid direct values
```

## Validation

Commands run:

```text
python -m py_compile F:\unified-sel\core\rewrite_proposal_replay.py F:\unified-sel\tests\test_rewrite_proposal_replay.py
python F:\unified-sel\tests\test_rewrite_proposal_replay.py
python F:\unified-sel\experiments\capability\rewrite_proposal_replay_fixture.py --label phase11_audit_fix
python F:\unified-sel\experiments\capability\rewrite_source_distribution_compare.py --label phase11_audit_fix
python -m py_compile F:\unified-sel\core\rewrite_proxy_mediator.py F:\unified-sel\tests\test_rewrite_proxy_mediator.py
python F:\unified-sel\tests\test_rewrite_proxy_mediator.py
python F:\unified-sel\tests\test_rewrite_proxy_calibration.py
python F:\unified-sel\tests\test_rewrite_provider_capture.py
python F:\unified-sel\tests\test_rewrite_source_distribution.py
python F:\unified-sel\tests\test_rewrite_live_provider_capture.py
python F:\unified-sel\tests\test_rewrite_dynamics.py
python F:\unified-sel\tests\test_rewrite_proposal_provider.py
python F:\unified-sel\tests\smoke_test.py
```

Observed status:

```text
All rewrite proposal replay tests passed
All rewrite proxy mediator tests passed
All rewrite proxy calibration tests passed
All rewrite provider capture tests passed
All rewrite source distribution tests passed
All rewrite live provider capture tests passed
All rewrite dynamics tests passed
All rewrite proposal provider tests passed
All smoke tests passed
```

The smoke run emitted an unrelated Hugging Face unauthenticated-request warning while loading `sentence-transformers/all-MiniLM-L6-v2`; it did not fail the run.

## Interpretation

This phase strengthens the core LSG boundary:

> explicit gates are typed authority facts, not truthy/falsy strings or coerced values.

The project is now stricter at the exact point where untrusted provider/capture/replay data crosses into trusted explicit state.

## Remaining Audit Targets

Future audits should inspect:

- whether direct `CandidateObservation` construction should also validate types and score ranges
- whether `ProposalEnvelope` should validate non-empty ids and proxy type at dataclass construction time
- whether `RewriteDynamicsConfig` should reject invalid threshold orderings
- whether source-distribution comparison should separately count expected schema interceptions vs unexpected schema failures

