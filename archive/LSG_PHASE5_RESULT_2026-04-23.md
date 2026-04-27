# LSG Phase 5 Result - Proxy Calibration Fixture

Date: 2026-04-23

## Decision

Phase 5 starts proxy calibration without connecting a real LLM provider.

The purpose is to measure the gap between:

- model-suggested proxy values
- system-owned explicit proxy values
- final mediated commit outcomes

This keeps the project on the LSG boundary:

> model proposal is diagnostic input, not rewrite authority.

## Implemented Artifacts

- `core/rewrite_proxy_calibration.py`
  - `ProxyCalibrationCase`
  - `ProxyCalibrationRow`
  - `proxy_abs_deltas`
  - `evaluate_proxy_calibration_case`
  - `summarize_proxy_calibration`

- `tests/test_rewrite_proxy_calibration.py`
  - checks proxy delta calculation
  - checks aligned, overclaim, and underclaim calibration fixtures
  - checks summary metrics

- `experiments/capability/rewrite_proxy_calibration_fixture.py`
  - writes a JSON fixture result with per-case proxy deltas and D/S changes

- `tests/smoke_test.py`
  - includes a minimal proxy calibration regression in the LSG smoke path

## Fixture Families

### aligned_commit

Model proxy and system explicit proxy agree.

Expected:

```text
committed = true
```

### overclaim_blocked

Model claims high disturbance and low stability, and requests authority gates.
System explicit state says disturbance is low, stability is high, and gates are closed.

Expected:

```text
committed = false
```

### underclaim_corrected

Model claims low disturbance and high stability.
System explicit state says disturbance is high, stability is low, and gates are open.

Expected:

```text
committed = true
```

## Result

Observed summary:

```text
num_cases: 3
passed: true
pass_rate: 1.0
false_commit_count: 0
missed_commit_count: 0
override_rate: 0.6666666666666666
authority_request_rate: 0.3333333333333333
mean_abs_delta: 0.5666666666666667
max_abs_delta: 0.85
```

Generated artifact:

```text
results/capability_generalization/rewrite_proxy_calibration_fixture_phase5_smoke.json
```

## Validation

Commands run:

```text
python -m py_compile F:\unified-sel\core\rewrite_proxy_calibration.py F:\unified-sel\tests\test_rewrite_proxy_calibration.py F:\unified-sel\experiments\capability\rewrite_proxy_calibration_fixture.py F:\unified-sel\tests\smoke_test.py
python F:\unified-sel\tests\test_rewrite_proxy_calibration.py
python F:\unified-sel\experiments\capability\rewrite_proxy_calibration_fixture.py --label phase5_smoke
python F:\unified-sel\tests\test_rewrite_proxy_mediator.py
python F:\unified-sel\tests\test_rewrite_dynamics.py
python F:\unified-sel\tests\test_rewrite_proposal_provider.py
python F:\unified-sel\tests\smoke_test.py
```

Observed status:

```text
All rewrite proxy calibration tests passed
All rewrite proxy mediator tests passed
All rewrite dynamics tests passed
All rewrite proposal provider tests passed
All smoke tests passed
```

The smoke run emitted an unrelated Hugging Face unauthenticated-request warning while loading `sentence-transformers/all-MiniLM-L6-v2`; it did not fail the run.

## Interpretation

Phase 5 turns proxy mediation from a yes/no boundary into a measurable calibration surface.

The important new diagnostic is not accuracy alone.  It is:

```text
suggested proxy -> explicit proxy -> effective D/S -> commit outcome
```

That chain lets future LLM or small-head integrations be judged without giving them authority over formal state rewrite.

## Next Phase

Phase 6 should introduce a local proposal replay dataset:

- store proposal JSON examples in a fixture file
- replay them through schema validation, mediation, and calibration
- classify failures as schema error, authority request, proxy disagreement, false commit, or missed commit
- keep MiniMax optional and outside the trusted path

