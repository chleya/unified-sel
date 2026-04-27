# LSG Phase 9 Result - Proposal Source Distribution Comparison

Date: 2026-04-23

## Decision

Phase 9 adds source-level distribution comparison for proposal streams.

The comparison layer treats every source as replayable evidence:

```text
source file -> replay rows -> source summary -> cross-source comparison
```

It does not call live providers and does not affect commit authority.

## Implemented Artifacts

- `core/rewrite_source_distribution.py`
  - `ProposalSourceSpec`
  - `ProposalSourceSummary`
  - `rows_from_source`
  - `summarize_source`
  - `compare_sources`

- `tests/test_rewrite_source_distribution.py`
  - verifies replay JSON source loading
  - verifies capture JSONL source loading
  - verifies source-level rates
  - verifies cross-source comparison

- `experiments/capability/rewrite_source_distribution_compare.py`
  - compares default local sources
  - accepts additional sources in `name:kind:path` form

- `tests/smoke_test.py`
  - includes source-distribution checks in the LSG smoke path

## Default Sources

### hand_authored_replay

```text
kind: replay_json
path: data/lsg/proposal_replay_v0.json
```

### provider_capture_fixture

```text
kind: capture_jsonl
path: data/lsg/provider_capture_v0.jsonl
```

## Result

Observed comparison summary:

```text
num_sources: 2
passed: true
```

### hand_authored_replay

```text
num_cases: 4
passed: true
schema_error_rate: 0.25
authority_request_rate: 0.25
proxy_disagreement_rate: 0.25
false_commit_count: 0
missed_commit_count: 0
mean_proxy_delta: 0.5666666666666667
max_proxy_delta: 0.85
failure_class_counts:
  none: 1
  authority_request: 1
  proxy_disagreement: 1
  schema_error: 1
```

### provider_capture_fixture

```text
num_cases: 2
passed: true
schema_error_rate: 0.5
authority_request_rate: 0.0
proxy_disagreement_rate: 0.0
false_commit_count: 0
missed_commit_count: 0
mean_proxy_delta: 0.0
max_proxy_delta: 0.0
failure_class_counts:
  none: 1
  schema_error: 1
```

Generated artifact:

```text
results/capability_generalization/rewrite_source_distribution_compare_phase9_smoke.json
```

## Validation

Commands run:

```text
python -m py_compile F:\unified-sel\core\rewrite_source_distribution.py F:\unified-sel\tests\test_rewrite_source_distribution.py F:\unified-sel\experiments\capability\rewrite_source_distribution_compare.py F:\unified-sel\tests\smoke_test.py
python F:\unified-sel\tests\test_rewrite_source_distribution.py
python F:\unified-sel\experiments\capability\rewrite_source_distribution_compare.py --label phase9_smoke
python F:\unified-sel\tests\test_rewrite_live_provider_capture.py
python F:\unified-sel\tests\test_rewrite_provider_capture.py
python F:\unified-sel\tests\test_rewrite_proposal_replay.py
python F:\unified-sel\tests\test_rewrite_proxy_calibration.py
python F:\unified-sel\tests\test_rewrite_proxy_mediator.py
python F:\unified-sel\tests\test_rewrite_dynamics.py
python F:\unified-sel\tests\test_rewrite_proposal_provider.py
python F:\unified-sel\experiments\capability\rewrite_live_provider_capture.py --label phase9_dry_run
python F:\unified-sel\tests\smoke_test.py
```

Observed status:

```text
All rewrite source distribution tests passed
All rewrite live provider capture tests passed
All rewrite provider capture tests passed
All rewrite proposal replay tests passed
All rewrite proxy calibration tests passed
All rewrite proxy mediator tests passed
All rewrite dynamics tests passed
All rewrite proposal provider tests passed
All smoke tests passed
```

The smoke run emitted an unrelated Hugging Face unauthenticated-request warning while loading `sentence-transformers/all-MiniLM-L6-v2`; it did not fail the run.

## Interpretation

Phase 9 gives LSG a source-quality dashboard.

Future provider captures can be compared against hand-authored fixtures by the same rates:

- schema error rate
- authority request rate
- proxy disagreement rate
- false commit count
- missed commit count
- mean proxy delta
- max proxy delta

The important boundary remains:

> source comparison is observational; it does not change commit authority.

## Next Phase

Phase 10 should produce a consolidated LSG milestone report:

- architecture chain
- implemented phases
- current guarantees
- remaining non-guarantees
- command checklist
- safe MiniMax/live-provider usage recipe

