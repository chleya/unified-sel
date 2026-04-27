# LSG Phase 13 Result - Commit Event and Schema Semantics Hardening

Date: 2026-04-23

## Decision

Phase 13 continues boundary hardening in two places:

1. `CommitEvent` construction
2. schema-error reporting semantics

## Part 1: CommitEvent Construction Hardening

Updated:

- `core/rewrite_dynamics.py`
- `tests/test_rewrite_dynamics.py`

`CommitEvent.__post_init__()` now validates:

- `event_id` is non-empty
- `step_index` is an integer and nonnegative
- `candidate_id` is non-empty
- `from_phase` and `to_phase` are non-empty
- `disturbance` and `stability` are in `[0, 1]`
- `ratio` is nonnegative
- `evidence_open`, `constitution_open`, `log_ready`, `commit_executed` are booleans
- executed acknowledgement events require all gates open

This moves invalid commit rejection earlier:

```text
bad CommitEvent construction fails before projection
```

Projection still rejects unexecuted events.

## Part 2: Schema Semantics Split

Updated:

- `core/rewrite_proposal_replay.py`
- `core/rewrite_source_distribution.py`
- `tests/test_rewrite_proposal_replay.py`
- `tests/test_rewrite_source_distribution.py`

Replay rows now carry:

```text
expected_error
```

Replay summary now separates:

```text
schema_error_count
expected_schema_interception_count
unexpected_schema_failure_count
```

Source distribution now reports:

```text
schema_error_rate
expected_schema_interception_rate
unexpected_schema_failure_rate
```

This prevents expected schema interceptions from being misread as unexpected provider failures.

## Result

Source distribution after the change:

```text
hand_authored_replay:
  schema_error_rate: 0.25
  expected_schema_interception_rate: 0.25
  unexpected_schema_failure_rate: 0.0
  authority_request_rate: 0.25
  proxy_disagreement_rate: 0.25
  false_commit_count: 0
  missed_commit_count: 0

provider_capture_fixture:
  schema_error_rate: 0.5
  expected_schema_interception_rate: 0.5
  unexpected_schema_failure_rate: 0.0
  authority_request_rate: 0.0
  proxy_disagreement_rate: 0.0
  false_commit_count: 0
  missed_commit_count: 0
```

Generated artifact:

```text
results/capability_generalization/rewrite_source_distribution_compare_phase13_schema_semantics.json
```

## Validation

Commands run:

```text
python -m py_compile F:\unified-sel\core\rewrite_dynamics.py F:\unified-sel\tests\test_rewrite_dynamics.py
python F:\unified-sel\tests\test_rewrite_dynamics.py
python F:\unified-sel\experiments\capability\rewrite_dynamics_cee_projection.py
python F:\unified-sel\experiments\capability\rewrite_dynamics_cee_roundtrip.py
python F:\unified-sel\tests\test_rewrite_proposal_provider.py
python F:\unified-sel\tests\test_rewrite_proxy_mediator.py
python F:\unified-sel\tests\test_rewrite_proposal_replay.py
python F:\unified-sel\tests\test_rewrite_source_distribution.py
python -m py_compile F:\unified-sel\core\rewrite_proposal_replay.py F:\unified-sel\core\rewrite_source_distribution.py F:\unified-sel\tests\test_rewrite_proposal_replay.py F:\unified-sel\tests\test_rewrite_source_distribution.py
python F:\unified-sel\experiments\capability\rewrite_source_distribution_compare.py --label phase13_schema_semantics
python F:\unified-sel\tests\smoke_test.py
```

Observed status:

```text
All rewrite dynamics tests passed
All rewrite proposal provider tests passed
All rewrite proxy mediator tests passed
All rewrite proposal replay tests passed
All rewrite source distribution tests passed
All smoke tests passed
```

The smoke run emitted an unrelated Hugging Face unauthenticated-request warning while loading `sentence-transformers/all-MiniLM-L6-v2`; it did not fail the run.

## Interpretation

Phase 13 makes two project claims sharper:

1. commit events are authority-bearing objects and must be valid at construction time
2. schema errors are not all the same; expected interceptions and unexpected failures must be reported separately

## Remaining Useful Work

Next high-value validation targets:

- duplicate candidate IDs and repeated candidate observations
- multi-step replay where gates open over time
- config sweep for pathological threshold settings
- commit log invariant checker over arbitrary timelines

