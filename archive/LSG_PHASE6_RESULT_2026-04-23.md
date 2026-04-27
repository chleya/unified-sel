# LSG Phase 6 Result - Local Proposal Replay Dataset

Date: 2026-04-23

## Decision

Phase 6 introduces a local replay dataset for untrusted proposal JSON.

The replay path is:

```text
model_json -> schema validation -> mediation -> calibration -> commit outcome -> failure class
```

No network provider is used in this phase.  MiniMax remains outside the trusted path.

## Implemented Artifacts

- `data/lsg/proposal_replay_v0.json`
  - local replay fixture dataset

- `core/rewrite_proposal_replay.py`
  - `ProposalReplayRow`
  - `load_replay_dataset`
  - `explicit_state_from_json`
  - `classify_failure`
  - `replay_proposal_case`
  - `summarize_replay`
  - `replay_dataset`

- `tests/test_rewrite_proposal_replay.py`
  - validates explicit state requirements
  - validates failure classification priority
  - validates replay summary
  - validates expected schema-error interception

- `experiments/capability/rewrite_proposal_replay_fixture.py`
  - writes a JSON replay result artifact

- `tests/smoke_test.py`
  - includes replay summary checks in the LSG smoke path

## Dataset Cases

### valid_aligned_commit

Valid JSON, aligned model/system proxy, gates open.

Expected:

```text
committed = true
failure_class = none
```

### authority_overclaim_blocked

Valid JSON, model overclaims pressure and requests gate/threshold authority.
System explicit state blocks authority and gates remain closed.

Expected:

```text
committed = false
failure_class = authority_request
```

### underclaim_corrected_commit

Valid JSON, model underclaims pressure.
System explicit state corrects proxy values and gates are open.

Expected:

```text
committed = true
failure_class = proxy_disagreement
```

### schema_bad_score

Invalid JSON because one proxy score is outside `[0, 1]`.

Expected:

```text
status = passed
failure_class = schema_error
committed = null
```

This is a successful interception, not a project failure.

## Result

Observed replay summary:

```text
num_cases: 4
passed: true
failed_count: 0
false_commit_count: 0
missed_commit_count: 0
schema_error_count: 1
authority_request_count: 1
proxy_disagreement_count: 1
failure_class_counts:
  none: 1
  authority_request: 1
  proxy_disagreement: 1
  schema_error: 1
```

Generated artifact:

```text
results/capability_generalization/rewrite_proposal_replay_fixture_phase6_smoke.json
```

## Validation

Commands run:

```text
python -m py_compile F:\unified-sel\core\rewrite_proposal_replay.py F:\unified-sel\tests\test_rewrite_proposal_replay.py F:\unified-sel\experiments\capability\rewrite_proposal_replay_fixture.py F:\unified-sel\tests\smoke_test.py
python F:\unified-sel\tests\test_rewrite_proposal_replay.py
python F:\unified-sel\experiments\capability\rewrite_proposal_replay_fixture.py --label phase6_smoke
python F:\unified-sel\tests\test_rewrite_proxy_calibration.py
python F:\unified-sel\tests\test_rewrite_proxy_mediator.py
python F:\unified-sel\tests\test_rewrite_dynamics.py
python F:\unified-sel\tests\test_rewrite_proposal_provider.py
python F:\unified-sel\tests\smoke_test.py
```

Observed status:

```text
All rewrite proposal replay tests passed
All rewrite proxy calibration tests passed
All rewrite proxy mediator tests passed
All rewrite dynamics tests passed
All rewrite proposal provider tests passed
All smoke tests passed
```

The smoke run emitted an unrelated Hugging Face unauthenticated-request warning while loading `sentence-transformers/all-MiniLM-L6-v2`; it did not fail the run.

## Interpretation

Phase 6 turns future LLM integration into replayable evidence.

The important boundary is:

> a model provider can add proposal JSON samples, but those samples must pass schema validation, mediation, calibration, and commit-gate replay before they mean anything.

This prevents "LLM says it should commit" from becoming an authority path.

## Next Phase

Phase 7 should add a provider-output capture protocol:

- define a stable prompt and response schema for proposal generation
- capture provider outputs as replay JSON
- keep provider calls optional and offline-replayable
- compare mock, hand-authored, and provider-generated proposal distributions
- do not let provider output bypass replay

