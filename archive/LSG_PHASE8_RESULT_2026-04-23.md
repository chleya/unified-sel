# LSG Phase 8 Result - Optional Live Provider Capture Adapter

Date: 2026-04-23

## Decision

Phase 8 adds an optional live provider capture adapter.

No live provider call is made by tests or smoke.

The only trusted live-provider path is:

```text
explicit --live command
-> environment-only key/url/model
-> provider response
-> captured JSONL
-> replay
-> schema validation
-> mediation
-> calibration
-> commit outcome
```

Provider output still has no direct commit authority.

## Implemented Artifacts

- `core/rewrite_live_provider_capture.py`
  - `LiveProviderConfig`
  - `JsonTransport`
  - `UrllibJsonTransport`
  - `build_live_provider_payload`
  - `extract_json_object_from_text`
  - `extract_model_json_from_response`
  - `capture_live_provider_output`

- `tests/test_rewrite_live_provider_capture.py`
  - verifies live config requires environment variables
  - verifies JSON extraction from plain text, fenced JSON, and response shapes
  - verifies fake-transport capture
  - verifies payload construction

- `experiments/capability/rewrite_live_provider_capture.py`
  - default mode is dry-run
  - `--live` is required to make a provider call
  - API key/url/model are read only from environment
  - captured output is written to `data/lsg/provider_captures/`
  - captured output is immediately replayed

## Required Environment For Live Calls

For `--provider-name minimax`, the live command requires:

```text
MINIMAX_API_KEY
MINIMAX_API_URL
MINIMAX_MODEL
```

No API key can be supplied by CLI argument.

## Dry Run

Dry run command:

```text
python F:\unified-sel\experiments\capability\rewrite_live_provider_capture.py --label phase8_dry_run
```

Observed:

```text
mode: dry_run
provider_name: minimax
required_env:
  MINIMAX_API_KEY
  MINIMAX_API_URL
  MINIMAX_MODEL
```

The dry run prints the provider prompt and makes no network request.

## Regression Results

Capture replay:

```text
num_cases: 2
passed: true
failed_count: 0
false_commit_count: 0
missed_commit_count: 0
schema_error_count: 1
```

Proposal replay:

```text
num_cases: 4
passed: true
failed_count: 0
false_commit_count: 0
missed_commit_count: 0
schema_error_count: 1
authority_request_count: 1
proxy_disagreement_count: 1
```

Generated artifacts:

```text
results/capability_generalization/rewrite_provider_capture_replay_phase8_regression.json
results/capability_generalization/rewrite_proposal_replay_fixture_phase8_regression.json
```

## Validation

Commands run:

```text
python -m py_compile F:\unified-sel\core\rewrite_live_provider_capture.py F:\unified-sel\tests\test_rewrite_live_provider_capture.py F:\unified-sel\experiments\capability\rewrite_live_provider_capture.py
python F:\unified-sel\tests\test_rewrite_live_provider_capture.py
python F:\unified-sel\experiments\capability\rewrite_live_provider_capture.py --label phase8_dry_run
python F:\unified-sel\tests\test_rewrite_provider_capture.py
python F:\unified-sel\tests\test_rewrite_proposal_replay.py
python F:\unified-sel\tests\test_rewrite_proxy_calibration.py
python F:\unified-sel\tests\test_rewrite_proxy_mediator.py
python F:\unified-sel\tests\test_rewrite_dynamics.py
python F:\unified-sel\tests\test_rewrite_proposal_provider.py
python F:\unified-sel\experiments\capability\rewrite_provider_capture_replay.py --label phase8_regression
python F:\unified-sel\experiments\capability\rewrite_proposal_replay_fixture.py --label phase8_regression
python F:\unified-sel\tests\smoke_test.py
```

Observed status:

```text
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

Phase 8 creates the integration seam for MiniMax without weakening LSG.

The important constraint is preserved:

> live model output is converted into replayable evidence, not trusted state.

## Next Phase

Phase 9 should add distribution comparison for proposal sources:

- hand-authored fixtures
- mock provider captures
- optional live provider captures

Metrics:

- schema error rate
- authority request rate
- proxy disagreement rate
- false commit count
- missed commit count
- mean/max proxy delta

