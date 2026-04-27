# LSG Phase 7 Result - Provider Output Capture Protocol

Date: 2026-04-23

## Decision

Phase 7 defines an offline provider-output capture protocol.

The trusted path remains:

```text
captured provider JSONL -> replay dataset -> schema validation -> mediation -> calibration -> commit outcome
```

Provider output is never executed directly.

## Implemented Artifacts

- `core/rewrite_provider_capture.py`
  - `ProviderCaptureRequest`
  - `ProviderCaptureRecord`
  - `build_provider_prompt`
  - `capture_provider_output`
  - `capture_record_to_json`
  - `write_capture_jsonl`
  - `load_capture_jsonl`
  - `capture_json_to_replay_case`
  - `capture_jsonl_to_replay_dataset`

- `data/lsg/provider_capture_v0.jsonl`
  - local capture fixture with one valid provider output and one schema-error output

- `tests/test_rewrite_provider_capture.py`
  - verifies prompt includes all required proxy fields
  - verifies capture record serialization
  - verifies captured JSONL replays through the trusted boundary

- `experiments/capability/rewrite_provider_capture_replay.py`
  - converts captured provider outputs to replay cases and evaluates them

- `tests/smoke_test.py`
  - includes capture replay summary checks in the LSG smoke path

## Prompt Boundary

The provider prompt asks for exactly one JSON object with:

- `proposal_id`
- `candidate_id`
- `candidate_summary`
- `proxy` with exactly 8 fields:
  - `u1_conflict`
  - `u2_mismatch`
  - `n1_goal_loss_if_ignored`
  - `n2_commitment_carry_cost`
  - `a1_institutional_level`
  - `a2_current_anchor_strength`
  - `p1_dependency_fanout`
  - `p2_rollback_cost`

The prompt explicitly states that requested gates or threshold updates are audit-only.

## Result

Observed capture replay summary:

```text
num_cases: 2
passed: true
failed_count: 0
false_commit_count: 0
missed_commit_count: 0
schema_error_count: 1
authority_request_count: 0
proxy_disagreement_count: 0
failure_class_counts:
  none: 1
  schema_error: 1
```

Generated artifact:

```text
results/capability_generalization/rewrite_provider_capture_replay_phase7_smoke.json
```

## Validation

Commands run:

```text
python -m py_compile F:\unified-sel\core\rewrite_provider_capture.py F:\unified-sel\tests\test_rewrite_provider_capture.py F:\unified-sel\experiments\capability\rewrite_provider_capture_replay.py F:\unified-sel\tests\smoke_test.py
python F:\unified-sel\tests\test_rewrite_provider_capture.py
python F:\unified-sel\experiments\capability\rewrite_provider_capture_replay.py --label phase7_smoke
python F:\unified-sel\tests\test_rewrite_proposal_replay.py
python F:\unified-sel\tests\test_rewrite_proxy_calibration.py
python F:\unified-sel\tests\test_rewrite_proxy_mediator.py
python F:\unified-sel\tests\test_rewrite_dynamics.py
python F:\unified-sel\tests\test_rewrite_proposal_provider.py
python F:\unified-sel\tests\smoke_test.py
```

Observed status:

```text
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

Phase 7 makes future MiniMax or LLM integration evidential instead of authoritative.

The provider can now be used to generate proposal candidates, but those candidates must be captured as replayable JSONL and evaluated offline before they affect the project.

This preserves the main LSG rule:

> provider output is proposal evidence, not state rewrite authority.

## Next Phase

Phase 8 can safely add an optional provider adapter behind this capture protocol:

- read API key only from environment
- make no provider call from smoke tests
- write raw outputs to `data/lsg/provider_captures/`
- immediately replay the capture file
- report schema error, authority request, proxy disagreement, false commit, and missed commit counts

