# LSG Phase 20 Result - Revision Requests in Sequence Replay

Date: 2026-04-24

## Decision

Phase 20 moves Phase 19's acknowledged-state revision rule into multi-step replay:

```text
sequence step observations may acknowledge a candidate
same or later step revision_requests may record audit-only revision proposals
revision_requests do not mutate acknowledged candidate state
revision_requests do not append acknowledgement commit events
```

This makes revision semantics testable in fixture data, not only in direct dynamics unit tests.

## Implemented Artifacts

Updated:

- `core/rewrite_sequence_replay.py`
  - replaced black-box `simulate_case` replay with explicit per-step `step_system`
  - added `revision_requests` handling after step observations
  - added `revision_invariants` using `check_revision_log_invariants`
  - added `revision_log_count`, `expected_revision_count`, `total_revision_log_count`
  - added serializable `commit_log` and `revision_log` rows to replay output

- `data/lsg/proposal_sequence_replay_v0.json`
  - added `revision_request_after_acknowledgement_audit_only`
  - case first acknowledges `c_revision_anchor`
  - then records one audit-only revision request with lower disturbance / higher stability
  - later observation tries low-pressure rewrite and is ignored by acknowledged absorption

- `tests/test_rewrite_sequence_replay.py`
  - updated fixture baseline to 5 cases, 11 commit events, 1 revision event
  - added `test_revision_request_after_acknowledgement_is_audit_only`

- `tests/smoke_test.py`
  - updated sequence replay smoke expectations for revision invariants and revision count

## Result

Artifact command:

```text
python experiments\capability\rewrite_sequence_replay_fixture.py --label phase20_revision_replay
```

Artifact summary:

```text
passed: true
num_cases: 5
failed_count: 0
invariant_failed_count: 0
revision_invariant_failed_count: 0
identity_failed_count: 0
total_commit_log_count: 11
total_revision_log_count: 1
```

Result artifact:

```text
results/capability_generalization/rewrite_sequence_replay_fixture_phase20_revision_replay.json
```

## Validation

Commands run:

```text
python -m json.tool data\lsg\proposal_sequence_replay_v0.json
python tests\test_rewrite_sequence_replay.py
python experiments\capability\rewrite_sequence_replay_fixture.py --label phase20_revision_replay
python tests\smoke_test.py
```

Observed status:

```text
All rewrite sequence replay tests passed
All smoke tests passed
```

The smoke run emitted the existing Hugging Face unauthenticated-request warning while loading `sentence-transformers/all-MiniLM-L6-v2`; it did not fail the run.

## Interpretation

Phase 20 closes the replay coverage gap from Phase 19:

> acknowledged-state revision is now represented as durable replay data, not only as an in-memory dynamics API.

The current rule remains intentionally conservative:

```text
revision request = audit record
revision execution = not implemented yet
```

## Remaining Useful Work

- define explicit revision execution / rollback protocol with approval
- add negative sequence cases for invalid revision targets
- extend randomized sequence fuzz to generate revision requests after acknowledgements
