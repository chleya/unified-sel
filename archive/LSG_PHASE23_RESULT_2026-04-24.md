# LSG Phase 23 Result - Approval Is Not Execution

Date: 2026-04-24

## Decision

Phase 23 explicitly separates revision approval from revision execution:

```text
approval_open=True means the approval gate is open
approval_open=True does not create version 2
approval_open=True does not mutate acknowledged state
approval_open=True does not mean revision_executed=True
```

This protects the future execution protocol from being accidentally smuggled into the audit layer.

## Implemented Artifacts

Updated:

- `core/rewrite_dynamics.py`
  - `check_revision_log_invariants` now reports:
    - `approved_revision_count`
    - `executed_revision_count`

- `core/rewrite_sequence_replay.py`
  - sequence summaries now report:
    - `total_approved_revision_count`
    - `total_executed_revision_count`

- `data/lsg/proposal_sequence_replay_v0.json`
  - added `approved_revision_request_remains_audit_only`
  - case records an approved revision request against version 1
  - case verifies candidate remains acknowledged version 1

- `tests/test_rewrite_sequence_replay.py`
  - fixture baseline updated to 6 cases, 12 commit events, 2 revision events
  - added `test_approved_revision_request_still_does_not_execute`

- `tests/smoke_test.py`
  - smoke now asserts one approved revision and zero executed revisions in sequence replay

## Result

Artifact command:

```text
python experiments\capability\rewrite_sequence_replay_fixture.py --label phase23_approval_not_execution
```

Artifact summary:

```text
passed: true
num_cases: 6
failed_count: 0
invariant_failed_count: 0
revision_invariant_failed_count: 0
identity_failed_count: 0
total_commit_log_count: 12
total_revision_log_count: 2
total_approved_revision_count: 1
total_executed_revision_count: 0
```

Result artifact:

```text
results/capability_generalization/rewrite_sequence_replay_fixture_phase23_approval_not_execution.json
```

## Validation

Commands run:

```text
python -m json.tool data\lsg\proposal_sequence_replay_v0.json
python tests\test_rewrite_dynamics.py
python tests\test_rewrite_sequence_replay.py
python tests\test_rewrite_sequence_fuzz.py
python experiments\capability\rewrite_sequence_replay_fixture.py --label phase23_approval_not_execution
python tests\smoke_test.py
```

Observed status:

```text
All rewrite dynamics tests passed
All rewrite sequence replay tests passed
All rewrite sequence fuzz tests passed
All smoke tests passed
```

The smoke run emitted the existing Hugging Face unauthenticated-request warning while loading `sentence-transformers/all-MiniLM-L6-v2`; it did not fail the run.

## Interpretation

Phase 23 makes the current boundary explicit:

> revision approval is a gate state inside an audit event, not an executed transition.

Current protocol:

```text
revision request without approval -> audit only
revision request with approval -> audit only
revision execution -> not implemented
rollback -> not implemented
```

## Remaining Useful Work

- define `RevisionExecutionEvent` as a separate event type
- require execution to reference an approved `RevisionProposalEvent`
- decide whether execution creates `CandidateState.version + 1` or a replacement candidate line
- keep rollback as a separate protocol from revision execution
