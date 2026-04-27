# LSG Phase 25 Result - Revision Execution Draft Log

Date: 2026-04-24

## Decision

Phase 25 adds a separate `revision_execution_log` for drafted execution events.

This still does not execute revision:

```text
approved RevisionProposalEvent
-> RevisionExecutionEvent draft
-> revision_execution_log append
-> no CandidateState.version change
-> no rollback
```

## Implemented Artifacts

Updated:

- `core/rewrite_dynamics.py`
  - added `RewriteSystemState.revision_execution_log`
  - added `draft_revision_execution_for_candidate`
  - added `record_revision_execution_draft`
  - added `check_revision_execution_log_invariants`

- `core/rewrite_sequence_replay.py`
  - added `revision_execution_drafts` replay payload
  - added `apply_revision_execution_draft`
  - timeline rows now include `revision_execution_events`
  - replay output includes serialized `revision_execution_log`
  - replay output includes `revision_execution_invariants`
  - sequence summary includes:
    - `total_revision_execution_draft_count`
    - `total_executed_revision_execution_count`
    - `revision_execution_invariant_failed_count`

- `data/lsg/proposal_sequence_replay_v0.json`
  - approved revision fixture now drafts one execution event via `revision_execution_drafts`

- `tests/test_rewrite_dynamics.py`
  - added execution-log test proving draft append does not mutate candidate state

- `tests/test_rewrite_sequence_replay.py`
  - asserts one execution draft is logged
  - asserts execution draft has `from_version=1`, `to_version=2`, `execution_executed=false`

- `tests/smoke_test.py`
  - smoke now records an execution draft and checks execution-log invariants
  - smoke checks sequence replay has one draft and zero executed execution events

## Result

Artifact command:

```text
python experiments\capability\rewrite_sequence_replay_fixture.py --label phase25_execution_log
```

Artifact summary:

```text
passed: true
num_cases: 6
failed_count: 0
invariant_failed_count: 0
revision_invariant_failed_count: 0
revision_execution_invariant_failed_count: 0
identity_failed_count: 0
total_commit_log_count: 12
total_revision_log_count: 2
total_approved_revision_count: 1
total_executed_revision_count: 0
total_revision_execution_draft_count: 1
total_executed_revision_execution_count: 0
```

Result artifact:

```text
results/capability_generalization/rewrite_sequence_replay_fixture_phase25_execution_log.json
```

## Validation

Commands run:

```text
python -m json.tool data\lsg\proposal_sequence_replay_v0.json
python tests\test_rewrite_dynamics.py
python tests\test_rewrite_sequence_replay.py
python tests\test_rewrite_sequence_fuzz.py
python experiments\capability\rewrite_sequence_replay_fixture.py --label phase25_execution_log
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

Phase 25 makes execution drafting durable and replayable:

> execution draft is now an audit-log event, not an ephemeral Python object.

The current line remains:

```text
revision proposal log = request layer
revision execution log = draft execution layer
CandidateState.version mutation = still not implemented
rollback = still separate future protocol
```

## Remaining Useful Work

- define the exact state transition for executed revision
- decide whether execution updates existing candidate to `version + 1` or creates a replacement candidate line
- add a negative execution-draft replay case for unapproved proposals
- keep rollback separate from revision execution
