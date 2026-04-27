# LSG Phase 17 Result - Proposal/Candidate Identity Boundary

Date: 2026-04-24

## Decision

Phase 17 locks the distinction between proposal-event identity and candidate-state identity.

The boundary is:

```text
proposal_id = one proposal event
candidate_id = durable candidate state updated by observations
```

Model/provider output may propose a candidate, but it must not collapse proposal identity into candidate identity, and a proposal event ID must not be reused inside a sequence.

## Implemented Artifacts

Updated:

- `core/rewrite_dynamics.py`
  - `ProposalEnvelope.__post_init__()` now rejects `proposal_id == candidate_id`.

- `core/rewrite_sequence_replay.py`
  - added `check_sequence_identity_invariants(raw_case)`
  - sequence replay results now include `identity_invariants`
  - sequence summaries now include `identity_failed_count`
  - `replay_sequence_case()` only passes when commit-log invariants and identity invariants both pass

- `tests/test_rewrite_proposal_provider.py`
  - validates schema rejection for proposal/candidate ID collision

- `tests/test_rewrite_sequence_replay.py`
  - validates normal fixture identity pass
  - validates duplicate proposal ID detection
  - validates cross-candidate proposal ID reuse detection

- `tests/smoke_test.py`
  - checks `identity_failed_count == 0` for the sequence replay fixture

## Identity Invariants

The sequence identity checker reports:

```text
duplicate_proposal_ids
cross_candidate_reuse
proposal_candidate_collisions
```

Failure cases:

- same `proposal_id` appears more than once in a sequence
- same `proposal_id` is reused for different `candidate_id` values
- `proposal_id == candidate_id`

## Result

Artifact command:

```text
python experiments/capability/rewrite_sequence_replay_fixture.py --label phase17_identity
```

Artifact summary:

```text
passed: true
num_cases: 3
failed_count: 0
invariant_failed_count: 0
identity_failed_count: 0
total_commit_log_count: 5
```

Result artifact:

```text
results/capability_generalization/rewrite_sequence_replay_fixture_phase17_identity.json
```

## Validation

Commands run:

```text
python -m py_compile core\rewrite_dynamics.py core\rewrite_sequence_replay.py tests\test_rewrite_proposal_provider.py tests\test_rewrite_sequence_replay.py tests\smoke_test.py
python tests\test_rewrite_proposal_provider.py
python tests\test_rewrite_sequence_replay.py
python tests\smoke_test.py
python experiments\capability\rewrite_sequence_replay_fixture.py --label phase17_identity
```

Observed status:

```text
All rewrite proposal provider tests passed
All rewrite sequence replay tests passed
All smoke tests passed
```

The smoke run emitted the existing Hugging Face unauthenticated-request warning while loading `sentence-transformers/all-MiniLM-L6-v2`; it did not fail the run.

## Interpretation

Phase 17 strengthens the LSG authority boundary:

> provider/model proposal events can update a candidate state only through explicit candidate identity, and proposal-event identity cannot be reused or collapsed into candidate identity.

This prevents a common class of audit ambiguity where an event ID becomes indistinguishable from the durable state key it is trying to modify.

## Remaining Useful Work

- explicit multi-step refilling behavior after top candidates commit
- optional revision/rollback protocol for acknowledged candidates
- larger provider capture set for identity/error distribution checks
