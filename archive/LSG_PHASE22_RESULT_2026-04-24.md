# LSG Phase 22 Result - Versioned Candidate Identity

Date: 2026-04-24

## Decision

Phase 22 adds candidate version identity before any rollback/revision execution protocol:

```text
CandidateState.version
CommitEvent.candidate_version
RevisionProposalEvent.target_version
```

This does not implement rollback. It only ensures that future rollback/revision execution can target a specific acknowledged candidate version instead of ambiguously rewriting a durable candidate ID.

## Implemented Artifacts

Updated:

- `core/rewrite_dynamics.py`
  - added `CandidateState.version`, default `1`
  - added `CommitEvent.candidate_version`
  - added `RevisionProposalEvent.target_version`
  - `propose_revision_for_acknowledged_candidate` accepts optional `target_version`
  - stale `target_version` is rejected
  - revision invariants now check target version matches current acknowledged candidate version
  - timelines include candidate version
  - CEE projection raw value includes `candidate_version`

- `core/rewrite_sequence_replay.py`
  - sequence timeline rows include candidate version
  - revision request payload may include `target_version`
  - stale or invalid target versions are rejected
  - commit-log invariants check commit event version matches candidate version

- `data/lsg/proposal_sequence_replay_v0.json`
  - revision fixture now includes `"target_version": 1`

- `experiments/capability/rewrite_sequence_fuzz.py`
  - fuzz timeline rows include candidate version

- `tests/test_rewrite_dynamics.py`
  - commit/revision event validation now covers version fields
  - audit-only revision proposal checks commit and revision versions
  - added stale-version rejection test

- `tests/test_rewrite_sequence_replay.py`
  - replay revision fixture checks commit/revision version fields
  - added stale target version rejection test

- `tests/smoke_test.py`
  - LSG smoke checks commit/revision version binding

## Result

Replay artifact command:

```text
python experiments\capability\rewrite_sequence_replay_fixture.py --label phase22_versioned_identity
```

Replay artifact summary:

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

Replay result artifact:

```text
results/capability_generalization/rewrite_sequence_replay_fixture_phase22_versioned_identity.json
```

Fuzz artifact command:

```text
python experiments\capability\rewrite_sequence_fuzz.py --seeds 0,1,2,3,4 --num-steps 12 --num-candidates 8 --max-observations-per-step 5 --bandwidth-limit 3 --label phase22_versioned_fuzz
```

Fuzz artifact summary:

```text
passed: true
num_seeds: 5
failed_seeds: []
max_active_observed: 3
total_commit_log_count: 13
total_revision_log_count: 31
```

Fuzz result artifact:

```text
results/capability_generalization/rewrite_sequence_fuzz_phase22_versioned_fuzz.json
```

## Validation

Commands run:

```text
python tests\test_rewrite_dynamics.py
python tests\test_rewrite_sequence_replay.py
python tests\test_rewrite_sequence_fuzz.py
python -m json.tool data\lsg\proposal_sequence_replay_v0.json
python experiments\capability\rewrite_sequence_replay_fixture.py --label phase22_versioned_identity
python experiments\capability\rewrite_sequence_fuzz.py --seeds 0,1,2,3,4 --num-steps 12 --num-candidates 8 --max-observations-per-step 5 --bandwidth-limit 3 --label phase22_versioned_fuzz
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

Phase 22 prevents a future rollback protocol from being underspecified:

> durable candidate ID identifies the line of state; candidate version identifies the specific acknowledged state revision.

Current behavior is still conservative:

```text
all candidates start at version 1
acknowledgement records version 1
revision audit targets version 1
no execution creates version 2 yet
```

## Remaining Useful Work

- design explicit revision execution event
- decide whether execution creates `version + 1` or a linked replacement candidate
- define rollback semantics separately from revision semantics
- add sequence fixture for approved-but-not-executed revision before implementing execution
