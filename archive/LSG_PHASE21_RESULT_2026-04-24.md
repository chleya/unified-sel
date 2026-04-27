# LSG Phase 21 Result - Revision Fuzz and Negative Replay Cases

Date: 2026-04-24

## Decision

Phase 21 strengthens Phase 20 in two ways:

```text
1. randomized sequence fuzz now mixes audit-only revision requests after acknowledgement
2. sequence replay unit tests now reject invalid revision targets
```

This verifies that revision audit semantics hold outside the hand-authored happy-path fixture.

## Implemented Artifacts

Updated:

- `experiments/capability/rewrite_sequence_fuzz.py`
  - replaced `simulate_case` with explicit per-step `step_system`
  - injects random audit-only revision requests for acknowledged candidates
  - validates `check_revision_log_invariants`
  - checks revision requests do not mutate acknowledged candidate disturbance/stability
  - reports `revision_log_count`, `total_revision_log_count`, `revision_invariants`, and `revision_mutation_errors`

- `tests/test_rewrite_sequence_fuzz.py`
  - asserts fuzz produces revision audit events
  - asserts revision invariants pass
  - asserts revision requests do not mutate acknowledged state

- `tests/test_rewrite_sequence_replay.py`
  - added negative case: revision request against unacknowledged target is rejected
  - added negative case: revision request against missing target is rejected

## Result

Artifact command:

```text
python experiments\capability\rewrite_sequence_fuzz.py --seeds 0,1,2,3,4 --num-steps 12 --num-candidates 8 --max-observations-per-step 5 --bandwidth-limit 3 --label phase21_revision_fuzz
```

Artifact summary:

```text
passed: true
num_seeds: 5
failed_seeds: []
max_active_observed: 3
total_commit_log_count: 13
total_revision_log_count: 31
```

Result artifact:

```text
results/capability_generalization/rewrite_sequence_fuzz_phase21_revision_fuzz.json
```

## Validation

Commands run:

```text
python tests\test_rewrite_sequence_replay.py
python tests\test_rewrite_sequence_fuzz.py
python experiments\capability\rewrite_sequence_fuzz.py --seeds 0,1,2,3,4 --num-steps 12 --num-candidates 8 --max-observations-per-step 5 --bandwidth-limit 3 --label phase21_revision_fuzz
python tests\smoke_test.py
```

Observed status:

```text
All rewrite sequence replay tests passed
All rewrite sequence fuzz tests passed
All smoke tests passed
```

The smoke run emitted the existing Hugging Face unauthenticated-request warning while loading `sentence-transformers/all-MiniLM-L6-v2`; it did not fail the run.

## Interpretation

Phase 21 gives the current revision semantics three independent checks:

```text
direct dynamics unit tests
hand-authored sequence replay fixture
randomized sequence fuzz with revision requests
```

The rule remains:

```text
revision request = durable audit record
revision request != state mutation
revision request != rollback execution
```

Invalid revision targets are rejected instead of being silently logged.

## Remaining Useful Work

- define explicit approval-based revision execution / rollback protocol
- decide whether revision execution should create a new candidate version or revoke an acknowledged candidate
- add versioned candidate identity before implementing actual rollback
