# LSG Phase 16 Result - Randomized Sequence Fuzz Smoke Coverage

Date: 2026-04-24

## Decision

Phase 16 formalizes randomized sequence fuzzing for the LSG multi-candidate replay path.

The goal is not to prove arbitrary correctness.  The goal is to make the existing sequence invariants run against randomized multi-step, multi-candidate observations so future changes are less likely to silently break bandwidth, commit-log, or acknowledged-state behavior.

## Implemented Artifacts

Updated:

- `tests/smoke_test.py`
  - adds fixed-seed randomized sequence fuzz coverage to the LSG smoke path
  - checks `passed`, `failed_seeds`, `max_active_observed`, per-seed commit-log invariants, and acknowledged-state mutation errors

Existing artifacts used:

- `experiments/capability/rewrite_sequence_fuzz.py`
- `tests/test_rewrite_sequence_fuzz.py`
- `core/rewrite_sequence_replay.py`

## Smoke Fix

The first full smoke run failed before reaching LSG fuzz coverage.

Failure:

```text
PermissionError during tempfile.TemporaryDirectory cleanup in test_transfer_matrix_runner
```

Cause:

```text
test_transfer_matrix_runner created an unused system temporary directory.
The test output is written to results/capability_generalization, so the temp directory was unnecessary.
```

Fix:

- removed the unused `TemporaryDirectory()` wrapper from `test_transfer_matrix_runner`
- preserved the actual transfer-matrix subprocess execution and result JSON assertions

## Fuzz Configuration

Smoke-path fixed seeds:

```text
seeds: [0, 1, 2]
num_steps: 8
num_candidates: 6
max_observations_per_step: 4
bandwidth_limit: 3
```

Artifact run:

```text
python experiments/capability/rewrite_sequence_fuzz.py --seeds 0,1,2,3,4 --num-steps 12 --num-candidates 8 --max-observations-per-step 5 --bandwidth-limit 3 --label phase16_smoke
```

Observed artifact summary:

```text
passed: true
num_seeds: 5
failed_seeds: []
max_active_observed: 3
total_commit_log_count: 13
```

Result artifact:

```text
results/capability_generalization/rewrite_sequence_fuzz_phase16_smoke.json
```

## Validation

Commands run:

```text
python tests/test_rewrite_sequence_fuzz.py
python tests/smoke_test.py
python experiments/capability/rewrite_sequence_fuzz.py --seeds 0,1,2,3,4 --num-steps 12 --num-candidates 8 --max-observations-per-step 5 --bandwidth-limit 3 --label phase16_smoke
```

Observed status:

```text
All rewrite sequence fuzz tests passed
All smoke tests passed
```

The smoke run emitted the existing Hugging Face unauthenticated-request warning while loading `sentence-transformers/all-MiniLM-L6-v2`; it did not fail the run.

## Interpretation

Phase 16 strengthens the LSG engineering guarantee:

> randomized multi-candidate sequence pressure does not violate the current commit-log invariants, bandwidth cap, or acknowledged-state absorption checks on the fixed smoke/fuzz seeds.

This remains an engineering regression guard, not a theorem-level proof.

## Remaining Useful Work

- repeated proposal IDs distinct from candidate IDs
- explicit multi-step refilling behavior after top candidates commit
- optional revision/rollback protocol for acknowledged candidates
