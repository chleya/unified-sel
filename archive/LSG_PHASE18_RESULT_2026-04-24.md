# LSG Phase 18 Result - Multi-Step Refill Semantics

Date: 2026-04-24

## Decision

Phase 18 makes the bandwidth refill rule explicit:

```text
same step: no refill after top candidates commit
next step: lower-priority candidates may refill if observed again and still qualify
```

This preserves Phase 15's no-same-step-refill semantics while validating that over-bandwidth candidates are not permanently starved.

## Implemented Artifacts

Updated:

- `data/lsg/proposal_sequence_replay_v0.json`
  - added `multi_step_refill_after_top3_commit`
  - first step presents 5 high-pressure candidates with bandwidth limit 3
  - top 3 commit in step 1
  - lower 2 remain uncommitted in step 1
  - lower 2 are observed again in step 2 and commit

- `tests/test_rewrite_sequence_replay.py`
  - updated sequence summary baseline to 4 cases and 10 total commits
  - added `test_multi_step_refill_after_top3_commit`

- `tests/smoke_test.py`
  - updated sequence replay smoke baseline to 4 cases and 10 total commits

## Result

Artifact command:

```text
python experiments\capability\rewrite_sequence_replay_fixture.py --label phase18_refill
```

Artifact summary:

```text
passed: true
num_cases: 4
failed_count: 0
invariant_failed_count: 0
identity_failed_count: 0
total_commit_log_count: 10
```

Result artifact:

```text
results/capability_generalization/rewrite_sequence_replay_fixture_phase18_refill.json
```

## Validation

Commands run:

```text
python -m json.tool data\lsg\proposal_sequence_replay_v0.json
python -m py_compile core\rewrite_sequence_replay.py tests\test_rewrite_sequence_replay.py tests\smoke_test.py
python tests\test_rewrite_sequence_replay.py
python tests\smoke_test.py
python experiments\capability\rewrite_sequence_replay_fixture.py --label phase18_refill
```

Observed status:

```text
All rewrite sequence replay tests passed
All smoke tests passed
```

The smoke run emitted the existing Hugging Face unauthenticated-request warning while loading `sentence-transformers/all-MiniLM-L6-v2`; it did not fail the run.

## Interpretation

Phase 18 closes the remaining refill ambiguity from Phase 15:

> bandwidth overflow suppresses candidates for the current step only; future observations can bring those candidates back into active selection.

This means LSG's current bandwidth semantics are conservative within a step but not permanently dropping lower-priority candidates.

## Remaining Useful Work

- optional revision/rollback protocol for acknowledged candidates
- larger provider capture set for identity/error distribution checks
- stress/fuzz case that mixes refill, duplicate proposal interception, and acknowledged absorption
