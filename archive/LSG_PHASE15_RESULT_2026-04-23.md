# LSG Phase 15 Result - Multi-Candidate Sequence and Commit Log Invariants

Date: 2026-04-23

## Decision

Phase 15 extends temporal validation from one candidate to multiple candidates competing under bandwidth pressure.

It also adds a commit log invariant checker for sequence replay.

## Implemented Artifacts

Updated:

- `core/rewrite_sequence_replay.py`
- `data/lsg/proposal_sequence_replay_v0.json`
- `tests/test_rewrite_sequence_replay.py`
- `tests/test_rewrite_dynamics.py`
- `tests/smoke_test.py`

## Commit Log Invariant Checker

Added:

```text
check_commit_log_invariants(state)
```

It verifies:

- no duplicate commit for the same candidate
- every commit references an existing candidate
- every committed candidate has a commit log event
- every logged commit candidate is finally committed
- every logged commit candidate is finally acknowledged
- every commit event has evidence, constitution, and log gates open
- every commit event is executed

Sequence replay now includes:

```text
invariants
invariant_failed_count
```

## Multi-Candidate Fixture

Added dataset case:

```text
multi_candidate_bandwidth_top3
```

One step presents 5 high-pressure candidates with all gates open.

With bandwidth limit 3, expected behavior:

```text
c_bw_0 committed
c_bw_1 committed
c_bw_2 committed
c_bw_3 not committed
c_bw_4 not committed
commit_log_count: 3
invariants.passed: true
```

This clarifies current step semantics:

> candidates over the bandwidth limit do not commit in the same step, even if their gates are open.

Also clarified:

> after top candidates commit, the same step does not refill active slots with lower-priority candidates.

## Result

Sequence replay summary:

```text
num_cases: 3
passed: true
failed_count: 0
invariant_failed_count: 0
total_commit_log_count: 5
```

Generated artifacts:

```text
results/capability_generalization/rewrite_sequence_replay_fixture_phase15_multicandidate.json
results/capability_generalization/rewrite_sequence_replay_fixture_phase15_multicandidate_fixed.json
results/capability_generalization/rewrite_source_distribution_compare_phase15_regression.json
```

## Validation

Commands run:

```text
python -m py_compile F:\unified-sel\core\rewrite_sequence_replay.py F:\unified-sel\tests\test_rewrite_sequence_replay.py F:\unified-sel\experiments\capability\rewrite_sequence_replay_fixture.py
python F:\unified-sel\tests\test_rewrite_sequence_replay.py
python F:\unified-sel\experiments\capability\rewrite_sequence_replay_fixture.py --label phase15_multicandidate
python F:\unified-sel\tests\test_rewrite_dynamics.py
python F:\unified-sel\experiments\capability\rewrite_sequence_replay_fixture.py --label phase15_multicandidate_fixed
python F:\unified-sel\tests\test_rewrite_proposal_provider.py
python F:\unified-sel\tests\test_rewrite_proxy_mediator.py
python F:\unified-sel\tests\test_rewrite_proposal_replay.py
python F:\unified-sel\tests\test_rewrite_source_distribution.py
python F:\unified-sel\tests\test_rewrite_proxy_calibration.py
python F:\unified-sel\tests\test_rewrite_provider_capture.py
python F:\unified-sel\tests\test_rewrite_live_provider_capture.py
python F:\unified-sel\experiments\capability\rewrite_source_distribution_compare.py --label phase15_regression
python F:\unified-sel\tests\smoke_test.py
```

Observed status:

```text
All rewrite sequence replay tests passed
All rewrite dynamics tests passed
All rewrite proposal provider tests passed
All rewrite proxy mediator tests passed
All rewrite proposal replay tests passed
All rewrite source distribution tests passed
All rewrite proxy calibration tests passed
All rewrite provider capture tests passed
All rewrite live provider capture tests passed
All smoke tests passed
```

The first smoke attempt failed because smoke still expected the old sequence total commit count of 2.  The dataset now has 3 cases and 5 total commits.  The smoke assertion was updated and the rerun passed.

The smoke run emitted an unrelated Hugging Face unauthenticated-request warning while loading `sentence-transformers/all-MiniLM-L6-v2`; it did not fail the run.

## Interpretation

Phase 15 strengthens the bandwidth claim:

> commit eligibility is not enough; a candidate must also survive active bandwidth selection in that step.

It also adds a reusable invariant checker for future sequence and stress tests.

## Remaining Useful Work

Next useful targets:

- repeated proposal IDs distinct from candidate IDs
- explicit multi-step refilling behavior after top candidates commit
- randomized sequence fuzzing with invariant checks
- optional revision/rollback protocol for acknowledged candidates

