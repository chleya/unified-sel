# LSG Phase 14 Result - Multi-Step Replay and Absorbing Acknowledgement

Date: 2026-04-23

## Decision

Phase 14 validates temporal boundaries:

- repeated candidate observations
- gates opening after earlier closed-gate pressure
- duplicate high-pressure observations after commit
- acknowledged candidates remaining stable under later observations

This phase moves beyond single-row replay.

## Implemented Artifacts

- `core/rewrite_sequence_replay.py`
  - `SequenceReplayResult`
  - `observation_from_sequence_step`
  - `replay_sequence_case`
  - `summarize_sequence_results`

- `data/lsg/proposal_sequence_replay_v0.json`
  - multi-step replay fixture dataset

- `tests/test_rewrite_sequence_replay.py`
  - validates sequence replay summary
  - validates late gate commit once
  - validates acknowledged candidate absorbs later observation

- `experiments/capability/rewrite_sequence_replay_fixture.py`
  - writes multi-step replay result artifact

- `tests/test_rewrite_dynamics.py`
  - adds direct dynamics tests for late gate commit and absorbing acknowledgement

- `tests/smoke_test.py`
  - includes sequence replay summary in the LSG smoke path

## Semantic Fix

During Phase 14, a sharper issue appeared:

```text
acknowledged candidates kept their phase, but later observations could still update D/S values
```

That made `acknowledged` only phase-absorbing, not state-absorbing.

Updated:

- `core/rewrite_dynamics.py`

`update_candidate()` now returns immediately for:

```text
candidate.committed == true
or
candidate.phase == acknowledged
```

So acknowledged candidates no longer silently change disturbance/stability under ordinary later observations.

## Dataset Cases

### late_gate_commit_once

Same candidate appears across multiple steps:

1. high pressure, evidence closed
2. same candidate, evidence opens
3. same candidate repeats high pressure

Expected:

```text
committed: true
commit_log_count: 1
final_phase: acknowledged
```

### acknowledged_absorbs_later_low_pressure

Candidate commits first, then receives later low-pressure closed-gate observation.

Expected:

```text
committed: true
commit_log_count: 1
final_phase: acknowledged
D/S unchanged after acknowledgement
```

## Result

Observed sequence replay summary:

```text
num_cases: 2
passed: true
failed_count: 0
total_commit_log_count: 2
```

Generated artifacts:

```text
results/capability_generalization/rewrite_sequence_replay_fixture_phase14_sequence.json
results/capability_generalization/rewrite_sequence_replay_fixture_phase14_absorbing_fix.json
```

## Validation

Commands run:

```text
python -m py_compile F:\unified-sel\core\rewrite_sequence_replay.py F:\unified-sel\tests\test_rewrite_sequence_replay.py F:\unified-sel\experiments\capability\rewrite_sequence_replay_fixture.py F:\unified-sel\tests\smoke_test.py
python F:\unified-sel\tests\test_rewrite_dynamics.py
python F:\unified-sel\tests\test_rewrite_sequence_replay.py
python F:\unified-sel\experiments\capability\rewrite_sequence_replay_fixture.py --label phase14_sequence
python F:\unified-sel\experiments\capability\rewrite_sequence_replay_fixture.py --label phase14_absorbing_fix
python F:\unified-sel\tests\test_rewrite_proposal_provider.py
python F:\unified-sel\tests\test_rewrite_proxy_mediator.py
python F:\unified-sel\tests\test_rewrite_proposal_replay.py
python F:\unified-sel\tests\test_rewrite_source_distribution.py
python F:\unified-sel\tests\test_rewrite_proxy_calibration.py
python F:\unified-sel\tests\test_rewrite_provider_capture.py
python F:\unified-sel\tests\test_rewrite_live_provider_capture.py
python F:\unified-sel\experiments\capability\rewrite_source_distribution_compare.py --label phase14_regression
python F:\unified-sel\tests\smoke_test.py
```

Observed status:

```text
All rewrite dynamics tests passed
All rewrite sequence replay tests passed
All rewrite proposal provider tests passed
All rewrite proxy mediator tests passed
All rewrite proposal replay tests passed
All rewrite source distribution tests passed
All rewrite proxy calibration tests passed
All rewrite provider capture tests passed
All rewrite live provider capture tests passed
All smoke tests passed
```

The smoke run emitted an unrelated Hugging Face unauthenticated-request warning while loading `sentence-transformers/all-MiniLM-L6-v2`; it did not fail the run.

## Interpretation

Phase 14 strengthens the temporal claim:

> formal acknowledgement is not just a phase label; it is an absorbing state for that candidate unless an explicit future revision mechanism is introduced.

That is important because LSG's current commit path has no rollback/revision protocol.  Ordinary observations must not silently mutate acknowledged candidates.

## Remaining Useful Work

Next high-value targets:

- multi-candidate sequence replay with bandwidth pressure
- duplicate proposal IDs vs duplicate candidate IDs
- explicit revision/rollback protocol for acknowledged candidates
- commit log invariant checker over arbitrary timelines

