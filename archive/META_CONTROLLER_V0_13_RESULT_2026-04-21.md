# Meta-Controller V0.13 Result - 2026-04-21

## Status

V0.13 tests B3 transfer.

New profile:

- `v02`

Transfer setup:

- training configs remain v01-like.
- evaluation configs use:
  - horizon `100`
  - more regime shifts
  - denser delayed memory queries
  - additional noisy clues

Validation:

```powershell
python -m pytest F:\unified-sel\tests\test_meta_controller_protocol.py -q -p no:cacheprovider
```

Result:

- `6 passed`

Transfer run:

```powershell
python -m experiments.meta_controller.run_experiment --profile v02 --mode train-eval --train-episodes 240 --eval-episodes 60 --seed 0 --table
```

Read-cost stress:

```powershell
python -m experiments.meta_controller.run_experiment --profile v01 --mode train-eval --train-episodes 240 --eval-episodes 60 --seed 1 --read-cost 0.20 --table
python -m experiments.meta_controller.run_experiment --profile v01 --mode train-eval --train-episodes 240 --eval-episodes 60 --seed 2 --read-cost 0.20 --table
```

## V02 Transfer Result

| controller | task_success | total_reward | planner_calls | memory_reads | drift |
|---|---:|---:|---:|---:|---:|
| fixed_rule_controller | 0.973 | 80.627 | 66.000 | 10.333 | 0.044 |
| shielded_relaxed_dominance_controller | 1.000 | 86.063 | 62.333 | 10.667 | 0.000 |
| habit_safe_set_h2_controller | 1.000 | 86.693 | 59.183 | 10.667 | 0.000 |
| planner_necessity_controller | 1.000 | 86.521 | 60.033 | 10.667 | 0.000 |
| planner_necessity_loose_controller | 1.000 | 86.693 | 59.183 | 10.667 | 0.000 |

Important observation:

- fixed rule no longer preserves perfect success in v02.
- B3 mainline reaches `1.000` success and `0.000` drift.
- planner calls drop from `66.000` to `59.183`.

This is stronger than merely saving compute; B3 transfers better than the hand-written fixed rule under altered regime/memory structure.

## V02 B3 Signal Masking

| ablation | task_success | planner_calls | memory_reads | drift |
|---|---:|---:|---:|---:|
| planner_necessity_loose_controller | 1.000 | 59.183 | 10.667 | 0.000 |
| b3_mask_surprise | 0.967 | 60.883 | 11.983 | 0.042 |
| b3_mask_memory | 0.949 | 64.433 | 0.000 | 0.070 |
| b3_mask_cost | 0.941 | 41.133 | 10.333 | 0.076 |
| b3_mask_conflict | 0.891 | 0.900 | 8.667 | 0.135 |
| b3_mask_drift | 1.000 | 59.850 | 10.667 | 0.000 |
| b3_mask_core_signals | 0.886 | 0.000 | 1.233 | 0.157 |

Masking confirms the same causal pattern as V0.12:

- conflict is the key dominance signal.
- memory relevance is required for delayed query success.
- surprise supports recovery.
- drift is still redundant in this environment.

## Read-Cost 0.20 Stress

### Seed 1

| controller | task_success | total_reward | planner_calls | memory_reads | drift |
|---|---:|---:|---:|---:|---:|
| fixed_rule_controller | 1.000 | 67.297 | 54.000 | 7.667 | 0.000 |
| planner_necessity_loose_controller | 1.000 | 68.153 | 49.917 | 7.667 | 0.000 |

### Seed 2

| controller | task_success | total_reward | planner_calls | memory_reads | drift |
|---|---:|---:|---:|---:|---:|
| fixed_rule_controller | 1.000 | 67.297 | 54.000 | 7.667 | 0.000 |
| planner_necessity_loose_controller | 1.000 | 68.170 | 49.833 | 7.667 | 0.000 |

Together with V0.11 seed 0 read-cost stress, B3 passes read-cost `0.20` on seeds 0, 1, 2.

## Interpretation

V0.13 is a strong positive transfer result.

B3 now satisfies:

- lower planner calls than fixed rule
- equal or better task success
- equal or lower drift
- memory reads within the fixed-rule guard band
- robustness under read-cost stress
- degradation under relevant signal masking

This is the first point where the experiment starts to support the original "learned unified mechanism" claim:

> the controller has learned a transferable dominance law over habit/planner/memory use, not merely a fixed-rule imitation.

## Branch Decision

B3 remains the mainline.

Next step:

- freeze the B3 interface as the V0 causal core.
- write a small integration contract before touching old F-drive projects.

Recommended V0.14:

- define the interface:
  - workspace state fields
  - safe-set training labels
  - planner-necessity labels
  - controller outputs
  - required logs/metrics
- then map old F-drive components to that interface without importing them yet.

