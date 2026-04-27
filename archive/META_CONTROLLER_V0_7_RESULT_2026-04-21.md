# Meta-Controller V0.7 Result - 2026-04-21

## Status

V0.7 tests short-horizon counterfactual rollout for dominance arbitration.

Read/write gates remain frozen from imitation. Only dominance is updated.

New controllers:

- `rollout_dominance_controller`
- `risk_averse_rollout_controller`

Rollout horizon:

- K = 3

## Command

```powershell
python -m experiments.meta_controller.run_experiment --profile v01 --mode train-eval --train-episodes 240 --eval-episodes 60 --seed 0 --table
```

## Key Result

| controller | task_success | total_reward | compute_cost | planner_calls | memory_reads | read_precision | read_recall | drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_rule_controller | 1.000 | 68.063 | 11.937 | 54.000 | 7.667 | 0.738 | 1.000 | 0.000 |
| counterfactual_dominance_controller | 0.970 | 65.917 | 9.667 | 42.850 | 7.667 | 0.738 | 1.000 | 0.029 |
| rollout_dominance_controller | 0.952 | 66.925 | 5.958 | 24.550 | 7.333 | 0.774 | 1.000 | 0.051 |
| risk_averse_rollout_controller | 0.919 | 65.978 | 2.038 | 5.017 | 7.333 | 0.774 | 1.000 | 0.087 |

## Acceptance

| check | pass | value | baseline |
|---|---:|---:|---:|
| rollout_success_ge_099 | false | 0.952 | 0.990 |
| rollout_reads_near_fixed | true | 7.333 | 7.667 |
| rollout_planner_below_fixed | true | 24.550 | 54.000 |
| risk_rollout_success_ge_099 | false | 0.919 | 0.990 |
| risk_rollout_reads_near_fixed | true | 7.333 | 7.667 |
| risk_rollout_planner_below_fixed | true | 5.017 | 54.000 |

## Interpretation

V0.7 is another useful negative result.

K-step rollout reduced planner use aggressively, but it did not preserve success:

- planner calls dropped from 54.000 to 24.550
- memory discipline stayed good
- success dropped from 1.000 to 0.952
- drift rose from 0.000 to 0.051

The risk-averse variant was worse:

- planner calls dropped to 5.017
- success dropped to 0.919
- drift rose to 0.087

This means the current rollout objective is still too cost-seeking. It learns that habit often works locally and underestimates the safety/recovery role of planner.

## Conclusion

V0.7 does not solve dominance arbitration.

The next missing piece is not a longer horizon by itself. The objective must include explicit safety constraints:

- maintain success >= 0.99
- avoid drift increase
- preserve recovery after regime shifts
- only reduce planner when these constraints remain satisfied

## Next Step

V0.8 should switch from reward-difference dominance learning to constrained dominance learning:

- freeze read/write
- start from imitation/fixed dominance
- permit planner reduction only if a validation window shows:
  - no success drop
  - no drift increase
  - no recovery slowdown
- use a constraint-first update:
  - if safety violation appears, restore planner
  - if safe for N windows, reduce planner threshold locally

Target:

> reduce planner calls while preserving task_success >= 0.99, drift <= fixed + 0.005, and memory_reads <= fixed + 1.0.

Do not integrate external F-drive projects yet.
