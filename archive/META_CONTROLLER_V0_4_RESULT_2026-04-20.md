# Meta-Controller V0.4 Result - 2026-04-20

## Status

V0.4 keeps the V0.1 pressure environment fixed and replaces destructive factored online fine-tuning with a conservative update rule.

New controller:

- `conservative_factored_controller`

Guardrails:

- imitation warm-start
- lower learning rate
- update only when advantage magnitude exceeds a margin
- read gate updates only in memory-relevant states
- write gate frozen after warm-start

## Command

```powershell
python -m experiments.meta_controller.run_experiment --profile v01 --mode train-eval --train-episodes 240 --eval-episodes 60 --seed 0 --table
```

## Key Result

| controller | task_success | total_reward | arbitration_regret | compute_cost | planner_calls | memory_reads | memory_writes | memory_read_precision | action_habit_rate | action_planner_rate | action_read_rate | action_write_rate | drift_under_horizon |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_rule_controller | 1.000 | 68.063 | 0.149 | 11.937 | 54.000 | 7.667 | 1.000 | 0.738 | 0.325 | 0.675 | 0.096 | 0.012 | 0.000 |
| imitation_controller | 1.000 | 68.063 | 0.149 | 11.937 | 54.000 | 7.667 | 1.000 | 0.738 | 0.325 | 0.675 | 0.096 | 0.012 | 0.000 |
| factored_warm_controller | 1.000 | 66.840 | 0.164 | 13.160 | 60.083 | 7.667 | 1.000 | 0.738 | 0.249 | 0.751 | 0.096 | 0.012 | 0.000 |
| conservative_factored_controller | 0.997 | 68.258 | 0.147 | 11.232 | 50.017 | 10.117 | 1.000 | 0.275 | 0.375 | 0.625 | 0.126 | 0.012 | 0.002 |
| learned_contextual_bandit | 0.946 | 57.194 | 0.289 | 14.938 | 49.783 | 44.000 | 3.000 | 0.038 | 0.378 | 0.622 | 0.550 | 0.084 | 0.061 |

## Acceptance Checks

| check | pass | value | baseline |
|---|---:|---:|---:|
| conservative_not_below_warm_reward | true | 68.258 | 66.840 |
| conservative_reads_not_above_warm | false | 10.117 | 7.667 |

## Interpretation

V0.4 fixes the main V0.3 failure.

The conservative fine-tune no longer collapses to all habit. It improves over warm-start and slightly exceeds fixed/imitation reward in this run:

- reward: 68.258 vs fixed 68.063
- compute cost: 11.232 vs fixed 11.937
- planner calls: 50.017 vs fixed 54.000

But it pays for that by reading memory more often:

- memory reads: 10.117 vs fixed/warm 7.667
- memory read precision: 0.275 vs fixed/warm 0.738

So the current best statement is:

> Conservative factored fine-tuning can improve control cost and reward without collapse, but its read gate still over-generalizes.

This is the first V0 result where a learned controller variant beats the fixed-rule total reward, but it does not yet beat fixed-rule cleanly because memory read precision is worse.

## Next Step

V0.5 should focus only on read-gate discipline:

1. Add a direct read false-positive penalty to the controller update.
2. Track `memory_read_recall` as well as precision.
3. Add a separate conservative margin for read updates.
4. Compare reward under stricter memory read cost, e.g. 0.15 and 0.20.
5. Require conservative fine-tune to satisfy both:
   - total reward >= fixed-rule - 0.5
   - memory reads <= fixed-rule + 1.0

Do not add new environment complexity yet.
