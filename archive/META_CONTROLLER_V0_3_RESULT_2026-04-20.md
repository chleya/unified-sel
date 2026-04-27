# Meta-Controller V0.3 Result - 2026-04-20

## Status

V0.3 adds a factored learned controller while keeping the V0.1 pressure environment fixed.

The controller separates the meta-action into learned gates:

- dominance: habit vs planner
- memory read
- memory write

Two variants are reported:

- `factored_warm_controller`: imitation warm-start only
- `factored_controller`: imitation warm-start plus online reward fine-tuning

## Command

```powershell
python -m experiments.meta_controller.run_experiment --profile v01 --mode train-eval --train-episodes 240 --eval-episodes 60 --seed 0 --table
```

## Key Result

| controller | task_success | total_reward | arbitration_regret | compute_cost | planner_calls | memory_reads | memory_writes | action_habit_rate | action_planner_rate | action_read_rate | action_write_rate | drift_under_horizon |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_rule_controller | 1.000 | 68.063 | 0.149 | 11.937 | 54.000 | 7.667 | 1.000 | 0.325 | 0.675 | 0.096 | 0.012 | 0.000 |
| imitation_controller | 1.000 | 68.063 | 0.149 | 11.937 | 54.000 | 7.667 | 1.000 | 0.325 | 0.675 | 0.096 | 0.012 | 0.000 |
| learned_contextual_bandit | 0.946 | 57.194 | 0.289 | 14.938 | 49.783 | 44.000 | 3.000 | 0.378 | 0.622 | 0.550 | 0.084 | 0.061 |
| factored_warm_controller | 1.000 | 66.840 | 0.164 | 13.160 | 60.083 | 7.667 | 1.000 | 0.249 | 0.751 | 0.096 | 0.012 | 0.000 |
| factored_controller | 0.904 | 65.810 | 0.189 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.104 |

## Acceptance Checks

| check | pass | value | baseline |
|---|---:|---:|---:|
| factored_warm_near_imitation_reward | true | 66.840 | 68.063 |
| factored_warm_reads_below_flat_bandit | true | 7.667 | 44.000 |
| factored_near_imitation_reward | false | 65.810 | 68.063 |
| factored_reads_below_flat_bandit | true | 0.000 | 44.000 |

## Interpretation

The factored architecture helps, but the current online fine-tuning rule is destructive.

Evidence:

- `factored_warm_controller` sharply reduces memory over-reading versus the flat bandit.
- `factored_warm_controller` reaches 66.840 reward, close to imitation/fixed at 68.063.
- `factored_warm_controller` keeps memory reads at 7.667, matching fixed/imitation.
- `factored_controller` collapses to all habit after online fine-tuning.

So V0.3 separates two facts:

1. Factored gates plus imitation warm-start are a good direction.
2. The current reward fine-tuning update is not acceptable.

## Next Step

V0.4 should keep `factored_warm_controller` as the base and replace the online update with a safer method:

- freeze the write gate after warm-start
- use lower learning rate for read gate
- update only when advantage magnitude exceeds a margin
- add off-policy counterfactual scoring before applying an update
- compare warm-start only vs conservative fine-tune

The immediate goal is not to beat fixed-rule yet. The next valid target is:

> conservative factored fine-tuning must not collapse below warm-start, and should reduce planner calls without increasing memory over-read.
