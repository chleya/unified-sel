# Meta-Controller V0.2 Result - 2026-04-20

## Status

V0.2 keeps the V0.1 pressure environment fixed and adds an oracle/imitation diagnostic.

Purpose:

- determine whether V0.1 failure comes from an impossible environment or from a weak learned controller
- add a supervised macro-action upper-bound style baseline
- avoid increasing environment complexity before learning catches up

## New Baselines

- `oracle_macro_controller`
- `imitation_controller`

The oracle macro controller is a transparent rule-labeler over the existing workspace state:

- write early clue
- read memory in memory-query phase
- call planner under surprise/conflict
- otherwise use habit

The imitation controller is a small multiclass linear perceptron trained on oracle macro-action labels collected from training environments.

## Command

```powershell
python -m experiments.meta_controller.run_experiment --profile v01 --mode train-eval --train-episodes 240 --eval-episodes 60 --seed 0 --table
```

## Key Result

| controller | task_success | total_reward | arbitration_regret | compute_cost | planner_calls | memory_reads | memory_writes | action_habit_rate | action_planner_rate | action_read_rate | action_write_rate | drift_under_horizon |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_rule_controller | 1.000 | 68.063 | 0.149 | 11.937 | 54.000 | 7.667 | 1.000 | 0.325 | 0.675 | 0.096 | 0.012 | 0.000 |
| oracle_macro_controller | 1.000 | 68.063 | 0.149 | 11.937 | 54.000 | 7.667 | 1.000 | 0.325 | 0.675 | 0.096 | 0.012 | 0.000 |
| imitation_controller | 1.000 | 68.063 | 0.149 | 11.937 | 54.000 | 7.667 | 1.000 | 0.325 | 0.675 | 0.096 | 0.012 | 0.000 |
| learned_contextual_bandit | 0.946 | 57.194 | 0.289 | 14.938 | 49.783 | 44.000 | 3.000 | 0.378 | 0.622 | 0.550 | 0.084 | 0.061 |
| learned_mask_core_signals | 0.904 | 64.097 | 0.206 | 1.779 | 7.683 | 0.000 | 0.000 | 0.904 | 0.096 | 0.000 | 0.000 | 0.104 |

## Acceptance Checks

| check | pass | learned | baseline |
|---|---:|---:|---:|
| learned_success_beats_habit | true | 0.946 | 0.904 |
| learned_compute_below_planner_always | true | 14.938 | 16.000 |
| learned_reads_below_memory_always | true | 44.000 | 80.000 |
| learned_regret_below_random | true | 0.289 | 0.366 |
| learned_near_fixed_rule_reward | false | 57.194 | 68.063 |
| imitation_near_oracle_reward | true | 68.063 | 68.063 |
| learned_near_imitation_reward | false | 57.194 | 68.063 |

## Interpretation

V0.2 answers the immediate question:

The V0.1 task is learnable from the bounded workspace state.

Evidence:

- imitation matches oracle/fixed on held-out V0.1 configs
- action distributions match exactly at this aggregate level
- the contextual bandit fails mainly by over-reading memory and spending too much control cost

Therefore the next bottleneck is the online learned meta-controller, not the environment or the workspace representation.

## Next Step

V0.3 should replace the flat contextual bandit with a factored learned controller:

- head 1: dominance selection, `habit` vs `planner`
- head 2: memory read gate
- head 3: memory write gate
- optional head 4: broadcast gate

Training should use:

- imitation warm-start from oracle labels
- online reward fine-tuning after warm-start
- direct penalty for read/write overuse
- delayed credit for write-read success pairs

Do not integrate external F-drive projects yet. The clean next test is whether a factored learned policy can match imitation/fixed on V0.1 while remaining trainable from interaction.
