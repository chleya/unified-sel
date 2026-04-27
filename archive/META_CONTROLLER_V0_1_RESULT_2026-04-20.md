# Meta-Controller V0.1 Result - 2026-04-20

## Status

V0.1 adds pressure to the V0 experiment rather than making the result easier.

Changes:

- optional `--profile v01`
- 3-action hidden regimes instead of 2-action regimes
- more regime shifts
- noisy clue steps
- higher memory read cost
- action distribution metrics

Command:

```powershell
python -m experiments.meta_controller.run_experiment --profile v01 --mode train-eval --train-episodes 240 --eval-episodes 60 --seed 0 --table
```

## Result

V0.1 is a useful failure case for the current learned contextual-bandit controller.

The learned controller still beats weak baselines on some checks:

- success above `habit_only`
- compute below `planner_always`
- memory reads below `memory_always`
- regret below `random_controller`

But it fails the important fixed-rule comparison:

- `learned_contextual_bandit` total reward: 57.194
- `fixed_rule_controller` total reward: 68.063
- `learned_near_fixed_rule_reward`: false

## Key Table

| controller | task_success | total_reward | arbitration_regret | compute_cost | planner_calls | memory_reads | memory_writes | action_habit_rate | action_planner_rate | action_read_rate | action_write_rate | drift_under_horizon |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| habit_only | 0.904 | 65.810 | 0.189 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.104 |
| planner_always | 0.955 | 57.467 | 0.282 | 16.000 | 80.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.050 |
| memory_always | 0.929 | 43.225 | 0.460 | 26.400 | 80.000 | 80.000 | 3.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.080 |
| fixed_rule_controller | 1.000 | 68.063 | 0.149 | 11.937 | 54.000 | 7.667 | 1.000 | 0.325 | 0.675 | 0.096 | 0.012 | 0.000 |
| random_controller | 0.903 | 50.758 | 0.366 | 14.923 | 45.600 | 34.183 | 1.383 | 0.430 | 0.570 | 0.427 | 0.431 | 0.101 |
| learned_contextual_bandit | 0.946 | 57.194 | 0.289 | 14.938 | 49.783 | 44.000 | 3.000 | 0.378 | 0.622 | 0.550 | 0.084 | 0.061 |
| learned_mask_core_signals | 0.904 | 64.097 | 0.206 | 1.779 | 7.683 | 0.000 | 0.000 | 0.904 | 0.096 | 0.000 | 0.000 | 0.104 |

## Interpretation

What V0.1 shows:

- The environment pressure is now strong enough to expose over-reading.
- The current contextual bandit learns to write, but it reads too often.
- The current learned controller spends nearly planner-always compute without achieving fixed-rule reward.
- Core signal masking still changes behavior strongly, collapsing toward mostly habit.

What this means:

- The experiment is now useful as a falsification harness.
- The current learning rule is too weak for the harder control problem.
- The next step should improve the meta-controller, not integrate old F-drive modules.

## Next Step

V0.2 should improve learning, with the environment held fixed:

1. Add delayed credit assignment for write/read pairs.
2. Add action-specific priors or separate heads for write, read, and planner choice.
3. Add explicit oracle arbitration regret using counterfactual action evaluation.
4. Penalize memory reads more directly in the controller reward.
5. Compare against a small supervised imitation controller trained from oracle macro-actions.

Do not increase environment complexity again until learned control beats or matches the fixed-rule baseline in V0.1.
