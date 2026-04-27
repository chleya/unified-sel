# Meta-Controller V0.9 Result - 2026-04-21

## Status

V0.9 implements Branch B2: learned habit safe set.

New controllers:

- `habit_safe_set_controller`
- `habit_safe_set_h2_controller`

Design:

- read/write gates are still learned from oracle/imitation examples.
- dominance is no longer trained as raw planner-vs-habit reward difference.
- instead, a separate `habit_safe(g_t)` classifier is trained from short-horizon counterfactual labels.
- if `habit_safe(g_t)` is true, the controller may use habit.
- otherwise it falls back to planner while preserving the learned read/write gates.

Safe-set labels:

- `habit_safe_set_controller`: habit must succeed for 3 rollout steps without drift increase.
- `habit_safe_set_h2_controller`: habit must succeed for 2 rollout steps without drift increase.

## Validation

```powershell
python -m pytest F:\unified-sel\tests\test_meta_controller_protocol.py -q -p no:cacheprovider
```

Result:

- `5 passed`

Main runs:

```powershell
python -m experiments.meta_controller.run_experiment --profile v01 --mode train-eval --train-episodes 240 --eval-episodes 60 --seed 0 --table
python -m experiments.meta_controller.run_experiment --profile v01 --mode train-eval --train-episodes 240 --eval-episodes 60 --seed 1 --table
python -m experiments.meta_controller.run_experiment --profile v01 --mode train-eval --train-episodes 240 --eval-episodes 60 --seed 2 --table
```

## Key Results

### Seed 0

| controller | task_success | total_reward | planner_calls | memory_reads | drift |
|---|---:|---:|---:|---:|---:|
| fixed_rule_controller | 1.000 | 68.063 | 54.000 | 7.667 | 0.000 |
| shielded_relaxed_dominance_controller | 1.000 | 68.303 | 53.000 | 7.667 | 0.000 |
| habit_safe_set_controller | 1.000 | 68.920 | 49.917 | 7.667 | 0.000 |
| habit_safe_set_h2_controller | 1.000 | 68.920 | 49.917 | 7.667 | 0.000 |

### Seed 1

| controller | task_success | total_reward | planner_calls | memory_reads | drift |
|---|---:|---:|---:|---:|---:|
| fixed_rule_controller | 1.000 | 68.063 | 54.000 | 7.667 | 0.000 |
| shielded_relaxed_dominance_controller | 1.000 | 68.303 | 53.000 | 7.667 | 0.000 |
| habit_safe_set_controller | 1.000 | 67.903 | 55.000 | 7.667 | 0.000 |
| habit_safe_set_h2_controller | 1.000 | 68.303 | 53.000 | 7.667 | 0.000 |

### Seed 2

| controller | task_success | total_reward | planner_calls | memory_reads | drift |
|---|---:|---:|---:|---:|---:|
| fixed_rule_controller | 1.000 | 68.063 | 54.000 | 7.667 | 0.000 |
| shielded_relaxed_dominance_controller | 1.000 | 68.303 | 53.000 | 7.667 | 0.000 |
| habit_safe_set_controller | 1.000 | 68.714 | 50.967 | 7.667 | 0.000 |
| habit_safe_set_h2_controller | 1.000 | 68.303 | 53.000 | 7.667 | 0.000 |

## Interpretation

B2 is a stronger route than B1, but not yet final.

What improved:

- It keeps the safety properties that failed in V0.6/V0.7:
  - `task_success = 1.000`
  - `drift_under_horizon = 0.000`
  - memory reads stay near fixed rule
- It can reduce planner calls more than shielded dominance:
  - seed 0: `54.000 -> 49.917`
  - seed 2: `54.000 -> 50.967`

What remains unstable:

- The 3-step safe set is seed-sensitive:
  - seed 1 becomes too conservative: `55.000` planner calls.
- The 2-step safe set is more stable across seeds, but only matches the relaxed shield improvement:
  - `54.000 -> 53.000`.

Current reading:

> The workspace state contains learnable habit-safe regions, but the safe-set boundary is not robust enough yet. The next problem is calibration, not architecture.

## Branch Decision

B2 should continue, but with calibration diagnostics before adding new theory.

Next recommended branchlet:

- B2.1 calibrated safe-set threshold and label audit.

Minimum next checks:

- report positive/negative safe-label rate during training
- expose safe-set score distribution
- tune threshold instead of relying on a zero perceptron boundary
- validate across seeds 0, 1, 2 and read-cost stress

Do not move to old F-drive integration yet.

