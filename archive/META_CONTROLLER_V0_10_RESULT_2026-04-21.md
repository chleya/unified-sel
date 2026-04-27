# Meta-Controller V0.10 Result - 2026-04-21

## Status

V0.10 implements B2.1: safe-set calibration and label diagnostics.

New diagnostics:

- `safe_set_positive_rate`
- `safe_score_mean`
- `safe_label_positive_rate`
- `safe_train_score_mean`

New calibrated variants:

- `habit_safe_set_loose_controller`: lower safe threshold
- `habit_safe_set_tight_controller`: higher safe threshold

Validation:

```powershell
python -m pytest F:\unified-sel\tests\test_meta_controller_protocol.py -q -p no:cacheprovider
```

Result:

- `5 passed`

Evaluation:

```powershell
python -m experiments.meta_controller.run_experiment --profile v01 --mode train-eval --train-episodes 240 --eval-episodes 60 --seed 0 --table
python -m experiments.meta_controller.run_experiment --profile v01 --mode train-eval --train-episodes 240 --eval-episodes 60 --seed 1 --table
python -m experiments.meta_controller.run_experiment --profile v01 --mode train-eval --train-episodes 240 --eval-episodes 60 --seed 2 --table
```

## Key Results

### Seed 0

| controller | task_success | planner_calls | safe_rate | safe_label_rate | drift |
|---|---:|---:|---:|---:|---:|
| fixed_rule_controller | 1.000 | 54.000 | 0.000 | 0.000 | 0.000 |
| habit_safe_set_controller | 1.000 | 49.917 | 0.376 | 0.281 | 0.000 |
| habit_safe_set_h2_controller | 1.000 | 49.917 | 0.376 | 0.323 | 0.000 |
| habit_safe_set_loose_controller | 1.000 | 49.917 | 0.376 | 0.281 | 0.000 |
| habit_safe_set_tight_controller | 1.000 | 73.483 | 0.081 | 0.281 | 0.000 |

### Seed 1

| controller | task_success | planner_calls | safe_rate | safe_label_rate | drift |
|---|---:|---:|---:|---:|---:|
| fixed_rule_controller | 1.000 | 54.000 | 0.000 | 0.000 | 0.000 |
| habit_safe_set_controller | 1.000 | 55.000 | 0.313 | 0.280 | 0.000 |
| habit_safe_set_h2_controller | 1.000 | 53.000 | 0.338 | 0.323 | 0.000 |
| habit_safe_set_loose_controller | 0.915 | 12.417 | 0.845 | 0.281 | 0.094 |
| habit_safe_set_tight_controller | 1.000 | 74.667 | 0.067 | 0.281 | 0.000 |

### Seed 2

| controller | task_success | planner_calls | safe_rate | safe_label_rate | drift |
|---|---:|---:|---:|---:|---:|
| fixed_rule_controller | 1.000 | 54.000 | 0.000 | 0.000 | 0.000 |
| habit_safe_set_controller | 1.000 | 50.967 | 0.363 | 0.281 | 0.000 |
| habit_safe_set_h2_controller | 1.000 | 53.000 | 0.338 | 0.322 | 0.000 |
| habit_safe_set_loose_controller | 1.000 | 50.900 | 0.364 | 0.281 | 0.000 |
| habit_safe_set_tight_controller | 1.000 | 75.000 | 0.062 | 0.281 | 0.000 |

## Interpretation

B2.1 gives a useful negative result for simple threshold calibration.

Findings:

- The training label positive rate is stable:
  - h3 labels around `0.28`
  - h2 labels around `0.32`
- The deployed safe rate is also interpretable:
  - h3 works when deployed safe rate is around `0.36-0.38`
  - h3 seed 1 is too conservative at `0.313`
  - loose seed 1 becomes unsafe at `0.845`
  - tight is always over-conservative around `0.06-0.08`

The important point:

> The safe-set problem is not solved by a global threshold. A small threshold shift can either do nothing or open a large unsafe region, depending on the learned score geometry.

## Branch Decision

B2 remains valuable, but B2.1 shows that linear safe-set thresholding is too crude.

Current best stable controller:

- `habit_safe_set_h2_controller`

It preserves safety on seeds 0, 1, 2 and reduces planner calls to `49.917`, `53.000`, `53.000`.

Current best higher-upside controller:

- `habit_safe_set_controller`

It preserves safety on seeds 0, 1, 2 but is seed-sensitive in planner usage: `49.917`, `55.000`, `50.967`.

Next route:

- Do not keep hand-tuning global thresholds.
- Move to B3: expected value of computation / planner necessity.

Reason:

- Safe-set labels show which states are safe, but not how much planner value is at stake.
- The next control signal should estimate planner necessity or expected computational value, then use safe-set as a hard safety constraint.

Recommended V0.11:

- add a planner-necessity score trained from short-horizon planner-vs-habit value gap
- choose habit only when:
  - habit safe-set passes
  - planner necessity is below threshold
- otherwise planner

