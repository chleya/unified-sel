# Meta-Controller V0.6 Result - 2026-04-21

## Status

V0.6 keeps read/write fixed and tests dominance arbitration only.

New controller:

- `counterfactual_dominance_controller`

Training idea:

- warm-start read/write/dominance from oracle labels
- freeze read/write gates
- update only dominance using same-step counterfactual value:
  - habit value
  - planner value
  - compute delta

## Command

```powershell
python -m experiments.meta_controller.run_experiment --profile v01 --mode train-eval --train-episodes 240 --eval-episodes 60 --seed 0 --table
```

## Key Result

| controller | task_success | total_reward | compute_cost | planner_calls | memory_reads | read_precision | read_recall | read_false_positive | drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_rule_controller | 1.000 | 68.063 | 11.937 | 54.000 | 7.667 | 0.738 | 1.000 | 0.027 | 0.000 |
| read_disciplined_factored_controller | 0.971 | 66.852 | 8.877 | 40.367 | 5.150 | 0.948 | 0.847 | 0.005 | 0.025 |
| dominance_tuned_factored_controller | 0.965 | 65.891 | 8.866 | 38.833 | 7.667 | 0.738 | 1.000 | 0.027 | 0.033 |
| counterfactual_dominance_controller | 0.970 | 65.917 | 9.667 | 42.850 | 7.667 | 0.738 | 1.000 | 0.027 | 0.029 |

## Acceptance

| check | pass | value | baseline |
|---|---:|---:|---:|
| counterfactual_success_ge_099 | false | 0.970 | 0.990 |
| counterfactual_reads_near_fixed | true | 7.667 | 7.667 |
| counterfactual_planner_below_fixed | true | 42.850 | 54.000 |

## Interpretation

V0.6 is a useful negative result.

What worked:

- read/write discipline was preserved
- memory reads exactly matched fixed-rule
- planner calls dropped from 54.000 to 42.850

What failed:

- task success dropped from 1.000 to 0.970
- total reward dropped from 68.063 to 65.917
- drift rose to 0.029

The failure mode is clear:

> Same-step counterfactual dominance is too myopic. It saves planner cost on steps where habit is immediately acceptable, but misses the longer-horizon value of planner for recovery and drift prevention.

This means V0.6 did not solve dominance arbitration. It did isolate the next missing mechanism: multi-step or recovery-aware counterfactual value.

## Next Step

V0.7 should replace same-step counterfactual value with short-horizon counterfactual rollouts.

Recommended design:

- keep read/write frozen from imitation
- for dominance updates, simulate both choices for K steps
- K can start at 3 or 5
- include:
  - accumulated reward
  - planner cost
  - drift penalty
  - recovery after regime shift
- update dominance only if rollout advantage exceeds margin

Target:

> planner_calls < fixed-rule while task_success >= 0.99 and memory_reads <= fixed + 1.0.

Do not add environment complexity or external project integration yet.
