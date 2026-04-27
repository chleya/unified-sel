# Meta-Controller V0.5 Result - 2026-04-21

## Status

V0.5 keeps the V0.1 pressure environment fixed and focuses on read-gate discipline.

New metrics:

- `memory_read_recall`
- `memory_read_false_positive_rate`

New controllers:

- `read_disciplined_factored_controller`
- `dominance_tuned_factored_controller`

## Command

```powershell
python -m experiments.meta_controller.run_experiment --profile v01 --mode train-eval --train-episodes 240 --eval-episodes 60 --seed 0 --table
```

Read-cost stress check:

```powershell
python -m experiments.meta_controller.run_experiment --profile v01 --mode train-eval --train-episodes 240 --eval-episodes 60 --seed 0 --read-cost 0.20 --table
```

## Default Read Cost Result

| controller | task_success | total_reward | compute_cost | planner_calls | memory_reads | read_precision | read_recall | read_false_positive | drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_rule_controller | 1.000 | 68.063 | 11.937 | 54.000 | 7.667 | 0.738 | 1.000 | 0.027 | 0.000 |
| conservative_factored_controller | 0.971 | 66.705 | 9.024 | 40.550 | 6.667 | 0.135 | 0.182 | 0.076 | 0.025 |
| read_disciplined_factored_controller | 0.971 | 66.852 | 8.877 | 40.367 | 5.150 | 0.948 | 0.847 | 0.005 | 0.025 |
| dominance_tuned_factored_controller | 0.965 | 65.891 | 8.866 | 38.833 | 7.667 | 0.738 | 1.000 | 0.027 | 0.033 |
| learned_contextual_bandit | 0.946 | 57.194 | 14.938 | 49.783 | 44.000 | 0.038 | 0.288 | 0.570 | 0.061 |

## Interpretation

V0.5 succeeds on read-gate discipline but not on total control quality.

Read discipline improves sharply:

- `read_disciplined_factored_controller` cuts memory reads from flat bandit's 44.000 to 5.150.
- false-positive read rate drops from 0.570 to 0.005.
- read precision rises from 0.038 to 0.948.
- read recall remains useful at 0.847.

But reward is still below fixed-rule:

- fixed reward: 68.063
- read-disciplined reward: 66.852

The reason is no longer memory over-read. The main remaining issue is dominance arbitration:

- read-disciplined planner calls: 40.367
- fixed planner calls: 54.000
- read-disciplined task success: 0.971
- fixed task success: 1.000

The controller is saving compute too aggressively and letting habit handle cases where planner is still needed.

## Read Cost 0.20 Stress

At read cost 0.20:

| controller | task_success | total_reward | planner_calls | memory_reads | read_precision | read_recall | read_false_positive |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_rule_controller | 1.000 | 67.297 | 54.000 | 7.667 | 0.738 | 1.000 | 0.027 |
| read_disciplined_factored_controller | 0.968 | 65.578 | 39.417 | 7.000 | 0.813 | 1.000 | 0.018 |
| conservative_factored_controller | 0.963 | 66.637 | 38.083 | 0.150 | 0.150 | 0.025 | 0.000 |

This confirms the same pattern:

- read discipline can be maintained under higher read cost
- dominance arbitration is still too compute-saving
- success loss offsets the cost savings

## Acceptance

Passed:

- read false positives are controlled
- read precision is much better than flat bandit
- memory reads are near or below fixed-rule
- all controllers remain runnable under read-cost changes

Failed:

- reward is not within 0.5 of fixed-rule under default read cost
- dominance tuning does not yet preserve fixed-rule success

## Next Step

V0.6 should stop changing read/write and focus on dominance arbitration.

Recommended next method:

- freeze read and write gates from imitation/warm-start
- train only dominance with counterfactual scoring
- before applying a dominance update, compare:
  - actual chosen module reward
  - simulated alternative module reward
  - compute delta
- update dominance only when counterfactual advantage exceeds margin

Target:

> reduce planner calls below fixed-rule while keeping task_success >= 0.99 and memory_reads <= fixed + 1.0.

Do not add new environment complexity and do not integrate external F-drive projects yet.
