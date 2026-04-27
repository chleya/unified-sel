# Meta-Controller V0.11 Result - 2026-04-21

## Status

V0.11 implements B3: planner necessity / expected value of computation.

New controllers:

- `planner_necessity_controller`
- `planner_necessity_loose_controller`

Design:

- keep the learned habit safe-set as a hard safety gate.
- train an additional planner-necessity classifier from short-horizon planner-vs-habit rollout value gap.
- choose habit only when:
  - habit safe-set passes
  - planner necessity is not active
- otherwise fall back to planner while preserving read/write gates.

New metrics:

- `necessity_positive_rate`
- `necessity_score_mean`
- `necessity_label_positive_rate`
- `necessity_train_score_mean`

## Validation

```powershell
python -m pytest F:\unified-sel\tests\test_meta_controller_protocol.py -q -p no:cacheprovider
```

Result:

- `5 passed`

Default evaluation:

```powershell
python -m experiments.meta_controller.run_experiment --profile v01 --mode train-eval --train-episodes 240 --eval-episodes 60 --seed 0 --table
python -m experiments.meta_controller.run_experiment --profile v01 --mode train-eval --train-episodes 240 --eval-episodes 60 --seed 1 --table
python -m experiments.meta_controller.run_experiment --profile v01 --mode train-eval --train-episodes 240 --eval-episodes 60 --seed 2 --table
```

Read-cost stress:

```powershell
python -m experiments.meta_controller.run_experiment --profile v01 --mode train-eval --train-episodes 240 --eval-episodes 60 --seed 0 --read-cost 0.20 --table
```

## Key Results

### Default Read Cost

| controller | seed | task_success | total_reward | planner_calls | memory_reads | drift |
|---|---:|---:|---:|---:|---:|---:|
| fixed_rule_controller | 0 | 1.000 | 68.063 | 54.000 | 7.667 | 0.000 |
| planner_necessity_controller | 0 | 1.000 | 68.843 | 50.300 | 7.667 | 0.000 |
| planner_necessity_loose_controller | 0 | 1.000 | 68.967 | 49.917 | 7.333 | 0.000 |
| fixed_rule_controller | 1 | 1.000 | 68.063 | 54.000 | 7.667 | 0.000 |
| planner_necessity_controller | 1 | 1.000 | 68.920 | 49.917 | 7.667 | 0.000 |
| planner_necessity_loose_controller | 1 | 1.000 | 68.920 | 49.917 | 7.667 | 0.000 |
| fixed_rule_controller | 2 | 1.000 | 68.063 | 54.000 | 7.667 | 0.000 |
| planner_necessity_controller | 2 | 1.000 | 68.937 | 49.833 | 7.667 | 0.000 |
| planner_necessity_loose_controller | 2 | 1.000 | 68.937 | 49.833 | 7.667 | 0.000 |

### Read Cost 0.20, Seed 0

| controller | task_success | total_reward | planner_calls | memory_reads | drift |
|---|---:|---:|---:|---:|---:|
| fixed_rule_controller | 1.000 | 67.297 | 54.000 | 7.667 | 0.000 |
| planner_necessity_controller | 1.000 | 68.077 | 50.300 | 7.667 | 0.000 |
| planner_necessity_loose_controller | 1.000 | 68.233 | 49.917 | 7.333 | 0.000 |

## Interpretation

B3 is the first clearly positive dominance result.

Compared with B2:

- B2 h3 had high upside but was seed-sensitive:
  - `49.917`, `55.000`, `50.967` planner calls.
- B2 h2 was stable but modest:
  - `49.917`, `53.000`, `53.000` planner calls.
- B3 is stable and stronger:
  - `50.300`, `49.917`, `49.833` for the default controller.
  - `49.917`, `49.917`, `49.833` for the loose necessity variant.

The key seed 1 failure was corrected:

- h3 safe-set alone: `55.000` planner calls.
- planner necessity: `49.917` planner calls.
- success and drift remained perfect.

This supports the branch hypothesis:

> "habit is safe" and "planner is worth calling" are separate signals. Combining them produces a more stable dominance law than either safe-set thresholding or reward-difference dominance alone.

## Branch Decision

B3 should become the current mainline.

Current best controller:

- `planner_necessity_loose_controller`

Reason:

- preserves `task_success = 1.000`
- preserves `drift_under_horizon = 0.000`
- keeps memory reads at or below fixed rule
- consistently reduces planner calls below fixed rule across seeds 0, 1, 2
- survives read-cost 0.20 stress on seed 0

Next checks:

- run transfer configs and signal masking for the B3 controller
- add direct acceptance checks comparing B3 against fixed rule
- then decide whether B4 option termination is needed

Do not integrate old F-drive projects yet.

