# Meta-Controller V0.19 V03b Drift Variant Transfer - 2026-04-21

## Purpose

V0.19 tests whether the V0.18 B7 drift repair threshold transfers beyond the exact `v03` drift profile.

Question:

> Does the hand-calibrated `0.08` repair gate remain robust when invariant guard schedules, guard thresholds, drift pressure, and repair strength change?

## Implementation

New profile:

- `v03b`

Files changed:

- `experiments/meta_controller/env.py`
- `experiments/meta_controller/run_experiment.py`
- `experiments/meta_controller/adapters/sel_lab_benchmark.py`
- `experiments/meta_controller/README.md`
- `tests/test_meta_controller_protocol.py`

Profile semantics:

- train configs remain `v03_train_configs`.
- eval configs use new `v03b_drift_variant_configs`.
- the variant changes:
  - invariant guard schedule
  - invariant guard threshold
  - unguarded drift increase
  - guarded drift decay

CLI:

```powershell
python -m experiments.meta_controller.run_experiment --profile v03b --mode train-eval --train-episodes 240 --eval-episodes 60 --seed 0 --drift-threshold-sweep 0.08,0.10,0.12,0.14,0.16 --table
```

New suite:

- `v03b_drift_variant_transfer`

## Seed 0

| controller | success | reward | planner calls | repair rate | drift |
|---|---:|---:|---:|---:|---:|
| fixed rule | 0.964 | 96.516 | 82.000 | 0.000 | 0.161 |
| B3 | 1.000 | 107.031 | 75.467 | 0.000 | 0.184 |
| threshold 0.08 | 0.964 | 95.131 | 77.517 | 0.107 | 0.033 |
| threshold 0.10 | 0.967 | 93.730 | 86.400 | 0.052 | 0.016 |
| threshold 0.12 | 0.986 | 99.012 | 83.600 | 0.044 | 0.014 |
| threshold 0.14 | 0.965 | 92.724 | 89.533 | 0.069 | 0.026 |
| threshold 0.16 | 0.986 | 99.273 | 82.400 | 0.032 | 0.014 |

## Seed 1

| controller | success | reward | planner calls | repair rate | drift |
|---|---:|---:|---:|---:|---:|
| fixed rule | 0.964 | 96.516 | 82.000 | 0.000 | 0.161 |
| B3 | 0.956 | 103.113 | 59.050 | 0.000 | 0.366 |
| threshold 0.08 | 1.000 | 103.138 | 78.467 | 0.025 | 0.005 |
| threshold 0.10 | 0.983 | 98.250 | 83.417 | 0.054 | 0.016 |
| threshold 0.12 | 0.975 | 96.151 | 84.850 | 0.058 | 0.021 |
| threshold 0.14 | 0.999 | 103.051 | 77.650 | 0.018 | 0.006 |
| threshold 0.16 | 1.000 | 103.378 | 77.133 | 0.014 | 0.006 |

## Seed 2

| controller | success | reward | planner calls | repair rate | drift |
|---|---:|---:|---:|---:|---:|
| fixed rule | 0.964 | 96.516 | 82.000 | 0.000 | 0.161 |
| B3 | 0.967 | 97.237 | 82.967 | 0.000 | 0.174 |
| threshold 0.08 | 1.000 | 103.111 | 78.367 | 0.025 | 0.005 |
| threshold 0.10 | 0.965 | 95.606 | 76.183 | 0.109 | 0.032 |
| threshold 0.12 | 0.980 | 97.225 | 84.733 | 0.054 | 0.018 |
| threshold 0.14 | 1.000 | 103.304 | 77.367 | 0.017 | 0.005 |
| threshold 0.16 | 0.966 | 95.216 | 79.600 | 0.089 | 0.034 |

## Interpretation

The `0.08` threshold transfers as a robust repair default:

- success is >= fixed rule on all seeds.
- drift is far below fixed rule on all seeds.
- planner calls stay below fixed rule on all seeds.

It is not reward-optimal on every variant:

- seed 0 favors `0.16` by reward.
- seed 1 favors `0.16` by reward.
- seed 2 favors `0.14` by reward.

The V0.18 conclusion still holds: threshold response remains non-monotonic, and the best reward threshold can shift under v03 variants. However, `0.08` is the strongest conservative default because it preserves the invariant-repair objective across variant pressure without increasing planner calls above fixed rule.

## Decision

Keep B7 main baseline:

- `drift_aware_planner_necessity_controller`: threshold `0.08`
- `drift_aware_planner_necessity_loose_controller`: threshold `0.14`

Do not build a learned repair classifier yet.

Next, if we revisit learning the gate, it should be framed as a surprise/residual gate rather than a direct classifier over the current v03 traces.

## Verification

Protocol tests:

```powershell
python -m pytest F:\unified-sel\tests\test_meta_controller_protocol.py -q
```

Result:

- `9 passed`

Smoke tests:

```powershell
python F:\unified-sel\tests\smoke_test.py
```

Result:

- `All smoke tests passed`

