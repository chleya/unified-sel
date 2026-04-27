# Meta-Controller V0.18 Drift Threshold Sweep - 2026-04-21

## Purpose

V0.18 calibrates the B7 drift repair threshold.

The question:

> Is there a stable hand-calibrated drift repair threshold before we invest in a learned repair gate?

Thresholds swept:

- `0.08`
- `0.10`
- `0.12`
- `0.14`
- `0.16`

Profile:

- `v03`

Command shape:

```powershell
python -m experiments.meta_controller.run_experiment --profile v03 --mode train-eval --train-episodes 240 --eval-episodes 60 --seed <seed> --drift-threshold-sweep 0.08,0.10,0.12,0.14,0.16 --table
```

## Seed 0

| controller | success | reward | planner calls | repair rate | drift |
|---|---:|---:|---:|---:|---:|
| fixed rule | 0.964 | 95.576 | 82.000 | 0.000 | 0.109 |
| B3 | 1.000 | 106.601 | 75.467 | 0.000 | 0.160 |
| threshold 0.08 | 0.964 | 95.518 | 74.933 | 0.080 | 0.026 |
| threshold 0.10 | 0.967 | 93.624 | 86.833 | 0.052 | 0.015 |
| threshold 0.12 | 0.986 | 99.113 | 82.833 | 0.034 | 0.011 |
| threshold 0.14 | 0.965 | 92.609 | 89.733 | 0.068 | 0.022 |
| threshold 0.16 | 0.986 | 99.113 | 82.833 | 0.034 | 0.011 |

## Seed 1

| controller | success | reward | planner calls | repair rate | drift |
|---|---:|---:|---:|---:|---:|
| fixed rule | 0.964 | 95.576 | 82.000 | 0.000 | 0.109 |
| B3 | 0.958 | 103.592 | 58.883 | 0.000 | 0.364 |
| threshold 0.08 | 1.000 | 103.301 | 77.467 | 0.017 | 0.003 |
| threshold 0.10 | 0.983 | 98.415 | 82.217 | 0.040 | 0.012 |
| threshold 0.12 | 0.975 | 96.110 | 84.717 | 0.055 | 0.017 |
| threshold 0.14 | 0.999 | 103.035 | 77.583 | 0.018 | 0.004 |
| threshold 0.16 | 1.000 | 103.254 | 77.467 | 0.017 | 0.003 |

## Seed 2

| controller | success | reward | planner calls | repair rate | drift |
|---|---:|---:|---:|---:|---:|
| fixed rule | 0.964 | 95.576 | 82.000 | 0.000 | 0.109 |
| B3 | 0.967 | 96.974 | 82.967 | 0.000 | 0.159 |
| threshold 0.08 | 1.000 | 103.274 | 77.367 | 0.017 | 0.003 |
| threshold 0.10 | 0.965 | 96.044 | 73.367 | 0.081 | 0.026 |
| threshold 0.12 | 0.980 | 97.258 | 84.300 | 0.046 | 0.014 |
| threshold 0.14 | 1.000 | 103.274 | 77.367 | 0.017 | 0.003 |
| threshold 0.16 | 0.966 | 95.250 | 78.767 | 0.081 | 0.029 |

## Interpretation

There is no monotonic threshold curve. The interaction between learned safe/necessity weights and the repair threshold makes some thresholds duplicate trajectories on one seed and diverge on another.

However, `0.08` is the most robust default:

- passes success >= fixed on all seeds
- keeps drift far below fixed on all seeds
- keeps planner calls below fixed on all seeds
- avoids planner-always by a wide margin

Tradeoff:

- seed 0 reward is roughly tied with fixed rather than B3.
- repair rate is highest on seed 0 at `0.080`.

This is acceptable for a drift-pressure controller because the purpose of B7 is invariant repair, not maximum immediate reward.

## Decision

Update the B7 main baseline:

- `drift_aware_planner_necessity_controller`: threshold `0.08`
- `drift_aware_planner_necessity_loose_controller`: keep threshold `0.14`

## Next Step

Do not build a learned repair classifier yet.

First create a small v03 variant set so the repair threshold is tested against changed drift schedules, not just changed seeds.

Recommended next branch:

- V0.19 `v03b_drift_variant_transfer`

