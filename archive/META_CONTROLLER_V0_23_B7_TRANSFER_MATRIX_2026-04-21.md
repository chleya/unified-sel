# Meta-Controller V0.23 B7 Transfer Matrix - 2026-04-21

## Purpose

V0.23 freezes the B7 claim into a compact transfer matrix.

Claim:

> B7 is a conservative invariant-coverage repair gate. It reduces drift and high-drift no-repair exposure under drift pressure, while accepting short-term reward cost.

This is still not a learned gate.

## Implementation

Files changed:

- `experiments/meta_controller/run_experiment.py`
- `experiments/meta_controller/README.md`
- `tests/test_meta_controller_protocol.py`

New function:

- `run_b7_transfer_matrix`

New CLI:

```powershell
python -m experiments.meta_controller.run_experiment --b7-transfer-matrix --matrix-profiles v01,v02,v03,v03b --matrix-seeds 0,1,2 --train-episodes 240 --eval-episodes 60 --repair-benefit-horizon 3 --table
```

The full 4-profile command timed out at 360s in this workspace. The matrix was completed by running each profile separately with the same settings.

## Matrix Columns

All deltas are B7 minus fixed rule:

- `success_delta`
- `planner_delta`
- `drift_delta`
- `reward_delta`
- `high_drift_no_repair_delta`

Repair-benefit columns are horizon-3 planner-first vs habit-first counterfactuals on high-drift states:

- `benefit_samples`
- `terminal_drift_benefit`
- `cumulative_drift_benefit`
- `cumulative_reward_benefit`

## V01

| run | success delta | planner delta | drift delta | reward delta | high drift no repair delta | samples | terminal drift benefit | cumulative drift benefit | cumulative reward benefit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v01 seed 0 | -0.008 | -5.200 | 0.005 | -0.179 | 0.000 | 0 | 0.000 | 0.000 | 0.000 |
| v01 seed 1 | -0.011 | -6.733 | 0.007 | -0.388 | 0.000 | 0 | 0.000 | 0.000 | 0.000 |
| v01 seed 2 | -0.018 | -2.367 | 0.011 | -2.241 | 0.000 | 407 | 0.057 | 0.170 | 0.238 |

## V02

| run | success delta | planner delta | drift delta | reward delta | high drift no repair delta | samples | terminal drift benefit | cumulative drift benefit | cumulative reward benefit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v02 seed 0 | 0.025 | -6.600 | -0.043 | 5.601 | -0.140 | 0 | 0.000 | 0.000 | 0.000 |
| v02 seed 1 | 0.023 | -4.000 | -0.042 | 4.860 | -0.140 | 0 | 0.000 | 0.000 | 0.000 |
| v02 seed 2 | 0.001 | -1.500 | -0.028 | -0.013 | -0.140 | 303 | 0.048 | 0.145 | 0.213 |

## V03

| run | success delta | planner delta | drift delta | reward delta | high drift no repair delta | samples | terminal drift benefit | cumulative drift benefit | cumulative reward benefit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v03 seed 0 | 0.003 | 4.150 | -0.098 | -1.961 | -0.189 | 579 | 0.410 | 1.037 | -0.339 |
| v03 seed 1 | -0.006 | 6.950 | -0.094 | -4.609 | -0.189 | 120 | 0.600 | 1.360 | -0.404 |
| v03 seed 2 | 0.011 | -5.067 | -0.089 | 1.990 | -0.189 | 120 | 0.600 | 1.360 | -0.404 |

## V03b

| run | success delta | planner delta | drift delta | reward delta | high drift no repair delta | samples | terminal drift benefit | cumulative drift benefit | cumulative reward benefit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v03b seed 0 | 0.003 | 5.083 | -0.146 | -3.016 | -0.239 | 768 | 0.352 | 0.897 | -0.243 |
| v03b seed 1 | -0.006 | 7.783 | -0.141 | -5.630 | -0.239 | 180 | 0.364 | 0.789 | -0.318 |
| v03b seed 2 | 0.011 | -2.250 | -0.136 | 0.592 | -0.239 | 180 | 0.364 | 0.789 | -0.318 |

## Interpretation

The matrix supports the frozen B7 claim:

- On v03/v03b, B7 consistently reduces drift and high-drift no-repair exposure versus fixed rule.
- On v03/v03b, repair-first counterfactuals have positive terminal and cumulative drift benefit.
- On v03/v03b, cumulative reward benefit is usually negative, confirming the short-term cost of repair.

Cross-profile behavior:

- v01 has little or no relevant drift-pressure signal; B7 is not useful there and can slightly hurt success/reward.
- v02 benefits on seeds 0/1 and is near neutral on seed 2 by reward, while reducing drift/high-drift exposure.
- v03/v03b are the intended invariant-pressure regimes and show the clearest B7 value.

The correct claim is not "B7 improves all metrics everywhere."

The correct claim is:

> B7 converts high-drift exposure into planner repair under invariant pressure, reducing drift debt across base and variant drift profiles. This can cost short-horizon reward, so the repair objective must be represented as invariant coverage or constrained value rather than raw reward maximization.

## Decision

Freeze B7 as the current drift-pressure branch:

- default threshold: `0.08`
- purpose: invariant coverage and drift repair
- not a learned classifier
- not a short-horizon reward optimizer

Next work should package this as a report/acceptance artifact and avoid adding a learned gate until a long-horizon constrained value target is defined.

## Verification

Protocol tests:

```powershell
python -m pytest F:\unified-sel\tests\test_meta_controller_protocol.py -q
```

Result:

- `11 passed`

Smoke tests:

```powershell
python F:\unified-sel\tests\smoke_test.py
```

Result:

- `All smoke tests passed`

