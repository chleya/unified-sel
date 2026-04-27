# Meta-Controller V0.20 Repair Residual Instrumentation - 2026-04-21

## Purpose

V0.20 adds drift repair diagnostics without changing controller behavior.

Question:

> Does the B7 `0.08` repair gate have mechanistic support beyond success/reward aggregates?

The intended evidence is repair-local:

- drift before repair
- drift after repair
- repair delta
- positive repair delta rate
- high-drift states that were not repaired
- next drift delta after high-drift no-repair states

## Implementation

Files changed:

- `experiments/meta_controller/metrics.py`
- `experiments/meta_controller/run_experiment.py`
- `experiments/meta_controller/report.py`
- `tests/test_meta_controller_protocol.py`

New metrics:

- `drift_repair_pre_mean`
- `drift_repair_post_mean`
- `drift_repair_delta_mean`
- `drift_repair_delta_positive_rate`
- `drift_residual_after_repair`
- `high_drift_no_repair_rate`
- `high_drift_no_repair_next_delta_mean`
- `repair_efficiency`

Instrumentation threshold:

- `DRIFT_REPAIR_OBSERVATION_THRESHOLD = 0.08`

This observation threshold is fixed across controllers. It is not the controller's own repair threshold. This lets the report compare whether high-drift states at the B7 default threshold were left unrepaired by fixed, B3, or looser threshold gates.

## Commands

```powershell
python -m experiments.meta_controller.run_experiment --profile v03 --mode train-eval --train-episodes 240 --eval-episodes 60 --seed <seed> --drift-threshold-sweep 0.08,0.10,0.12,0.14,0.16 --table
python -m experiments.meta_controller.run_experiment --profile v03b --mode train-eval --train-episodes 240 --eval-episodes 60 --seed <seed> --drift-threshold-sweep 0.08,0.10,0.12,0.14,0.16 --table
```

Seeds:

- `0`
- `1`
- `2`

## V03 Residual Summary

| seed | threshold | success | reward | repair rate | repair delta | positive rate | residual | high drift no repair |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.08 | 0.964 | 95.518 | 0.080 | 0.144 | 0.939 | 0.144 | 0.000 |
| 1 | 0.08 | 1.000 | 103.301 | 0.017 | 0.160 | 1.000 | 0.020 | 0.000 |
| 2 | 0.08 | 1.000 | 103.274 | 0.017 | 0.160 | 1.000 | 0.020 | 0.000 |

Reference high-drift no-repair rates:

| seed | fixed rule | B3 |
|---:|---:|---:|
| 0 | 0.189 | 0.222 |
| 1 | 0.189 | 0.475 |
| 2 | 0.189 | 0.261 |

## V03b Residual Summary

| seed | threshold | success | reward | repair rate | repair delta | positive rate | residual | high drift no repair |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.08 | 0.964 | 95.131 | 0.107 | 0.110 | 0.948 | 0.149 | 0.000 |
| 1 | 0.08 | 1.000 | 103.138 | 0.025 | 0.098 | 1.000 | 0.064 | 0.000 |
| 2 | 0.08 | 1.000 | 103.111 | 0.025 | 0.098 | 1.000 | 0.064 | 0.000 |

Reference high-drift no-repair rates:

| seed | fixed rule | B3 |
|---:|---:|---:|
| 0 | 0.239 | 0.243 |
| 1 | 0.239 | 0.478 |
| 2 | 0.239 | 0.263 |

## Interpretation

The residual instrumentation supports the V0.18/V0.19 decision:

- `0.08` does not leave high-drift states unrepaired in any v03 or v03b seed.
- repair deltas are positive in nearly all repair events.
- fixed rule and B3 leave many high-drift states unrepaired, especially on v03b and seed 1.

However, `0.08` is not a single-step residual optimizer:

- seed 0 has higher post-repair residual than seeds 1/2.
- looser thresholds sometimes show higher repair efficiency because they repair fewer, larger events.
- reward-optimal thresholds still shift across v03/v03b.

The correct interpretation is therefore:

> `0.08` is a conservative invariant-coverage gate, not a reward-optimal or repair-efficiency-optimal threshold.

## Decision

Keep B7 default:

- `drift_aware_planner_necessity_controller`: threshold `0.08`

Do not train a repair classifier yet.

If learning is introduced later, the target should be expected residual reduction or counterfactual repair benefit, not direct imitation of the threshold gate.

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
