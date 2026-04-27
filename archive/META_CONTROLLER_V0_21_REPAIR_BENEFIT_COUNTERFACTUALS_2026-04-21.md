# Meta-Controller V0.21 Repair Benefit Counterfactuals - 2026-04-21

## Purpose

V0.21 estimates the counterfactual benefit of B7 repair on high-drift states.

Question:

> When drift is above the B7 observation threshold, does forcing planner repair produce measurable benefit versus habit/no-repair?

This is still analysis-only. It does not change controller behavior and does not train a learned gate.

## Implementation

Files changed:

- `experiments/meta_controller/run_experiment.py`
- `tests/test_meta_controller_protocol.py`

New function:

- `run_repair_benefit_analysis`

New CLI:

```powershell
python -m experiments.meta_controller.run_experiment --profile v03 --mode train-eval --train-episodes 240 --eval-episodes 60 --seed 0 --repair-benefit-analysis --table
```

Analysis method:

- Fit the B7 drift-aware planner-necessity controller.
- During eval, whenever `obs.drift >= 0.08`, clone the environment.
- Compare one-step forced `planner` repair against one-step forced `habit` no-repair.
- Record:
  - post-drift difference
  - immediate reward difference
  - immediate success difference
  - success rates under each branch

## V03 Results

| seed | samples | drift benefit | drift benefit positive | reward benefit | reward benefit positive | success benefit |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 579 | 0.300 | 1.000 | -0.243 | 0.038 | 0.009 |
| 1 | 120 | 0.300 | 1.000 | -0.245 | 0.000 | 0.000 |
| 2 | 120 | 0.300 | 1.000 | -0.245 | 0.000 | 0.000 |

Post-drift comparison:

| seed | planner post drift | habit post drift |
|---:|---:|---:|
| 0 | 0.155 | 0.456 |
| 1 | 0.020 | 0.320 |
| 2 | 0.020 | 0.320 |

## V03b Results

| seed | samples | drift benefit | drift benefit positive | reward benefit | reward benefit positive | success benefit |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 768 | 0.262 | 0.852 | -0.149 | 0.073 | 0.051 |
| 1 | 180 | 0.160 | 0.556 | -0.224 | 0.000 | 0.000 |
| 2 | 180 | 0.160 | 0.556 | -0.224 | 0.000 | 0.000 |

Post-drift comparison:

| seed | planner post drift | habit post drift |
|---:|---:|---:|
| 0 | 0.171 | 0.434 |
| 1 | 0.067 | 0.227 |
| 2 | 0.067 | 0.227 |

## Interpretation

The counterfactual result is clean:

- planner repair has strong positive drift benefit on every v03/v03b seed.
- one-step immediate reward benefit is negative on every seed.
- one-step success benefit is small or zero except v03b seed 0.

This confirms that B7 should not be justified as an immediate reward improvement. It is an invariant repair mechanism that accepts short-term cost to reduce drift pressure.

This also explains why reward-optimal thresholds drift across V0.18/V0.19:

- thresholds that repair fewer events can look better by immediate reward.
- thresholds that repair earlier can look better by invariant coverage.

## Decision

Keep B7 default `0.08`.

Do not train a repair classifier yet.

If a learned gate is pursued later, the learning target should be multi-step expected drift reduction or counterfactual repair value, not one-step reward and not direct threshold imitation.

## Verification

Protocol tests:

```powershell
python -m pytest F:\unified-sel\tests\test_meta_controller_protocol.py -q
```

Result:

- `10 passed`

Smoke tests:

```powershell
python F:\unified-sel\tests\smoke_test.py
```

Result:

- `All smoke tests passed`

