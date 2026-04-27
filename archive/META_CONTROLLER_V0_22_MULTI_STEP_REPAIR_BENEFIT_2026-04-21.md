# Meta-Controller V0.22 Multi-Step Repair Benefit - 2026-04-21

## Purpose

V0.22 extends the V0.21 one-step repair counterfactual to a short horizon.

Question:

> Does B7 repair produce multi-step drift benefit even when short-term reward benefit is negative?

This remains analysis-only. It does not change controller behavior and does not train a learned gate.

## Implementation

Files changed:

- `experiments/meta_controller/run_experiment.py`
- `tests/test_meta_controller_protocol.py`

Extended CLI:

```powershell
python -m experiments.meta_controller.run_experiment --profile v03 --mode train-eval --train-episodes 240 --eval-episodes 60 --seed 0 --repair-benefit-analysis --repair-benefit-horizon 3 --table
```

Method:

- Collect high-drift states where `obs.drift >= 0.08`.
- Clone the environment.
- Force first step to:
  - planner repair
  - habit/no-repair
- Roll forward for horizon `3`.
- After the first forced step, use the oracle-style local decision used by existing rollout probes.
- Record:
  - terminal drift benefit
  - cumulative drift benefit
  - cumulative reward benefit
  - horizon success benefit

## V03 Horizon 3 Results

| seed | samples | terminal drift benefit | cumulative drift benefit | cumulative reward benefit | reward positive rate |
|---:|---:|---:|---:|---:|---:|
| 0 | 579 | 0.410 | 1.037 | -0.339 | 0.038 |
| 1 | 120 | 0.600 | 1.360 | -0.404 | 0.000 |
| 2 | 120 | 0.600 | 1.360 | -0.404 | 0.000 |

Terminal drift:

| seed | planner-first | habit-first |
|---:|---:|---:|
| 0 | 0.235 | 0.644 |
| 1 | 0.000 | 0.600 |
| 2 | 0.000 | 0.600 |

## V03b Horizon 3 Results

| seed | samples | terminal drift benefit | cumulative drift benefit | cumulative reward benefit | reward positive rate |
|---:|---:|---:|---:|---:|---:|
| 0 | 768 | 0.352 | 0.897 | -0.243 | 0.073 |
| 1 | 180 | 0.364 | 0.789 | -0.318 | 0.000 |
| 2 | 180 | 0.364 | 0.789 | -0.318 | 0.000 |

Terminal drift:

| seed | planner-first | habit-first |
|---:|---:|---:|
| 0 | 0.252 | 0.604 |
| 1 | 0.018 | 0.382 |
| 2 | 0.018 | 0.382 |

## Interpretation

The multi-step result strengthens the V0.21 conclusion:

- repair-first has positive terminal drift benefit on all v03/v03b seeds.
- repair-first has positive cumulative drift benefit on all v03/v03b seeds.
- cumulative reward benefit remains negative on all v03/v03b seeds.

So B7 is not merely producing a one-step artifact. The drift benefit persists and grows over a short horizon.

However, immediate and short-horizon reward remain the wrong target for a learned gate. A reward-only learner would likely suppress repairs because the planner/repair intervention pays a short-term cost while reducing latent/invariant debt.

## Decision

Keep B7 default `0.08`.

Do not train a reward-based repair classifier.

If a learned gate is eventually pursued, candidate targets should be constrained or multi-objective:

- expected terminal drift reduction
- expected cumulative drift reduction
- expected invariant debt reduction
- reward subject to drift budget

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

