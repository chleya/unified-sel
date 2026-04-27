# Meta-Controller V0.24 B7 Acceptance Artifact - 2026-04-21

## Purpose

V0.24 turns the frozen B7 claim into explicit acceptance checks.

Frozen claim:

> B7 is an invariant-coverage repair gate. On drift-pressure profiles it should reduce drift debt and high-drift no-repair exposure, while reporting reward tradeoffs rather than hiding them. On non-pressure profiles it should avoid large regressions.

## Implementation

Files changed:

- `experiments/meta_controller/run_experiment.py`
- `experiments/meta_controller/README.md`
- `tests/test_meta_controller_protocol.py`

New functions:

- `evaluate_b7_acceptance`
- `run_b7_acceptance_artifact`

New CLI:

```powershell
python -m experiments.meta_controller.run_experiment --b7-acceptance-artifact --matrix-profiles v01,v02,v03,v03b --matrix-seeds 0,1,2 --train-episodes 240 --eval-episodes 60 --repair-benefit-horizon 3 --table
```

As with V0.23, the full 4-profile run is operationally heavy. The official run was completed per profile with the same settings.

## Acceptance Policy

Pressure profiles:

- `v03`
- `v03b`

Required:

- drift improves by at least `0.05` versus fixed rule.
- high-drift no-repair exposure improves by at least `0.10` versus fixed rule.
- terminal drift benefit is positive.
- cumulative drift benefit is positive.
- success delta is not below `-0.01`.

Reported, not pass/fail target:

- reward delta versus fixed rule.
- cumulative reward benefit.

Regression profiles:

- `v01`
- `v02`

Required:

- success delta is not below `-0.02`.
- planner calls do not increase by more than `2.0`.
- drift does not worsen by more than `0.02`.
- reward delta is not below `-3.0`.

Reported:

- reward tradeoff.

## Official Results

All profile-specific acceptance runs passed.

### V01

| check group | result |
|---|---|
| seed 0 bounded regression | PASS |
| seed 1 bounded regression | PASS |
| seed 2 bounded regression | PASS |

Observed worst values:

- worst success delta: `-0.018`
- worst drift delta: `0.011`
- worst reward delta: `-2.241`
- planner deltas were all negative.

### V02

| check group | result |
|---|---|
| seed 0 bounded regression | PASS |
| seed 1 bounded regression | PASS |
| seed 2 bounded regression | PASS |

Observed worst values:

- worst success delta: `0.001`
- worst drift delta: `-0.028`
- worst reward delta: `-0.013`
- planner deltas were all negative.

### V03

| check group | result |
|---|---|
| seed 0 pressure acceptance | PASS |
| seed 1 pressure acceptance | PASS |
| seed 2 pressure acceptance | PASS |

Observed ranges:

- drift delta: `-0.098`, `-0.094`, `-0.089`
- high-drift no-repair delta: `-0.189` on all seeds
- terminal drift benefit: `0.410`, `0.600`, `0.600`
- cumulative drift benefit: `1.037`, `1.360`, `1.360`
- reward delta is mixed and reported, not hidden: `-1.961`, `-4.609`, `1.990`

### V03b

| check group | result |
|---|---|
| seed 0 pressure acceptance | PASS |
| seed 1 pressure acceptance | PASS |
| seed 2 pressure acceptance | PASS |

Observed ranges:

- drift delta: `-0.146`, `-0.141`, `-0.136`
- high-drift no-repair delta: `-0.239` on all seeds
- terminal drift benefit: `0.352`, `0.364`, `0.364`
- cumulative drift benefit: `0.897`, `0.789`, `0.789`
- reward delta is mixed and reported, not hidden: `-3.016`, `-5.630`, `0.592`

## Decision

B7 passes the frozen acceptance artifact.

The claim is now stable enough to move from experiment-building to claim/evidence packaging.

Do not build a learned repair gate yet.

Next step should be a claim-evidence map and paper/report figure plan for:

- B7 invariant repair gate
- threshold sweep
- v03b transfer
- repair residual instrumentation
- one-step and multi-step counterfactual benefit
- acceptance matrix

## Verification

Protocol tests:

```powershell
python -m pytest F:\unified-sel\tests\test_meta_controller_protocol.py -q
```

Result:

- `12 passed`

Smoke tests:

```powershell
python F:\unified-sel\tests\smoke_test.py
```

Result:

- `All smoke tests passed`

