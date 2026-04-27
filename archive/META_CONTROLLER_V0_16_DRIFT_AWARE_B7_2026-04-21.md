# Meta-Controller V0.16 Drift-Aware B7 - 2026-04-21

## Purpose

V0.16 implements B7:

- `DriftAwarePlannerNecessityController`

This is a narrow extension of B3. It keeps:

- habit-safe gate
- planner-necessity gate
- gated memory read/write

and adds:

- a drift repair gate based on `invariant_violation`

## Implementation

Files:

- `experiments/meta_controller/meta_controller.py`
- `experiments/meta_controller/baselines.py`
- `experiments/meta_controller/run_experiment.py`
- `experiments/meta_controller/adapters/sel_lab_benchmark.py`
- `tests/test_meta_controller_protocol.py`

New controllers:

- `drift_aware_planner_necessity_controller`
- `drift_aware_planner_necessity_loose_controller`

Mechanism:

- call B3 normally
- if `state.invariant_violation >= drift_repair_threshold`, force planner dominance
- preserve memory read when selected
- do not force planner-always

Thresholds:

- main drift-aware: `0.10`
- loose drift-aware: `0.14`

## Full V03 Suite Result

Command:

```powershell
python -m experiments.meta_controller.run_experiment --suite long_horizon_drift_v03 --table
```

### Seed 0

| controller | success | reward | planner calls | drift |
|---|---:|---:|---:|---:|
| fixed rule | 0.964 | 95.576 | 82.000 | 0.109 |
| B3 mainline | 1.000 | 106.601 | 75.467 | 0.160 |
| B7 drift-aware | 0.967 | 93.615 | 86.150 | 0.011 |
| B7 loose | 1.000 | 103.347 | 77.467 | 0.003 |
| shielded | 1.000 | 102.670 | 80.667 | 0.003 |

### Seed 1

| controller | success | reward | planner calls | drift |
|---|---:|---:|---:|---:|
| fixed rule | 0.964 | 95.576 | 82.000 | 0.109 |
| B3 mainline | 0.965 | 94.670 | 86.583 | 0.095 |
| B7 drift-aware | 0.978 | 97.021 | 82.533 | 0.011 |
| B7 loose | 0.970 | 94.800 | 84.383 | 0.019 |
| shielded | 0.967 | 93.252 | 87.333 | 0.006 |

### Seed 2

| controller | success | reward | planner calls | drift |
|---|---:|---:|---:|---:|
| fixed rule | 0.964 | 95.576 | 82.000 | 0.109 |
| B3 mainline | 0.967 | 95.983 | 83.867 | 0.115 |
| B7 drift-aware | 1.000 | 103.367 | 77.367 | 0.003 |
| B7 loose | 0.969 | 94.072 | 86.717 | 0.018 |
| shielded | 0.967 | 93.332 | 87.333 | 0.006 |

## Acceptance

In the full v03 suite:

- B7 drift-aware passes drift <= fixed on all 3 seeds.
- B7 drift-aware passes success >= fixed on all 3 seeds.
- B7 does not collapse to planner-always; planner calls remain far below 120.
- B7 main is stronger than B3 on drift.
- B7 loose is sometimes better on reward/planner calls, but less stable.

Signal masking:

- `b3_mask_conflict` and core masking strongly increase drift on all seeds.
- `b3_mask_drift` increases drift on seeds 1 and 2.
- `b3_mask_drift` does not increase drift on seed 0, so drift is not the only causal route in v03.

## Interpretation

B7 is a positive result for the new v03 pressure profile.

It demonstrates that `invariant_violation` can be turned into a useful control signal when the environment separates short-term success from long-horizon drift.

The result is not yet final:

- B7 main sacrifices reward on seed 0 relative to B3 and shielded.
- B7 loose is more efficient on seed 0 but weaker on seeds 1 and 2.
- There is no explicit metric yet for drift-repair interventions.
- B7 has not been checked against v01/v02 regressions in the suite matrix.

## Next Step

V0.17 should harden B7:

1. add a `drift_repair_rate` metric
2. add B7 to mainline/transfer suites or create a cross-profile B7 suite
3. run v01/v02/v03 regression checks
4. decide whether B7 main or B7 loose should become the v03 mainline

