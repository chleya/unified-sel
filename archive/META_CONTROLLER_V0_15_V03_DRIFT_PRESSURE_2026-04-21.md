# Meta-Controller V0.15 V03 Drift Pressure - 2026-04-21

## Purpose

V0.15 adds a long-horizon drift pressure profile:

- `v03`
- suite: `long_horizon_drift_v03`

The purpose is to test whether `invariant_violation` can become a causal control signal instead of remaining redundant.

## Mechanism

`EnvConfig` now supports invariant guard dynamics:

- `invariant_guard_steps`
- `invariant_guard_threshold`
- `unguarded_drift_increase`
- `guarded_drift_decay`

At invariant guard states, the immediate task action can still succeed. However:

- if the dominant module is not `planner`, drift increases
- if the dominant module is `planner`, drift is repaired
- if drift crosses the threshold, subsequent steps continue requiring planner repair until drift drops

This creates the intended pressure:

> short-term action success is not enough; the controller must sometimes allocate dominance to planner to preserve long-horizon invariants.

## Code Changes

Files:

- `experiments/meta_controller/env.py`
- `experiments/meta_controller/run_experiment.py`
- `experiments/meta_controller/adapters/sel_lab_benchmark.py`
- `tests/test_meta_controller_protocol.py`
- `tests/test_meta_controller_adapter_sel_lab.py`

New environment functions:

- `v03_train_configs`
- `v03_heldout_configs`

New CLI profile:

```powershell
python -m experiments.meta_controller.run_experiment --profile v03 --mode train-eval --train-episodes 240 --eval-episodes 60 --seed 0 --table
```

New suite:

```powershell
python -m experiments.meta_controller.run_experiment --suite long_horizon_drift_v03 --table
```

## Validation

Commands:

```powershell
python -m py_compile F:\unified-sel\experiments\meta_controller\env.py F:\unified-sel\experiments\meta_controller\run_experiment.py F:\unified-sel\experiments\meta_controller\adapters\sel_lab_benchmark.py F:\unified-sel\tests\test_meta_controller_protocol.py F:\unified-sel\tests\test_meta_controller_adapter_sel_lab.py
python -m pytest F:\unified-sel\tests\test_meta_controller_adapter_sel_lab.py F:\unified-sel\tests\test_meta_controller_protocol.py -q -p no:cacheprovider
python -m experiments.meta_controller.run_experiment --profile v03 --mode train-eval --train-episodes 12 --eval-episodes 6 --seed 6 --table
```

Result:

- combined tests: `15 passed`
- quick v03 run completed

## Quick V03 Signal

The quick run is not a final benchmark. It is a profile smoke test.

Observed pattern:

- `habit_only` gets high apparent reward but accumulates very high drift.
- `planner_always` keeps drift near zero but pays high compute cost.
- B3 mainline still reduces some planner use, but does not yet reliably beat fixed-rule drift guard.
- `b3_mask_drift` increases drift relative to B3 mainline in the quick run.
- `b3_mask_conflict` and core-signal masking remain strongly damaging.

Interpretation:

- v03 successfully creates a pressure that separates short-term success from long-horizon invariant preservation.
- `invariant_violation` is now at least partially active.
- The current B3 mainline is not yet a solved controller for this setting.

## Research Consequence

This is useful because it creates a harder falsification target.

The next branch should not claim victory for B3. It should ask:

> Can a learned or calibrated drift-aware dominance rule preserve B3's planner savings while keeping v03 drift below fixed-rule or shielded baselines?

Candidate next branch:

- B7 `Drift-Aware Planner Necessity`

Minimum idea:

- keep B3's habit-safe and planner-necessity gates
- add a learned or calibrated drift repair gate
- planner is forced only when invariant violation predicts unrecovered drift
- compare against fixed rule, shielded dominance, and planner-always

