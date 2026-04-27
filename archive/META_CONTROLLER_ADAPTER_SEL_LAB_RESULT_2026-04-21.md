# Meta-Controller Sel-Lab Adapter Result - 2026-04-21

## Purpose

This records the first local adapter implemented after the V0.14 interface freeze.

The adapter borrows the `sel-lab` benchmark-discipline idea, but does not import or install `F:\sel-lab`.

## Implemented

Files:

- `experiments/meta_controller/adapters/__init__.py`
- `experiments/meta_controller/adapters/sel_lab_benchmark.py`
- `tests/test_meta_controller_adapter_sel_lab.py`

The adapter defines:

- `BenchmarkSuite`
- `RunSpec`
- `AcceptanceRule`
- `default_suites()`
- `expand_suites()`
- `validate_suites()`
- `evaluate_acceptance()`
- `controller_families()`

## Suites

Current default suites:

1. `mainline_acceptance`
2. `transfer_v02`
3. `read_cost_stress`
4. `b3_signal_masking`

These encode the manual runs already used for V0.11-V0.13:

- v01 mainline seeds
- v02 transfer seeds
- read-cost `0.20` stress
- B3 signal masking

## Boundaries Preserved

The adapter is intentionally outside the causal core.

It does not:

- import `F:\sel-lab`
- call old F-drive code
- change `WorkspaceState`
- change `ControlDecision`
- change environment dynamics
- change controller behavior
- change metric definitions
- execute long benchmark sweeps automatically

It only creates deterministic run specifications and acceptance checks.

## Validation

Commands:

```powershell
python -m py_compile F:\unified-sel\experiments\meta_controller\adapters\sel_lab_benchmark.py F:\unified-sel\tests\test_meta_controller_adapter_sel_lab.py
python -m pytest F:\unified-sel\tests\test_meta_controller_adapter_sel_lab.py -q -p no:cacheprovider
python -m pytest F:\unified-sel\tests\test_meta_controller_protocol.py F:\unified-sel\tests\test_meta_controller_adapter_sel_lab.py -q -p no:cacheprovider
python -m experiments.meta_controller.run_experiment --help
```

Result:

- adapter tests before CLI path: `6 passed`
- combined meta-controller tests before CLI path: `12 passed`
- adapter tests after CLI path: `7 passed`
- combined meta-controller tests after CLI path: `13 passed`
- CLI exposes `--suite {all,mainline_acceptance,transfer_v02,read_cost_stress,b3_signal_masking}`

## Interpretation

This is a safe first integration step.

It upgrades the experiment from hand-managed command lists to an explicit benchmark matrix while keeping old F-drive projects as reference-only material.

The next useful step is one of:

1. add a `--suite` CLI path that runs these `RunSpec` objects
2. create `v03_long_horizon_drift`

## CLI Path

Implemented:

```powershell
python -m experiments.meta_controller.run_experiment --suite mainline_acceptance --table
python -m experiments.meta_controller.run_experiment --suite transfer_v02 --table
python -m experiments.meta_controller.run_experiment --suite read_cost_stress --table
python -m experiments.meta_controller.run_experiment --suite b3_signal_masking --table
```

The suite path:

- expands the adapter's deterministic run specs
- reuses `run_train_eval_suite`
- filters the displayed results to the suite's declared controllers and mask rows
- reports acceptance checks per seed
- does not alter the underlying experiment

Long suite sweeps were not launched during this implementation pass.

## Next Step

The recommended next step is:

1. run `--suite mainline_acceptance --table` as the first benchmark-matrix smoke sweep
2. then build `v03_long_horizon_drift`
