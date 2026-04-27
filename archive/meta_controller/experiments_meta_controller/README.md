# Meta-Controller V0 Experiment

This is a minimal, self-contained experiment for testing learned arbitration.

It intentionally does not import other F-drive projects. The first goal is to
make the causal question measurable before integrating larger prototypes.

## Run

```powershell
python -m experiments.meta_controller.run_experiment --episodes 12 --seed 0
```

Train/evaluate split with signal masking ablations:

```powershell
python -m experiments.meta_controller.run_experiment --mode train-eval --train-episodes 120 --eval-episodes 30 --seed 0
```

V0.1 pressure profile:

```powershell
python -m experiments.meta_controller.run_experiment --profile v01 --mode train-eval --train-episodes 240 --eval-episodes 60 --seed 0 --table
```

V0.2 transfer profile:

```powershell
python -m experiments.meta_controller.run_experiment --profile v02 --mode train-eval --train-episodes 240 --eval-episodes 60 --seed 0 --table
```

V0.3 long-horizon drift pressure profile:

```powershell
python -m experiments.meta_controller.run_experiment --profile v03 --mode train-eval --train-episodes 240 --eval-episodes 60 --seed 0 --table
```

V0.3b drift variant transfer profile:

```powershell
python -m experiments.meta_controller.run_experiment --profile v03b --mode train-eval --train-episodes 240 --eval-episodes 60 --seed 0 --table
```

Benchmark suite matrix:

```powershell
python -m experiments.meta_controller.run_experiment --suite mainline_acceptance --table
```

B7 transfer matrix:

```powershell
python -m experiments.meta_controller.run_experiment --b7-transfer-matrix --train-episodes 240 --eval-episodes 60 --repair-benefit-horizon 3 --table
```

B7 acceptance artifact:

```powershell
python -m experiments.meta_controller.run_experiment --b7-acceptance-artifact --train-episodes 240 --eval-episodes 60 --repair-benefit-horizon 3 --table
```

Available suites:

- `mainline_acceptance`
- `transfer_v02`
- `read_cost_stress`
- `b3_signal_masking`
- `long_horizon_drift_v03`
- `v03b_drift_variant_transfer`
- `b7_cross_profile_v01`
- `b7_cross_profile_v02`
- `all`

## Claim Under Test

A learned meta-controller should learn when to:

- continue a cheap habit policy
- call an expensive planner
- read episodic memory
- write useful clues into memory
- broadcast a control change

The environment has hidden regime shifts and delayed memory queries. The secret
clue is visible early, but later steps require recalling it. Planner calls and
memory operations have explicit costs.

## First Baselines

- `habit_only`
- `planner_always`
- `memory_always`
- `fixed_rule_controller`
- `random_controller`
- `learned_contextual_bandit`

## Current Mainline

The current V0 causal core is:

- `planner_necessity_loose_controller`

It combines:

- habit-safe initiation
- planner-necessity estimation
- gated memory read/write

The frozen interface contract is documented in:

- `F:\unified-sel\META_CONTROLLER_V0_14_INTERFACE_2026-04-21.md`

The current drift-pressure branch is:

- `drift_aware_planner_necessity_controller`

It is documented in:

- `F:\unified-sel\META_CONTROLLER_V0_16_DRIFT_AWARE_B7_2026-04-21.md`

Old F-drive projects should not be imported directly. They should only be adapted after matching the V0.14 interface contract.

The first local adapter is documented in:

- `F:\unified-sel\META_CONTROLLER_ADAPTER_SEL_LAB_RESULT_2026-04-21.md`

It adds benchmark suite/run-spec organization only; it does not import `F:\sel-lab` or change controller behavior.

## Primary Metrics

- task success
- arbitration regret
- switch latency
- recovery after surprise
- compute cost
- memory read/write precision
- drift under horizon
