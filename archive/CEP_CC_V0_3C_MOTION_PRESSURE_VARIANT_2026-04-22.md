# CEP-CC V0.3c Motion Pressure Variant - 2026-04-22

## Purpose

V0.3c tests whether strengthening motion dependence makes motion acquire a targeted communication segment role.

Question:

> If motion dependence is strengthened, does motion acquire a targeted communication segment role without destroying q/relation roles?

## Implementation

Files changed:

- `experiments/cep_cc/env.py`
- `experiments/cep_cc/run_experiment.py`
- `experiments/cep_cc/metrics.py`
- `tests/test_cep_cc_protocol.py`

Added:

- `EnvConfig.rule_mode="motion_pressure"`
- `run_c3c_motion_pressure_variant`
- CLI flag: `--c3c-motion-pressure`
- table field: `rule_mode_motion_pressure`

The `motion_pressure` rule increases:

- direct motion factor weight.
- q-motion interaction.
- motion-relation interaction.

It keeps continuous communication and the existing compositional curriculum.

## Validation

Protocol tests:

```powershell
python -m pytest F:\unified-sel\tests\test_cep_cc_protocol.py -q
```

Result:

- `28 passed`
- Pytest cache warning only.

CLI smoke:

```powershell
python -m experiments.cep_cc.run_experiment --c3c-motion-pressure --seeds 0 --nuisance-modes none --stage1-episodes 2 --curriculum-episodes 2 --stage2-episodes 2 --batch-size 8 --lr 0.002 --stage2-lr 0.001 --lambda-comm 0.10 --factor-names motion --factor-consistency-sweep 0.01 --audit-batches 2 --table
```

Result:

- CLI exposes and runs `--c3c-motion-pressure`.

## Official Probe

```powershell
python -m experiments.cep_cc.run_experiment --c3c-motion-pressure --seeds 0,1,2 --nuisance-modes none --stage1-episodes 600 --curriculum-episodes 300 --stage2-episodes 120 --batch-size 128 --lr 0.005 --stage2-lr 0.001 --lambda-comm 0.10 --factor-names motion --factor-consistency-sweep 0.03 --audit-batches 12 --table
```

## Compact Results

Task accuracy:

| seed | accuracy |
|---|---:|
| 0 | 0.439 |
| 1 | 0.443 |
| 2 | 0.438 |

Per-factor targeted ratios:

| seed | q | motion | relation |
|---|---:|---:|---:|
| 0 | 1.871 | 0.057 | 2.505 |
| 1 | 1.852 | 0.053 | 2.627 |
| 2 | 1.632 | 0.054 | 2.425 |

Best segment assignment:

| factor | assignment |
|---|---|
| q | late |
| motion | early, but ratio is near zero |
| relation | early |

## Interpretation

Supported:

- q and relation roles persist under the motion-pressure variant.
- relation remains the strongest targeted factor.

Not supported:

- motion targeted control.
- task performance comparable to temporal_memory.
- simple scalar weight/interaction pressure as a route to motion compositionality.

Interesting finding:

- Motion pressure does not make motion independent. It mostly makes the task harder while the protocol still routes useful control through q and relation.
- This suggests motion may be too entangled with relation/nearest-neighbor structure in the current environment.

## Decision

C3c is a negative but informative result.

Supported claim:

> Simply increasing motion weight and motion interactions is not enough to produce a motion-specific communication segment.

Next branch:

- C3d: Motion Decoupling Environment.

Recommended focus:

- redesign the environment so motion cannot be inferred from relation/position proxies.
- introduce an independent motion-only target component.
- audit whether motion obtains a dedicated segment without sacrificing q/relation roles.
