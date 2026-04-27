# CEP-CC V0.3d Motion Decoupling Environment - 2026-04-22

## Purpose

V0.3d tests whether motion acquires a targeted communication segment when the motion factor is made independent of relation and position proxies.

Question:

> If motion is made independent of nearest-neighbor relation/position structure, does it acquire a dedicated communication segment?

## Implementation

Files changed:

- `experiments/cep_cc/env.py`
- `experiments/cep_cc/run_experiment.py`
- `experiments/cep_cc/metrics.py`
- `tests/test_cep_cc_protocol.py`

Added:

- `EnvConfig.rule_mode="motion_decoupled"`
- `run_c3d_motion_decoupling_variant`
- CLI flag: `--c3d-motion-decoupling`
- table field: `rule_mode_motion_decoupled`

The `motion_decoupled` rule replaces nearest-neighbor velocity alignment with an independent velocity projection:

- motion depends on object velocity and a hidden goal dimension.
- motion does not depend on positions or nearest-neighbor relation.
- q and relation factors remain available so the C3b q/relation roles can be checked for persistence.

This preserves the no-token/no-codebook constraint. No discrete symbol module or classifier was introduced.

## Validation

Protocol tests:

```powershell
python -m pytest F:\unified-sel\tests\test_cep_cc_protocol.py -q
```

Result:

- `30 passed`
- Pytest cache warning only.

CLI smoke:

```powershell
python -m experiments.cep_cc.run_experiment --c3d-motion-decoupling --seeds 0 --nuisance-modes none --stage1-episodes 2 --curriculum-episodes 2 --stage2-episodes 2 --batch-size 8 --lr 0.002 --stage2-lr 0.001 --lambda-comm 0.10 --factor-names motion --factor-consistency-sweep 0.01 --audit-batches 2 --table
```

Result:

- CLI exposes and runs `--c3d-motion-decoupling`.

## Official Probe

```powershell
python -m experiments.cep_cc.run_experiment --c3d-motion-decoupling --seeds 0,1,2 --nuisance-modes none --stage1-episodes 600 --curriculum-episodes 300 --stage2-episodes 120 --batch-size 128 --lr 0.005 --stage2-lr 0.001 --lambda-comm 0.10 --factor-names motion --factor-consistency-sweep 0.03 --audit-batches 12 --table
```

## Compact Results

Task accuracy:

| seed | accuracy |
|---|---:|
| 0 | `0.562` |
| 1 | `0.568` |
| 2 | `0.709` |

Per-factor targeted ratios:

| seed | q | motion | relation |
|---|---:|---:|---:|
| 0 | `1.278` | `0.487` | `1.519` |
| 1 | `1.309` | `0.480` | `1.537` |
| 2 | `1.368` | `0.473` | `1.779` |

Best segment assignment:

| factor | assignment |
|---|---|
| q | middle/late |
| motion | late, but targeted ratio remains below 1 |
| relation | early |

## Environment Dependency Diagnostic

One-off oracle ablation on `motion_decoupled`, batch size `8192`, seed `123`:

| ablation | target agreement with full rule |
|---|---:|
| no q | `0.660` |
| no motion | `0.479` |
| no relation | `0.825` |

This confirms that motion is genuinely task-critical in the environment. Removing motion changes the target more than removing q or relation.

## Interpretation

Supported:

- Decoupling motion from relation/position makes the task learnable again compared with V0.3c.
- relation still maps to the early segment.
- q keeps a weak but stable local role.
- motion is task-critical in the target rule.

Not supported:

- a motion-specific locally replaceable communication segment.
- full three-factor local compositionality.

Interesting finding:

- Even when motion is the most target-critical factor, the protocol does not expose motion as a clean segment-level control variable.
- The late segment affects actions strongly, but not in a semantically targeted motion-only way.
- This suggests the current recurrent continuous protocol may compress motion into an entangled decision trajectory rather than a reusable local unit.

## Decision

C3d is a strong negative/informative result.

Supported claim:

> Task-critical factors do not automatically become locally compositional communication segments under the current CEP-CC pressure set.

Refined claim:

> CEP-CC currently shows robust two-factor local roles for q/relation, but motion remains entangled even after decoupling from relation/position proxies.

Next branch:

- C3e: Motion Readout And Temporal-Position Audit.

Recommended focus:

- audit whether motion is present as a distributed or late global readout rather than a segment-local variable.
- add factor-specific probes for motion and relation across early/middle/late segments.
- avoid learned discrete classifiers or codebooks until the continuous audit explains where motion information lives.
