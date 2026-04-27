# CEP-CC V0.2c Factor-Separated Invariance Audit - 2026-04-22

## Purpose

V0.2c resolves the ambiguity in V0.2b by separating invariant factors from coordinate-dependent or equivariant factors.

Question:

> Which semantic factors are invariant, and which transform predictably, under nuisance transformations?

## Implementation

Files changed:

- `experiments/cep_cc/metrics.py`
- `experiments/cep_cc/run_experiment.py`
- `tests/test_cep_cc_protocol.py`

Added per-factor paired metrics:

- `paired_pos_factor_corr`
- `paired_q_score_factor_corr`
- `paired_motion_factor_corr`
- `paired_relation_factor_corr`
- `paired_invariant_factor_corr`
- `paired_equivariant_factor_corr`

Added:

- `run_c2c_factor_separated_invariance`
- CLI flag: `--c2c-factor-invariance`

The runner reuses the V0.2b paired clean/nuisance protocol and reports factor-separated correlations.

## Validation

Protocol tests:

```powershell
python -m pytest F:\unified-sel\tests\test_cep_cc_protocol.py -q
```

Result:

- `24 passed`
- Pytest cache warning only.

CLI smoke:

```powershell
python -m experiments.cep_cc.run_experiment --c2c-factor-invariance --seeds 0 --nuisance-modes mirror_x --stage1-episodes 2 --curriculum-episodes 2 --stage2-episodes 2 --batch-size 8 --lr 0.002 --stage2-lr 0.001 --lambda-comm 0.10 --factor-names q --factor-consistency-sweep 0.01 --audit-batches 2 --table
```

Result:

- CLI exposes and runs `--c2c-factor-invariance`.

## Official Run

```powershell
python -m experiments.cep_cc.run_experiment --c2c-factor-invariance --seeds 0,1,2 --nuisance-modes mirror_x,rotate90,velocity_scale --stage1-episodes 600 --curriculum-episodes 300 --stage2-episodes 120 --batch-size 128 --lr 0.005 --stage2-lr 0.001 --lambda-comm 0.10 --factor-names q --factor-consistency-sweep 0.03 --audit-batches 4 --table
```

## Compact Results

Invariant q-score factor correlation:

| seed | mirror_x | rotate90 | velocity_scale |
|---|---:|---:|---:|
| 0 | 0.840 | 0.911 | 1.000 |
| 1 | 0.851 | 0.903 | 1.000 |
| 2 | 0.866 | 0.920 | 1.000 |

Hidden q object-value correlation:

| seed | mirror_x | rotate90 | velocity_scale |
|---|---:|---:|---:|
| 0 | 0.908 | 0.952 | 1.000 |
| 1 | 0.901 | 0.945 | 1.000 |
| 2 | 0.918 | 0.956 | 1.000 |

Position factor correlation:

| seed | mirror_x | rotate90 | velocity_scale |
|---|---:|---:|---:|
| 0 | 0.656 | 0.850 | 1.000 |
| 1 | 0.630 | 0.848 | 1.000 |
| 2 | 0.665 | 0.877 | 1.000 |

Motion factor correlation:

| seed | mirror_x | rotate90 | velocity_scale |
|---|---:|---:|---:|
| 0 | 0.742 | 0.849 | 1.000 |
| 1 | 0.704 | 0.818 | 1.000 |
| 2 | 0.705 | 0.873 | 1.000 |

Relation factor correlation:

| seed | mirror_x | rotate90 | velocity_scale |
|---|---:|---:|---:|
| 0 | -0.934 | 0.047 | 1.000 |
| 1 | -0.929 | 0.067 | 1.000 |
| 2 | -0.927 | 0.129 | 1.000 |

Mean equivariant factor correlation:

| seed | mirror_x | rotate90 | velocity_scale |
|---|---:|---:|---:|
| 0 | 0.155 | 0.582 | 1.000 |
| 1 | 0.135 | 0.578 | 1.000 |
| 2 | 0.147 | 0.626 | 1.000 |

## Interpretation

Supported:

- Hidden q and q-score semantics are stable under all tested nuisance transforms.
- Velocity scaling is effectively fully invariant for all audited factors.
- Position and motion are moderately aligned under mirror and strongly aligned under rotation.
- The relation factor is sign-reversing under mirror and weakly aligned under rotation, which explains the low aggregate latent correlation in V0.2b.

Key clarification:

- The low aggregate latent correlation in V0.2b does not mean the communication protocol lost semantic stability.
- It mostly reflects that relation is coordinate/sign dependent under mirror and rotation.
- Treating invariant and equivariant factors separately is necessary.

## Decision

C2c is positive.

Supported claim:

> The protocol preserves invariant hidden-q semantics, while coordinate-dependent factors transform nontrivially but predictably under nuisance transformations.

Remaining boundary:

> This still does not establish full local compositional syntax.

Next:

- C3a: Local Compositionality With Targeted Segment Intervention.

Recommended focus:

- use the robust C1l/C2 curriculum temporal_memory protocol.
- test targeted segment effects by factor, not just generic late-segment causal importance.
- separate invariant hidden-q interventions from relation/motion interventions.
