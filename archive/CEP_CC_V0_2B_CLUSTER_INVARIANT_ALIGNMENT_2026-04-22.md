# CEP-CC V0.2b Cluster-Invariant Alignment Under Reparameterization - 2026-04-22

## Purpose

V0.2b tests paired clean/nuisance alignment for the V0.1l curriculum temporal_memory protocol.

Question:

> Do communication prototype assignments and latent-factor alignments remain stable when the same random world sample is nuisance-transformed?

## Implementation

Files changed:

- `experiments/cep_cc/metrics.py`
- `experiments/cep_cc/run_experiment.py`
- `tests/test_cep_cc_protocol.py`

Added:

- `paired_nuisance_alignment_metrics`
- `_summarize_collected_audit`
- `run_c2b_cluster_invariant_alignment`
- CLI flag: `--c2b-cluster-alignment`

Paired metrics:

- `paired_target_agreement`
- `paired_proto_assignment_stability`
- `paired_comm_distance_ratio`
- `paired_hidden_q_corr`
- `paired_latent_corr`

The protocol is trained once per seed using the V0.1l curriculum temporal_memory setup, then evaluated on paired clean and nuisance-transformed batches generated from the same sample seeds.

## Validation

Protocol tests:

```powershell
python -m pytest F:\unified-sel\tests\test_cep_cc_protocol.py -q
```

Result:

- `23 passed`
- Pytest cache warning only.

CLI smoke:

```powershell
python -m experiments.cep_cc.run_experiment --c2b-cluster-alignment --seeds 0 --nuisance-modes mirror_x --stage1-episodes 2 --curriculum-episodes 2 --stage2-episodes 2 --batch-size 8 --lr 0.002 --stage2-lr 0.001 --lambda-comm 0.10 --factor-names q --factor-consistency-sweep 0.01 --audit-batches 2 --table
```

Result:

- CLI exposes and runs `--c2b-cluster-alignment`.

Note:

- The first full C2b run timed out because previous Python experiment processes remained alive after timeout and consumed CPU. After stopping stale processes and avoiding duplicate audit collection, the official run completed.

## Official Run

```powershell
python -m experiments.cep_cc.run_experiment --c2b-cluster-alignment --seeds 0,1,2 --nuisance-modes mirror_x,rotate90,velocity_scale --stage1-episodes 600 --curriculum-episodes 300 --stage2-episodes 120 --batch-size 128 --lr 0.005 --stage2-lr 0.001 --lambda-comm 0.10 --factor-names q --factor-consistency-sweep 0.03 --audit-batches 4 --table
```

## Compact Results

Paired target agreement:

| seed | mirror_x | rotate90 | velocity_scale |
|---|---:|---:|---:|
| 0 | 0.609 | 0.809 | 1.000 |
| 1 | 0.598 | 0.797 | 1.000 |
| 2 | 0.629 | 0.828 | 1.000 |

Prototype assignment stability:

| seed | mirror_x | rotate90 | velocity_scale |
|---|---:|---:|---:|
| 0 | 0.975 | 0.977 | 0.996 |
| 1 | 0.973 | 0.965 | 0.988 |
| 2 | 0.982 | 0.988 | 0.998 |

Paired communication distance ratio:

| seed | mirror_x | rotate90 | velocity_scale |
|---|---:|---:|---:|
| 0 | 0.065 | 0.059 | 0.012 |
| 1 | 0.046 | 0.053 | 0.008 |
| 2 | 0.043 | 0.046 | 0.008 |

Hidden q correlation:

| seed | mirror_x | rotate90 | velocity_scale |
|---|---:|---:|---:|
| 0 | 0.908 | 0.952 | 1.000 |
| 1 | 0.901 | 0.945 | 1.000 |
| 2 | 0.918 | 0.956 | 1.000 |

Latent-factor correlation:

| seed | mirror_x | rotate90 | velocity_scale |
|---|---:|---:|---:|
| 0 | 0.326 | 0.664 | 1.000 |
| 1 | 0.314 | 0.659 | 1.000 |
| 2 | 0.327 | 0.699 | 1.000 |

## Interpretation

Supported:

- Prototype assignment stability is high under all nuisance transforms.
- Paired communication distance is much smaller than unrelated within-batch distances.
- Hidden q alignment is high under mirror and rotation, and exact under velocity scaling.
- Velocity scaling is nearly perfectly invariant.

Mixed:

- Paired target agreement is partial under mirror and rotation because the target index can change when coordinate-dependent motion/relation factors are reparameterized.
- Mean latent-factor correlation is low for mirror and moderate for rotation because some audited latent factors are coordinate/relationship dependent.

Not supported:

- Full cluster identity invariance under all reparameterizations.
- Full latent-factor invariance for coordinate-dependent factors.

## Decision

C2b is mixed-positive.

Supported claim:

> The protocol's communication geometry and hidden-q semantics remain strongly aligned under paired nuisance transforms.

Qualified boundary:

> The protocol is not invariant to all audited latent factors, because some factors are intentionally coordinate/relationship dependent.

Next branch:

- C2c: Factor-Separated Invariance Audit.

Recommended focus:

- split invariant factors from equivariant factors.
- report hidden q separately from position/motion/relation.
- avoid treating all latent dimensions as if they should be invariant.
- define which semantic variables should remain invariant and which should transform predictably.
