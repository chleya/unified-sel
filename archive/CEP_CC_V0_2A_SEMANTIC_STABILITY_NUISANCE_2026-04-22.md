# CEP-CC V0.2a Semantic Stability Under Nuisance Transformations - 2026-04-22

## Purpose

V0.2a tests whether the V0.1l curriculum temporal_memory protocol preserves task behavior and communication semantics under continuous nuisance reparameterizations.

Question:

> Does the learned continuous communication protocol remain semantically stable when the world is mirrored, rotated, or velocity-scaled?

## Implementation

Files changed:

- `experiments/cep_cc/env.py`
- `experiments/cep_cc/run_experiment.py`
- `experiments/cep_cc/metrics.py`
- `tests/test_cep_cc_protocol.py`

Added:

- `EnvConfig.nuisance_mode`
- nuisance modes:
  - `none`
  - `mirror_x`
  - `rotate90`
  - `velocity_scale`
- `run_c2a_semantic_stability`
- CLI flag: `--c2a-semantic-stability`
- CLI flag: `--nuisance-modes`
- clean-delta metrics:
  - `delta_clean_task_accuracy`
  - `delta_clean_audit_hidden_q_probe_r2`
  - `delta_clean_segment_late_swap_action_change_rate`
  - `delta_clean_segment_late_ablation_action_change_rate`
  - `delta_clean_segment_late_ablation_target_logit_drop`

The primary setup is V0.1l:

- compositional curriculum pretraining.
- temporal_memory training.
- frozen-Listener compression with factor consistency.
- continuous communication only.

## Validation

Protocol tests:

```powershell
python -m pytest F:\unified-sel\tests\test_cep_cc_protocol.py -q
```

Result:

- `22 passed`
- Pytest cache warning only.

CLI smoke:

```powershell
python -m experiments.cep_cc.run_experiment --c2a-semantic-stability --seeds 0 --nuisance-modes none,mirror_x --stage1-episodes 2 --curriculum-episodes 2 --stage2-episodes 2 --batch-size 8 --lr 0.002 --stage2-lr 0.001 --lambda-comm 0.10 --factor-names q --factor-consistency-sweep 0.01 --audit-batches 2 --table
```

Result:

- CLI exposes and runs `--c2a-semantic-stability`.

## Official Run

```powershell
python -m experiments.cep_cc.run_experiment --c2a-semantic-stability --seeds 0,1,2 --nuisance-modes none,mirror_x,rotate90,velocity_scale --stage1-episodes 600 --curriculum-episodes 300 --stage2-episodes 120 --batch-size 128 --lr 0.005 --stage2-lr 0.001 --lambda-comm 0.10 --factor-names q --factor-consistency-sweep 0.03 --audit-batches 8 --table
```

## Compact Results

Task accuracy:

| seed | clean | mirror_x | rotate90 | velocity_scale |
|---|---:|---:|---:|---:|
| 0 | 0.559 | 0.572 | 0.568 | 0.557 |
| 1 | 0.697 | 0.691 | 0.695 | 0.695 |
| 2 | 0.673 | 0.682 | 0.673 | 0.670 |

Task delta from clean:

| seed | mirror_x | rotate90 | velocity_scale |
|---|---:|---:|---:|
| 0 | +0.014 | +0.010 | -0.002 |
| 1 | -0.006 | -0.002 | -0.002 |
| 2 | +0.009 | +0.000 | -0.003 |

Hidden q probe R2:

| seed | clean | mirror_x | rotate90 | velocity_scale |
|---|---:|---:|---:|---:|
| 0 | 0.642 | 0.639 | 0.606 | 0.639 |
| 1 | 0.648 | 0.646 | 0.652 | 0.648 |
| 2 | 0.629 | 0.632 | 0.633 | 0.627 |

Late segment causal effects:

| seed | mode | late swap action change | late ablation action change | late ablation logit drop |
|---|---|---:|---:|---:|
| 0 | clean | 0.503 | 0.421 | 0.455 |
| 0 | mirror_x | 0.501 | 0.405 | 0.473 |
| 0 | rotate90 | 0.493 | 0.409 | 0.459 |
| 0 | velocity_scale | 0.505 | 0.423 | 0.455 |
| 1 | clean | 0.529 | 0.396 | 0.728 |
| 1 | mirror_x | 0.528 | 0.382 | 0.740 |
| 1 | rotate90 | 0.532 | 0.391 | 0.759 |
| 1 | velocity_scale | 0.531 | 0.393 | 0.727 |
| 2 | clean | 0.568 | 0.384 | 0.775 |
| 2 | mirror_x | 0.571 | 0.383 | 0.772 |
| 2 | rotate90 | 0.575 | 0.386 | 0.757 |
| 2 | velocity_scale | 0.573 | 0.387 | 0.775 |

## Interpretation

Supported:

- Task success is stable under mirror, rotation, and velocity scaling.
- Hidden q communication semantics are mostly stable.
- Late-segment causal effects are stable under nuisance transforms.
- This strengthens the V0.1l claim that the communication is not just memorizing one coordinate parameterization.

Remaining limitations:

- This is not yet full semantic invariance under arbitrary distribution shift.
- The strongest causal effect remains concentrated in the late segment.
- Factor-specific local compositionality is still not established.

## Decision

C2a is positive.

Supported claim:

> The curriculum temporal_memory protocol preserves task-relevant communication semantics under simple continuous nuisance reparameterizations.

Next branch:

- C2b: Cluster-Invariant Alignment Under Reparameterization.

Recommended focus:

- compare prototype assignments across clean and nuisance batches.
- measure invariant-factor alignment directly.
- test whether communication clusters/probes map to the same latent variables across reparameterizations.
