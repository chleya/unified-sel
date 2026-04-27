# CEP-CC V0.1f Stronger/Frozen Two-Stage Compression Ablation - 2026-04-21

## Purpose

V0.1f tests whether stage-2 compression can simplify communication geometry when compression is stronger, the stage-2 learning rate is lower, and optionally the Listener is frozen.

Question:

> Can stage-2 compression reduce communication energy, L1, and effective dimension while preserving task performance above no-communication?

## Implementation

Files changed:

- `experiments/cep_cc/run_experiment.py`
- `tests/test_cep_cc_protocol.py`

Added:

- `freeze_listener_stage2`
- `stage2_lr`
- `run_c1f_ablation`
- CLI flag: `--c1f-ablation`
- CLI flag: `--stage2-lr`
- CLI flag: `--freeze-listener-stage2`

## Validation

Protocol tests:

```powershell
python -m pytest F:\unified-sel\tests\test_cep_cc_protocol.py -q
```

Result:

- `12 passed`
- Pytest cache warning only.

## Official Run

```powershell
python -m experiments.cep_cc.run_experiment --c1f-ablation --seeds 0,1,2 --stage1-episodes 300 --stage2-episodes 120 --batch-size 128 --lambda-sweep 0.03,0.06,0.10 --lr 0.005 --stage2-lr 0.001 --table
```

## Compact Results

Stage-1 high-bandwidth baselines:

| seed | accuracy | energy | l1 | effective dim | target alignment |
|---|---:|---:|---:|---:|---:|
| 0 | 0.668 | 0.774 | 0.843 | 4.336 | 0.552 |
| 1 | 0.859 | 0.727 | 0.810 | 4.225 | 0.614 |
| 2 | 0.828 | 0.702 | 0.793 | 4.623 | 0.647 |

Best joint stage-2 rows:

| seed | lambda | accuracy | delta acc | delta energy | delta l1 | delta dim | delta align |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.10 | 0.727 | 0.059 | -0.118 | -0.096 | -0.376 | 0.012 |
| 1 | 0.10 | 0.881 | 0.021 | -0.032 | -0.023 | 0.001 | 0.019 |
| 2 | 0.10 | 0.883 | 0.055 | -0.064 | -0.049 | 0.160 | -0.024 |

Best freeze-Listener stage-2 rows:

| seed | lambda | accuracy | delta acc | delta energy | delta l1 | delta dim | delta align |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.10 | 0.732 | 0.064 | -0.116 | -0.095 | -0.328 | 0.010 |
| 1 | 0.10 | 0.879 | 0.020 | -0.019 | -0.013 | -0.370 | -0.027 |
| 2 | 0.10 | 0.891 | 0.062 | -0.051 | -0.038 | -0.173 | -0.033 |

## Interpretation

C1f improves over C1e.

What now holds:

- stage-2 task accuracy improves on all seeds.
- communication energy decreases at the strongest tested compression.
- communication L1 decreases at the strongest tested compression.
- freeze-Listener stage 2 decreases effective dimension on all seeds.
- task accuracy remains well above no-communication.

What is still not established:

- target alignment is mixed.
- prototype metrics are still fallback proxies.
- no semantic stability test has been run.
- no segment-level compositionality intervention has been run.
- this is compression fine-tuning after a learned protocol, not spontaneous proto-symbol proof from scratch.

## Decision

C1f succeeds as a compression-control result:

> Two-stage training with stronger compression and lower stage-2 learning rate can reduce communication magnitude while preserving or improving task success.

The strongest current candidate is:

- stage 1: high-bandwidth learned, 300 episodes.
- stage 2: freeze Listener, lambda `0.10`, 120 episodes, lr `0.001`.

This is now strong enough to justify a geometry-focused C1g before C2.

Do not move directly to semantic stability yet.

Recommended next branch:

- C1g: Communication Geometry Audit

Purpose:

- replace fallback cluster proxies with stronger measurements.
- export communication trajectories and latent factors.
- measure target/invariant alignment directly.
- decide whether the compressed protocol has proto-symbol-like partitions.

