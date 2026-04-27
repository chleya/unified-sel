# CEP-CC V0.1g Communication Geometry Audit - 2026-04-21

## Purpose

V0.1g audits whether the compressed two-stage protocol from V0.1f contains proto-symbol-like trajectory partitions aligned with task-relevant invariants.

Question:

> Does freeze-Listener two-stage compression at lambda `0.10` create a compact, invariant-aligned communication geometry beyond task success and lower magnitude?

## Implementation

Files changed:

- `experiments/cep_cc/metrics.py`
- `experiments/cep_cc/run_experiment.py`
- `tests/test_cep_cc_protocol.py`

Added metrics:

- `audit_target_purity`
- `audit_within_between_ratio`
- `audit_nearest_proto_stability`
- `audit_latent_probe_r2`
- `audit_hidden_q_probe_r2`

Added runner:

- `run_c1g_geometry_audit`

Added CLI:

- `--c1g-geometry-audit`
- `--audit-batches`

## Validation

Protocol tests:

```powershell
python -m pytest F:\unified-sel\tests\test_cep_cc_protocol.py -q
```

Result:

- `13 passed`
- Pytest cache warning only.

## Official Run

```powershell
python -m experiments.cep_cc.run_experiment --c1g-geometry-audit --seeds 0,1,2 --stage1-episodes 300 --stage2-episodes 120 --batch-size 128 --lambda-comm 0.10 --lr 0.005 --stage2-lr 0.001 --audit-batches 8 --table
```

## Results

| run | accuracy | energy | l1 | dim | target align | target purity | within/between | proto stability | latent probe r2 | hidden q probe r2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seed 0 stage1 | 0.671 | 0.779 | 0.846 | 4.241 | 0.556 | 0.556 | 0.949 | 0.984 | -0.013 | 0.651 |
| seed 0 compressed | 0.720 | 0.666 | 0.755 | 3.908 | 0.557 | 0.557 | 0.947 | 0.995 | 0.042 | 0.648 |
| seed 1 stage1 | 0.865 | 0.728 | 0.811 | 4.223 | 0.598 | 0.598 | 0.952 | 0.995 | -0.030 | 0.597 |
| seed 1 compressed | 0.896 | 0.706 | 0.795 | 3.871 | 0.638 | 0.638 | 0.950 | 0.994 | -0.001 | 0.637 |
| seed 2 stage1 | 0.856 | 0.704 | 0.794 | 4.472 | 0.588 | 0.588 | 0.950 | 0.994 | 0.015 | 0.590 |
| seed 2 compressed | 0.903 | 0.652 | 0.756 | 4.350 | 0.567 | 0.567 | 0.947 | 0.987 | 0.034 | 0.612 |

## Interpretation

Compression-control result still holds:

- compressed accuracy improves on every seed.
- energy decreases on every seed.
- L1 decreases on every seed.
- effective dimension decreases on every seed.

Hidden continuous factor encoding is present:

- hidden `q` probe R2 is strong and stable:
  - seed 0: `0.651` -> `0.648`
  - seed 1: `0.597` -> `0.637`
  - seed 2: `0.590` -> `0.612`

But proto-symbol partition evidence is weak:

- target purity is only around `0.56-0.64`.
- within/between target trajectory distance ratio is about `0.947-0.952`, close to no separation.
- nearest-prototype stability is high, but this mostly shows local geometric smoothness under small noise, not semantic discreteness.
- full latent-factor probe R2 is near zero.

## Decision

C1g does not justify moving to C2 semantic stability.

Supported claim:

> Two-stage compressed continuous communication preserves task performance and encodes a hidden continuous task factor while reducing communication magnitude and effective dimension.

Unsupported claim:

> The compressed communication has clear proto-symbol partitions.

Next work should remain in C1 and add pressure that specifically encourages reusable partitions:

- consistency loss for semantically similar samples.
- contrastive/prototype regularization over target-relevant invariants.
- bottleneck scheduling after protocol discovery.

Recommended next branch:

- C1h: Consistency-Pressure Protocol Partitioning

