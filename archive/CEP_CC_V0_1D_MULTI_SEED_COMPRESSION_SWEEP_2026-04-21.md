# CEP-CC V0.1d Multi-Seed Compression Sweep - 2026-04-21

## Purpose

V0.1d tests whether the seed-0 light-compression improvement from V0.1c reproduces across seeds.

Question:

> Does learned continuous communication beat no-communication across seeds, and does light compression reliably improve or preserve the learned protocol while changing communication geometry?

## Implementation

Files changed:

- `experiments/cep_cc/run_experiment.py`
- `tests/test_cep_cc_protocol.py`

Added:

- `run_multiseed_compression_sweep`
- CLI flag: `--multiseed-compression-sweep`
- CLI flag: `--seeds`

Official command:

```powershell
python -m experiments.cep_cc.run_experiment --multiseed-compression-sweep --seeds 0,1,2 --episodes 300 --batch-size 128 --lambda-sweep 0.0,0.001,0.003,0.006,0.01 --lr 0.005 --table
```

## Validation

Protocol tests:

```powershell
python -m pytest F:\unified-sel\tests\test_cep_cc_protocol.py -q
```

Result:

- `10 passed`
- Pytest cache warning only.

## Results

| run | eval accuracy | train accuracy | comm energy | comm l1 | comm effective dim | prototype reuse | compactness | target alignment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| seed 0 no communication | 0.455 | 0.398 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| seed 0 lambda 0 | 0.668 | 0.680 | 0.774 | 0.843 | 4.336 | 0.186 | 2.714 | 0.552 |
| seed 0 lambda 0.001 | 0.846 | 0.836 | 0.695 | 0.786 | 4.091 | 0.162 | 2.257 | 0.626 |
| seed 0 lambda 0.003 | 0.807 | 0.758 | 0.588 | 0.707 | 3.914 | 0.201 | 2.171 | 0.520 |
| seed 0 lambda 0.006 | 0.668 | 0.664 | 0.385 | 0.519 | 3.085 | 0.240 | 2.763 | 0.482 |
| seed 0 lambda 0.01 | 0.467 | 0.398 | 0.001 | 0.011 | 1.084 | 0.199 | 5.363 | 0.334 |
| seed 1 no communication | 0.488 | 0.453 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| seed 1 lambda 0 | 0.859 | 0.836 | 0.727 | 0.810 | 4.225 | 0.184 | 2.076 | 0.614 |
| seed 1 lambda 0.001 | 0.844 | 0.836 | 0.638 | 0.747 | 3.959 | 0.180 | 2.276 | 0.569 |
| seed 1 lambda 0.003 | 0.537 | 0.508 | 0.322 | 0.445 | 1.405 | 0.150 | 6.731 | 0.434 |
| seed 1 lambda 0.006 | 0.484 | 0.461 | 0.001 | 0.017 | 1.178 | 0.209 | 4.011 | 0.323 |
| seed 1 lambda 0.01 | 0.484 | 0.445 | 0.000 | 0.007 | 1.065 | 0.162 | 6.265 | 0.310 |
| seed 2 no communication | 0.457 | 0.484 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| seed 2 lambda 0 | 0.828 | 0.875 | 0.702 | 0.793 | 4.623 | 0.150 | 2.259 | 0.647 |
| seed 2 lambda 0.001 | 0.670 | 0.656 | 0.727 | 0.812 | 3.385 | 0.156 | 3.298 | 0.507 |
| seed 2 lambda 0.003 | 0.463 | 0.477 | 0.010 | 0.053 | 1.039 | 0.182 | 7.267 | 0.318 |
| seed 2 lambda 0.006 | 0.537 | 0.484 | 0.249 | 0.383 | 1.550 | 0.162 | 6.094 | 0.404 |
| seed 2 lambda 0.01 | 0.500 | 0.508 | 0.191 | 0.325 | 1.223 | 0.248 | 7.538 | 0.437 |

## Seed-Level Decisions

Seed 0:

- learned communication beats no-communication.
- light compression improves accuracy.
- best observed setting: lambda `0.001`.

Seed 1:

- learned communication beats no-communication.
- high-bandwidth is best.
- lambda `0.001` nearly preserves high-bandwidth accuracy while lowering energy, L1, and effective dimension.
- lambda `0.003+` collapses toward no-communication.

Seed 2:

- learned communication beats no-communication.
- high-bandwidth is best.
- lambda `0.001` still beats no-communication but loses substantial accuracy.
- lambda `0.003` collapses near no-communication.
- lambda `0.006/0.01` partially recover accuracy but with lower alignment and low effective dimension.

## Interpretation

The communication-necessity result is stable:

- no-communication accuracy is around `0.455-0.488`.
- learned high-bandwidth accuracy is `0.668`, `0.859`, `0.828`.

The light-compression result is not stable enough for a phase-transition claim:

- seed 0 improves under lambda `0.001`.
- seed 1 nearly preserves performance under lambda `0.001`.
- seed 2 degrades under lambda `0.001`.
- stronger compression often collapses communication geometry and task accuracy.

The evidence supports:

> Continuous learned communication can emerge from task loss and beat no-communication.

The evidence does not yet support:

> Compression reliably induces proto-symbol partitions across seeds.

## Decision

Do not move to C2 semantic stability yet.

Next work should remain in C1:

- add a more robust geometry metric for trajectory partitions.
- reduce seed sensitivity before claiming compression-induced proto-symbol emergence.
- separate communication discovery from compression by using a two-stage protocol:
  - stage 1: train high-bandwidth communication.
  - stage 2: fine-tune with compression.

Recommended next branch:

- C1e: Two-Stage Compression Fine-Tuning

