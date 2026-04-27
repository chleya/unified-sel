# CEP-CC V0.1h Consistency-Pressure Protocol Partitioning - 2026-04-21

## Purpose

V0.1h tests whether explicit communication consistency pressure over semantically similar samples can turn the useful continuous code into reusable trajectory partitions.

Question:

> Does target-conditioned consistency pressure improve target purity and within/between trajectory separation without destroying task performance or hidden-factor encoding?

## Implementation

Files changed:

- `experiments/cep_cc/losses.py`
- `experiments/cep_cc/run_experiment.py`
- `experiments/cep_cc/metrics.py`
- `tests/test_cep_cc_protocol.py`

Added:

- `communication_consistency`
- `lambda_consistency`
- `run_c1h_consistency_partitioning`
- CLI flag: `--c1h-consistency`
- CLI flag: `--consistency-sweep`

Consistency definition:

- within each mini-batch, group samples by final target index.
- pull communication trajectories toward their target-group centroid.
- this uses task supervision, not a communication token vocabulary.

## Validation

Protocol tests:

```powershell
python -m pytest F:\unified-sel\tests\test_cep_cc_protocol.py -q
```

Result:

- `14 passed`
- Pytest cache warning only.

## Official Run

```powershell
python -m experiments.cep_cc.run_experiment --c1h-consistency --seeds 0,1,2 --stage1-episodes 300 --stage2-episodes 120 --batch-size 128 --lambda-comm 0.10 --consistency-sweep 0.01,0.03,0.10 --lr 0.005 --stage2-lr 0.001 --audit-batches 8 --table
```

## Compact Results

Best consistency rows by seed:

| seed | consistency | accuracy | delta acc | energy delta | l1 delta | dim delta | target purity | within/between | hidden q r2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.10 | 0.719 | 0.048 | -0.127 | -0.106 | -0.283 | 0.548 | 0.941 | 0.641 |
| 1 | 0.10 | 0.898 | 0.033 | -0.031 | -0.023 | -0.302 | 0.631 | 0.948 | 0.644 |
| 2 | 0.03 | 0.901 | 0.045 | -0.062 | -0.046 | -0.133 | 0.628 | 0.947 | 0.616 |

Baseline stage-1 target purity:

| seed | stage1 target purity | best consistency target purity |
|---|---:|---:|
| 0 | 0.556 | 0.565 at consistency `0.01` |
| 1 | 0.598 | 0.638 at consistency `0.01` |
| 2 | 0.588 | 0.628 at consistency `0.03` |

Within/between target trajectory ratio:

| seed | stage1 | best observed |
|---|---:|---:|
| 0 | 0.949 | 0.941 |
| 1 | 0.952 | 0.948 |
| 2 | 0.950 | 0.946 |

## Interpretation

Consistency pressure helps, but does not produce a strong partition.

What improves:

- task accuracy remains above stage 1 on all seeds.
- energy and L1 decrease on all consistency settings.
- effective dimension decreases on all consistency settings.
- hidden `q` probe remains strong.
- target purity improves clearly on seeds 1 and 2, and slightly on seed 0 at low consistency.
- within/between target trajectory ratio improves slightly on all seeds.

What remains weak:

- within/between ratio remains near `0.94-0.95`, far from strong target separation.
- high consistency can trade off target purity on seed 0.
- this is target-conditioned consistency, so it is a supervised partition pressure, not spontaneous proto-symbol emergence.

## Decision

C1h partially succeeds.

Supported:

> Consistency pressure improves alignment and preserves compressed hidden-factor communication.

Not supported:

> Clear proto-symbol partitions have emerged.

Do not move to C2 semantic stability yet.

Next work should test whether richer semantic pressure creates reusable local partitions:

- relation/attribute factor consistency instead of only target-index consistency.
- task variants requiring composition of hidden factors.
- segment-level communication analysis.

Recommended next branch:

- C1i: Factor-Conditioned Consistency And Segment Audit

