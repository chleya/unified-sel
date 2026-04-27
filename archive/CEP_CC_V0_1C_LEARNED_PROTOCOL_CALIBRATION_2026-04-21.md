# CEP-CC V0.1c Learned Protocol Calibration - 2026-04-21

## Purpose

V0.1c tests whether a learned continuous Speaker can discover useful communication from task loss, after V0.1b established that the Listener and environment can use continuous communication.

This version still does not claim proto-symbol emergence.

It establishes the precondition for the real compression sweep:

> learned continuous communication must beat no-communication before compression geometry can be interpreted.

## Implementation

Files changed:

- `experiments/cep_cc/run_experiment.py`
- `experiments/cep_cc/losses.py`
- `experiments/cep_cc/metrics.py`
- `experiments/cep_cc/models.py`
- `tests/test_cep_cc_protocol.py`

Added:

- `train_task_accuracy`
- `--lr`
- `--c1c-calibration`
- `lambda_comm_distill`
- `learned_distilled` diagnostic condition

The score-distillation path is diagnostic only. It is not evidence of emergent protocol.

## Fixed-Batch Capacity Diagnostic

Speaker-only distillation on a fixed batch can fit the continuous teacher signal:

| step | teacher MSE |
|---:|---:|
| 0 | 0.358 |
| 10 | 0.228 |
| 50 | 0.015 |
| 100 | 0.001 |

Interpretation:

- Speaker capacity is sufficient.
- The earlier failure was not a broken gradient path.
- Cross-batch protocol discovery requires enough optimization budget.

## Official C1c Profile

Settings:

- episodes: `300`
- batch size: `128`
- learning rate: `0.005`
- seed: `0`

Commands:

```powershell
python -m experiments.cep_cc.run_experiment --episodes 300 --batch-size 128 --baseline no-communication --lr 0.005 --seed 0 --table
python -m experiments.cep_cc.run_experiment --episodes 300 --batch-size 128 --baseline teacher-signal --lr 0.005 --seed 0 --table
python -m experiments.cep_cc.run_experiment --episodes 300 --batch-size 128 --baseline high-bandwidth --lr 0.005 --seed 0 --table
python -m experiments.cep_cc.run_experiment --episodes 300 --batch-size 128 --lambda-comm 0.001 --lr 0.005 --seed 0 --table
python -m experiments.cep_cc.run_experiment --episodes 300 --batch-size 128 --lambda-comm 0.003 --lr 0.005 --seed 0 --table
python -m experiments.cep_cc.run_experiment --episodes 300 --batch-size 128 --lambda-comm 0.01 --lr 0.005 --seed 0 --table
```

## Results

| run | eval accuracy | train accuracy | comm energy | comm l1 | comm effective dim | prototype reuse | compactness | target alignment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no-communication | 0.455 | 0.398 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| teacher-signal | 0.973 | 0.984 | 0.348 | 0.532 | 4.515 | 0.170 | 1.975 | 0.596 |
| learned high-bandwidth | 0.668 | 0.680 | 0.774 | 0.843 | 4.336 | 0.186 | 2.714 | 0.552 |
| learned lambda 0.001 | 0.846 | 0.836 | 0.695 | 0.786 | 4.091 | 0.162 | 2.257 | 0.626 |
| learned lambda 0.003 | 0.807 | 0.758 | 0.588 | 0.707 | 3.914 | 0.201 | 2.171 | 0.520 |
| learned lambda 0.01 | 0.467 | 0.398 | 0.001 | 0.011 | 1.084 | 0.199 | 5.363 | 0.334 |

Protocol tests:

```powershell
python -m pytest F:\unified-sel\tests\test_cep_cc_protocol.py -q
```

Result:

- `9 passed`
- Pytest cache warning only.

## Interpretation

The communication-necessity gate now passes.

Learned continuous communication beats no-communication:

- no-communication: `0.455`
- learned high-bandwidth: `0.668`
- learned lambda `0.001`: `0.846`
- learned lambda `0.003`: `0.807`

The compression response is already suggestive:

- light compression improves both task accuracy and target/cluster alignment.
- moderate compression preserves strong task gain.
- strong compression collapses communication energy and returns accuracy near no-communication.

This is not yet a phase-transition claim because:

- only one seed was run.
- only a small number of lambda values were sampled.
- cluster metrics are fallback proxies.
- no semantic-stability or compositionality intervention has been run.

## Decision

C1c succeeds.

The next step can be the real V0.1 proto-symbol sweep:

- multiple seeds.
- lambda grid around `0.0`, `0.001`, `0.003`, `0.006`, `0.01`.
- same official profile: `episodes=300`, `batch_size=128`, `lr=0.005`.
- compare task accuracy, effective dimension, prototype reuse, compactness, and target alignment.

Do not proceed to C2 semantic stability until the multi-seed compression sweep is recorded.

