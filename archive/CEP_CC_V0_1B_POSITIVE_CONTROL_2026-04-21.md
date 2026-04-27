# CEP-CC V0.1b Positive Control - 2026-04-21

## Purpose

V0.1b calibrates the first CEP-CC scaffold.

The question:

> Can the Listener and environment use continuous communication when the communication signal contains task-relevant continuous hidden factors?

This is a positive control, not evidence of emergent language-like protocol.

## Changes

Files changed:

- `experiments/cep_cc/env.py`
- `experiments/cep_cc/models.py`
- `experiments/cep_cc/run_experiment.py`
- `experiments/cep_cc/README.md`
- `experiments/cep_cc/PROJECT_SPEC.md`
- `experiments/cep_cc/IMPLEMENTATION_PLAN.md`
- `tests/test_cep_cc_protocol.py`

Implemented:

- object-wise Listener scoring head.
- per-object communication decoder inside Listener.
- `teacher-signal` baseline.
- stronger hidden-factor dependence in the target rule.

## Teacher-Signal Baseline

The `teacher-signal` baseline is continuous.

It does not use:

- tokens
- codebooks
- categorical communication actions
- language data
- next-token prediction

It sends a continuous trajectory derived from the same hidden continuous scoring factors that define the task.

Purpose:

- verify that the Listener can bind continuous communication to object-wise decisions.
- verify that the task has genuine hidden information that communication can carry.

It is not a learned emergent protocol.

## Results

Protocol tests:

```powershell
python -m pytest F:\unified-sel\tests\test_cep_cc_protocol.py -q
```

Result:

- `7 passed`
- Pytest cache warning only.

Positive-control comparison:

```powershell
python -m experiments.cep_cc.run_experiment --episodes 40 --batch-size 64 --baseline teacher-signal --seed 0 --table
python -m experiments.cep_cc.run_experiment --episodes 40 --batch-size 64 --baseline no-communication --seed 0 --table
```

| run | task accuracy | comm energy | comm l1 | comm effective dim | prototype reuse | compactness | target alignment |
|---|---:|---:|---:|---:|---:|---:|---:|
| teacher-signal | 0.863 | 0.350 | 0.534 | 4.498 | 0.176 | 2.003 | 0.657 |
| no-communication | 0.398 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |

End-to-end learned Speaker sanity:

```powershell
python -m experiments.cep_cc.run_experiment --episodes 80 --batch-size 64 --baseline high-bandwidth --seed 0 --table
python -m experiments.cep_cc.run_experiment --episodes 80 --batch-size 64 --baseline no-communication --seed 0 --table
```

| run | task accuracy | comm energy | comm l1 | comm effective dim | prototype reuse | compactness | target alignment |
|---|---:|---:|---:|---:|---:|---:|---:|
| high-bandwidth | 0.398 | 0.225 | 0.383 | 1.074 | 0.141 | 7.328 | 0.330 |
| no-communication | 0.422 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |

## Interpretation

The positive control passes.

Continuous communication can carry task-relevant hidden factors, and the Listener can use it. This clears the first calibration gate.

The learned Speaker does not yet pass.

High-bandwidth learned communication did not beat no-communication in the 80-episode sanity run. Therefore, CEP-CC still has no positive emergent-protocol result.

## Decision

Do not run the full compression phase-transition sweep yet.

Next work is C1c:

- improve end-to-end learned communication under task loss.
- keep teacher-signal as the positive control.
- compare no-communication, teacher-signal, high-bandwidth learned, and compressed learned under the same run profile.

Candidate C1c work:

- add a staged task curriculum.
- add an auxiliary continuous score-distillation control, clearly labeled as non-emergent.
- try longer official train budgets after the training signal is stable.
- log train accuracy so optimization failure and generalization failure can be separated.

