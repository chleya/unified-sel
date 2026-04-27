# CEP-CC V0.1i Factor-Conditioned Consistency And Segment Audit - 2026-04-21

## Purpose

V0.1i tests whether consistency over task-relevant latent factors and segment-level auditing reveal reusable sub-trajectory structure.

Question:

> Do different communication segments align with different latent factors, suggesting early compositional structure?

## Implementation

Files changed:

- `experiments/cep_cc/losses.py`
- `experiments/cep_cc/metrics.py`
- `experiments/cep_cc/run_experiment.py`
- `tests/test_cep_cc_protocol.py`

Added:

- `factor_bin_consistency`
- `lambda_factor_consistency`
- factor names:
  - `q`
  - `motion`
  - `relation`
  - `pos`
  - `hidden_q_score`
- `segment_audit_metrics`
- `run_c1i_factor_segment_audit`
- CLI flag: `--c1i-factor-segment`
- CLI flags:
  - `--factor-names`
  - `--factor-consistency-sweep`

## Validation

Protocol tests:

```powershell
python -m pytest F:\unified-sel\tests\test_cep_cc_protocol.py -q
```

Result:

- `15 passed`
- Pytest cache warning only.

## Official Run

```powershell
python -m experiments.cep_cc.run_experiment --c1i-factor-segment --seeds 0,1,2 --stage1-episodes 300 --stage2-episodes 120 --batch-size 128 --lambda-comm 0.10 --factor-names q,motion,relation --factor-consistency-sweep 0.03 --lr 0.005 --stage2-lr 0.001 --audit-batches 8 --table
```

## Compact Results

Best factor-conditioned rows by seed:

| seed | factor | accuracy | energy delta | l1 delta | dim delta | target purity | within/between | hidden q r2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | q | 0.717 | -0.120 | -0.098 | -0.382 | 0.564 | 0.946 | 0.648 |
| 1 | motion/relation | 0.896 | -0.025 | -0.018 | -0.335 | 0.638 | 0.949 | 0.639 |
| 2 | motion/relation | 0.902 | -0.063 | -0.046 | -0.132 | 0.631 | 0.947 | 0.617 |

Segment q-probe behavior:

| seed | stage1 q segment range | factor-conditioned q segment range | interpretation |
|---|---:|---:|---|
| 0 | `0.575-0.643` | `0.611-0.641` | no stable specialization |
| 1 | `0.554-0.574` | `0.577-0.594` | all segments improve together |
| 2 | `0.510-0.557` | `0.589-0.598` | all segments improve together |

Segment target-purity behavior:

- segment-level target purity is noisy.
- no consistent early/middle/late division appears.
- factor-conditioned `motion` and `relation` produce almost identical results in this environment.

## Interpretation

Factor-conditioned consistency is useful, but not compositional yet.

What improves:

- accuracy remains high.
- energy/L1/effective dimension decrease from stage 1.
- hidden q encoding remains strong.
- target purity reaches `0.63-0.64` on seeds 1 and 2.

What does not appear:

- no stable segment specialization.
- no clear factor-specific segment assignment.
- no strong trajectory partition; within/between remains about `0.946-0.949`.
- motion and relation consistency are effectively indistinguishable under the current task.

The current environment likely allows a single continuous hidden-factor code to solve the task, so the model has no strong reason to factor communication into reusable sub-parts.

## Decision

C1i is a useful negative/boundary result.

Supported:

> Factor-conditioned consistency preserves compressed communication and improves continuous factor encoding.

Not supported:

> Local compositional protocol structure has emerged.

Do not proceed to C2 semantic stability yet.

Next work should change the environment/task pressure, not continue tuning the same consistency objective.

Recommended next branch:

- C1j: Compositional Task Variant

Requirements:

- task must require separable latent factors.
- communication should be analyzable by segment.
- target rule should include explicit independent subdecisions, for example:
  - object identity factor.
  - relation factor.
  - motion factor.
- evaluate segment swap or segment probe after training.

