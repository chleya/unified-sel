# CEP-CC V0.3e Motion Readout Audit - 2026-04-22

## Purpose

V0.3e tests whether motion exists in the learned continuous communication as a distributed readout rather than a locally swappable segment.

Question:

> If motion is task-critical but not segment-local, can it be decoded from the full communication trajectory or from any early/middle/late segment?

## Implementation

Files changed:

- `experiments/cep_cc/metrics.py`
- `experiments/cep_cc/run_experiment.py`
- `tests/test_cep_cc_protocol.py`

Added audit fields:

- full-trajectory factor probes:
  - `audit_pos_probe_r2`
  - `audit_q_probe_r2`
  - `audit_motion_probe_r2`
  - `audit_relation_probe_r2`
- segment factor probes:
  - `segment_{early,middle,late}_{pos,q_score,motion,relation}_probe_r2`
  - factor specialization gaps.
- runner: `run_c3e_motion_readout_audit`
- CLI flag: `--c3e-motion-readout`

No model change was introduced. C3e reuses `rule_mode="motion_decoupled"` from V0.3d.

## Validation

Protocol tests:

```powershell
python -m pytest F:\unified-sel\tests\test_cep_cc_protocol.py -q
```

Result:

- `31 passed`
- Pytest cache warning only.

CLI smoke:

```powershell
python -m experiments.cep_cc.run_experiment --c3e-motion-readout --seeds 0 --nuisance-modes none --stage1-episodes 2 --curriculum-episodes 2 --stage2-episodes 2 --batch-size 8 --lr 0.002 --stage2-lr 0.001 --lambda-comm 0.10 --factor-names motion --factor-consistency-sweep 0.01 --audit-batches 2 --table
```

Result:

- CLI exposes and runs `--c3e-motion-readout`.
- factor probe fields appear in the table.

## Official Probe

```powershell
python -m experiments.cep_cc.run_experiment --c3e-motion-readout --seeds 0,1,2 --nuisance-modes none --stage1-episodes 600 --curriculum-episodes 300 --stage2-episodes 120 --batch-size 128 --lr 0.005 --stage2-lr 0.001 --lambda-comm 0.10 --factor-names motion --factor-consistency-sweep 0.03 --audit-batches 12 --table
```

## Compact Results

Task accuracy:

| seed | accuracy | no-comm accuracy |
|---|---:|---:|
| 0 | `0.583` | `0.264` |
| 1 | `0.581` | `0.279` |
| 2 | `0.733` | `0.240` |

Full-trajectory factor probe R2:

| seed | q-score | motion | relation | hidden q |
|---|---:|---:|---:|---:|
| 0 | `0.092` | `0.064` | `-0.069` | `0.223` |
| 1 | `0.066` | `0.005` | `-0.049` | `0.323` |
| 2 | `0.035` | `0.008` | `-0.030` | `0.289` |

Segment motion probe R2:

| seed | early | middle | late |
|---|---:|---:|---:|
| 0 | `0.022` | `0.038` | `0.036` |
| 1 | `-0.015` | `-0.005` | `0.008` |
| 2 | `-0.015` | `-0.008` | `0.004` |

Targeted intervention ratios:

| seed | q | motion | relation |
|---|---:|---:|---:|
| 0 | `1.313` | `0.491` | `1.573` |
| 1 | `1.345` | `0.507` | `1.582` |
| 2 | `1.267` | `0.486` | `1.831` |

Late segment action effect:

| seed | late swap action change | late ablation action change |
|---|---:|---:|
| 0 | `0.521` | `0.520` |
| 1 | `0.490` | `0.441` |
| 2 | `0.523` | `0.334` |

## Interpretation

Supported:

- communication is necessary; no-communication remains near chance.
- the late communication segment strongly controls listener action.
- motion is not linearly readable from the full communication trajectory.
- motion is not linearly readable from any local segment.
- motion also fails targeted intervention as a locally swappable semantic factor.

Not supported:

- motion as a segment-local proto-symbol.
- motion as a simple distributed linear readout.

Interesting finding:

- The system solves a motion-critical task without exposing motion as a recoverable semantic variable.
- The communication likely carries an action/target decision manifold rather than a factorized description of q, motion, and relation.
- This is a sharper boundary on the current CEP-CC claim: task coordination emerged, but language-like compositional semantics is still partial.

## Decision

C3e is a negative but important diagnostic.

Supported claim:

> The current CEP-CC protocol uses continuous communication for task coordination, but motion is encoded, if at all, as an entangled decision trajectory rather than an interpretable semantic factor.

Next branch:

- C3f: Factorized-Pressure Objective.

Recommended focus:

- keep the no-token constraint.
- add continuous consistency pressure for multiple factors at once, not only one selected factor.
- compare q-only, motion-only, q+motion, and q+motion+relation consistency.
- acceptance requires motion probe/intervention improvement without collapse in task accuracy.
