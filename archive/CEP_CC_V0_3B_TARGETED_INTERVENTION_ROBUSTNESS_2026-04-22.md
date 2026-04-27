# CEP-CC V0.3b Targeted Intervention Robustness And Segment Assignment - 2026-04-22

## Purpose

V0.3b tests whether the factor-targeted intervention effects from V0.3a are robust under nuisance transformations and larger audit batches.

Question:

> Are factor-targeted intervention effects stable under nuisance transforms, and do they use consistent communication segments?

## Implementation

Files changed:

- `experiments/cep_cc/run_experiment.py`
- `tests/test_cep_cc_protocol.py`

Added:

- `run_c3b_targeted_intervention_robustness`
- CLI flag: `--c3b-targeted-robustness`

Method:

- train the robust curriculum temporal_memory protocol.
- evaluate targeted segment interventions under:
  - `none`
  - `mirror_x`
  - `rotate90`
  - `velocity_scale`
- use larger audit batches than V0.3a.

## Validation

Protocol tests:

```powershell
python -m pytest F:\unified-sel\tests\test_cep_cc_protocol.py -q
```

Result:

- `26 passed`
- Pytest cache warning only.

CLI smoke:

```powershell
python -m experiments.cep_cc.run_experiment --c3b-targeted-robustness --seeds 0 --nuisance-modes none,mirror_x --stage1-episodes 2 --curriculum-episodes 2 --stage2-episodes 2 --batch-size 8 --lr 0.002 --stage2-lr 0.001 --lambda-comm 0.10 --factor-names q --factor-consistency-sweep 0.01 --audit-batches 2 --table
```

Result:

- CLI exposes and runs `--c3b-targeted-robustness`.

## Official Run

```powershell
python -m experiments.cep_cc.run_experiment --c3b-targeted-robustness --seeds 0,1,2 --nuisance-modes none,mirror_x,rotate90,velocity_scale --stage1-episodes 600 --curriculum-episodes 300 --stage2-episodes 120 --batch-size 128 --lr 0.005 --stage2-lr 0.001 --lambda-comm 0.10 --factor-names q --factor-consistency-sweep 0.03 --audit-batches 12 --table
```

## Compact Results

Relation targeted ratio:

| seed | none | mirror_x | rotate90 | velocity_scale |
|---|---:|---:|---:|---:|
| 0 | 2.972 | 2.771 | 2.928 | 2.831 |
| 1 | 4.126 | 3.738 | 4.122 | 3.748 |
| 2 | 4.561 | 4.401 | 4.584 | 4.000 |

Q targeted ratio:

| seed | none | mirror_x | rotate90 | velocity_scale |
|---|---:|---:|---:|---:|
| 0 | 1.882 | 1.879 | 1.843 | 1.802 |
| 1 | 1.784 | 1.729 | 1.814 | 1.651 |
| 2 | 1.777 | 1.781 | 1.890 | 1.739 |

Motion targeted ratio:

| seed | none | mirror_x | rotate90 | velocity_scale |
|---|---:|---:|---:|---:|
| 0 | 0.046 | 0.049 | 0.047 | 0.105 |
| 1 | 0.055 | 0.054 | 0.058 | 0.125 |
| 2 | 0.064 | 0.063 | 0.065 | 0.144 |

Best segment assignment:

| factor | assignment |
|---|---|
| q | late segment across all seeds and nuisance modes |
| relation | early segment across all seeds and nuisance modes |
| motion | no meaningful targeted control |

Task accuracy remains stable under nuisance transforms:

- seed 0: `0.544-0.572`
- seed 1: `0.681-0.684`
- seed 2: `0.664-0.678`

## Interpretation

Supported:

- Relation-targeted intervention is robust under nuisance transforms.
- Q-targeted intervention is weaker but stable.
- Segment assignment is stable:
  - q maps to late segment.
  - relation maps to early segment.
- This is the first clear evidence of reusable local segment roles.

Not supported:

- Motion-specific local control.
- Full three-factor compositional syntax.

## Decision

C3b is positive but scoped.

Supported claim:

> The learned continuous protocol has stable local segment roles for at least two semantic factors: relation and q.

Qualified boundary:

> Motion remains unresolved, so this is partial local compositionality rather than a complete compositional grammar.

Next branch:

- C3c: Motion Pressure Variant.

Recommended focus:

- increase the task's dependence on motion.
- audit whether motion can obtain a targeted segment role.
- preserve the q/relation roles while adding motion pressure.
