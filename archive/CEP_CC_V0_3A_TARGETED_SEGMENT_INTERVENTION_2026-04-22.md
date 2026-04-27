# CEP-CC V0.3a Targeted Segment Intervention - 2026-04-22

## Purpose

V0.3a tests whether communication segments have factor-targeted effects, rather than only generic causal importance.

Question:

> Does replacing a communication segment primarily change one latent factor more than off-target factors?

## Implementation

Files changed:

- `experiments/cep_cc/metrics.py`
- `experiments/cep_cc/run_experiment.py`
- `tests/test_cep_cc_protocol.py`

Added:

- `_targeted_segment_intervention_metrics`
- `run_c3a_targeted_segment_intervention`
- CLI flag: `--c3a-targeted-intervention`

Metrics:

- `intervention_best_targeted_effect`
- `intervention_best_offtarget_effect`
- `intervention_best_targeted_ratio`
- `intervention_q_best_ratio`
- `intervention_motion_best_ratio`
- `intervention_relation_best_ratio`
- `intervention_q_best_segment`
- `intervention_motion_best_segment`
- `intervention_relation_best_segment`

Method:

- train the robust curriculum temporal_memory protocol.
- collect clean audit batches.
- for each factor `q`, `motion`, `relation`:
  - sort examples by the selected target object's factor value.
  - use reversed-order examples as donors.
  - replace one communication segment at a time.
  - recompute Listener outputs.
  - compare selected-object latent factor changes.
- compute targeted/off-target ratio for each factor and segment.

## Validation

Protocol tests:

```powershell
python -m pytest F:\unified-sel\tests\test_cep_cc_protocol.py -q
```

Result:

- `25 passed`
- Pytest cache warning only.

CLI smoke:

```powershell
python -m experiments.cep_cc.run_experiment --c3a-targeted-intervention --seeds 0 --stage1-episodes 2 --curriculum-episodes 2 --stage2-episodes 2 --batch-size 8 --lr 0.002 --stage2-lr 0.001 --lambda-comm 0.10 --factor-names q --factor-consistency-sweep 0.01 --audit-batches 2 --table
```

Result:

- CLI exposes and runs `--c3a-targeted-intervention`.

## Official Run

```powershell
python -m experiments.cep_cc.run_experiment --c3a-targeted-intervention --seeds 0,1,2 --stage1-episodes 600 --curriculum-episodes 300 --stage2-episodes 120 --batch-size 128 --lr 0.005 --stage2-lr 0.001 --lambda-comm 0.10 --factor-names q --factor-consistency-sweep 0.03 --audit-batches 8 --table
```

## Compact Results

Task accuracy:

| seed | accuracy | hidden q probe R2 |
|---|---:|---:|
| 0 | 0.559 | 0.642 |
| 1 | 0.697 | 0.648 |
| 2 | 0.673 | 0.629 |

Best targeted intervention:

| seed | best targeted effect | best off-target effect | best ratio |
|---|---:|---:|---:|
| 0 | 0.143 | 0.049 | 2.942 |
| 1 | 0.124 | 0.032 | 3.852 |
| 2 | 0.106 | 0.023 | 4.605 |

Per-factor best ratios:

| seed | q ratio | motion ratio | relation ratio |
|---|---:|---:|---:|
| 0 | 1.805 | 0.050 | 2.942 |
| 1 | 1.686 | 0.060 | 3.852 |
| 2 | 1.915 | 0.062 | 4.605 |

Best segment index:

| seed | q | motion | relation |
|---|---:|---:|---:|
| 0 | 2 late | 1 middle | 1 middle |
| 1 | 2 late | 0 early | 0 early |
| 2 | 2 late | 0 early | 0 early |

## Interpretation

Supported:

- targeted intervention effects exist.
- relation has a strong targeted/off-target ratio across all seeds.
- q has a weaker but consistent targeted effect.
- motion has no meaningful targeted effect in this setup.

Not supported:

- full local compositionality.
- stable one-factor-per-segment assignment.
- motion-specific segment control.

Key boundary:

- C3a shows factor-targeted causal effects, especially for relation.
- It does not yet show a clean reusable local grammar.

## Decision

C3a is partial-positive.

Supported claim:

> Some communication segments have factor-targeted causal effects, with relation showing the strongest and most stable targeted control.

Not supported:

> The protocol has developed full language-like local compositional syntax.

Next branch:

- C3b: Targeted Intervention Robustness And Segment Assignment.

Recommended focus:

- repeat targeted interventions under nuisance transforms.
- check whether relation targeting remains stable.
- test segment assignment consistency under larger audit batches.
- explore whether motion requires a stronger task pressure or explicit motion-target intervention design.
