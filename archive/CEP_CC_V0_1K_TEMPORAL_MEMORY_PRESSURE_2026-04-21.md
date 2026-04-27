# CEP-CC V0.1k Temporal Memory Pressure Variant - 2026-04-21

## Purpose

V0.1k tests whether a stronger memory-like target rule creates causal segment structure in continuous communication.

Question:

> If hidden evidence must be carried through the continuous communication trajectory, do segment swaps or ablations reveal reusable memory-bearing substructure?

## Implementation

Files changed:

- `experiments/cep_cc/env.py`
- `experiments/cep_cc/run_experiment.py`
- `experiments/cep_cc/metrics.py`
- `tests/test_cep_cc_protocol.py`

Added:

- `EnvConfig.rule_mode="temporal_memory"`
- `run_c1k_temporal_memory_variant`
- CLI flag: `--c1k-temporal-memory`
- `--rule-mode temporal_memory`
- segment ablation audit:
  - `segment_early_ablation_action_change_rate`
  - `segment_middle_ablation_action_change_rate`
  - `segment_late_ablation_action_change_rate`
  - `segment_early_ablation_target_logit_drop`
  - `segment_middle_ablation_target_logit_drop`
  - `segment_late_ablation_target_logit_drop`

The temporal-memory rule is still a minimal proxy, not a full two-phase episode API. It weights hidden q as early evidence and motion/relation as late plan factors, with interaction terms that make the hidden evidence matter for final target selection.

## Validation

Protocol tests:

```powershell
python -m pytest F:\unified-sel\tests\test_cep_cc_protocol.py -q
```

Result:

- `19 passed`
- Pytest cache warning only.

CLI smoke:

```powershell
python -m experiments.cep_cc.run_experiment --c1k-temporal-memory --seeds 0 --stage1-episodes 2 --stage2-episodes 2 --batch-size 8 --lr 0.002 --stage2-lr 0.001 --lambda-comm 0.10 --factor-names q --factor-consistency-sweep 0.01 --audit-batches 2 --table
```

Result:

- CLI exposes and runs `--c1k-temporal-memory`.

## Official Run

```powershell
python -m experiments.cep_cc.run_experiment --c1k-temporal-memory --seeds 0,1,2 --stage1-episodes 600 --stage2-episodes 120 --batch-size 128 --lr 0.005 --stage2-lr 0.001 --lambda-comm 0.10 --factor-names q,motion,relation --factor-consistency-sweep 0.03 --audit-batches 8 --table
```

Additional seed-1 budget probe:

```powershell
python -m experiments.cep_cc.run_experiment --episodes 1200 --batch-size 128 --seed 1 --lr 0.005 --baseline high-bandwidth --rule-mode temporal_memory --table
```

Result:

- seed 1 high-bandwidth only reached `0.357`, so the failure is not just a 600-episode budget issue.

## Compact Results

No-communication vs high-bandwidth:

| seed | no-comm acc | high-bandwidth acc | interpretation |
|---|---:|---:|---|
| 0 | 0.283 | 0.636 | strong communication gain |
| 1 | 0.215 | 0.250 | failed/highly weak gain |
| 2 | 0.262 | 0.567 | strong communication gain |

Best compressed rows:

| seed | best factor | acc | delta acc | delta energy | hidden q r2 |
|---|---|---:|---:|---:|---:|
| 0 | q | 0.679 | +0.043 | -0.068 | 0.685 |
| 1 | q/motion/relation | 0.245 | -0.005 | -0.189 | 0.372-0.384 |
| 2 | motion/relation | 0.594 | +0.026 | -0.039 | 0.564 |

Segment causal audit:

| seed | strongest swap action change | strongest ablation action change | strongest ablation target-logit drop |
|---|---:|---:|---:|
| 0 | late `0.542` | late `0.428` | late `0.987` |
| 1 | late `0.002` | all `0.000` | near `0.000` |
| 2 | late `0.460` | late `0.399` | late `0.357` |

Segment q-probe pattern:

| seed | pattern | specialization gap |
|---|---|---:|
| 0 | middle/late high | 0.049 |
| 1 | weak/no meaningful structure | 0.068-0.088 |
| 2 | middle high, late close | 0.062 |

## Interpretation

Supported:

- Temporal-memory pressure creates much stronger communication necessity on seeds 0 and 2 than C1j.
- On successful seeds, late communication segments have clear causal influence under swap and ablation.
- Compression with frozen Listener preserves or improves task success on successful seeds.
- The learned communication carries hidden q information strongly on successful seeds.

Not supported:

- The effect is not seed-stable.
- Seed 1 does not recover even with a 1200-episode high-bandwidth probe.
- Factor consistency over q/motion/relation still does not create clean factor-specific segment roles.
- This is not yet sufficient evidence for robust language-like local compositionality.

## Decision

C1k is a borderline-positive result.

It is the strongest evidence so far that temporal/memory pressure is the right lever: successful seeds show high communication necessity and segment-level causal effects. But the result is not stable enough to advance to C2 semantic stability.

Next branch:

- C1l: Temporal Robustness And Bootstrap.

Recommended focus:

- reduce seed sensitivity before adding new semantic claims.
- compare:
  - teacher-signal warm start or distillation.
  - longer/wider Listener memory.
  - staged curriculum from compositional to temporal_memory.
  - explicit segment dropout during training.
- acceptance:
  - high-bandwidth temporal_memory succeeds on all seeds.
  - compressed stage preserves success.
  - segment ablation remains causal on all successful seeds.
