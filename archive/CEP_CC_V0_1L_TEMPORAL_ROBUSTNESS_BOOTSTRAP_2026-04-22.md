# CEP-CC V0.1l Temporal Robustness And Bootstrap - 2026-04-22

## Purpose

V0.1l tests whether the V0.1k temporal-memory seed failure can be repaired without introducing token/codebook machinery.

Question:

> Can temporal_memory communication become stable across seeds through continuous training bootstraps?

## Implementation

Files changed:

- `experiments/cep_cc/run_experiment.py`
- `experiments/cep_cc/metrics.py`
- `tests/test_cep_cc_protocol.py`

Added:

- `TrainConfig.segment_dropout_prob`
- training-time segment dropout for continuous communication
- `run_c1l_temporal_bootstrap`
- CLI flag: `--c1l-bootstrap`
- CLI flags:
  - `--bootstrap-modes`
  - `--curriculum-episodes`
  - `--segment-dropout-prob`

Bootstrap modes:

- `direct`: temporal_memory training from scratch.
- `curriculum`: compositional pretraining, then temporal_memory.
- `segment_dropout`: temporal_memory training with random communication segment dropout.

No discrete token, codebook, classifier, or external symbolic module was added.

## Validation

Protocol tests:

```powershell
python -m pytest F:\unified-sel\tests\test_cep_cc_protocol.py -q
```

Result:

- `20 passed`
- Pytest cache warning only.

CLI smoke:

```powershell
python -m experiments.cep_cc.run_experiment --c1l-bootstrap --seeds 0 --bootstrap-modes segment_dropout --stage1-episodes 2 --curriculum-episodes 2 --stage2-episodes 2 --batch-size 8 --lr 0.002 --stage2-lr 0.001 --lambda-comm 0.10 --factor-names q --factor-consistency-sweep 0.01 --segment-dropout-prob 0.25 --audit-batches 2 --table
```

Result:

- CLI exposes and runs `--c1l-bootstrap`.

## Seed-1 Rescue Probe

```powershell
python -m experiments.cep_cc.run_experiment --c1l-bootstrap --seeds 1 --bootstrap-modes direct,curriculum,segment_dropout --stage1-episodes 600 --curriculum-episodes 300 --stage2-episodes 120 --batch-size 128 --lr 0.005 --stage2-lr 0.001 --lambda-comm 0.10 --factor-names q --factor-consistency-sweep 0.03 --segment-dropout-prob 0.25 --audit-batches 8 --table
```

Seed 1 result:

| mode | stage1 acc | compressed acc | note |
|---|---:|---:|---|
| direct | 0.250 | 0.245 | repeats V0.1k failure |
| curriculum | 0.673 | 0.697 | strong rescue |
| segment_dropout | 0.660 | 0.678 | rescue, but weaker causal concentration |

## Official Bootstrap Comparison

```powershell
python -m experiments.cep_cc.run_experiment --c1l-bootstrap --seeds 0,1,2 --bootstrap-modes direct,curriculum,segment_dropout --stage1-episodes 600 --curriculum-episodes 300 --stage2-episodes 120 --batch-size 128 --lr 0.005 --stage2-lr 0.001 --lambda-comm 0.10 --factor-names q --factor-consistency-sweep 0.03 --segment-dropout-prob 0.25 --audit-batches 8 --table
```

Compressed accuracy:

| seed | direct | curriculum | segment_dropout |
|---|---:|---:|---:|
| 0 | 0.679 | 0.559 | 0.430 |
| 1 | 0.245 | 0.697 | 0.678 |
| 2 | 0.593 | 0.673 | 0.530 |

Interpretation:

- direct is unstable because seed 1 fails.
- segment_dropout rescues seed 1 but hurts seeds 0 and 2.
- curriculum is the only mode that succeeds on all three seeds.

## Official Curriculum Factor Sweep

```powershell
python -m experiments.cep_cc.run_experiment --c1l-bootstrap --seeds 0,1,2 --bootstrap-modes curriculum --stage1-episodes 600 --curriculum-episodes 300 --stage2-episodes 120 --batch-size 128 --lr 0.005 --stage2-lr 0.001 --lambda-comm 0.10 --factor-names q,motion,relation --factor-consistency-sweep 0.03 --segment-dropout-prob 0.25 --audit-batches 8 --table
```

Best curriculum rows:

| seed | high-bandwidth acc | best compressed acc | energy delta | strongest late swap | strongest late ablation logit drop |
|---|---:|---:|---:|---:|---:|
| 0 | 0.556 | 0.559 | -0.038 | 0.503 | 0.459 |
| 1 | 0.673 | 0.698 | -0.023 | 0.531 | 0.730 |
| 2 | 0.660 | 0.674 | -0.046 | 0.572 | 0.777 |

Factor consistency behavior:

- `q`, `motion`, and `relation` rows remain very similar.
- The robustness gain comes from curriculum, not factor-specific consistency.
- Segment q-probe specialization gaps remain small:
  - seed 0: about `0.069-0.077`
  - seed 1: about `0.020-0.022`
  - seed 2: about `0.016-0.021`

## Interpretation

Supported:

- Temporal_memory can be stabilized across seeds without token/codebook machinery.
- Compositional-to-temporal curriculum is the best bootstrap.
- Compression with frozen Listener preserves or slightly improves task success under curriculum.
- Late communication segments have consistent causal influence across all seeds under curriculum.

Not supported:

- There is still no clean factor-specific local grammar.
- Factor consistency does not separate q/motion/relation roles.
- Segment causal effects are strong but mostly concentrated late, so this is closer to an external memory/control segment than full compositional syntax.

## Decision

C1l upgrades C1k from borderline-positive to robust-positive for temporal memory pressure.

Supported claim:

> Continuous communication can become a robust task-necessary external memory/control protocol under temporal pressure and curriculum bootstrapping.

Still not supported:

> A fully language-like local compositional protocol has emerged.

Next branch can move to C2 semantic stability, with the curriculum temporal_memory setup as the primary system.

Recommended next:

- C2a: Semantic Stability Under Nuisance Transformations.

Acceptance for C2a:

- preserve task success under nuisance transforms.
- communication cluster/probe semantics remain aligned with hidden q and temporal factors.
- late segment causal effect remains stable under transforms.
