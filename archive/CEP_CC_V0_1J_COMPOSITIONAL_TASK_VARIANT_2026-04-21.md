# CEP-CC V0.1j Compositional Task Variant - 2026-04-21

## Purpose

V0.1j tests whether an explicitly compositional target rule creates stronger pressure for reusable segment-level communication structure.

Question:

> If target selection depends on multiple hidden latent factors and factor interactions, does continuous communication develop local compositional segments?

## Implementation

Files changed:

- `experiments/cep_cc/env.py`
- `experiments/cep_cc/run_experiment.py`
- `experiments/cep_cc/metrics.py`
- `tests/test_cep_cc_protocol.py`

Added:

- `EnvConfig.rule_mode`
- `rule_mode="compositional"`
- `run_c1j_compositional_variant`
- CLI flag: `--c1j-compositional`
- CLI flag: `--rule-mode scalar|compositional`
- segment swap audit metrics:
  - `segment_early_swap_action_change_rate`
  - `segment_middle_swap_action_change_rate`
  - `segment_late_swap_action_change_rate`
  - `segment_early_swap_target_logit_drop`
  - `segment_middle_swap_target_logit_drop`
  - `segment_late_swap_target_logit_drop`

The compositional rule uses hidden q, motion alignment, and relation-to-anchor factors with interaction terms. It keeps all communication continuous and does not add token/codebook machinery.

## Validation

Protocol tests:

```powershell
python -m pytest F:\unified-sel\tests\test_cep_cc_protocol.py -q
```

Result:

- `17 passed`
- Pytest cache warning only.

CLI smoke:

```powershell
python -m experiments.cep_cc.run_experiment --c1j-compositional --seeds 0 --stage1-episodes 2 --stage2-episodes 2 --batch-size 8 --lr 0.002 --stage2-lr 0.001 --lambda-comm 0.10 --factor-names q --factor-consistency-sweep 0.01 --audit-batches 2 --table
```

Result:

- CLI exposes and runs `--c1j-compositional`.

## Official Run

```powershell
python -m experiments.cep_cc.run_experiment --c1j-compositional --seeds 0,1,2 --stage1-episodes 600 --stage2-episodes 120 --batch-size 128 --lr 0.005 --stage2-lr 0.001 --lambda-comm 0.10 --factor-names q,motion,relation --factor-consistency-sweep 0.03 --audit-batches 8 --table
```

## Compact Results

No-communication vs high-bandwidth:

| seed | no-comm acc | high-bandwidth acc | high-bandwidth energy | high-bandwidth dim |
|---|---:|---:|---:|---:|
| 0 | 0.248 | 0.422 | 0.833 | 4.276 |
| 1 | 0.275 | 0.444 | 0.893 | 3.248 |
| 2 | 0.289 | 0.440 | 0.804 | 3.228 |

Best compressed rows:

| seed | factor | acc | delta acc | delta energy | delta L1 | target purity | hidden q r2 |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | motion/relation | 0.444 | +0.022 | -0.109 | -0.085 | 0.430 | 0.248 |
| 1 | q/motion/relation | 0.459 | +0.015 | -0.062 to -0.066 | -0.048 to -0.051 | 0.440-0.443 | 0.287-0.293 |
| 2 | motion | 0.456 | +0.016 | -0.173 | -0.149 | 0.426 | 0.350 |

Segment audit:

| seed | q segment r2 pattern | q specialization gap | strongest swap segment |
|---|---|---:|---|
| 0 | middle highest | 0.078-0.079 | late |
| 1 | middle/late close | 0.018-0.021 | late |
| 2 | middle highest | 0.033-0.036 | middle/late |

## Interpretation

Supported:

- learned continuous communication beats no-communication on the compositional task variant.
- stage2 compression preserves or slightly improves success across all seeds.
- compression reduces communication energy and L1 while keeping the Listener frozen.
- segment swap has measurable effects, especially late-segment swaps.

Not supported:

- factor-specific consistency does not separate q, motion, and relation.
- q/motion/relation rows remain nearly identical.
- segment probe differences are small.
- segment swap effects are not targeted enough to claim local compositionality.

## Decision

C1j is a useful partial-positive boundary result.

It establishes a learnable compositional task variant and shows that continuous communication remains necessary under that variant. It does not yet show language-like local compositional structure.

Do not move to C2 semantic stability yet as if proto-symbol/compositionality were established.

Next branch should add temporal or memory pressure:

- delayed evidence.
- weak Listener memory.
- multi-step communication.
- forced reuse of earlier communication for later decisions.

Recommended next branch:

- C1k: Temporal Memory Pressure Variant.
