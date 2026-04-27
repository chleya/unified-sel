# CEP-CC Handoff - 2026-04-21

## One-Line State

CEP-CC has established learned continuous communication, but has not yet established compression-induced proto-symbol emergence.

Next task:

- C1e: Two-Stage Compression Fine-Tuning

## Project Boundary

CEP-CC stands for:

> Compressed Emergent Protocol under Continuous Communication.

Research question:

> Can continuous multi-agent systems develop reusable language-like protocol structure without tokens, text data, next-token prediction, or categorical communication actions?

Hard constraints:

- no token vocabulary.
- no next-token prediction objective.
- no language data.
- no discrete communication codebook as the main mechanism.
- final object selection may be categorical, but communication must stay continuous.

Do not import B7/meta-controller evidence as proof for CEP-CC. B7 is background only.

## Current Claim Boundary

Supported:

- continuous communication can carry hidden task-relevant information.
- a Listener can use continuous communication for object-wise decisions.
- learned continuous Speaker/Listener communication can beat no-communication across seeds.

Not supported yet:

- compression reliably induces proto-symbol partitions.
- there is a phase transition.
- communication clusters have semantic stability.
- communication has local compositionality.
- the system is language-like in the strong sense.

The current wording should be:

> Learned continuous communication emerges under task coordination pressure.

Do not say:

> Language emerged.

Do not say:

> Compression produced proto-symbols.

## File Map

Project code:

- `experiments/cep_cc/env.py`: continuous object-selection environment.
- `experiments/cep_cc/models.py`: Speaker, Listener, teacher-signal, random projection, system wrapper.
- `experiments/cep_cc/losses.py`: task loss, communication penalties, effective dimension, distillation diagnostic.
- `experiments/cep_cc/metrics.py`: accuracy, communication metrics, fallback prototype metrics, table formatting.
- `experiments/cep_cc/run_experiment.py`: train/eval CLI and sweep runners.
- `tests/test_cep_cc_protocol.py`: protocol and smoke tests.

Project docs:

- `experiments/cep_cc/README.md`
- `experiments/cep_cc/PROJECT_SPEC.md`
- `experiments/cep_cc/ROADMAP.md`
- `experiments/cep_cc/IMPLEMENTATION_PLAN.md`
- `CEP_CC_BRANCH_LEDGER_2026-04-21.md`

Result docs:

- `CEP_CC_V0_1_SCAFFOLD_2026-04-21.md`
- `CEP_CC_V0_1B_POSITIVE_CONTROL_2026-04-21.md`
- `CEP_CC_V0_1C_LEARNED_PROTOCOL_CALIBRATION_2026-04-21.md`
- `CEP_CC_V0_1D_MULTI_SEED_COMPRESSION_SWEEP_2026-04-21.md`

Meta-controller split handoff:

- `META_CONTROLLER_TO_CEP_CC_HANDOFF_2026-04-21.md`

## Implemented CLI

Single train/eval:

```powershell
python -m experiments.cep_cc.run_experiment --episodes 300 --batch-size 128 --baseline high-bandwidth --lr 0.005 --seed 0 --table
```

No communication:

```powershell
python -m experiments.cep_cc.run_experiment --episodes 300 --batch-size 128 --baseline no-communication --lr 0.005 --seed 0 --table
```

Teacher-signal positive control:

```powershell
python -m experiments.cep_cc.run_experiment --episodes 300 --batch-size 128 --baseline teacher-signal --lr 0.005 --seed 0 --table
```

Compressed learned communication:

```powershell
python -m experiments.cep_cc.run_experiment --episodes 300 --batch-size 128 --lambda-comm 0.001 --lr 0.005 --seed 0 --table
```

C1c calibration:

```powershell
python -m experiments.cep_cc.run_experiment --c1c-calibration --episodes 300 --batch-size 128 --lambda-comm 0.001 --lr 0.005 --seed 0 --table
```

C1d multi-seed from-scratch compression sweep:

```powershell
python -m experiments.cep_cc.run_experiment --multiseed-compression-sweep --seeds 0,1,2 --episodes 300 --batch-size 128 --lambda-sweep 0.0,0.001,0.003,0.006,0.01 --lr 0.005 --table
```

Tests:

```powershell
python -m pytest F:\unified-sel\tests\test_cep_cc_protocol.py -q
```

Current expected test result:

- `10 passed`
- Pytest cache warning may appear and is not a CEP-CC failure.

## Result Summary

V0.1 scaffold:

- executable environment/model/loss/metric/CLI/test scaffold exists.
- no positive result claimed.

V0.1b positive control:

- teacher-signal continuous communication beats no-communication.
- 40 episode sanity:
  - teacher-signal: `0.863`
  - no-communication: `0.398`
- conclusion: channel and Listener are viable.

V0.1c learned protocol calibration:

- official seed-0 profile: `episodes=300`, `batch_size=128`, `lr=0.005`.
- no-communication: `0.455`
- learned high-bandwidth: `0.668`
- learned lambda `0.001`: `0.846`
- learned lambda `0.003`: `0.807`
- learned lambda `0.01`: `0.467`
- conclusion: learned communication works on seed 0; light compression looks promising on seed 0 only.

V0.1d multi-seed sweep:

Learned high-bandwidth beats no-communication on all seeds:

| seed | no communication | learned high-bandwidth |
|---|---:|---:|
| 0 | 0.455 | 0.668 |
| 1 | 0.488 | 0.859 |
| 2 | 0.457 | 0.828 |

Light compression is seed-sensitive:

| seed | high-bandwidth | lambda 0.001 | interpretation |
|---|---:|---:|---|
| 0 | 0.668 | 0.846 | improves |
| 1 | 0.859 | 0.844 | nearly preserves |
| 2 | 0.828 | 0.670 | degrades |

Strong compression often collapses communication toward no-communication.

Decision:

- stay in C1.
- do not move to C2 semantic stability yet.

## Next Task: C1e

Task name:

- Two-Stage Compression Fine-Tuning

Question:

> If a learned high-bandwidth protocol is established first, can compression fine-tuning simplify communication geometry without destroying task performance?

Why:

- From-scratch compression is seed-sensitive.
- Protocol discovery and protocol compression are likely different phases.
- C1e tests compression after communication has already emerged.

Recommended implementation:

1. Add a runner that trains high-bandwidth learned communication for stage 1.
2. Reuse the same model weights.
3. Fine-tune with each compression lambda in stage 2.
4. Compare to V0.1d from-scratch compressed runs.

Recommended first profile:

- seeds: `0,1,2`
- stage 1 episodes: `300`
- stage 2 episodes: `120`
- batch size: `128`
- lr: `0.005`
- lambdas: `0.001,0.003,0.006,0.01`

Expected CLI shape:

```powershell
python -m experiments.cep_cc.run_experiment --two-stage-compression --seeds 0,1,2 --stage1-episodes 300 --stage2-episodes 120 --batch-size 128 --lambda-sweep 0.001,0.003,0.006,0.01 --lr 0.005 --table
```

Required output columns:

- eval accuracy
- train accuracy
- communication energy
- communication L1
- communication effective dimension
- prototype reuse
- cluster compactness
- target/cluster alignment
- baseline high-bandwidth metrics before fine-tune
- delta from high-bandwidth baseline

Acceptance to proceed to C2:

- high-bandwidth stage-1 beats no-communication on most or all seeds.
- at least one compression fine-tune lambda preserves most task accuracy across seeds.
- communication energy/L1/effective dimension decrease after fine-tune.
- target alignment does not collapse at the selected lambda.

Reject or hold condition:

- fine-tuning collapses task accuracy on most seeds.
- compression only works on seed 0.
- metrics show lower communication magnitude but no stable cluster/target alignment.

If rejected, stay in C1 and improve either optimization or geometry metrics.

## Engineering Notes

Keep `teacher-signal` as positive control.

Keep `lambda_comm_distill` clearly labeled as diagnostic and non-emergent.

Do not remove no-communication baseline from any official sweep.

Do not treat `prototype_reuse_rate=1.0` on zero communication as a real proto-symbol. It is a degenerate zero-variance signal.

The current cluster metrics are fallback proxies, not paper-grade. Before strong claims, add more robust geometry metrics or export communication trajectories for external analysis.

## Handoff Warning

The project is at a fragile but useful point:

- communication emergence is real enough to continue.
- compression/proto-symbol emergence is not yet proven.
- the next experiment should test staged compression, not semantic stability.

