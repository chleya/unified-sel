# CEP-CC Branch Ledger - 2026-04-21

## Project

CEP-CC: Compressed Emergent Protocol under Continuous Communication.

Location:

- `experiments/cep_cc`

## North Star

Test whether language-like structure can emerge as a compressed coordination protocol in continuous multi-agent dynamics without tokens, next-token prediction, or language data.

Core chain:

```text
interaction pressure
+ compression pressure
+ temporal/memory pressure
=> symbol-like continuous trajectory partition
=> reusable protocol
=> local compositional structure
```

## Boundary

Do not import the meta-controller B7 claim as evidence.

B7 is useful background because it showed compression/constraint pressure can expose useful structure, but CEP-CC needs its own evidence:

- proto-symbol trajectory partitions
- semantic stability
- local compositionality
- phase transition under compression sweep

## Created Artifacts

- `META_CONTROLLER_TO_CEP_CC_HANDOFF_2026-04-21.md`
- `experiments/cep_cc/README.md`
- `experiments/cep_cc/PROJECT_SPEC.md`
- `experiments/cep_cc/ROADMAP.md`
- `experiments/cep_cc/IMPLEMENTATION_PLAN.md`
- `experiments/cep_cc/__init__.py`

## C0: Project Split And Spec

Status: completed.

Outcome:

- CEP-CC is split from the meta-controller/B7 line.
- B7 W23 is paused, not discarded.
- CEP-CC now has a project spec, roadmap, and implementation plan.

## C1: V0.1 Proto-Symbol Emergence Scaffold

Status: completed as V0.1 scaffold.

Question:

> Under communication compression, do continuous communication trajectories form stable reusable proto-symbol partitions while preserving task performance?

Recommended implementation:

- `experiments/cep_cc/env.py`
- `experiments/cep_cc/models.py`
- `experiments/cep_cc/losses.py`
- `experiments/cep_cc/metrics.py`
- `experiments/cep_cc/run_experiment.py`
- `tests/test_cep_cc_protocol.py`

Minimum CLI:

```powershell
python -m experiments.cep_cc.run_experiment --mode train-eval --episodes 2000 --batch-size 128 --lambda-sweep 0.0,0.001,0.003,0.01,0.03 --seed 0 --table
```

Minimum acceptance:

- compressed communication beats no-communication on task success.
- compressed communication has lower communication effective dimension than high-bandwidth communication.
- compressed communication shows stronger prototype reuse or clustering quality than high-bandwidth communication.

Reject condition:

- no task gain over no communication.
- no reusable trajectory structure under compression.
- clusters fail to align with task-relevant latent invariants.

Result document:

- `CEP_CC_V0_1_SCAFFOLD_2026-04-21.md`

Outcome:

- executable CEP-CC scaffold exists.
- continuous environment, Speaker/Listener models, losses, metrics, CLI, and tests are implemented.
- protocol tests pass.
- no positive proto-symbol claim yet.
- quick sanity runs show stable code and measurable communication, but high-bandwidth communication does not yet clearly beat no-communication.

## C1b: V0.1 Training Calibration And Positive Control

Status: completed as V0.1b.

Question:

> Can the V0.1 environment and model be calibrated so learned continuous communication clearly improves task success over no communication?

Recommended next actions:

- simplify or stage the target rule so hidden continuous factors create a learnable communication need.
- strengthen the Listener object-wise head so decoded communication can affect each object score directly.
- add a positive-control continuous teacher-signal baseline.
- define a bounded official run profile before running the full compression sweep.

Do not proceed to C2 semantic stability until C1b establishes communication necessity.

Result document:

- `CEP_CC_V0_1B_POSITIVE_CONTROL_2026-04-21.md`

Outcome:

- continuous `teacher-signal` positive control added.
- Listener now has object-wise scoring and per-object communication decoding.
- target rule depends more strongly on hidden continuous factors.
- protocol tests pass: `7 passed`.
- teacher-signal clearly beats no-communication in the 40-episode sanity run:
  - teacher-signal accuracy: `0.863`
  - no-communication accuracy: `0.398`
- high-bandwidth learned Speaker still does not beat no-communication in the 80-episode sanity run.
- conclusion: communication channel and Listener are viable; emergent learned protocol is not yet established.

## C1c: End-To-End Learned Protocol Calibration

Status: completed as V0.1c.

Question:

> Can a learned continuous Speaker discover a useful communication protocol from task loss alone, before compression is increased?

Recommended next actions:

- log train accuracy and eval accuracy separately.
- compare no-communication, teacher-signal, high-bandwidth learned, and compressed learned under one bounded profile.
- add a staged curriculum or easier rule profile if high-bandwidth learned remains at no-communication level.
- optionally add a continuous score-distillation control, clearly marked non-emergent, to distinguish architecture capacity from task-gradient discovery.

Do not claim proto-symbol emergence until learned continuous communication beats no-communication and compression changes the communication geometry.

Result document:

- `CEP_CC_V0_1C_LEARNED_PROTOCOL_CALIBRATION_2026-04-21.md`

Outcome:

- added train/eval split metrics.
- added C1c calibration runner.
- added learning-rate CLI.
- added continuous score-distillation diagnostic.
- protocol tests pass: `9 passed`.
- learned high-bandwidth communication beats no-communication under the official C1c profile:
  - no-communication: `0.455`
  - learned high-bandwidth: `0.668`
- light compression improves the learned protocol in the seed-0 profile:
  - lambda `0.001`: `0.846`
  - lambda `0.003`: `0.807`
  - lambda `0.01`: `0.467`
- conclusion: communication necessity and end-to-end learned communication are now established for the seed-0 calibration profile.

## C1d: Multi-Seed Compression Sweep

Status: completed as V0.1d.

Question:

> Does the light-compression improvement and communication-geometry change reproduce across seeds?

Recommended run profile:

```powershell
python -m experiments.cep_cc.run_experiment --episodes 300 --batch-size 128 --lambda-sweep 0.0,0.001,0.003,0.006,0.01 --lr 0.005 --seed <seed> --table
```

Recommended seeds:

- `0`
- `1`
- `2`

Acceptance for moving to C2:

- learned communication beats no-communication on most seeds.
- at least one light-compression setting preserves or improves task accuracy against high-bandwidth learned communication.
- communication energy/L1/effective dimension shift with lambda.
- target/cluster alignment does not collapse at the best light-compression setting.

If the sweep is unstable:

- stay in C1 and tune the environment/model.
- do not move to semantic stability.

Result document:

- `CEP_CC_V0_1D_MULTI_SEED_COMPRESSION_SWEEP_2026-04-21.md`

Outcome:

- protocol tests pass: `10 passed`.
- learned high-bandwidth communication beats no-communication on all seeds:
  - seed 0: `0.668` vs `0.455`
  - seed 1: `0.859` vs `0.488`
  - seed 2: `0.828` vs `0.457`
- light compression is seed-sensitive:
  - seed 0 lambda `0.001` improves to `0.846`.
  - seed 1 lambda `0.001` nearly preserves high-bandwidth accuracy at `0.844`.
  - seed 2 lambda `0.001` drops to `0.670`.
- stronger compression often collapses toward no-communication.
- conclusion: learned continuous communication is stable; compression-induced proto-symbol emergence is not yet established.

## C1e: Two-Stage Compression Fine-Tuning

Status: completed as V0.1e.

Question:

> If a learned high-bandwidth protocol is first established, can compression fine-tuning simplify communication geometry without destroying task performance?

Recommended implementation:

- train high-bandwidth learned communication for stage 1.
- continue training same model with lambda sweep in stage 2.
- report task accuracy, communication energy/L1/effective dimension, compactness, and target alignment.
- compare to from-scratch compressed runs from V0.1d.

Recommended first profile:

- seeds: `0,1,2`
- stage 1 episodes: `300`
- stage 2 episodes: `120`
- batch size: `128`
- lr: `0.005`
- lambdas: `0.001,0.003,0.006,0.01`

Do not move to C2 until compression can be applied reproducibly after protocol discovery.

Result document:

- `CEP_CC_V0_1E_TWO_STAGE_COMPRESSION_FINETUNE_2026-04-21.md`

Outcome:

- protocol tests pass: `11 passed`.
- two-stage fine-tuning runner implemented.
- stage-2 fine-tuning improves task accuracy on all seeds:
  - seed 0: `0.668` -> `0.703-0.713`
  - seed 1: `0.859` -> `0.877-0.881`
  - seed 2: `0.828` -> `0.863-0.865`
- stage-2 does not simplify communication geometry:
  - energy/L1 mostly increase or barely decrease.
  - effective dimension increases on every stage-2 run.
  - target alignment is mixed.
- conclusion: two-stage training avoids task collapse but does not yet demonstrate compression-induced protocol simplification.

## C1f: Stronger/Frozen Two-Stage Compression Ablation

Status: completed as V0.1f.

Question:

> Can stage-2 compression simplify communication geometry if we reduce task fine-tuning drift or increase compression pressure?

Recommended ablations:

- stronger stage-2 lambdas:
  - `0.03`
  - `0.06`
  - `0.10`
- lower stage-2 learning rate:
  - `0.001`
- freeze Listener during stage 2 and update Speaker only.
- compare against C1e and V0.1d.

Acceptance:

- task accuracy remains above no-communication.
- communication energy/L1/effective dimension decrease from stage 1 on most seeds.
- target alignment does not collapse.

If this fails:

- compression objective is not yet shaping proto-symbol structure.
- next work should improve geometry metrics or add a bottleneck/consistency pressure, not move to C2.

Result document:

- `CEP_CC_V0_1F_STRONGER_FROZEN_COMPRESSION_ABLATION_2026-04-21.md`

Outcome:

- protocol tests pass: `12 passed`.
- C1f ablation runner implemented.
- stronger compression with lower stage-2 lr preserves or improves task accuracy on all seeds.
- freeze-Listener stage 2 with lambda `0.10` gives the cleanest compression-control result:
  - seed 0: accuracy `0.732`, energy `-0.116`, L1 `-0.095`, dim `-0.328`
  - seed 1: accuracy `0.879`, energy `-0.019`, L1 `-0.013`, dim `-0.370`
  - seed 2: accuracy `0.891`, energy `-0.051`, L1 `-0.038`, dim `-0.173`
- target alignment is mixed but does not collapse.
- conclusion: two-stage stronger/frozen compression can reduce communication magnitude and effective dimension while preserving task success.

## C1g: Communication Geometry Audit

Status: completed as V0.1g.

Question:

> Does the compressed two-stage protocol contain proto-symbol-like trajectory partitions aligned with task-relevant continuous invariants?

Recommended implementation:

- export communication trajectories, target index, latent factors, and object scores for:
  - high-bandwidth stage 1.
  - selected compressed stage 2: freeze Listener, lambda `0.10`.
- add direct invariant alignment metrics:
  - cluster purity by target object.
  - regression/probe from communication to latent factors.
  - within-target vs between-target trajectory distance.
  - nearest-prototype stability.
- compare compressed stage 2 to high-bandwidth stage 1.

Do not move to C2 semantic stability until C1g shows a real geometry signal beyond fallback cluster compactness.

Result document:

- `CEP_CC_V0_1G_COMMUNICATION_GEOMETRY_AUDIT_2026-04-21.md`

Outcome:

- protocol tests pass: `13 passed`.
- C1g geometry audit runner implemented.
- compressed freeze-Listener lambda `0.10` improves accuracy and reduces energy/L1/effective dimension on all seeds.
- hidden `q` probe R2 is strong and stable around `0.59-0.65`.
- target purity remains modest around `0.56-0.64`.
- within/between target trajectory ratio remains near `0.95`, indicating weak partition separation.
- full latent-factor probe R2 is near zero.
- conclusion: compressed continuous communication encodes a hidden continuous task factor, but does not yet show strong proto-symbol partitions.

## C1h: Consistency-Pressure Protocol Partitioning

Status: completed as V0.1h.

Question:

> Does adding explicit consistency pressure over semantically similar samples turn the useful continuous code into reusable trajectory partitions?

Recommended implementation:

- define semantic similarity using target index and/or target hidden `q` quantile.
- add a communication consistency loss within mini-batches.
- train high-bandwidth stage 1 as before.
- stage 2: freeze Listener, lambda `0.10`, consistency pressure, low stage2 lr.
- rerun C1g geometry audit metrics.

Acceptance:

- task accuracy remains above no-communication.
- communication energy/L1/effective dimension stay below stage 1.
- target purity improves materially.
- within/between trajectory ratio drops below the current `~0.95`.
- hidden q probe remains non-collapsed.

If this fails:

- proto-symbol emergence likely requires richer task composition or memory/temporal pressure, not compression alone.

Result document:

- `CEP_CC_V0_1H_CONSISTENCY_PROTOCOL_PARTITIONING_2026-04-21.md`

Outcome:

- protocol tests pass: `14 passed`.
- target-conditioned communication consistency implemented.
- consistency pressure preserves or improves task accuracy on all seeds.
- energy/L1/effective dimension remain below stage 1.
- hidden `q` probe remains strong.
- target purity improves on seeds 1 and 2, and slightly on seed 0 at low consistency:
  - seed 0: `0.556` -> `0.565`
  - seed 1: `0.598` -> `0.638`
  - seed 2: `0.588` -> `0.628`
- within/between trajectory ratio improves only slightly:
  - seed 0: `0.949` -> `0.941`
  - seed 1: `0.952` -> `0.948`
  - seed 2: `0.950` -> `0.946`
- conclusion: consistency helps alignment but does not yet create strong proto-symbol partitions.

## C1i: Factor-Conditioned Consistency And Segment Audit

Status: completed as V0.1i.

Question:

> Does consistency over task-relevant latent factors, plus segment-level analysis, reveal reusable sub-trajectory structure?

Recommended next actions:

- define factor bins for hidden `q`, motion alignment, and relation-to-anchor.
- add factor-conditioned consistency variants.
- audit trajectory segments rather than only full trajectories.
- measure whether different communication segments align with different latent factors.

Do not move to C2 until factor-level or segment-level reusable structure is visible.

Result document:

- `CEP_CC_V0_1I_FACTOR_CONSISTENCY_SEGMENT_AUDIT_2026-04-21.md`

Outcome:

- protocol tests pass: `15 passed`.
- factor-bin consistency implemented.
- segment audit implemented.
- factor-conditioned consistency preserves compressed communication and task performance.
- target purity reaches about `0.63-0.64` on seeds 1 and 2.
- segment q probes improve together rather than separating by segment.
- no stable early/middle/late factor assignment appears.
- motion and relation consistency behave almost identically under the current task.
- conclusion: current environment supports continuous hidden-factor encoding, but does not force local compositional protocol structure.

## C1j: Compositional Task Variant

Status: completed as V0.1j.

Question:

> If the task requires separable latent subdecisions, do communication trajectories develop reusable segment-level structure?

Recommended implementation:

- add a new environment profile or target-rule mode with explicit factor composition.
- keep communication continuous.
- require independent hidden factors, for example:
  - hidden q/category-like scalar.
  - relation-to-anchor factor.
  - motion-alignment factor.
- design target rule so no single scalar hidden code is sufficient.
- preserve C1f/C1h training style:
  - stage 1 high-bandwidth.
  - stage 2 freeze Listener, compression, factor/target consistency.
- audit segment probes and segment swap interventions.

Acceptance:

- learned communication beats no-communication.
- compression preserves task success.
- at least one segment specializes to a different latent factor.
- segment swap produces targeted changes larger than off-target changes.

If this fails:

- the current minimal setup may need explicit memory/temporal pressure before language-like compositionality can emerge.

Outcome:

- implemented `EnvConfig.rule_mode="compositional"`.
- implemented `run_c1j_compositional_variant`.
- implemented segment swap audit metrics.
- protocol tests pass: `17 passed`.
- official run:

```powershell
python -m experiments.cep_cc.run_experiment --c1j-compositional --seeds 0,1,2 --stage1-episodes 600 --stage2-episodes 120 --batch-size 128 --lr 0.005 --stage2-lr 0.001 --lambda-comm 0.10 --factor-names q,motion,relation --factor-consistency-sweep 0.03 --audit-batches 8 --table
```

Compact result:

- no-communication accuracy:
  - seed 0: `0.248`
  - seed 1: `0.275`
  - seed 2: `0.289`
- high-bandwidth compositional accuracy:
  - seed 0: `0.422`
  - seed 1: `0.444`
  - seed 2: `0.440`
- best compressed/factor-conditioned accuracy:
  - seed 0: `0.444`
  - seed 1: `0.459`
  - seed 2: `0.456`
- compressed rows reduce communication energy and L1 across all seeds.
- q/motion/relation consistency rows remain almost identical.
- segment q specialization gaps remain small, about `0.018-0.079`.
- segment swap effects are measurable but not factor-targeted enough for a local-compositionality claim.

Decision:

- V0.1j is partial-positive:
  - communication is necessary under the compositional variant.
  - compression preserves success.
- V0.1j is not a compositionality win:
  - no robust factor-specific segment assignment.
  - no targeted segment-swap evidence.

Next:

- C1k: Temporal Memory Pressure Variant.

Rationale:

- Repeated factor-consistency tuning is no longer the right lever.
- The system likely needs delayed evidence or externalized memory pressure before local reusable sub-protocols become necessary.

## C1k: Temporal Memory Pressure Variant

Status: completed as V0.1k.

Question:

> If part of the task-relevant evidence is only available early and the Listener must act later, does continuous communication become an external memory protocol with reusable local structure?

Recommended implementation:

- add an environment mode with two observation phases:
  - early hidden evidence visible to Speaker.
  - later decision observation visible to Listener.
- keep communication continuous.
- restrict direct Listener access to hidden evidence.
- evaluate:
  - no-communication baseline.
  - high-bandwidth learned communication.
  - compressed/frozen Listener stage2.
  - segment audit.
  - segment swap audit.
- acceptance:
  - learned communication beats no-communication.
  - compression preserves success.
  - at least one segment has consistent causal effect under swap or ablation.
  - preferably, different segments affect different latent subdecisions.

Outcome:

- implemented `EnvConfig.rule_mode="temporal_memory"`.
- implemented `run_c1k_temporal_memory_variant`.
- added CLI flag `--c1k-temporal-memory`.
- added segment ablation audit metrics.
- protocol tests pass: `19 passed`.
- official run:

```powershell
python -m experiments.cep_cc.run_experiment --c1k-temporal-memory --seeds 0,1,2 --stage1-episodes 600 --stage2-episodes 120 --batch-size 128 --lr 0.005 --stage2-lr 0.001 --lambda-comm 0.10 --factor-names q,motion,relation --factor-consistency-sweep 0.03 --audit-batches 8 --table
```

Compact result:

- no-communication accuracy:
  - seed 0: `0.283`
  - seed 1: `0.215`
  - seed 2: `0.262`
- high-bandwidth temporal_memory accuracy:
  - seed 0: `0.636`
  - seed 1: `0.250`
  - seed 2: `0.567`
- best compressed accuracy:
  - seed 0: `0.679`
  - seed 1: `0.245`
  - seed 2: `0.594`
- strongest segment causal effects:
  - seed 0: late swap action change `0.542`, late ablation action change `0.428`, late ablation logit drop `0.987`
  - seed 1: no meaningful segment causal effect
  - seed 2: late swap action change `0.460`, late ablation action change `0.399`, late ablation logit drop `0.357`
- seed 1 high-bandwidth 1200-episode probe only reached `0.357`, so the failure is not just the 600-episode budget.

Decision:

- V0.1k is borderline-positive:
  - temporal/memory pressure is the strongest lever so far.
  - successful seeds show strong communication necessity and segment-level causal effects.
- V0.1k is not yet robust:
  - seed 1 is a hard failure.
  - factor consistency still does not create clean q/motion/relation segment roles.

Do not move to C2 yet.

Next:

- C1l: Temporal Robustness And Bootstrap.

## C1l: Temporal Robustness And Bootstrap

Status: completed as V0.1l.

Question:

> Can temporal_memory communication be made stable across seeds without adding token/codebook machinery?

Recommended implementation:

- test teacher-signal warm start or distillation on `temporal_memory`.
- test curriculum:
  - train on `compositional`.
  - continue on `temporal_memory`.
- test segment dropout/noise during training to force distributed robust memory.
- optionally widen Listener/Speaker state only if the above fails.

Acceptance:

- high-bandwidth temporal_memory beats no-communication on seeds 0/1/2.
- compressed/frozen Listener stage preserves success.
- segment ablation remains causal on all successful seeds.
- no discrete token/codebook mechanism is introduced.

Outcome:

- implemented `TrainConfig.segment_dropout_prob`.
- implemented training-time continuous segment dropout.
- implemented `run_c1l_temporal_bootstrap`.
- added CLI flag `--c1l-bootstrap`.
- protocol tests pass: `20 passed`.
- no token/codebook/classifier introduced.

Seed-1 rescue probe:

```powershell
python -m experiments.cep_cc.run_experiment --c1l-bootstrap --seeds 1 --bootstrap-modes direct,curriculum,segment_dropout --stage1-episodes 600 --curriculum-episodes 300 --stage2-episodes 120 --batch-size 128 --lr 0.005 --stage2-lr 0.001 --lambda-comm 0.10 --factor-names q --factor-consistency-sweep 0.03 --segment-dropout-prob 0.25 --audit-batches 8 --table
```

- direct compressed accuracy: `0.245`
- curriculum compressed accuracy: `0.697`
- segment_dropout compressed accuracy: `0.678`

Official bootstrap comparison:

```powershell
python -m experiments.cep_cc.run_experiment --c1l-bootstrap --seeds 0,1,2 --bootstrap-modes direct,curriculum,segment_dropout --stage1-episodes 600 --curriculum-episodes 300 --stage2-episodes 120 --batch-size 128 --lr 0.005 --stage2-lr 0.001 --lambda-comm 0.10 --factor-names q --factor-consistency-sweep 0.03 --segment-dropout-prob 0.25 --audit-batches 8 --table
```

Compressed accuracy:

| seed | direct | curriculum | segment_dropout |
|---|---:|---:|---:|
| 0 | `0.679` | `0.559` | `0.430` |
| 1 | `0.245` | `0.697` | `0.678` |
| 2 | `0.593` | `0.673` | `0.530` |

Official curriculum factor sweep:

```powershell
python -m experiments.cep_cc.run_experiment --c1l-bootstrap --seeds 0,1,2 --bootstrap-modes curriculum --stage1-episodes 600 --curriculum-episodes 300 --stage2-episodes 120 --batch-size 128 --lr 0.005 --stage2-lr 0.001 --lambda-comm 0.10 --factor-names q,motion,relation --factor-consistency-sweep 0.03 --segment-dropout-prob 0.25 --audit-batches 8 --table
```

Best curriculum rows:

- seed 0: high-bandwidth `0.556`, best compressed `0.559`
- seed 1: high-bandwidth `0.673`, best compressed `0.698`
- seed 2: high-bandwidth `0.660`, best compressed `0.674`

Decision:

- V0.1l is robust-positive for temporal memory pressure.
- curriculum is the best bootstrap:
  - stabilizes temporal_memory across seeds.
  - preserves success under frozen-Listener compression.
  - keeps late segment causal effects across all seeds.
- factor-specific segment roles still do not appear.

Supported claim:

> Continuous communication can become a robust task-necessary external memory/control protocol under temporal pressure and curriculum bootstrapping.

Not supported:

> A fully language-like local compositional protocol has emerged.

Next:

- C2a: Semantic Stability Under Nuisance Transformations.

## C2: V0.2 Semantic Stability

Status: in progress.

Start from the V0.1l curriculum temporal_memory setup.

Tasks:

- nuisance transformations.
- invariant probes.
- cluster/invariant alignment under reparameterization.

## C2a: Semantic Stability Under Nuisance Transformations

Status: completed as V0.2a.

Question:

> Does the learned continuous communication protocol remain semantically stable when the world is mirrored, rotated, or velocity-scaled?

Outcome:

- implemented `EnvConfig.nuisance_mode`.
- implemented nuisance modes:
  - `none`
  - `mirror_x`
  - `rotate90`
  - `velocity_scale`
- implemented `run_c2a_semantic_stability`.
- added CLI flag `--c2a-semantic-stability`.
- added CLI flag `--nuisance-modes`.
- protocol tests pass: `22 passed`.

Official run:

```powershell
python -m experiments.cep_cc.run_experiment --c2a-semantic-stability --seeds 0,1,2 --nuisance-modes none,mirror_x,rotate90,velocity_scale --stage1-episodes 600 --curriculum-episodes 300 --stage2-episodes 120 --batch-size 128 --lr 0.005 --stage2-lr 0.001 --lambda-comm 0.10 --factor-names q --factor-consistency-sweep 0.03 --audit-batches 8 --table
```

Task accuracy:

| seed | clean | mirror_x | rotate90 | velocity_scale |
|---|---:|---:|---:|---:|
| 0 | `0.559` | `0.572` | `0.568` | `0.557` |
| 1 | `0.697` | `0.691` | `0.695` | `0.695` |
| 2 | `0.673` | `0.682` | `0.673` | `0.670` |

Hidden q probe R2:

| seed | clean | mirror_x | rotate90 | velocity_scale |
|---|---:|---:|---:|---:|
| 0 | `0.642` | `0.639` | `0.606` | `0.639` |
| 1 | `0.648` | `0.646` | `0.652` | `0.648` |
| 2 | `0.629` | `0.632` | `0.633` | `0.627` |

Late segment causal effects stay stable:

- seed 0 late swap action change: clean `0.503`, nuisance range `0.493-0.505`.
- seed 1 late swap action change: clean `0.529`, nuisance range `0.528-0.532`.
- seed 2 late swap action change: clean `0.568`, nuisance range `0.571-0.575`.

Decision:

- V0.2a is positive.
- task success, hidden-q semantics, and late-segment causal effects remain stable under simple continuous nuisance reparameterizations.

Supported claim:

> The curriculum temporal_memory protocol preserves task-relevant communication semantics under simple continuous nuisance reparameterizations.

Next:

- C2b: Cluster-Invariant Alignment Under Reparameterization.

## C2b: Cluster-Invariant Alignment Under Reparameterization

Status: completed as V0.2b.

Question:

> Do communication prototype assignments and latent-factor probes align across clean and nuisance-transformed worlds?

Recommended implementation:

- collect paired clean/nuisance batches from the same seeds.
- compare communication nearest-prototype assignments.
- measure latent invariant alignment across nuisance modes.
- report cluster stability, hidden-q alignment, and late-segment causal preservation.

Outcome:

- implemented `paired_nuisance_alignment_metrics`.
- implemented `run_c2b_cluster_invariant_alignment`.
- added CLI flag `--c2b-cluster-alignment`.
- protocol tests pass: `23 passed`.

Official run:

```powershell
python -m experiments.cep_cc.run_experiment --c2b-cluster-alignment --seeds 0,1,2 --nuisance-modes mirror_x,rotate90,velocity_scale --stage1-episodes 600 --curriculum-episodes 300 --stage2-episodes 120 --batch-size 128 --lr 0.005 --stage2-lr 0.001 --lambda-comm 0.10 --factor-names q --factor-consistency-sweep 0.03 --audit-batches 4 --table
```

Paired target agreement:

| seed | mirror_x | rotate90 | velocity_scale |
|---|---:|---:|---:|
| 0 | `0.609` | `0.809` | `1.000` |
| 1 | `0.598` | `0.797` | `1.000` |
| 2 | `0.629` | `0.828` | `1.000` |

Prototype assignment stability:

| seed | mirror_x | rotate90 | velocity_scale |
|---|---:|---:|---:|
| 0 | `0.975` | `0.977` | `0.996` |
| 1 | `0.973` | `0.965` | `0.988` |
| 2 | `0.982` | `0.988` | `0.998` |

Paired communication distance ratio:

| seed | mirror_x | rotate90 | velocity_scale |
|---|---:|---:|---:|
| 0 | `0.065` | `0.059` | `0.012` |
| 1 | `0.046` | `0.053` | `0.008` |
| 2 | `0.043` | `0.046` | `0.008` |

Hidden q correlation:

| seed | mirror_x | rotate90 | velocity_scale |
|---|---:|---:|---:|
| 0 | `0.908` | `0.952` | `1.000` |
| 1 | `0.901` | `0.945` | `1.000` |
| 2 | `0.918` | `0.956` | `1.000` |

Decision:

- V0.2b is mixed-positive.
- communication geometry and hidden-q semantics are strongly aligned under paired nuisance transforms.
- full latent-factor invariance is not supported because some audited factors are coordinate/relationship dependent.

Next:

- C2c: Factor-Separated Invariance Audit.

## C2c: Factor-Separated Invariance Audit

Status: completed as V0.2c.

Question:

> Which semantic factors are invariant, and which are merely equivariant, under nuisance transformations?

Recommended implementation:

- split factor alignment metrics:
  - hidden q should be invariant.
  - position/motion/relation may be equivariant or transform-dependent.
- report per-factor clean/nuisance correlations.
- avoid aggregating all latent dimensions into one invariance score.
- use C2b paired batches and C2a nuisance modes.

Outcome:

- implemented per-factor paired correlations:
  - `paired_pos_factor_corr`
  - `paired_q_score_factor_corr`
  - `paired_motion_factor_corr`
  - `paired_relation_factor_corr`
  - `paired_invariant_factor_corr`
  - `paired_equivariant_factor_corr`
- implemented `run_c2c_factor_separated_invariance`.
- added CLI flag `--c2c-factor-invariance`.
- protocol tests pass: `24 passed`.

Official run:

```powershell
python -m experiments.cep_cc.run_experiment --c2c-factor-invariance --seeds 0,1,2 --nuisance-modes mirror_x,rotate90,velocity_scale --stage1-episodes 600 --curriculum-episodes 300 --stage2-episodes 120 --batch-size 128 --lr 0.005 --stage2-lr 0.001 --lambda-comm 0.10 --factor-names q --factor-consistency-sweep 0.03 --audit-batches 4 --table
```

Key results:

- hidden q object-value correlation:
  - mirror_x: `0.901-0.918`
  - rotate90: `0.945-0.956`
  - velocity_scale: `1.000`
- q-score factor correlation:
  - mirror_x: `0.840-0.866`
  - rotate90: `0.903-0.920`
  - velocity_scale: `1.000`
- relation factor correlation:
  - mirror_x: about `-0.93`
  - rotate90: about `0.05-0.13`
  - velocity_scale: `1.000`

Decision:

- V0.2c is positive.
- low aggregate latent correlation in V0.2b was mainly caused by coordinate/sign-dependent relation factors.
- invariant hidden-q semantics remain stable.
- coordinate-dependent factors transform nontrivially but predictably.

Supported claim:

> The protocol preserves invariant hidden-q semantics, while coordinate-dependent factors transform nontrivially under nuisance transformations.

Next:

- C3a: Local Compositionality With Targeted Segment Intervention.

## C3: V0.3 Local Compositionality

Status: in progress.

Start from the robust C1l/C2 curriculum temporal_memory setup.

Tasks:

- segment slicing.
- segment swap interventions.
- targeted/off-target effect ratio.

## C3a: Local Compositionality With Targeted Segment Intervention

Status: completed as V0.3a.

Question:

> Does replacing a communication segment primarily change one latent factor more than off-target factors?

Outcome:

- implemented `_targeted_segment_intervention_metrics`.
- implemented `run_c3a_targeted_segment_intervention`.
- added CLI flag `--c3a-targeted-intervention`.
- protocol tests pass: `25 passed`.

Official run:

```powershell
python -m experiments.cep_cc.run_experiment --c3a-targeted-intervention --seeds 0,1,2 --stage1-episodes 600 --curriculum-episodes 300 --stage2-episodes 120 --batch-size 128 --lr 0.005 --stage2-lr 0.001 --lambda-comm 0.10 --factor-names q --factor-consistency-sweep 0.03 --audit-batches 8 --table
```

Best targeted intervention:

| seed | targeted effect | off-target effect | ratio |
|---|---:|---:|---:|
| 0 | `0.143` | `0.049` | `2.942` |
| 1 | `0.124` | `0.032` | `3.852` |
| 2 | `0.106` | `0.023` | `4.605` |

Per-factor best ratios:

| seed | q | motion | relation |
|---|---:|---:|---:|
| 0 | `1.805` | `0.050` | `2.942` |
| 1 | `1.686` | `0.060` | `3.852` |
| 2 | `1.915` | `0.062` | `4.605` |

Decision:

- V0.3a is partial-positive.
- relation has strong targeted/off-target causal effects across all seeds.
- q has weaker but consistent targeted effects.
- motion has no meaningful targeted effect.
- full local compositional syntax is not established.

Supported claim:

> Some communication segments have factor-targeted causal effects, with relation showing the strongest targeted control.

Next:

- C3b: Targeted Intervention Robustness And Segment Assignment.

## C3b: Targeted Intervention Robustness And Segment Assignment

Status: completed as V0.3b.

Question:

> Are factor-targeted intervention effects stable under nuisance transforms and larger audit batches?

Recommended implementation:

- repeat C3a targeted interventions under `mirror_x`, `rotate90`, and `velocity_scale`.
- increase audit batches for relation/q targeted effects.
- report segment assignment consistency.
- identify whether motion needs stronger task pressure.

Outcome:

- implemented `run_c3b_targeted_intervention_robustness`.
- added CLI flag `--c3b-targeted-robustness`.
- protocol tests pass: `26 passed`.

Official run:

```powershell
python -m experiments.cep_cc.run_experiment --c3b-targeted-robustness --seeds 0,1,2 --nuisance-modes none,mirror_x,rotate90,velocity_scale --stage1-episodes 600 --curriculum-episodes 300 --stage2-episodes 120 --batch-size 128 --lr 0.005 --stage2-lr 0.001 --lambda-comm 0.10 --factor-names q --factor-consistency-sweep 0.03 --audit-batches 12 --table
```

Relation targeted ratio:

| seed | none | mirror_x | rotate90 | velocity_scale |
|---|---:|---:|---:|---:|
| 0 | `2.972` | `2.771` | `2.928` | `2.831` |
| 1 | `4.126` | `3.738` | `4.122` | `3.748` |
| 2 | `4.561` | `4.401` | `4.584` | `4.000` |

Q targeted ratio:

| seed | none | mirror_x | rotate90 | velocity_scale |
|---|---:|---:|---:|---:|
| 0 | `1.882` | `1.879` | `1.843` | `1.802` |
| 1 | `1.784` | `1.729` | `1.814` | `1.651` |
| 2 | `1.777` | `1.781` | `1.890` | `1.739` |

Motion targeted ratio:

| seed | none | mirror_x | rotate90 | velocity_scale |
|---|---:|---:|---:|---:|
| 0 | `0.046` | `0.049` | `0.047` | `0.105` |
| 1 | `0.055` | `0.054` | `0.058` | `0.125` |
| 2 | `0.064` | `0.063` | `0.065` | `0.144` |

Decision:

- V0.3b is positive but scoped.
- q maps robustly to late segment.
- relation maps robustly to early segment.
- motion remains unresolved.

Supported claim:

> The learned continuous protocol has stable local segment roles for at least two semantic factors: relation and q.

Not supported:

> Full three-factor local compositional grammar.

Next:

- C3c: Motion Pressure Variant.

## C3c: Motion Pressure Variant

Status: completed as V0.3c.

Question:

> If motion dependence is strengthened, does motion acquire a targeted communication segment role without destroying q/relation roles?

Recommended implementation:

- add a `motion_pressure` target-rule mode or config knob.
- increase motion factor weight and interaction with q/relation.
- rerun C1l curriculum and C3b targeted robustness.
- acceptance:
  - task remains learnable across seeds.
  - q/relation roles remain present.
  - motion targeted ratio rises above off-target baseline and is stable.

Outcome:

- implemented `EnvConfig.rule_mode="motion_pressure"`.
- implemented `run_c3c_motion_pressure_variant`.
- added CLI flag `--c3c-motion-pressure`.
- protocol tests pass: `28 passed`.

Official probe:

```powershell
python -m experiments.cep_cc.run_experiment --c3c-motion-pressure --seeds 0,1,2 --nuisance-modes none --stage1-episodes 600 --curriculum-episodes 300 --stage2-episodes 120 --batch-size 128 --lr 0.005 --stage2-lr 0.001 --lambda-comm 0.10 --factor-names motion --factor-consistency-sweep 0.03 --audit-batches 12 --table
```

Task accuracy:

| seed | accuracy |
|---|---:|
| 0 | `0.439` |
| 1 | `0.443` |
| 2 | `0.438` |

Per-factor targeted ratios:

| seed | q | motion | relation |
|---|---:|---:|---:|
| 0 | `1.871` | `0.057` | `2.505` |
| 1 | `1.852` | `0.053` | `2.627` |
| 2 | `1.632` | `0.054` | `2.425` |

Decision:

- V0.3c is negative but informative.
- q/relation roles persist.
- motion remains unresolved.
- simple scalar motion weighting makes the task harder but does not create motion-specific communication.

Interesting finding:

- motion may be too entangled with relation/nearest-neighbor structure in the current environment.
- the protocol routes useful local control through q and relation rather than motion.

Next:

- C3d: Motion Decoupling Environment.

## C3d: Motion Decoupling Environment

Status: completed as V0.3d.

Question:

> If motion is made independent of relation/position proxies, does it acquire a dedicated communication segment?

Implementation:

- added `EnvConfig.rule_mode="motion_decoupled"`.
- replaced nearest-neighbor velocity alignment with an independent velocity projection.
- added `run_c3d_motion_decoupling_variant`.
- added CLI flag `--c3d-motion-decoupling`.
- protocol tests pass: `30 passed`.

Official probe:

```powershell
python -m experiments.cep_cc.run_experiment --c3d-motion-decoupling --seeds 0,1,2 --nuisance-modes none --stage1-episodes 600 --curriculum-episodes 300 --stage2-episodes 120 --batch-size 128 --lr 0.005 --stage2-lr 0.001 --lambda-comm 0.10 --factor-names motion --factor-consistency-sweep 0.03 --audit-batches 12 --table
```

Task accuracy:

| seed | accuracy |
|---|---:|
| 0 | `0.562` |
| 1 | `0.568` |
| 2 | `0.709` |

Per-factor targeted ratios:

| seed | q | motion | relation |
|---|---:|---:|---:|
| 0 | `1.278` | `0.487` | `1.519` |
| 1 | `1.309` | `0.480` | `1.537` |
| 2 | `1.368` | `0.473` | `1.779` |

Environment dependency diagnostic:

| ablation | target agreement with full rule |
|---|---:|
| no q | `0.660` |
| no motion | `0.479` |
| no relation | `0.825` |

Decision:

- V0.3d is a strong negative/informative result.
- motion is genuinely task-critical after decoupling.
- motion still does not become a locally replaceable communication segment.
- relation remains early; q remains weak but stable; motion appears entangled in the decision trajectory.

Refined claim:

> Task-critical factors do not automatically become locally compositional communication segments under the current CEP-CC pressure set.

Next:

- C3e: Motion Readout And Temporal-Position Audit.
- audit whether motion is present as distributed information rather than segment-local control.
- add motion/relation probes across early/middle/late communication segments before changing the model or adding symbol machinery.

## C3e: Motion Readout And Temporal-Position Audit

Status: completed as V0.3e.

Question:

> If motion is task-critical but not segment-local, can it be decoded from the full communication trajectory or from any early/middle/late segment?

Implementation:

- added full-trajectory probes for pos/q/motion/relation factors.
- added early/middle/late segment probes for pos/q/motion/relation factors.
- added factor specialization gaps.
- added `run_c3e_motion_readout_audit`.
- added CLI flag `--c3e-motion-readout`.
- protocol tests pass: `31 passed`.

Official probe:

```powershell
python -m experiments.cep_cc.run_experiment --c3e-motion-readout --seeds 0,1,2 --nuisance-modes none --stage1-episodes 600 --curriculum-episodes 300 --stage2-episodes 120 --batch-size 128 --lr 0.005 --stage2-lr 0.001 --lambda-comm 0.10 --factor-names motion --factor-consistency-sweep 0.03 --audit-batches 12 --table
```

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

Late segment action effect:

| seed | late swap action change | late ablation action change |
|---|---:|---:|
| 0 | `0.521` | `0.520` |
| 1 | `0.490` | `0.441` |
| 2 | `0.523` | `0.334` |

Decision:

- V0.3e is a negative but important diagnostic.
- communication is necessary.
- late segment strongly controls action.
- motion is not readable from the full communication trajectory by linear probe.
- motion is not readable from early/middle/late segments.
- current protocol carries an entangled target/action decision manifold rather than factorized motion semantics.

Refined claim:

> CEP-CC currently supports task coordination and partial q/relation roles, but not full factorized compositional semantics.

Next:

- C3f: Factorized-Pressure Objective.
- compare q-only, motion-only, q+motion, and q+motion+relation continuous consistency pressure.
- do not add discrete codebooks or learned symbol classifiers yet.

## C4: V0.4 Protocol As Cognitive Necessity

Status: pending.

Only start after C3.

Tasks:

- memory externalization pressure.
- delayed evidence tasks.
- multi-view complementarity.
- value coordination.
