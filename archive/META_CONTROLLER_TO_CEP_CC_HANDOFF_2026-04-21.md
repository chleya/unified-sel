# Meta-Controller To CEP-CC Handoff - 2026-04-21

## Decision

Open a new project for CEP-CC.

Reason:

- The meta-controller/B7 line is now an evidence-packaging line.
- CEP-CC is a different research question: continuous multi-agent protocol emergence without tokens, conditional probability modeling, or next-token prediction.
- Keeping CEP-CC inside the B7 ledger would mix a controller-repair claim with an emergent-protocol claim.

## Current Meta-Controller State

The B7 branch is stable enough to pause as a frozen research artifact.

Frozen claim:

> B7 is an invariant-coverage repair gate. On drift-pressure profiles, it converts high-drift exposure into planner repair, reducing drift debt and high-drift no-repair exposure. It can cost short-horizon reward, so the correct objective is invariant coverage or constrained value, not raw reward maximization.

Completed handoff artifacts:

- `META_CONTROLLER_V0_25_B7_CLAIM_EVIDENCE_MAP_2026-04-21.md`
- `META_CONTROLLER_V0_26_B7_REPORT_SECTION_AND_FIGURE_TABLE_PACK_2026-04-21.md`

Relevant prior evidence:

- `META_CONTROLLER_V0_18_DRIFT_THRESHOLD_SWEEP_2026-04-21.md`
- `META_CONTROLLER_V0_19_V03B_DRIFT_VARIANT_TRANSFER_2026-04-21.md`
- `META_CONTROLLER_V0_20_REPAIR_RESIDUAL_INSTRUMENTATION_2026-04-21.md`
- `META_CONTROLLER_V0_21_REPAIR_BENEFIT_COUNTERFACTUALS_2026-04-21.md`
- `META_CONTROLLER_V0_22_MULTI_STEP_REPAIR_BENEFIT_2026-04-21.md`
- `META_CONTROLLER_V0_23_B7_TRANSFER_MATRIX_2026-04-21.md`
- `META_CONTROLLER_V0_24_B7_ACCEPTANCE_ARTIFACT_2026-04-21.md`

Next meta-controller task if resumed:

- W23: B7 visual artifact generation.
- Convert V0.26 CSV-style data blocks into actual plots/tables.
- Do not add new controller logic unless report writing reveals a missing measurement.

## New Project

Project name:

- CEP-CC: Compressed Emergent Protocol under Continuous Communication

Location:

- `experiments/cep_cc`

Primary research question:

> In a continuous dynamical multi-agent system, do compression pressure, task coordination pressure, and temporal/memory pressure induce discrete, reusable, compositional protocol structure without tokenization or next-token prediction?

The new project should treat B7 as background motivation only. It should not reuse B7 acceptance claims as evidence for CEP-CC.

## Boundary Rules

CEP-CC must not assume:

- token vocabulary
- next-token prediction
- language data
- softmax over symbols
- manually assigned discrete labels as communication input

CEP-CC may use:

- continuous neural dynamics as function approximators
- continuous object states
- continuous communication trajectories
- task losses
- communication and state compression penalties
- post-hoc clustering/probing for analysis

## First Milestone

V0.1 should answer only this:

> Under communication compression, do continuous communication trajectories form stable reusable proto-symbol partitions while preserving task performance?

Do not start with full compositionality. First prove or falsify proto-symbol emergence.

