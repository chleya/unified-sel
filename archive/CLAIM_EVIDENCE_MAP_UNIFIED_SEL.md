# CLAIM_EVIDENCE_MAP_UNIFIED_SEL

## Goal

Map each paper claim to the concrete evidence already present in the repository.

## Claim 1

Claim:

- shared readout drift is a real source of forgetting in modular continual learners

Evidence:

- [COMPREHENSIVE_ANALYSIS_REPORT.md](F:/unified-sel/results/COMPREHENSIVE_ANALYSIS_REPORT.md)
- [20260409_110330.json](F:/unified-sel/results/continual_no_boundary/20260409_110330.json)
- route-cleanup notes in [EXPERIMENT_LOG.md](F:/unified-sel/EXPERIMENT_LOG.md)

Support type:

- mechanism interpretation
- benchmark improvement after explicit `W_out` protection

Risk:

- exact causal wording must stay careful because the benchmark is small

Safe wording:

- "shared readout drift acts as a meaningful and partly independent source of forgetting"

## Claim 2

Claim:

- on the cleaned route, light `W_out` protection beats EWC on mean forgetting and mean average accuracy

Evidence:

- [20260409_110330.json](F:/unified-sel/results/continual_no_boundary/20260409_110330.json)
- [20260405_213122_multi_seed.json](F:/unified-sel/results/baseline_ewc/20260405_213122_multi_seed.json)
- status summary in [STATUS.md](F:/unified-sel/STATUS.md)

Support type:

- direct benchmark result

Risk:

- no strong statistical support at `n=5`

Safe wording:

- "beats EWC on means under the current cleaned-route benchmark"

## Claim 3

Claim:

- after route cleanup, the dominant residual problem is seed-level control instability rather than Fisher strength alone

Evidence:

- [CLEANED_ROUTE_VARIANCE_DIAGNOSIS_2026-04-09.md](F:/unified-sel/results/CLEANED_ROUTE_VARIANCE_DIAGNOSIS_2026-04-09.md)
- `lambda=10` vs `lambda=20` runs

Support type:

- comparative diagnosis

Risk:

- this is about the current benchmark and current controller family only

Safe wording:

- "within the cleaned-route setup, residual variance is better explained by pool-control behavior than by coarse Fisher strength changes"

## Claim 4

Claim:

- output isolation is a real adaptation lever

Evidence:

- [SELECTIVE_READOUT_PROBES_2026-04-09.md](F:/unified-sel/results/SELECTIVE_READOUT_PROBES_2026-04-09.md)
- conservative gated local result:
  - [20260409_122332.json](F:/unified-sel/results/continual_no_boundary/20260409_122332.json)

Support type:

- probe result

Risk:

- should not be overclaimed as a solution

Safe wording:

- "local readout capacity can materially reopen late-task adaptation"

## Claim 5

Claim:

- simple output isolation is not a sufficient fix

Evidence:

- exclusive local negative result:
  - [20260409_133043.json](F:/unified-sel/results/continual_no_boundary/20260409_133043.json)
- strict event-window near-no-op control:
  - [20260409_133658.json](F:/unified-sel/results/continual_no_boundary/20260409_133658.json)
- pressure-window middle-regime probe:
  - [20260409_135433.json](F:/unified-sel/results/continual_no_boundary/20260409_135433.json)

Support type:

- negative ablations
- tradeoff evidence

Risk:

- needs careful phrasing so it does not sound universal

Safe wording:

- "in this system, neither full isolation nor overly strict gating yields a satisfactory retention-adaptation balance"

## Claim 6

Claim:

- the remaining failures are joint failures of shared-head interference and routing / specialization dynamics

Evidence:

- exclusive local still fails
- variance diagnosis identifies over-stabilization vs churn-without-recovery
- pressure-window gate improves mechanism cleanliness but not enough to win

Files:

- [20260409_133043.json](F:/unified-sel/results/continual_no_boundary/20260409_133043.json)
- [CLEANED_ROUTE_VARIANCE_DIAGNOSIS_2026-04-09.md](F:/unified-sel/results/CLEANED_ROUTE_VARIANCE_DIAGNOSIS_2026-04-09.md)
- [20260409_135433.json](F:/unified-sel/results/continual_no_boundary/20260409_135433.json)

Support type:

- synthesis across ablations

Risk:

- this is the most interpretive claim in the paper

Safe wording:

- "the ablations suggest that retention failure is jointly shaped by shared-readout interference and pool-level specialization behavior"

## Claim 7

Claim:

- this project should currently be framed as a mechanism study, not as a final reasoning architecture

Evidence:

- [PROJECT_MAINLINE_2026-04-09.md](F:/unified-sel/PROJECT_MAINLINE_2026-04-09.md)
- [MAINLINE_DECISION_NOTE_2026-04-09.md](F:/unified-sel/results/MAINLINE_DECISION_NOTE_2026-04-09.md)

Support type:

- scope decision

Use:

- discussion / limitations section

## Not-To-Claim List

1. Do not claim statistical superiority over EWC yet.
2. Do not claim the selective-readout line solved continual learning.
3. Do not claim `StructurePool` is a viable general reasoning engine from these results.
4. Do not claim the negative ablations generalize beyond this setup.

## Repository Anchor Set

If only a small set of files is cited in drafting, use these:

1. [20260409_110330.json](F:/unified-sel/results/continual_no_boundary/20260409_110330.json)
2. [20260405_213122_multi_seed.json](F:/unified-sel/results/baseline_ewc/20260405_213122_multi_seed.json)
3. [CLEANED_ROUTE_VARIANCE_DIAGNOSIS_2026-04-09.md](F:/unified-sel/results/CLEANED_ROUTE_VARIANCE_DIAGNOSIS_2026-04-09.md)
4. [SELECTIVE_READOUT_PROBES_2026-04-09.md](F:/unified-sel/results/SELECTIVE_READOUT_PROBES_2026-04-09.md)
5. [PROJECT_MAINLINE_2026-04-09.md](F:/unified-sel/PROJECT_MAINLINE_2026-04-09.md)
