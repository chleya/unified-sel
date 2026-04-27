# PAPER_FIGURE_PLAN_UNIFIED_SEL

## Goal

Define the minimum figure and table set needed to tell the `Unified-SEL` mechanism story cleanly.

## Figure 1: System Diagram

Content:

- input stream
- `StructurePool`
- pooled hidden route
- shared `W_out`
- optional local residual head
- pressure / stabilization events

Purpose:

- show where structure isolation exists
- show why a shared readout can still be a bottleneck

## Figure 2: Main Benchmark Comparison

Content:

- EWC vs cleaned-route `lambda=10`
- plot mean average accuracy and mean forgetting

Data:

- [20260409_110330.json](F:/unified-sel/results/continual_no_boundary/20260409_110330.json)
- [20260405_213122_multi_seed.json](F:/unified-sel/results/baseline_ewc/20260405_213122_multi_seed.json)

Purpose:

- establish that `W_out` protection matters at benchmark level

## Figure 3: Seed-Level Failure Mode Plot

Content options:

- per-seed forgetting vs task-1 scatter
- annotate seeds `8`, `9`, `11`
- optionally color by late stabilization count or churn count

Data:

- [CLEANED_ROUTE_VARIANCE_DIAGNOSIS_2026-04-09.md](F:/unified-sel/results/CLEANED_ROUTE_VARIANCE_DIAGNOSIS_2026-04-09.md)
- [20260409_110330.json](F:/unified-sel/results/continual_no_boundary/20260409_110330.json)

Purpose:

- show that variance is structured, not random noise

## Figure 4: Selective-Readout Ablation Ladder

Content:

- bar plot or connected plot across:
  - controller reference
  - conservative gated local head
  - exclusive local
  - strict event-window gate
  - pressure-window gate

Metrics:

- forgetting
- task-1 final accuracy

Data:

- [20260409_114650.json](F:/unified-sel/results/continual_no_boundary/20260409_114650.json)
- [20260409_122332.json](F:/unified-sel/results/continual_no_boundary/20260409_122332.json)
- [20260409_133043.json](F:/unified-sel/results/continual_no_boundary/20260409_133043.json)
- [20260409_133658.json](F:/unified-sel/results/continual_no_boundary/20260409_133658.json)
- [20260409_135433.json](F:/unified-sel/results/continual_no_boundary/20260409_135433.json)

Purpose:

- make the mechanism tradeoff visually obvious

## Figure 5: Activity vs Effect For Local Readout

Content:

- x-axis:
  - recent local gate rate
- y-axis:
  - forgetting or task-1 final accuracy
- one point per probe

Purpose:

- show:
  - strict gate is too inert
  - constant-on / exclusive are too aggressive
  - pressure-window sits in the only meaningful middle region

## Table 1: Main Benchmark Numbers

Rows:

- EWC
- cleaned-route `lambda=10`
- cleaned-route `lambda=20`

Columns:

- task 0 final
- task 1 final
- forgetting
- average accuracy
- seed count

## Table 2: Mechanism Ablations

Rows:

- controller reference
- conservative gated local
- exclusive local
- strict event-window gate
- pressure-window gate

Columns:

- task 0 final
- task 1 final
- forgetting
- local gate rate
- interpretation

## Optional Appendix Figure A1

Content:

- route mismatch diagnostic explanation
- why cleaned-route results are cleaner than pre-cleanup runs

Use only if space allows.

## Production Order

1. Table 1
2. Figure 4
3. Figure 3
4. Figure 1
5. Figure 2
6. Figure 5

This order maximizes writing velocity because it starts from already-known result files.
