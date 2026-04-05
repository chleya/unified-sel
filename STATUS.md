# STATUS.md - Current Progress

**Last Updated**: 2026-04-03
**Current Phase**: Phase 6 - Endogenous Boundary Diagnostics / Research Positioning

---

## Current Task

Re-center the project around endogenous task-boundary formation and evaluate interventions by whether they improve boundary emergence and stability, not only headline baseline metrics.

Current best-known setting:
- `SURPRISE_THRESHOLD = 0.60`
- `TENSION_THRESHOLD = 0.08`
- `UTILITY_DECAY = 0.005`
- `UTILITY_PRUNE = 0.08`
- `MATURE_AGE = 80`
- `MATURE_DECAY_SCALE = 0.35`
- `max_structures = 12`

Current best-known comparison:
- Fixed avg accuracy: `0.4844`
- EWC avg accuracy: `0.5020`
- Unified-SEL avg accuracy: `0.5113`
- Fixed forgetting: `0.8320`
- EWC forgetting: `0.0781`
- Unified-SEL forgetting: `0.1148`

Primary research target:
- produce a continual-learning system that can form, maintain, and update task-boundary structure from internal signals alone
- do this without explicit task labels or external boundary markers
- show that the resulting endogenous-boundary dynamics explain retention and transfer behavior

Validation targets:
- keep a competitive avg-accuracy profile against EWC
- reduce forgetting relative to the current best-known Unified-SEL result
- strengthen statistical evidence and benchmark discipline over time

New infrastructure now in place:
- `experiments/continual/no_boundary.py` records step traces, window summaries, checkpoint metrics, and config snapshots
- `analysis/compare.py` accepts explicit paths and reports sample-level summaries plus bootstrap delta intervals
- `core/experiment_config.py` is now wired into the no-boundary experiment path
- `analysis/boundary_diagnostics.py` extracts phase summaries, top boundary-pressure windows, candidate endogenous-boundary steps, and new boundary metrics for emergence / collapse / recurrence

Acceptance criteria for the next round:
- make endogenous boundary formation easier to observe, quantify, or stabilize
- improve the boundary-pressure mechanism itself, not only mature-structure retention
- avoid interventions that merely trade `task_0` retention against `task_1` adaptation
- treat beating EWC as a validation check, not the project definition

---

## Best References

Best run so far:
- [continual_no_boundary retention-aware run](F:\unified-sel\results\continual_no_boundary\20260403_135552.json)
- [analysis_compare retention-aware result](F:\unified-sel\results\analysis_compare\20260403_135605.json)

Latest verification after Track A/B/C integration:
- [continual_no_boundary diagnostics verification](F:\unified-sel\results\continual_no_boundary\20260403_133309.json)
- [analysis_compare explicit-path verification](F:\unified-sel\results\analysis_compare\20260403_133309.json)
- [continual_no_boundary 5-seed diagnostic rerun](F:\unified-sel\results\continual_no_boundary\20260403_134535.json)
- [boundary diagnostics 5-seed report](F:\unified-sel\results\analysis_boundary\20260403_134542.json)
- [boundary diagnostics retention-aware report](F:\unified-sel\results\analysis_boundary\20260403_135604.json)

---

## Known Issues

- Unified-SEL still does not beat EWC on forgetting.
- Baseline evidence is still asymmetric: Unified-SEL has 5 seeds, current Fixed/EWC baseline files are single-result summaries.
- The repo is now reproducible enough to diagnose behavior, but still not a full feature-aligned reproduction of the source projects.
- Boundary formation does happen, but the current mechanism does not reliably stabilize after the mid-stream regime shift.
- High-forgetting seeds show stronger sustained mid/late tension and surprise pressure than low-forgetting seeds.
- The new mature-structure retention term improves forgetting, but it does not materially change the boundary-pressure profile.
- A direct weakest-structure replacement rule under full-capacity pressure was tested and reverted; it sharply worsened forgetting and destabilized task-0 retention.
- A soft full-capacity competition rebalance was tested and reverted; it improved forgetting slightly but reduced avg accuracy by shifting performance away from task 1 instead of resolving pressure cleanly.
- The project can already screen baseline-level tradeoffs, but it still lacks a crisp quantitative account of when an endogenous boundary has actually formed and when it has stabilized.
- Under the current best-known run, the new boundary metrics classify `4/5` seeds as `recurrent_pressure` and only `1/5` as `transient`; no seed currently qualifies as stable endogenous-boundary behavior.
- A shared-output protection gate under full-capacity high pressure was tested and reverted; it beat EWC on forgetting but did so mainly by suppressing task-1 adaptation, and still produced no stable boundary seeds.
- A balanced pressure-routing attempt that shifted error away from structure updates and toward shared output updates was tested and reverted; it improved task-1 adaptation in some seeds but worsened forgetting and still produced no stable boundary seeds.

---

## Next Recommended Work

1. Upgrade diagnostics so each run can say when a boundary appears, whether it persists, and whether it predicts later retention or transfer.
2. Design pressure-resolution mechanisms that change shared learning dynamics under drift, but reject mechanisms that merely flip the task bias between structure-heavy retention and output-heavy adaptation.
3. Add multi-seed baseline generation so EWC remains a useful validation target instead of an under-sampled reference point.
