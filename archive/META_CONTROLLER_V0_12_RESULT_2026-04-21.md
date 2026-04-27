# Meta-Controller V0.12 Result - 2026-04-21

## Status

V0.12 validates the B3 mainline with direct acceptance checks and signal masking.

Mainline controller:

- `planner_necessity_loose_controller`

New ablations:

- `b3_mask_surprise`
- `b3_mask_memory`
- `b3_mask_cost`
- `b3_mask_conflict`
- `b3_mask_drift`
- `b3_mask_core_signals`

Validation:

```powershell
python -m pytest F:\unified-sel\tests\test_meta_controller_protocol.py -q -p no:cacheprovider
```

Result:

- `5 passed`

Evaluation:

```powershell
python -m experiments.meta_controller.run_experiment --profile v01 --mode train-eval --train-episodes 240 --eval-episodes 60 --seed 0 --table
```

## Mainline Result

| controller | task_success | total_reward | planner_calls | memory_reads | drift |
|---|---:|---:|---:|---:|---:|
| fixed_rule_controller | 1.000 | 68.063 | 54.000 | 7.667 | 0.000 |
| planner_necessity_loose_controller | 1.000 | 68.967 | 49.917 | 7.333 | 0.000 |

Direct checks:

- `b3_mainline_success_ge_099`: pass
- `b3_mainline_drift_guard`: pass
- `b3_mainline_reads_near_fixed`: pass
- `planner_necessity_loose_planner_below_fixed`: pass

## Signal Masking

| ablation | task_success | total_reward | planner_calls | memory_reads | drift |
|---|---:|---:|---:|---:|---:|
| planner_necessity_loose_controller | 1.000 | 68.967 | 49.917 | 7.333 | 0.000 |
| b3_mask_surprise | 0.966 | 64.285 | 47.283 | 9.050 | 0.037 |
| b3_mask_memory | 0.947 | 62.898 | 46.400 | 0.000 | 0.060 |
| b3_mask_cost | 0.977 | 67.509 | 39.750 | 7.183 | 0.017 |
| b3_mask_conflict | 0.908 | 65.148 | 0.833 | 7.333 | 0.090 |
| b3_mask_drift | 1.000 | 68.877 | 50.367 | 7.333 | 0.000 |
| b3_mask_core_signals | 0.898 | 65.038 | 0.000 | 0.500 | 0.114 |

## Interpretation

V0.12 strengthens the B3 claim.

The B3 mainline is not just a cosmetic orchestrator:

- masking core signals collapses success from `1.000` to `0.898`
- masking conflict nearly eliminates planner use and drops success to `0.908`
- masking memory removes required retrieval and drops success to `0.947`
- masking surprise also degrades success and drift
- masking drift alone does not hurt this environment

This identifies the current causal control signals:

- `conflict_score` is the strongest dominance signal.
- `memory_relevance` is essential for delayed memory queries.
- `surprise` contributes to recovery and robustness.
- `invariant_violation` is currently redundant because drift is indirectly controlled by conflict/recovery.

## Branch Decision

B3 remains the mainline.

Current claim:

> A two-gate controller, combining habit-safe initiation with planner-necessity estimation, forms a measurable control law that reduces planner use while preserving success, memory behavior, and drift.

Next step:

- V0.13 should test transfer, not add a new controller.

Recommended transfer checks:

- heldout seeds with altered regime shifts
- read-cost stress on seeds 1 and 2
- optionally a profile with longer horizon or denser memory queries

Do not integrate old F-drive projects yet.

