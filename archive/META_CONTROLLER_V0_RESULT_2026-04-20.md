# Meta-Controller V0 Result - 2026-04-20

## Status

V0 implementation is runnable and self-contained.

This is an early engineering result, not a final research claim. It shows that the experiment spine can measure arbitration, memory control, compute cost, drift, and signal ablations without importing other F-drive projects.

## Command

```powershell
python -m experiments.meta_controller.run_experiment --mode train-eval --train-episodes 120 --eval-episodes 30 --seed 0 --table
```

## Summary

The learned contextual-bandit meta-controller passes the first coarse acceptance checks:

- success above `habit_only`
- compute cost below `planner_always`
- memory reads below `memory_always`
- arbitration regret below `random_controller`
- total reward within 1.0 of `fixed_rule_controller`

The strongest falsification signal is `learned_mask_core_signals`: masking surprise, memory, cost, conflict, and drift together collapses the learned controller back near habit-only behavior.

## Key Result Table

| controller | task_success | total_reward | arbitration_regret | switch_latency | compute_cost | planner_calls | memory_reads | memory_writes | memory_read_precision | memory_write_precision | drift_under_horizon |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| habit_only | 0.930 | 69.765 | 0.136 | nan | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.075 |
| planner_always | 0.964 | 58.791 | 0.265 | 0.000 | 16.000 | 80.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.049 |
| memory_always | 1.000 | 57.600 | 0.280 | 0.000 | 22.400 | 80.000 | 80.000 | 1.000 | 0.071 | 1.000 | 0.000 |
| fixed_rule_controller | 1.000 | 70.913 | 0.114 | 5.000 | 9.087 | 42.667 | 5.667 | 1.000 | 1.000 | 1.000 | 0.000 |
| random_controller | 0.935 | 56.958 | 0.289 | 0.656 | 13.349 | 46.133 | 34.100 | 0.433 | 0.030 | 0.433 | 0.064 |
| learned_contextual_bandit | 1.000 | 70.565 | 0.118 | 2.111 | 9.435 | 38.433 | 25.100 | 1.000 | 0.089 | 1.000 | 0.000 |
| learned_mask_core_signals | 0.929 | 69.504 | 0.140 | nan | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.075 |

## Acceptance Checks

| check | pass | learned | baseline |
|---|---:|---:|---:|
| learned_success_beats_habit | true | 1.000 | 0.930 |
| learned_compute_below_planner_always | true | 9.435 | 16.000 |
| learned_reads_below_memory_always | true | 25.100 | 80.000 |
| learned_regret_below_random | true | 0.118 | 0.289 |
| learned_near_fixed_rule_reward | true | 70.565 | 70.913 |

## Interpretation

What V0 supports:

- The environment can force habit, planning, memory read/write, and cost tradeoffs into the same loop.
- The learned controller can reach fixed-rule-level reward while using less planner than planner-always and less memory than memory-always.
- The learned controller is not just random arbitration: regret and reward beat random strongly.
- Core signal masking breaks the learned policy back toward habit-only, so the measured state signals matter as a group.

What V0 does not yet prove:

- The learned policy does not beat the best fixed rule.
- Individual signal ablations are weak; several single-signal masks do not degrade performance.
- Memory read precision is low because learned policy over-reads after writing the clue.
- The environment is still simple and symbolic.

## Next Engineering Step

V0.1 should make the falsification harder:

1. Add delayed or noisy clues so memory write timing matters more.
2. Add a stricter memory-read cost to punish over-reading.
3. Add a held-out environment with more than two regimes.
4. Add an oracle arbitration baseline for cleaner regret.
5. Add action-distribution logging so learned policy changes can be inspected.

Do not integrate F-drive donor projects until these V0.1 pressures are in place.
