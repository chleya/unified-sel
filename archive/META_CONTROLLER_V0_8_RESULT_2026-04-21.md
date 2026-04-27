# Meta-Controller V0.8 Result - 2026-04-21

## Status

V0.8 implements Branch B1: shielded dominance control.

New controllers:

- `shielded_dominance_controller`
- `shielded_relaxed_dominance_controller`

New metrics:

- `shield_interventions`
- `shield_intervention_rate`

Design:

- read/write gates stay inherited from imitation/counterfactual dominance.
- learned dominance may propose `habit`.
- the shield overrides unsafe `habit` proposals back to `planner`.

Strict shield predicates:

- recent failure
- drift above threshold
- surprise above threshold
- conflict above threshold
- later memory-relevant state

Relaxed shield:

- keeps memory and drift protection,
- relaxes surprise/conflict thresholds to test the safe boundary.

## Validation

```powershell
python -m pytest F:\unified-sel\tests\test_meta_controller_protocol.py -q -p no:cacheprovider
```

Result:

- `5 passed`

Main run:

```powershell
python -m experiments.meta_controller.run_experiment --profile v01 --mode train-eval --train-episodes 240 --eval-episodes 60 --seed 0 --table
```

Cross-check:

```powershell
python -m experiments.meta_controller.run_experiment --profile v01 --mode train-eval --train-episodes 240 --eval-episodes 60 --seed 1 --table
```

## Seed 0 Key Result

| controller | task_success | total_reward | planner_calls | memory_reads | drift | shield_rate |
|---|---:|---:|---:|---:|---:|---:|
| fixed_rule_controller | 1.000 | 68.063 | 54.000 | 7.667 | 0.000 | 0.000 |
| counterfactual_dominance_controller | 0.970 | 65.917 | 42.850 | 7.667 | 0.029 | 0.000 |
| rollout_dominance_controller | 0.952 | 66.925 | 24.550 | 7.333 | 0.051 | 0.000 |
| shielded_dominance_controller | 1.000 | 68.110 | 54.000 | 7.333 | 0.000 | 0.101 |
| shielded_relaxed_dominance_controller | 1.000 | 68.303 | 53.000 | 7.667 | 0.000 | 0.088 |

## Seed 1 Cross-Check

| controller | task_success | total_reward | planner_calls | memory_reads | drift | shield_rate |
|---|---:|---:|---:|---:|---:|---:|
| fixed_rule_controller | 1.000 | 68.063 | 54.000 | 7.667 | 0.000 | 0.000 |
| counterfactual_dominance_controller | 0.969 | 65.605 | 43.183 | 7.667 | 0.031 | 0.000 |
| rollout_dominance_controller | 0.933 | 66.367 | 13.650 | 7.333 | 0.070 | 0.000 |
| shielded_dominance_controller | 1.000 | 68.063 | 54.000 | 7.667 | 0.000 | 0.101 |
| shielded_relaxed_dominance_controller | 1.000 | 68.303 | 53.000 | 7.667 | 0.000 | 0.093 |

## Acceptance

Strict shield:

- passes success guard
- passes drift guard
- passes memory-read guard
- fails planner reduction: `54.000 == fixed 54.000`

Relaxed shield:

- passes success guard
- passes drift guard
- passes memory-read guard
- passes planner reduction formally: `53.000 < fixed 54.000`

## Interpretation

B1 is a partial positive result.

The shield solves the V0.6/V0.7 failure mode: unconstrained dominance learning no longer saves compute by sacrificing success and drift. This establishes that dominance arbitration should be safety-constrained.

But the gain is too small:

- strict shield collapses to fixed-rule planner usage.
- relaxed shield saves only one planner call per episode on two seeds.
- the branch is still mostly predicate-shaped, not a learned broad safe-control law.

The result is useful because it narrows the next problem:

> The safe regions where habit can replace planner are small or not expressed cleanly by the current shield predicates.

## Branch Decision

B1 should remain as a safety baseline, but it is not enough as the main learning mechanism.

Next branch:

- B2 `HabitSafeSetController`

Reason:

- B1 asks "when must planner override habit?"
- B2 asks the sharper question: "where is habit safe to initiate?"

That is the right next diagnostic because it can show whether the current workspace state contains learnable habit-safe regions beyond the fixed rule.

