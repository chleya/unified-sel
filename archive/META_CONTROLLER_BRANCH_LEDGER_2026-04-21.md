# Meta-Controller Branch Ledger - 2026-04-21

## Purpose

This ledger records the active exploration branches for the learnable unified meta-controller experiment.

The current git branch is `master`, but the branches below are research/engineering routes, not Git branches. Each branch should produce either:

- a measurable improvement,
- a clean negative result,
- or a reason to split/retire the branch.

## Current Mainline

Main claim under test:

> A learnable meta-controller can form a stable control law for when to use habit, planner, memory read, and memory write under partial observability, regime shifts, delayed memory requirements, and compute costs.

Current experiment spine:

- project: `F:\unified-sel`
- code: `experiments/meta_controller`
- protocol: `META_CONTROLLER_EXPERIMENT_PROTOCOL_2026-04-20.md`
- latest route audit: `META_CONTROLLER_ROUTE_AUDIT_2026-04-21.md`

Latest bottleneck:

- Memory read/write control is mostly tractable.
- Dominance arbitration is not solved.
- V0.6/V0.7 reduce planner calls but lose success and increase drift.

Current safety anchor:

- `fixed_rule_controller`
- `imitation_controller`

V0.8 target:

- reduce planner calls below fixed rule,
- preserve `task_success >= 0.99`,
- preserve `drift_under_horizon <= fixed + 0.005`,
- keep `memory_reads <= fixed + 1.0`.

## Branch B1: Shielded Dominance Control

Status: partial positive result in V0.8; keep as safety baseline, but do not stop here.

Hypothesis:

Learned dominance can propose cheaper habit use, but a runtime shield must override unsafe habit proposals back to planner.

Why this branch exists:

- V0.7 showed rollout dominance is too cost-seeking.
- The missing mechanism is not another reward term; it is a safety guard around dominance changes.

Minimal implementation:

- Add `ShieldedDominanceController`.
- Freeze read/write gates from imitation.
- Let learned dominance propose `habit` or `planner`.
- If the proposal is `habit` and any safety predicate fires, override to `planner`.
- Record `shield_interventions`.

Initial shield predicates:

- `recent_failure_source > 0`
- `invariant_violation > threshold`
- `surprise > threshold`
- `conflict_score > threshold`
- memory-relevant later step without safe resolution

Metrics to add:

- `shield_interventions`
- `shield_intervention_rate`
- optional later: `false_safe_rate`, `false_planner_rate`

Acceptance:

- success remains at least `0.99`
- drift stays within guard band
- memory reads remain near fixed rule
- planner calls are below fixed rule
- shield intervention rate is explainable, not constant fallback

Falsification:

- If it preserves success only by calling planner as often as fixed rule, B1 is just fixed rule with extra machinery.
- If planner calls drop but success/drift fail, the shield predicates are insufficient.

Result:

- `shielded_dominance_controller` preserves success and drift but does not reduce planner calls.
- `shielded_relaxed_dominance_controller` preserves success and drift while reducing planner calls from `54.000` to `53.000` on seeds 0 and 1.

Interpretation:

- B1 proves that shielding fixes the unsafe compute-saving collapse from V0.6/V0.7.
- The planner reduction is too small to count as a strong unified-control result.
- The branch remains useful as a safety anchor for future learned dominance.

Result document:

- `META_CONTROLLER_V0_8_RESULT_2026-04-21.md`

Next action:

- Start B2 `HabitSafeSetController`.

## Branch B2: Habit Safe Set

Status: partial positive result in V0.9; continue with calibration.

Hypothesis:

The right abstraction is not "when planner is valuable", but "where habit is safe to initiate".

Minimal implementation:

- Train a binary `habit_safe(g_t)` classifier.
- Positive label: habit succeeds for a short window without drift/recovery degradation.
- Negative label: habit causes failure, drift increase, delayed recovery, or memory miss.
- Use planner outside the safe set.

Acceptance:

- safe set reduces planner calls while preserving fixed-rule success.
- safe set generalizes across seeds and read-cost variants.

Falsification:

- If the safe set mostly reproduces the fixed rule, there is not enough learnable structure in current `g_t`.
- If it over-expands and loses success, the labels/horizon are too weak.

Trigger:

- Started after B1 proved safe but planner-heavy.

Result:

- `habit_safe_set_controller` preserved success/drift and reduced planner calls on seeds 0 and 2.
- It was too conservative on seed 1.
- `habit_safe_set_h2_controller` was more stable but only matched the relaxed shield improvement.

Interpretation:

- The workspace state contains learnable habit-safe regions.
- The current safe-set boundary is seed-sensitive.
- The next issue is calibration and label diagnostics, not a new module.

Result document:

- `META_CONTROLLER_V0_9_RESULT_2026-04-21.md`

Next action:

- Keep `habit_safe_set_h2_controller` as the stable B2 baseline.
- Move to B3 expected value of computation / planner necessity.

## Branch B3: Expected Value Of Computation

Status: defer until B1/B2 establish a safe dominance baseline.

Hypothesis:

Planner calls should be treated as computational actions whose value is uncertainty reduction, recovery protection, and future error avoidance, not just immediate reward difference.

Minimal implementation:

- Add an estimated `planner_necessity` or `expected_planner_value` feature to workspace state.
- Estimate it from:
  - uncertainty reduction,
  - recent surprise,
  - predicted habit error,
  - drift risk,
  - recovery debt,
  - planner cost.

Acceptance:

- improves planner selectivity without losing success.
- explains planner calls better than raw surprise/uncertainty thresholds.

Falsification:

- If EVOC score becomes a noisy duplicate of existing surprise/conflict, it is not worth keeping.

Trigger:

- Start after B1 or B2 if planner calls remain too high but safe.

## Branch B4: Option Initiation And Termination

Status: future branch, not next.

Hypothesis:

Habit and planner should be modeled as options with initiation and termination boundaries.

Minimal implementation:

- Learn habit initiation set.
- Learn planner termination set.
- Track whether mode persistence beats stepwise switching.

Acceptance:

- fewer incoherent switches,
- equal or better recovery,
- lower planner calls than fixed rule.

Falsification:

- If stepwise shielded dominance already solves the task, option machinery is unnecessary for V0.

Trigger:

- Start if B1/B2 produce chattering or unstable handoff.

## Branch B5: Workspace Feature Repair

Status: supporting branch, only activate if a controller fails for lack of state.

Hypothesis:

Current `g_t` may be missing explicit safety features:

- regime-shift hazard,
- recovery debt,
- habit-validity confidence,
- planner safety value,
- counterfactual uncertainty.

Minimal implementation:

- Add one feature at a time.
- Run ablations to prove the feature changes controller behavior.

Acceptance:

- a new feature improves one failure mode and survives signal masking.

Falsification:

- if performance does not change or only overfits one seed, remove it.

Trigger:

- Start only after a concrete failure diagnosis from B1/B2/B3.

## Branch B6: Old F-Drive Integration

Status: explicitly deferred.

Hypothesis:

Parts of old F-drive projects may help after V0 has a stable causal shape.

Reusable ideas already mapped:

- `cognitive-execution-engine`: event log, replay, policy/state skeleton.
- `sel-lab`: benchmark and ablation patterns.
- `SDAS`: adaptation and structure pool ideas.
- `diff_world`: delta predictor and surprise signal ideas.

Current decision:

- Do not import old projects into V0.8.
- Use them as read-only references until the V0 meta-controller is stable.

Trigger:

- Revisit after one branch passes the V0.8 acceptance target or produces a stable interface requirement.

## Exploration Order

1. B1 `ShieldedDominanceController`
2. B2 `HabitSafeSetController`
3. B3 expected value of computation feature
4. B4 option initiation/termination
5. B5 workspace feature repair, only when failure diagnosis requires it
6. B6 old F-drive integration, only after V0 interface stabilizes

## Current Work Queue

### W1: Implement B1

Status: completed as V0.8.

Files touched:

- `experiments/meta_controller/meta_controller.py`
- `experiments/meta_controller/baselines.py`
- `experiments/meta_controller/metrics.py`
- `experiments/meta_controller/report.py`
- `tests/test_meta_controller_protocol.py`

Validation:

```powershell
python -m pytest F:\unified-sel\tests\test_meta_controller_protocol.py -q -p no:cacheprovider
python -m experiments.meta_controller.run_experiment --profile v01 --mode train-eval --train-episodes 240 --eval-episodes 60 --seed 0 --table
python -m experiments.meta_controller.run_experiment --profile v01 --mode train-eval --train-episodes 240 --eval-episodes 60 --seed 1 --table
```

Result document:

- `META_CONTROLLER_V0_8_RESULT_2026-04-21.md`

### W2: Implement B2

Status: completed as V0.9.

Files touched:

- `experiments/meta_controller/meta_controller.py`
- `experiments/meta_controller/baselines.py`
- `experiments/meta_controller/run_experiment.py`
- `experiments/meta_controller/report.py`
- `tests/test_meta_controller_protocol.py`

Implemented:

- add `HabitSafeSetController`
- generate short-horizon labels for whether habit is safe from `g_t`
- choose habit only inside the learned safe set
- otherwise fall back to planner while preserving read/write gates

Result document:

- `META_CONTROLLER_V0_9_RESULT_2026-04-21.md`

### W3: B2.1 Calibrate Safe Set

Status: completed as V0.10.

Files likely touched:

- `experiments/meta_controller/meta_controller.py`
- `experiments/meta_controller/run_experiment.py`
- `experiments/meta_controller/metrics.py`
- `experiments/meta_controller/report.py`

Implemented:

- record safe-label positive/negative rates
- expose safe-score distribution or threshold margin
- add a calibrated threshold variant instead of using score `>= 0`
- compare seeds 0, 1, 2 and read-cost stress

Result document:

- `META_CONTROLLER_V0_10_RESULT_2026-04-21.md`

Outcome:

- simple global threshold calibration is too crude.
- loose threshold can collapse safety on seed 1.
- tight threshold overuses planner.
- h2 is stable but modest.

### W4: Implement B3

Status: completed as V0.11.

Files likely touched:

- `experiments/meta_controller/meta_controller.py`
- `experiments/meta_controller/run_experiment.py`
- `experiments/meta_controller/baselines.py`
- `experiments/meta_controller/report.py`

Implemented:

- add `PlannerNecessityController`
- train planner-necessity score from short-horizon planner-vs-habit value gap
- use habit only when:
  - habit safe set says safe
  - planner necessity is below threshold
- otherwise use planner

Result document:

- `META_CONTROLLER_V0_11_RESULT_2026-04-21.md`

Outcome:

- B3 is the first clearly positive dominance result.
- `planner_necessity_loose_controller` preserved success and drift across seeds 0, 1, 2.
- planner calls were reduced from fixed rule's `54.000` to about `49.8-49.9`.
- read-cost 0.20 stress on seed 0 also passed.

Current mainline controller:

- `planner_necessity_loose_controller`

### W5: Validate B3 Mainline

Status: completed as V0.12.

Files likely touched:

- `experiments/meta_controller/report.py`
- `experiments/meta_controller/run_experiment.py`
- `tests/test_meta_controller_protocol.py`

Implemented:

- add direct B3 acceptance checks
- run signal masking on B3, not only flat learned bandit
- identify causal control signals

Result document:

- `META_CONTROLLER_V0_12_RESULT_2026-04-21.md`

Outcome:

- B3 mainline passes direct checks against fixed rule.
- signal masking shows conflict, memory, and surprise are causally important.
- drift masking does not hurt in the current environment, so drift is redundant for now.

### W6: Transfer B3 Mainline

Status: completed as V0.13.

Files likely touched:

- `experiments/meta_controller/env.py`
- `experiments/meta_controller/run_experiment.py`
- `experiments/meta_controller/report.py`

Implemented:

- run read-cost stress on seeds 1 and 2
- add a transfer profile with altered regime shifts or denser memory queries
- compare fixed rule, h2 safe set, and B3 mainline

Result document:

- `META_CONTROLLER_V0_13_RESULT_2026-04-21.md`

Outcome:

- B3 transferred to v02.
- In v02, fixed rule dropped to `0.973` success while B3 mainline reached `1.000`.
- B3 reduced planner calls from `66.000` to `59.183`.
- read-cost `0.20` passed on seeds 0, 1, and 2.

### W7: Freeze B3 Interface

Status: completed as V0.14.

Files likely touched:

- new protocol/interface document
- `experiments/meta_controller/README.md`

Implemented:

- define the B3 causal-core interface
- specify workspace fields, labels, outputs, and metrics
- map old F-drive projects to the interface as candidates only
- do not import old code yet

Result document:

- `META_CONTROLLER_V0_14_INTERFACE_2026-04-21.md`

Outcome:

- V0 causal core interface is frozen.
- old F-drive projects remain reference-only until they implement the contract through local adapters.
- README now points to the mainline controller and interface contract.

### W8: Select First Adapter Candidate

Status: completed as adapter plan.

Recommended order:

1. `sel-lab` benchmark registry adapter
2. `cognitive-execution-engine` event/replay adapter
3. `diff_world` surprise predictor adapter
4. `SDAS-LLM` gated memory adapter

Minimal plan:

- pick one candidate
- write a design note only
- map input/output to the V0.14 contract
- list tests required before implementation

Result document:

- `META_CONTROLLER_ADAPTER_PLAN_2026-04-21.md`

Outcome:

- First adapter candidate selected: `sel-lab` benchmark registry adapter.
- Scope is observational: benchmark suites, controller groups, masks, and acceptance rules only.
- Direct imports, editable installs, and behavior-changing integration remain forbidden.
- Next implementation must create only a local adapter skeleton and tests.

### W9: Decide Integration

Status: completed as local adapter skeleton.

After adapter plan:

- If an old project cleanly matches the frozen interface, plan a narrow adapter.
- If not, keep V0 causal core standalone and continue benchmark hardening.

Recommended next action:

- implement `experiments/meta_controller/adapters/sel_lab_benchmark.py`
- add `tests/test_meta_controller_adapter_sel_lab.py`
- validate that generated run matrices match the existing manual commands

Implemented:

- `experiments/meta_controller/adapters/__init__.py`
- `experiments/meta_controller/adapters/sel_lab_benchmark.py`
- `tests/test_meta_controller_adapter_sel_lab.py`

Result document:

- `META_CONTROLLER_ADAPTER_SEL_LAB_RESULT_2026-04-21.md`

Validation:

```powershell
python -m py_compile F:\unified-sel\experiments\meta_controller\adapters\sel_lab_benchmark.py F:\unified-sel\tests\test_meta_controller_adapter_sel_lab.py
python -m pytest F:\unified-sel\tests\test_meta_controller_adapter_sel_lab.py -q -p no:cacheprovider
python -m pytest F:\unified-sel\tests\test_meta_controller_protocol.py F:\unified-sel\tests\test_meta_controller_adapter_sel_lab.py -q -p no:cacheprovider
```

Outcome:

- local suite matrix adapter is implemented.
- no direct `sel-lab` import is used.
- no controller, environment, or metric behavior was changed.
- combined tests pass: `12 passed`.

### W10: Choose Next Research Pressure

Status: suite CLI completed; next pressure profile pending.

Implemented:

- `run_experiment.py` now accepts:

```powershell
python -m experiments.meta_controller.run_experiment --suite mainline_acceptance --table
python -m experiments.meta_controller.run_experiment --suite transfer_v02 --table
python -m experiments.meta_controller.run_experiment --suite read_cost_stress --table
python -m experiments.meta_controller.run_experiment --suite b3_signal_masking --table
```

Validation:

```powershell
python -m py_compile F:\unified-sel\experiments\meta_controller\run_experiment.py F:\unified-sel\experiments\meta_controller\adapters\sel_lab_benchmark.py F:\unified-sel\tests\test_meta_controller_adapter_sel_lab.py
python -m pytest F:\unified-sel\tests\test_meta_controller_adapter_sel_lab.py -q -p no:cacheprovider
python -m pytest F:\unified-sel\tests\test_meta_controller_protocol.py F:\unified-sel\tests\test_meta_controller_adapter_sel_lab.py -q -p no:cacheprovider
python -m experiments.meta_controller.run_experiment --help
```

Outcome:

- adapter tests: `7 passed`.
- combined meta-controller tests: `13 passed`.
- CLI exposes suite execution.
- long suite sweeps were not launched in this pass.

### W11: Build Long-Horizon Drift Pressure

Status: completed as V0.15 pressure profile.

Implemented:

- `v03_train_configs`
- `v03_heldout_configs`
- invariant guard dynamics in `EnvConfig`
- CLI profile `--profile v03`
- suite `long_horizon_drift_v03`

Result document:

- `META_CONTROLLER_V0_15_V03_DRIFT_PRESSURE_2026-04-21.md`

Validation:

```powershell
python -m pytest F:\unified-sel\tests\test_meta_controller_adapter_sel_lab.py F:\unified-sel\tests\test_meta_controller_protocol.py -q -p no:cacheprovider
```

Outcome:

- combined tests: `15 passed`.
- quick v03 run completed with `train_episodes=12`, `eval_episodes=6`, `seed=6`.
- v03 separates short-term success from long-horizon drift.
- B3 is not yet solved under v03; drift masking is now more informative but the mainline still needs a drift-aware branch.

### W12: Drift-Aware Planner Necessity

Status: completed as V0.16.

Recommended next action:

- add a B7 drift-aware dominance branch that uses `invariant_violation` as a repair gate without falling back to planner-always.

Alternative:

- run the full `long_horizon_drift_v03` suite first to quantify the current B3 failure envelope.

Implemented:

- `DriftAwarePlannerNecessityController`
- `drift_aware_planner_necessity_controller`
- `drift_aware_planner_necessity_loose_controller`
- B7 entries in `long_horizon_drift_v03`

Result document:

- `META_CONTROLLER_V0_16_DRIFT_AWARE_B7_2026-04-21.md`

Outcome:

- Full `long_horizon_drift_v03` suite completed.
- B7 main passes drift <= fixed and success >= fixed on seeds 0, 1, 2.
- B7 avoids planner-always; planner calls remain far below 120.
- B3 remains useful on reward/planner savings but is not drift-safe enough under v03.

### W13: Harden B7 Across Profiles

Status: completed as V0.17.

Recommended next action:

- add drift-repair intervention metrics.
- add B7 to cross-profile suites.
- run v01/v02/v03 regression to make sure B7 fixes v03 without damaging previous results.

Implemented:

- metrics: `drift_repairs`, `drift_repair_rate`
- suites: `b7_cross_profile_v01`, `b7_cross_profile_v02`
- B7 regression runs for v01 and v02

Result document:

- `META_CONTROLLER_V0_17_B7_CROSS_PROFILE_2026-04-21.md`

Outcome:

- combined tests: `16 passed`.
- v01 B7 repair rate stays `0.000`, preserving B3 behavior.
- v02 B7 repair rate stays `0.000`, preserving transfer behavior.
- v03 B7 repair rate is narrow and nonzero only under drift pressure.

### W14: Learn Or Calibrate Drift Repair Gate

Status: completed as V0.18 threshold sweep.

Recommended next action:

- replace the fixed B7 drift threshold with a learned or swept repair gate.
- compare learned repair against thresholds `0.08`, `0.10`, `0.12`, `0.14`, `0.16` on v03.
- keep v01/v02 regression checks in the loop.

Implemented:

- CLI: `--drift-threshold-sweep`
- dynamic threshold sweep for B7
- B7 main threshold calibrated from `0.10` to `0.08`

Result document:

- `META_CONTROLLER_V0_18_DRIFT_THRESHOLD_SWEEP_2026-04-21.md`

Outcome:

- `0.08` is the most robust v03 default across seeds 0, 1, 2.
- It passes success >= fixed, drift below fixed, and planner calls below fixed on all three seeds.
- Threshold response is non-monotonic, so a learned gate should wait until v03 variant transfer exists.

### W15: V03 Drift Variant Transfer

Status: completed as V0.19.

Recommended next action:

- create `v03b` or a v03 variant suite with altered invariant guard schedules and drift thresholds.
- test whether B7 threshold `0.08` transfers before learning a repair classifier.

Implemented:

- profile: `v03b`
- configs: `v03b_drift_variant_configs`
- suite: `v03b_drift_variant_transfer`
- protocol coverage for `v03b`

Result document:

- `META_CONTROLLER_V0_19_V03B_DRIFT_VARIANT_TRANSFER_2026-04-21.md`

Outcome:

- protocol tests: `9 passed`.
- smoke tests: `All smoke tests passed`.
- `0.08` transfers as a robust repair default across v03b seeds 0, 1, 2.
- It keeps success >= fixed, drift below fixed, and planner calls below fixed on all three seeds.
- Reward-optimal threshold shifts under v03b, so the curve remains non-monotonic.
- Do not build a learned repair classifier yet; if revisited, frame it as a surprise/residual gate.

### W16: Residual/Surprise Gate Instrumentation

Status: completed as V0.20.

Recommended next action:

- add lightweight post-repair residual metrics before any learned gate.
- record drift delta around repair events.
- compare threshold gates by residual reduction, not only success/reward/planner calls.

Implemented:

- repair-local drift instrumentation.
- fixed observation threshold for high-drift no-repair analysis: `0.08`.
- metrics:
  - `drift_repair_pre_mean`
  - `drift_repair_post_mean`
  - `drift_repair_delta_mean`
  - `drift_repair_delta_positive_rate`
  - `drift_residual_after_repair`
  - `high_drift_no_repair_rate`
  - `high_drift_no_repair_next_delta_mean`
  - `repair_efficiency`

Result document:

- `META_CONTROLLER_V0_20_REPAIR_RESIDUAL_INSTRUMENTATION_2026-04-21.md`

Outcome:

- protocol tests: `9 passed`.
- smoke tests: `All smoke tests passed`.
- `0.08` has high-drift no-repair rate `0.000` across v03 and v03b seeds 0, 1, 2.
- repair deltas are positive in nearly all repair events.
- `0.08` is now better described as a conservative invariant-coverage gate, not a reward-optimal threshold.

### W17: Gate Benefit Counterfactuals

Status: completed as V0.21.

Recommended next action:

- estimate counterfactual repair benefit for high-drift states.
- compare actual B7 repair against no-repair/planner-only counterfactual rollouts.
- keep this as analysis instrumentation first; do not train a learned gate yet.

Implemented:

- function: `run_repair_benefit_analysis`
- CLI: `--repair-benefit-analysis`
- one-step cloned counterfactual comparison:
  - forced planner repair
  - forced habit/no-repair

Result document:

- `META_CONTROLLER_V0_21_REPAIR_BENEFIT_COUNTERFACTUALS_2026-04-21.md`

Outcome:

- protocol tests: `10 passed`.
- smoke tests: `All smoke tests passed`.
- planner repair has positive drift benefit across v03/v03b seeds 0, 1, 2.
- one-step reward benefit is negative across v03/v03b seeds 0, 1, 2.
- B7 should be framed as invariant repair with short-term cost, not immediate reward gain.

### W18: Multi-Step Repair Benefit Horizon

Status: completed as V0.22.

Recommended next action:

- extend repair benefit counterfactuals from one-step to short horizon, likely `3` or `5`.
- track cumulative reward, cumulative drift, terminal drift, and success under planner-first vs habit-first branches.
- only after multi-step benefit is understood should a learned residual/value gate be considered.

Implemented:

- CLI extension: `--repair-benefit-horizon`
- horizon-3 cloned counterfactual rollouts.
- metrics:
  - terminal drift benefit
  - cumulative drift benefit
  - cumulative reward benefit
  - horizon success benefit

Result document:

- `META_CONTROLLER_V0_22_MULTI_STEP_REPAIR_BENEFIT_2026-04-21.md`

Outcome:

- protocol tests: `10 passed`.
- smoke tests: `All smoke tests passed`.
- planner-first repair has positive terminal and cumulative drift benefit across v03/v03b seeds 0, 1, 2.
- cumulative reward benefit remains negative across v03/v03b seeds 0, 1, 2.
- a learned gate should not use short-horizon reward as its target.

### W19: B7 Claim Freeze And Transfer Matrix

Status: completed as V0.23.

Recommended next action:

- freeze the current B7 claim as an invariant-coverage repair gate.
- create a compact transfer matrix across v01/v02/v03/v03b.
- include success, planner calls, drift, high-drift no-repair, terminal drift benefit, and reward tradeoff.
- turn the matrix into an acceptance/report artifact before any learned gate prototype.

Implemented:

- function: `run_b7_transfer_matrix`
- CLI: `--b7-transfer-matrix`
- compact matrix table output
- v01/v02/v03/v03b by seeds 0, 1, 2

Result document:

- `META_CONTROLLER_V0_23_B7_TRANSFER_MATRIX_2026-04-21.md`

Outcome:

- protocol tests: `11 passed`.
- smoke tests: `All smoke tests passed`.
- v03/v03b show consistent drift and high-drift no-repair reduction.
- v03/v03b repair-benefit counterfactuals show positive terminal/cumulative drift benefit and usually negative cumulative reward benefit.
- B7 claim is now frozen as invariant-coverage repair, not global reward improvement.

### W20: B7 Acceptance Artifact

Status: completed as V0.24.

Recommended next action:

- convert the B7 transfer matrix into explicit acceptance checks.
- separate drift-pressure profiles from non-pressure regressions.
- encode the frozen claim in report/acceptance terms:
  - v03/v03b drift and high-drift exposure must improve.
  - v01/v02 must not regress beyond bounded tolerance.
  - reward tradeoff must be reported, not hidden.

Implemented:

- function: `evaluate_b7_acceptance`
- function: `run_b7_acceptance_artifact`
- CLI: `--b7-acceptance-artifact`
- pressure-profile and regression-profile acceptance policies

Result document:

- `META_CONTROLLER_V0_24_B7_ACCEPTANCE_ARTIFACT_2026-04-21.md`

Outcome:

- protocol tests: `12 passed`.
- smoke tests: `All smoke tests passed`.
- v01/v02 bounded regression checks passed.
- v03/v03b drift-pressure checks passed.
- reward tradeoffs are explicitly reported and not used as hidden pass/fail targets.

### W21: Claim-Evidence Map For B7

Status: completed as V0.25.

Recommended next action:

- convert V0.18-V0.24 into a compact claim-evidence map.
- separate evidence by claim:
  - threshold calibration
  - variant transfer
  - residual repair mechanism
  - counterfactual drift benefit
  - bounded cross-profile regression
- identify figures/tables needed for a paper or technical report section.

Result document:

- `META_CONTROLLER_V0_25_B7_CLAIM_EVIDENCE_MAP_2026-04-21.md`

Outcome:

- B7 frozen claim is packaged as an invariant-coverage repair gate claim.
- V0.18-V0.24 are mapped into threshold calibration, transfer, residual mechanism, counterfactual benefit, transfer matrix, and acceptance evidence.
- unsupported claims are explicitly excluded:
  - no global reward-improvement claim.
  - no learned repair-classifier claim.
  - no global optimality claim for threshold `0.08`.
- report figures and tables are enumerated for the next handoff step.

### W22: B7 Report Section And Figure/Table Pack

Status: completed as V0.26.

Recommended next action:

- write a report-ready B7 section from the V0.25 claim-evidence map.
- create compact figure/table source artifacts for:
  - threshold sweep.
  - high-drift exposure and repair residuals.
  - one-step and horizon-3 repair benefit.
  - transfer matrix.
  - acceptance checks.
- keep this as packaging/report work unless a missing measurement is discovered.

Result document:

- `META_CONTROLLER_V0_26_B7_REPORT_SECTION_AND_FIGURE_TABLE_PACK_2026-04-21.md`

Outcome:

- report-ready B7 section drafted.
- figure captions added for:
  - B7 controller schematic.
  - threshold sweep.
  - high-drift exposure and repair residuals.
  - one-step and horizon-3 repair benefit.
- CSV-style data blocks added for:
  - threshold sweep summary.
  - exposure and residual summary.
  - counterfactual repair benefit.
  - transfer matrix.
  - acceptance summary.
- reward tradeoff and unsupported-claim boundaries remain explicit.

### W23: B7 Visual Artifact Generation

Status: paused by project split.

Recommended next action:

- convert V0.26 data blocks into actual figures:
  - threshold sweep panels.
  - exposure/residual bar chart.
  - counterfactual repair-benefit chart.
  - acceptance summary table.
- keep the source data traceable to V0.18-V0.24.
- only add plotting code if static Markdown tables are not enough for the intended report.

Pause note:

- The B7 line has enough handoff material for a normal LLM to resume from V0.25/V0.26.
- New research work is split into CEP-CC under `experiments/cep_cc`.
- Resume W23 only when the immediate need is paper/report visualization for B7.

Split handoff:

- `META_CONTROLLER_TO_CEP_CC_HANDOFF_2026-04-21.md`

### Side Note: WebVM

Status: assessed as future sandbox reference, not current dependency.

Source:

- `F:\workspace-ideas\webvm-main`

Assessment document:

- `WEBVM_REUSE_ASSESSMENT_2026-04-21.md`

Outcome:

- WebVM is not useful for the current V0/V1 causal core.
- It may be useful later as a browser-contained agent sandbox and tool-use telemetry surface.
- Keep it reference-only.
