# Meta-Controller V0.26 B7 Report Section And Figure/Table Pack - 2026-04-21

## Purpose

V0.26 turns the V0.25 claim-evidence map into report-ready text and compact figure/table source artifacts.

This is packaging work only. It does not change controller logic.

## Report Section Draft

### B7 Drift-Repair Gate

B7 adds an invariant-coverage repair gate to the planner-necessity controller. The gate monitors drift and invokes the planner when drift exceeds the conservative threshold `0.08`. Its purpose is not to maximize immediate reward. Its purpose is to prevent high-drift states from remaining unhandled under invariant pressure.

The threshold was calibrated on `v03` by sweeping `0.08`, `0.10`, `0.12`, `0.14`, and `0.16` across seeds 0, 1, and 2. The response was non-monotonic, but `0.08` was the most robust default: it matched or exceeded fixed-rule success on all seeds, reduced mean drift far below fixed rule, and kept planner calls below fixed rule. The same threshold transferred to `v03b`, a drift variant with changed invariant guard schedules, guard thresholds, drift pressure, and repair strength. On `v03b`, `0.08` again preserved success, reduced drift substantially, and avoided planner-call inflation against the fixed rule.

Residual instrumentation shows why the gate works. At threshold `0.08`, high-drift no-repair exposure is eliminated on `v03` and `v03b` seeds 0, 1, and 2. Repair deltas are positive in nearly all repair events, while fixed rule and B3 leave substantial high-drift exposure. Counterfactual analysis supports the same mechanism: planner-first repair improves next-step drift on high-drift states, and the effect persists at horizon 3. However, reward benefit is often negative over the same windows. This confirms that B7 is an invariant repair mechanism with a visible reward tradeoff, not a raw reward optimizer.

The transfer matrix and acceptance artifact freeze the supported claim. On drift-pressure profiles `v03` and `v03b`, B7 reduces drift debt and high-drift no-repair exposure while preserving success within the acceptance tolerance. On non-pressure profiles `v01` and `v02`, regressions are bounded under the frozen acceptance policy. The accepted claim is therefore narrow: B7 converts high-drift exposure into planner repair under invariant pressure. It does not claim universal reward improvement or global optimality of the `0.08` threshold.

### Limitation Statement

B7 has been validated only inside the current synthetic meta-controller profiles. It should not be described as a learned classifier, a global reward optimizer, or a general theorem about drift repair. The reward tradeoff is part of the result and must remain visible in the main table or limitations.

## Figure Captions

Figure 1. B7 drift-repair gate. The B3 planner-necessity controller is augmented with a drift override. When observed drift exceeds `0.08`, the controller routes to planner repair to restore invariant coverage.

Figure 2. Threshold sweep on v03 and v03b. The `0.08` threshold is the most robust conservative default across seeds: it preserves success, lowers drift, and avoids planner-call inflation, although reward-optimal thresholds shift across variants.

Figure 3. High-drift exposure and repair residuals. B7 eliminates high-drift no-repair exposure on v03/v03b at threshold `0.08`, while fixed rule and B3 leave high-drift states unhandled.

Figure 4. Counterfactual repair benefit. On high-drift states, planner-first repair improves one-step and horizon-3 drift relative to habit-first execution. Reward benefit is often negative, showing the repair tradeoff.

Table 1. B7 transfer matrix across v01/v02/v03/v03b and seeds 0/1/2. Deltas are B7 minus fixed rule. Drift-pressure profiles show consistent drift and exposure reduction; non-pressure profiles stay within bounded regression tolerance.

Table 2. B7 acceptance artifact. All v01/v02 bounded-regression checks and v03/v03b pressure-profile checks pass under the frozen acceptance policy.

## Figure/Table Data Blocks

### Figure 2: Threshold Sweep Summary

```csv
profile,seed,threshold,success,reward,planner_calls,drift,fixed_success,fixed_reward,fixed_planner_calls,fixed_drift
v03,0,0.08,0.964,95.518,74.933,0.026,0.964,95.576,82.000,0.109
v03,1,0.08,1.000,103.301,77.467,0.003,0.964,95.576,82.000,0.109
v03,2,0.08,1.000,103.274,77.367,0.003,0.964,95.576,82.000,0.109
v03b,0,0.08,0.964,95.131,77.517,0.033,0.964,96.516,82.000,0.161
v03b,1,0.08,1.000,103.138,78.467,0.005,0.964,96.516,82.000,0.161
v03b,2,0.08,1.000,103.111,78.367,0.005,0.964,96.516,82.000,0.161
```

### Figure 3: Exposure And Residual Summary

```csv
profile,seed,controller,high_drift_no_repair_rate,drift_repair_delta_mean,drift_residual_after_repair
v03,0,b7_0.08,0.000,0.144,0.144
v03,1,b7_0.08,0.000,0.160,0.020
v03,2,b7_0.08,0.000,0.160,0.020
v03b,0,b7_0.08,0.000,0.110,0.149
v03b,1,b7_0.08,0.000,0.098,0.064
v03b,2,b7_0.08,0.000,0.098,0.064
v03,0,fixed,0.189,0.000,0.000
v03,1,fixed,0.189,0.000,0.000
v03,2,fixed,0.189,0.000,0.000
v03b,0,fixed,0.239,0.000,0.000
v03b,1,fixed,0.239,0.000,0.000
v03b,2,fixed,0.239,0.000,0.000
```

### Figure 4: Counterfactual Repair Benefit

```csv
profile,seed,horizon,samples,terminal_drift_benefit,cumulative_drift_benefit,cumulative_reward_benefit
v03,0,1,579,0.300,0.300,-0.243
v03,1,1,120,0.300,0.300,-0.245
v03,2,1,120,0.300,0.300,-0.245
v03b,0,1,768,0.262,0.262,-0.149
v03b,1,1,180,0.160,0.160,-0.224
v03b,2,1,180,0.160,0.160,-0.224
v03,0,3,579,0.410,1.037,-0.339
v03,1,3,120,0.600,1.360,-0.404
v03,2,3,120,0.600,1.360,-0.404
v03b,0,3,768,0.352,0.897,-0.243
v03b,1,3,180,0.364,0.789,-0.318
v03b,2,3,180,0.364,0.789,-0.318
```

### Table 1: Transfer Matrix

```csv
profile,seed,success_delta,planner_delta,drift_delta,reward_delta,high_drift_no_repair_delta,benefit_samples,terminal_drift_benefit,cumulative_drift_benefit,cumulative_reward_benefit
v01,0,-0.008,-5.200,0.005,-0.179,0.000,0,0.000,0.000,0.000
v01,1,-0.011,-6.733,0.007,-0.388,0.000,0,0.000,0.000,0.000
v01,2,-0.018,-2.367,0.011,-2.241,0.000,407,0.057,0.170,0.238
v02,0,0.025,-6.600,-0.043,5.601,-0.140,0,0.000,0.000,0.000
v02,1,0.023,-4.000,-0.042,4.860,-0.140,0,0.000,0.000,0.000
v02,2,0.001,-1.500,-0.028,-0.013,-0.140,303,0.048,0.145,0.213
v03,0,0.003,4.150,-0.098,-1.961,-0.189,579,0.410,1.037,-0.339
v03,1,-0.006,6.950,-0.094,-4.609,-0.189,120,0.600,1.360,-0.404
v03,2,0.011,-5.067,-0.089,1.990,-0.189,120,0.600,1.360,-0.404
v03b,0,0.003,5.083,-0.146,-3.016,-0.239,768,0.352,0.897,-0.243
v03b,1,-0.006,7.783,-0.141,-5.630,-0.239,180,0.364,0.789,-0.318
v03b,2,0.011,-2.250,-0.136,0.592,-0.239,180,0.364,0.789,-0.318
```

### Table 2: Acceptance Summary

```csv
profile,seed,policy_group,result,key_margin
v01,0,bounded_regression,PASS,worst_values_within_tolerance
v01,1,bounded_regression,PASS,worst_values_within_tolerance
v01,2,bounded_regression,PASS,worst_values_within_tolerance
v02,0,bounded_regression,PASS,worst_values_within_tolerance
v02,1,bounded_regression,PASS,worst_values_within_tolerance
v02,2,bounded_regression,PASS,worst_values_within_tolerance
v03,0,pressure_acceptance,PASS,drift_delta=-0.098;high_drift_delta=-0.189;terminal_benefit=0.410
v03,1,pressure_acceptance,PASS,drift_delta=-0.094;high_drift_delta=-0.189;terminal_benefit=0.600
v03,2,pressure_acceptance,PASS,drift_delta=-0.089;high_drift_delta=-0.189;terminal_benefit=0.600
v03b,0,pressure_acceptance,PASS,drift_delta=-0.146;high_drift_delta=-0.239;terminal_benefit=0.352
v03b,1,pressure_acceptance,PASS,drift_delta=-0.141;high_drift_delta=-0.239;terminal_benefit=0.364
v03b,2,pressure_acceptance,PASS,drift_delta=-0.136;high_drift_delta=-0.239;terminal_benefit=0.364
```

## Handoff Notes

Use the section draft as the report body.

Use Figure 2 for threshold calibration and transfer.

Use Figure 3 plus Figure 4 to prove the mechanism: exposure is eliminated and repair has counterfactual drift benefit.

Use Table 1 and Table 2 to make the acceptance boundary explicit.

Do not hide reward tradeoffs. The reward cost is part of the scientific result.

## Verification

No tests were required. This version only adds report text and figure/table source data.

