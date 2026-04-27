# Meta-Controller V0.25 B7 Claim-Evidence Map - 2026-04-21

## Purpose

V0.25 packages the B7 drift-repair branch for handoff.

It converts V0.18-V0.24 into a compact claim-evidence map so the next operator can write the report section, build figures, or decide whether a learned repair gate is justified.

## Frozen Claim

B7 is an invariant-coverage repair gate.

On drift-pressure profiles, it converts high-drift exposure into planner repair, reducing drift debt and high-drift no-repair exposure. It can cost short-horizon reward, so the correct objective is invariant coverage or constrained value, not raw reward maximization.

This claim does not say that B7 improves every metric on every profile.

## Claim-Evidence Map

| claim | evidence | source | decision |
|---|---|---|---|
| `0.08` is the current conservative B7 threshold. | On v03 seeds 0/1/2, threshold `0.08` keeps success at or above fixed rule, lowers drift far below fixed rule, and keeps planner calls below fixed rule. The threshold curve is non-monotonic. | V0.18 drift threshold sweep | Keep `0.08` as the main B7 threshold. Do not treat it as reward-optimal. |
| The threshold transfers to a drift variant. | On v03b seeds 0/1/2, `0.08` again keeps success at or above fixed rule, lowers drift from fixed `0.161` to `0.033`, `0.005`, `0.005`, and keeps planner calls below fixed rule. | V0.19 v03b transfer | Accept `0.08` as a robust invariant-repair default across v03/v03b. |
| B7 is a repair/exposure mechanism, not a reward shortcut. | With threshold `0.08`, high-drift no-repair rate is `0.000` across v03 and v03b seeds. Repair deltas are positive in almost all repair events. Fixed rule and B3 leave substantial high-drift no-repair exposure. | V0.20 residual instrumentation | Frame B7 as invariant coverage. Reward is a reported tradeoff. |
| One-step repair has direct drift benefit. | Planner-first counterfactuals on high-drift states improve next-step drift by `0.300` on v03 seeds 0/1/2 and by `0.262`, `0.160`, `0.160` on v03b seeds 0/1/2. One-step reward benefit is negative. | V0.21 one-step repair benefit | The repair action is mechanistically useful for drift, but raw one-step reward is the wrong training target. |
| Multi-step repair benefit persists. | Horizon-3 terminal drift benefit is positive on v03 (`0.410`, `0.600`, `0.600`) and v03b (`0.352`, `0.364`, `0.364`). Cumulative drift benefit is also positive. Cumulative reward benefit is negative on pressure profiles. | V0.22 multi-step repair benefit | Use horizon or constrained value if learning a gate later. |
| B7 reduces drift debt under invariant pressure. | In the transfer matrix, v03 drift deltas are `-0.098`, `-0.094`, `-0.089`; v03b drift deltas are `-0.146`, `-0.141`, `-0.136`. High-drift no-repair deltas are `-0.189` on v03 and `-0.239` on v03b. | V0.23 transfer matrix | The strongest supported result is drift-pressure improvement, not universal reward gain. |
| Cross-profile regressions are bounded. | V0.24 acceptance passes v01/v02 bounded regression checks and v03/v03b pressure checks on seeds 0/1/2. | V0.24 acceptance artifact | B7 is stable enough to freeze as the current drift-pressure branch. |
| A learned repair classifier is not justified yet. | Reward-optimal threshold shifts under v03b, the threshold curve is non-monotonic, and counterfactual reward benefit is often negative even when drift benefit is positive. | V0.18-V0.24 combined | Do not train a direct classifier on current v03 traces. Define constrained value first. |

## Evidence Inventory

- `META_CONTROLLER_V0_18_DRIFT_THRESHOLD_SWEEP_2026-04-21.md`: calibrates `0.08` on v03 and records non-monotonic threshold response.
- `META_CONTROLLER_V0_19_V03B_DRIFT_VARIANT_TRANSFER_2026-04-21.md`: tests transfer under changed invariant guard schedule, guard threshold, drift pressure, and repair strength.
- `META_CONTROLLER_V0_20_REPAIR_RESIDUAL_INSTRUMENTATION_2026-04-21.md`: records repair pre/post drift, repair deltas, residual drift, and high-drift no-repair exposure.
- `META_CONTROLLER_V0_21_REPAIR_BENEFIT_COUNTERFACTUALS_2026-04-21.md`: measures one-step planner-first versus habit-first counterfactuals on high-drift states.
- `META_CONTROLLER_V0_22_MULTI_STEP_REPAIR_BENEFIT_2026-04-21.md`: extends repair-benefit analysis to horizon 3.
- `META_CONTROLLER_V0_23_B7_TRANSFER_MATRIX_2026-04-21.md`: summarizes v01/v02/v03/v03b by seeds 0/1/2.
- `META_CONTROLLER_V0_24_B7_ACCEPTANCE_ARTIFACT_2026-04-21.md`: converts the frozen claim into pass/fail acceptance checks.

## Figure And Table Plan

Figure 1: B7 controller schematic.

- Show B3 planner-necessity path plus B7 drift-repair override.
- Label `drift > 0.08` as the conservative repair trigger.
- Make clear that repair invokes planner under invariant pressure.

Figure 2: threshold sweep.

- Plot threshold on x-axis.
- Use panels or grouped lines for success, mean reward, planner calls, and mean drift.
- Include v03 and v03b seeds.
- Message: `0.08` is conservative and transferable; reward optimum shifts.

Figure 3: high-drift exposure and repair residuals.

- Compare fixed rule, B3, and B7 `0.08`.
- Use high-drift no-repair rate and repair delta/residual as bars.
- Message: B7 removes unhandled high-drift exposure on pressure profiles.

Figure 4: repair-benefit counterfactuals.

- Compare habit-first and planner-first on high-drift states.
- Include one-step and horizon-3 drift benefit.
- Message: repair improves drift state even when reward cost is immediate.

Table 1: B7 transfer matrix.

- Rows: v01/v02/v03/v03b x seed 0/1/2.
- Columns: success delta, planner delta, drift delta, reward delta, high-drift no-repair delta, terminal/cumulative drift benefit.
- Message: pressure profiles improve; non-pressure regressions are bounded.

Table 2: acceptance artifact.

- Rows: profile and seed.
- Columns: policy group, pass/fail, key margins.
- Message: all v01/v02/v03/v03b seed checks pass under the frozen acceptance policy.

Appendix Table: raw sweep tables.

- Include all threshold rows from V0.18 and V0.19.
- Keep this out of the main narrative unless space permits.

## Claims Not Supported

B7 is not a global reward improver.

B7 is not a learned repair classifier.

B7 has not been validated beyond the current synthetic meta-controller profiles.

The current evidence does not prove that `0.08` is globally optimal.

The reward tradeoff is real and must stay visible in report tables.

## Next Work

W22 should turn this evidence map into a report-ready B7 section and figure/table pack.

Recommended scope:

- write the B7 report section with the frozen claim and limitations.
- create machine-readable CSV or Markdown tables for Figures 2-4 and Tables 1-2.
- avoid new controller logic unless the report writing exposes a missing measurement.

Future learned-gate work should wait until the target is defined as constrained multi-step value or invariant coverage, not next-step reward.

