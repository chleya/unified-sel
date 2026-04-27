# Meta-Controller V0.17 B7 Cross-Profile Hardening - 2026-04-21

## Purpose

V0.17 hardens B7 after the positive v03 result.

It adds:

- explicit drift-repair metrics
- B7 cross-profile suites for v01 and v02
- regression checks to ensure B7 does not damage the previous B3 results

## Implemented

Files:

- `experiments/meta_controller/metrics.py`
- `experiments/meta_controller/report.py`
- `experiments/meta_controller/run_experiment.py`
- `experiments/meta_controller/adapters/sel_lab_benchmark.py`
- `tests/test_meta_controller_adapter_sel_lab.py`

New metrics:

- `drift_repairs`
- `drift_repair_rate`

New suites:

- `b7_cross_profile_v01`
- `b7_cross_profile_v02`

## Validation

Commands:

```powershell
python -m pytest F:\unified-sel\tests\test_meta_controller_adapter_sel_lab.py F:\unified-sel\tests\test_meta_controller_protocol.py -q -p no:cacheprovider
python -m experiments.meta_controller.run_experiment --profile v03 --mode train-eval --train-episodes 12 --eval-episodes 6 --seed 6 --table
python -m experiments.meta_controller.run_experiment --suite b7_cross_profile_v01 --table
python -m experiments.meta_controller.run_experiment --suite b7_cross_profile_v02 --table
```

Result:

- combined tests: `16 passed`
- v01 cross-profile suite: B7 passes all acceptance checks
- v02 cross-profile suite: B7 passes all acceptance checks

## V01 Regression

In v01:

- B7 repair rate: `0.000` on all three seeds
- B7 success: `1.000` on all three seeds
- B7 drift: `0.000` on all three seeds
- B7 planner calls remain at about B3 level, around `49.8-49.9`

Interpretation:

- B7 does not interfere with the original v01 B3 result.
- Since drift never rises, the repair gate stays inactive.

## V02 Regression

In v02:

- B7 repair rate: `0.000` on all three seeds
- B7 success: `1.000` on all three seeds
- B7 drift: `0.000` on all three seeds
- B7 planner calls match B3, around `59.0-59.2`

Interpretation:

- B7 preserves the v02 transfer result.
- The drift gate remains dormant when not needed.

## V03 Repair Check

Quick v03 check:

- B7 main repair rate: about `0.031`
- B7 loose repair rate: about `0.024`
- B7 main drift: about `0.008`
- B7 loose drift: about `0.006`

Interpretation:

- The repair gate is narrow.
- It is not planner-always.
- It activates only in the drift-pressure profile.

## Current Recommendation

B3 should remain the mainline for v01/v02.

B7 should become the conditional mainline for drift-pressure settings:

- use B3 when drift pressure is absent
- use B7 when invariant repair is part of the environment

## Remaining Gap

B7's repair gate is calibrated, not learned.

The next research step is to learn or adapt the drift threshold:

- from drift-repair labels
- from expected recovery value
- or from a small threshold sweep across v03 variants

This would turn the current hand-set repair gate into a learned part of the meta-controller.

