# EXPERIMENT_LOG.md - Experiment Record

Append one entry after each concrete experiment or verification step.

Template:

```text
## [date] [experiment name]

**Command**:
**Parameters**:
**Result**:
**Issues / Observations**:
**Next Step**:
```

---

## 2026-04-03 Framework Initialization

**Command**: none (initial setup)
**Parameters**: none
**Result**: created the initial project structure and core framework files
**Issues / Observations**:
- The design combines ideas from `F:\sel-lab`, `F:\SDAS`, and `F:\fcrs_mis`
- The workspace notes claimed several files already existed, but some verification files were still missing
**Next Step**: run the smoke test for the Phase 1 framework

---

## 2026-04-03 Smoke Test

**Command**: `python F:\unified-sel\tests\smoke_test.py`
**Parameters**: none
**Result**: smoke test passed
**Issues / Observations**:
- Added the missing Phase 1 verification files: `tests/smoke_test.py`, `experiments/baselines/fixed.py`, and `experiments/baselines/ewc.py`
- Verified these checks passed: `Structure` creation, `StructurePool.observe`, `UnifiedSELClassifier` forward/learn path, `EWCBaseline`, and `FixedNetwork`
- This verified only the skeleton framework, not full Phase 2 experiment readiness
**Next Step**: inspect the three source projects before claiming the baseline stage is ready

---

## 2026-04-03 Source Project Inspection

**Command**: read-only inspection of
- `F:\sel-lab\README.md`
- `F:\sel-lab\core\sel_core.py`
- `F:\sel-lab\core\phase3.py`
- `F:\SDAS\README.md`
- `F:\SDAS\src\structure_pool.py`
- `F:\fcrs_mis\README.md`
- `F:\fcrs_mis\src\fcrs\core\pool.py`
- `F:\fcrs_mis\src\fcrs\types.py`
**Parameters**: none
**Result**: confirmed the intended source alignment for `unified-sel`
**Issues / Observations**:
- `core/learner.py` mainly tracks ideas from `sel-lab` DFA learning
- `core/pool.py` mainly tracks ideas from `SDAS` surprise-driven structure management
- `core/structure.py` and some engineering style track ideas from `fcrs_mis`
- Current `unified-sel` is still a concept-integrated skeleton, not a feature-aligned implementation of the three source systems
- The current baseline files were runnable for smoke testing but were not yet proper Phase 2 experiment entrypoints
**Next Step**: add the minimum runnable baseline experiment scripts and then execute Phase 2

---

## 2026-04-03 Phase 2 Baseline Validation

**Command**:
- `python F:\unified-sel\experiments\baselines\fixed.py`
- `python F:\unified-sel\experiments\baselines\ewc.py`
- `python F:\unified-sel\tests\smoke_test.py`
**Parameters**:
- two-task continual benchmark
- input dimension: 4
- train samples per task: 256
- test samples per task: 256
- epochs per task: 6
- seed: 7
**Result**:
- Fixed baseline result saved to `F:\unified-sel\results\baseline_fixed\20260403_102846.json`
- EWC baseline result saved to `F:\unified-sel\results\baseline_ewc\20260403_102846.json`
- smoke test passed after the baseline script changes
**Issues / Observations**:
- `FixedNetwork` behaves as expected for a catastrophic-forgetting baseline
- `EWCBaseline` retains task 0 much better than `FixedNetwork`
**Next Step**: implement `experiments/continual/no_boundary.py`

---

## 2026-04-03 Phase 3 And Phase 4

**Command**:
- `python F:\unified-sel\experiments\continual\no_boundary.py --seeds 5`
- `python F:\unified-sel\analysis\compare.py`
**Parameters**:
- no-boundary stream length: 600 steps
- checkpoint step: 200
- evaluation samples per fixed task: 256
- seeds: `[7, 8, 9, 10, 11]`
**Result**:
- established the first runnable Unified-SEL main experiment and direct comparison pipeline
**Issues / Observations**:
- untuned Unified-SEL did not beat EWC on average accuracy or forgetting
- early pool behavior showed rapid saturation and almost no clone activity
**Next Step**: begin one-parameter-at-a-time tuning

---

## 2026-04-03 Phase 5 Tuning Iteration 1

**Parameters changed**:
- `SURPRISE_THRESHOLD: 0.45 -> 0.60`
**Result**:
- improved avg accuracy from `0.4891` to `0.4949`
- improved forgetting from `0.2617` to `0.1961`
**Observation**:
- helpful, but still not enough to beat EWC

---

## 2026-04-03 Phase 5 Tuning Iteration 2

**Parameters changed**:
- `TENSION_THRESHOLD: 0.15 -> 0.08`
**Result**:
- no measurable improvement over iteration 1
**Observation**:
- simply lowering tension threshold did not activate meaningful clone behavior

---

## 2026-04-03 Phase 5 Tuning Iteration 3

**Parameters changed**:
- effective experiment capacity: `max_structures 12 -> 20`
**Result**:
- avg accuracy worsened to `0.4879`
- forgetting worsened to `0.2422`
**Observation**:
- more capacity amplified branch/create growth without improving reuse

---

## 2026-04-03 Phase 5 Tuning Iteration 4

**Parameters changed**:
- `UTILITY_DECAY: 0.002 -> 0.005`
**Result**:
- avg accuracy improved to `0.5176`
- forgetting improved to `0.1344`
- Unified-SEL beat EWC on average accuracy for the first time
**Observation**:
- this was the first iteration where clone and prune activity became meaningfully active

---

## 2026-04-03 Phase 5 Tuning Iteration 5

**Parameters changed**:
- `UTILITY_PRUNE: 0.08 -> 0.10`
**Result**:
- result saved to `F:\unified-sel\results\continual_no_boundary\20260403_122351.json`
- comparison saved to `F:\unified-sel\results\analysis_compare\20260403_122356.json`
- avg accuracy changed from `0.5176` to `0.5086`
- forgetting changed from `0.1344` to `0.1641`
**Observation**:
- stronger prune pressure was slightly worse than the previous best setting
- best-known setting remains iteration 4
**Next Step**: revert to the iteration-4 setting and try a smaller change in surprise gating or another gentle retention-control parameter

---

## 2026-04-03 Project Gap Review

**Command**:
- top-down audit of current docs, core, experiments, analysis, and best-known results
**Parameters**: none
**Result**:
- merged project-level gap analysis and roadmap written to `F:\unified-sel\PROJECT_GAPS_AND_ROADMAP.md`
**Issues / Observations**:
- the main missing pieces are no longer basic implementation entrypoints
- the main remaining gaps are:
  - forgetting still worse than EWC
  - no statistical significance layer yet
  - experiment configuration and result selection are still too implicit
- the highest-value next work splits naturally into three parallel tracks:
  - forgetting diagnostics
  - compare/statistics upgrade
  - configuration and reproducibility backbone
**Next Step**: choose whether to continue tuning or begin the next-stage parallel work tracks from the roadmap document

---

## 2026-04-03 Track A/B/C Integration

**Command**:
- `python F:\unified-sel\tests\smoke_test.py`
- `python F:\unified-sel\experiments\continual\no_boundary.py --seeds 2 --max-structures 12 --window-size 50`
- `python F:\unified-sel\analysis\compare.py --fixed F:\unified-sel\results\baseline_fixed\20260403_102846.json --ewc F:\unified-sel\results\baseline_ewc\20260403_102846.json --unified F:\unified-sel\results\continual_no_boundary\20260403_122312.json`
**Parameters**:
- no-boundary verification with `2` seeds and `max_structures = 12`
- explicit result paths for fixed, EWC, and Unified-SEL comparison
**Result**:
- smoke test passed after wiring configuration through `core/pool.py`, `core/learner.py`, and `experiments/continual/no_boundary.py`
- diagnostic verification result saved to `F:\unified-sel\results\continual_no_boundary\20260403_133309.json`
- explicit compare verification saved to `F:\unified-sel\results\analysis_compare\20260403_133309.json`
**Issues / Observations**:
- Track A is now active in the main experiment entrypoint: step traces, window summaries, and checkpoint metrics are saved
- Track B is now active in the analysis entrypoint: explicit path selection, sample-level summaries, and bootstrap deltas are saved
- Track C is no longer only a placeholder file: no-boundary runs now save a structured config snapshot and pool hyperparameters flow through runtime construction
- the scientific gap remains unchanged: Unified-SEL still leads EWC on avg accuracy but still loses on forgetting
- current statistical strength is limited because the baseline side still comes from single-result files
**Next Step**: use the new diagnostics to identify where forgetting grows, then upgrade baseline generation to multi-seed runs for stronger comparison statistics

---

## 2026-04-03 Boundary Diagnostics Bootstrap

**Command**:
- `python F:\unified-sel\analysis\boundary_diagnostics.py --input F:\unified-sel\results\continual_no_boundary\20260403_133309.json`
- `python F:\unified-sel\experiments\continual\no_boundary.py --seeds 5 --max-structures 12 --window-size 50`
- `python F:\unified-sel\analysis\boundary_diagnostics.py --input F:\unified-sel\results\continual_no_boundary\20260403_134535.json`
- `python F:\unified-sel\tests\smoke_test.py`
**Parameters**:
- best-known pool setting retained:
  - `SURPRISE_THRESHOLD = 0.60`
  - `TENSION_THRESHOLD = 0.08`
  - `UTILITY_DECAY = 0.005`
  - `UTILITY_PRUNE = 0.08`
  - `max_structures = 12`
**Result**:
- added `analysis/boundary_diagnostics.py`
- 2-seed preview report saved to `F:\unified-sel\results\analysis_boundary\20260403_134527.json`
- 5-seed diagnostic rerun saved to `F:\unified-sel\results\continual_no_boundary\20260403_134535.json`
- 5-seed boundary report saved to `F:\unified-sel\results\analysis_boundary\20260403_134542.json`
- smoke test passed after adding the new analysis path
**Issues / Observations**:
- endogenous boundary formation is visible: the strongest pressure windows are not random; they cluster at the initial structure-fill stage and again near the mid-stream drift region
- high-forgetting seeds are now identifiable: seeds `8` and `9`
- the main gap is not absence of boundary signals; it is failure to absorb boundary pressure cleanly after the stream shifts
- high-forgetting seeds show stronger sustained mid/late tension and, in some cases, repeated mid-stream branch/create activity
- low-forgetting seeds can still show late tension growth, but they do not accumulate the same unstable forgetting outcome
**Next Step**: design the next change around persistent mid/late boundary pressure, not around more aggressive early growth

---

## 2026-04-03 Retention-Oriented Mature-Structure Decay

**Command**:
- `python F:\unified-sel\tests\smoke_test.py`
- `python F:\unified-sel\experiments\continual\no_boundary.py --seeds 5 --max-structures 12 --window-size 50`
- `python F:\unified-sel\analysis\compare.py --fixed F:\unified-sel\results\baseline_fixed\20260403_102846.json --ewc F:\unified-sel\results\baseline_ewc\20260403_102846.json --unified F:\unified-sel\results\continual_no_boundary\20260403_135552.json`
- `python F:\unified-sel\analysis\boundary_diagnostics.py --input F:\unified-sel\results\continual_no_boundary\20260403_135552.json`
**Parameters changed**:
- `MATURE_AGE = 80`
- `MATURE_DECAY_SCALE = 0.35`
**Result**:
- retention-aware main run saved to `F:\unified-sel\results\continual_no_boundary\20260403_135552.json`
- comparison saved to `F:\unified-sel\results\analysis_compare\20260403_135605.json`
- boundary report saved to `F:\unified-sel\results\analysis_boundary\20260403_135604.json`
- smoke test passed
**Issues / Observations**:
- forgetting improved from `0.1344` to `0.1148`
- avg accuracy decreased from `0.5176` to `0.5113`, but still remained above EWC `0.5020`
- this change helped retention without changing pool capacity or the surprise/tension thresholds
- boundary-pressure structure did not materially change; high-forgetting seeds are still `8` and `9`
- the intervention appears to help by preserving mature structures, not by resolving the underlying mid/late boundary-pressure mechanism
**Next Step**: design the next intervention around how full-capacity pools respond to persistent mid/late pressure

---

## 2026-04-03 Full-Capacity Replacement Attempt (Reverted)

**Command**:
- `python F:\unified-sel\tests\smoke_test.py`
- `python F:\unified-sel\experiments\continual\no_boundary.py --seeds 5 --max-structures 12 --window-size 50`
**Parameters changed**:
- temporary full-capacity replacement rule:
  - when pool is full and surprise remains in `branch/create` range, replace the weakest sufficiently old structure instead of falling back to reinforce
**Result**:
- temporary run saved to `F:\unified-sel\results\continual_no_boundary\20260403_140108.json`
- smoke test passed both before and after rollback
- code change was reverted after evaluation
**Issues / Observations**:
- this was a bad direction
- task-0 final accuracy collapsed to `0.3867` mean
- forgetting worsened sharply to `0.2422`
- the mechanism overreacted to pressure by destroying useful retention structure
- the failure mode confirms that current pressure is not solved by aggressive pool membership churn
**Next Step**: keep the mature-retention version as the code baseline and look for softer pressure-resolution mechanisms that act before replacement

---

## 2026-04-03 Full-Capacity Competition Rebalance (Reverted)

**Command**:
- `python F:\unified-sel\tests\smoke_test.py`
- `python F:\unified-sel\experiments\continual\no_boundary.py --seeds 5 --max-structures 12 --window-size 50`
- `python F:\unified-sel\analysis\compare.py --fixed F:\unified-sel\results\baseline_fixed\20260403_102846.json --ewc F:\unified-sel\results\baseline_ewc\20260403_102846.json --unified F:\unified-sel\results\continual_no_boundary\20260403_142141.json`
- `python F:\unified-sel\analysis\boundary_diagnostics.py --input F:\unified-sel\results\continual_no_boundary\20260403_142141.json`
**Parameters changed**:
- temporary full-capacity competition rebalance:
  - when pool is full and surprise is above the low-surprise reinforce range, add a small competition penalty to mature high-utility structures so near-tie alternatives can win the update instead of always reusing the incumbent mature structure
**Result**:
- temporary run saved to `F:\unified-sel\results\continual_no_boundary\20260403_142141.json`
- comparison saved to `F:\unified-sel\results\analysis_compare\20260403_142158.json`
- boundary report saved to `F:\unified-sel\results\analysis_boundary\20260403_142157.json`
- smoke test passed before rollback
- code change was reverted after evaluation
**Issues / Observations**:
- this direction was not a net improvement over the mature-retention baseline
- avg accuracy slipped from `0.5113` to `0.5094`
- forgetting improved from `0.1148` to `0.0969`, but the gain came with worse `task_1` final accuracy (`0.5086 -> 0.4867`)
- the mechanism only activated on some seeds and mainly redistributed competition rather than resolving the underlying mid/late pressure pattern
- high-forgetting seeds remained `8/9`, so the core instability did not disappear
**Next Step**: keep the mature-retention version as the code baseline and try pressure-resolution mechanisms that modulate shared learning dynamics or output competition without simply biasing winner selection back toward older structures

---

## 2026-04-03 Boundary Metrics Upgrade

**Command**:
- `python F:\unified-sel\analysis\boundary_diagnostics.py --input F:\unified-sel\results\continual_no_boundary\20260403_135552.json`
**Parameters changed**:
- upgraded `analysis/boundary_diagnostics.py` to compute per-run boundary metrics:
  - first emergence
  - active-window ratio
  - episode count
  - collapse / reactivation count
  - late-pressure recurrence
  - simple stability score and status label
**Result**:
- upgraded boundary report saved to `F:\unified-sel\results\analysis_boundary\20260403_144327.json`
**Issues / Observations**:
- this was an analysis-only change; no training code or experiment baseline changed
- the mature-retention best run still does not show stable endogenous-boundary behavior
- aggregate boundary summary on the current best run:
  - mean stability score: `0.0900`
  - mean active-window ratio: `0.2500`
- mean collapse count: `1.6`
- status counts: `recurrent_pressure=4`, `transient=1`, `stable=0`
- this sharpens the project claim: the system can express endogenous boundary pressure, but it still mostly re-enters pressure rather than settling into a stable internal boundary regime
**Next Step**: use these new boundary metrics as the primary screen for the next mechanism change, and prefer interventions that reduce late recurrence rather than only shifting accuracy/forgetting tradeoffs

---

## 2026-04-03 Shared Output Protection Under Full-Capacity Pressure (Reverted)

**Command**:
- `python F:\unified-sel\tests\smoke_test.py`
- `python F:\unified-sel\experiments\continual\no_boundary.py --seeds 5 --max-structures 12 --window-size 50`
- `python F:\unified-sel\analysis\compare.py --fixed F:\unified-sel\results\baseline_fixed\20260403_102846.json --ewc F:\unified-sel\results\baseline_ewc\20260403_102846.json --unified F:\unified-sel\results\continual_no_boundary\20260403_144915.json`
- `python F:\unified-sel\analysis\boundary_diagnostics.py --input F:\unified-sel\results\continual_no_boundary\20260403_144915.json`
**Parameters changed**:
- temporary shared-output protection gate:
  - when the pool is full and both `surprise` and `avg_tension` are high, reduce the shared output-layer update scale instead of changing pool membership or winner selection
**Result**:
- temporary run saved to `F:\unified-sel\results\continual_no_boundary\20260403_144915.json`
- comparison saved to `F:\unified-sel\results\analysis_compare\20260403_144934.json`
- boundary report saved to `F:\unified-sel\results\analysis_boundary\20260403_144933.json`
- smoke test passed before rollback
- code change was reverted after evaluation
**Issues / Observations**:
- this was not a clean endogenous-boundary improvement
- avg accuracy fell from `0.5113` to `0.5039`
- forgetting improved sharply from `0.1148` to `0.0273`, but the gain came with a strong `task_1` accuracy drop (`0.5086 -> 0.4164`)
- relative to EWC, the run now passed both avg accuracy and forgetting, but only with a very narrow avg-accuracy margin and an obvious task-bias tradeoff
- boundary metrics improved only slightly:
  - status counts moved from `recurrent_pressure=4, transient=1, stable=0` to `recurrent_pressure=3, transient=2, stable=0`
  - mean stability score rose from `0.0900` to `0.1050`
- the protection gate fired often enough to matter, especially late in some seeds, but the mechanism behaved more like shared-output freezing than true pressure resolution
**Next Step**: keep the mature-retention version as the code baseline and look for mechanisms that reduce late recurrence without suppressing task-1 adaptation or freezing shared readout learning

---

## 2026-04-03 Balanced Pressure Routing Between Structure And Shared Output (Reverted)

**Command**:
- `python F:\unified-sel\tests\smoke_test.py`
- `python F:\unified-sel\experiments\continual\no_boundary.py --seeds 5 --max-structures 12 --window-size 50`
- `python F:\unified-sel\analysis\compare.py --fixed F:\unified-sel\results\baseline_fixed\20260403_102846.json --ewc F:\unified-sel\results\baseline_ewc\20260403_102846.json --unified F:\unified-sel\results\continual_no_boundary\20260403_152346.json`
- `python F:\unified-sel\analysis\boundary_diagnostics.py --input F:\unified-sel\results\continual_no_boundary\20260403_152346.json`
**Parameters changed**:
- temporary balanced pressure-routing gate:
  - when the pool is full and both `surprise` and `avg_tension` are high, reduce active-structure learning rate and slightly increase shared output-layer update rate so error flows more through shared adaptation and less through direct structural overwrite
**Result**:
- temporary run saved to `F:\unified-sel\results\continual_no_boundary\20260403_152346.json`
- comparison saved to `F:\unified-sel\results\analysis_compare\20260403_152405.json`
- boundary report saved to `F:\unified-sel\results\analysis_boundary\20260403_152405.json`
- smoke test passed before rollback
- code change was reverted after evaluation
**Issues / Observations**:
- this was also not a clean endogenous-boundary improvement
- avg accuracy fell from `0.5113` to `0.5043`
- forgetting worsened from `0.1148` to `0.1836`
- the mechanism increased `task_1` final accuracy (`0.5086 -> 0.5617`) but reduced `task_0` final accuracy (`0.5141 -> 0.4469`), so it mainly flipped the task bias rather than resolving pressure
- boundary metrics improved only slightly:
  - status counts moved from `recurrent_pressure=4, transient=1, stable=0` to `recurrent_pressure=3, transient=2, stable=0`
  - mean stability score rose from `0.0900` to `0.1083`
- the routing gate fired often enough to matter, especially on seeds `8/9/10`, but the effect looked like shared adaptation bias, not stable endogenous boundary formation
**Next Step**: keep the mature-retention version as the code baseline and look for mechanisms that change pressure scheduling or credit assignment without simply shifting bias from one task side to the other
