# Project Gaps And Roadmap

This document is a top-down view of what `F:\unified-sel` still lacks, based on the current code, results, and experiment history.

## Executive View

The repository now has a runnable minimum evidence chain:

- runnable baselines
- runnable no-boundary Unified-SEL experiment
- runnable comparison script
- smoke verification
- a repeatable tuning loop

So the project is no longer blocked on missing entrypoints or missing core files.

What is still missing is **claim-quality evidence** and **research-grade experiment discipline** around the real project thesis:

- endogenous task-boundary formation
- boundary stability under drift and full-capacity pressure
- mechanistic explanation of retention and transfer through those internal boundaries

Baseline comparison still matters, but it is now a validation layer rather than the project definition.

## What Is Proven So Far

- The repository can run a baseline comparison end to end.
- Unified-SEL can run on a no-boundary continual stream.
- The current best tuning direction improves behavior enough for Unified-SEL to beat EWC on average accuracy.
- Unified-SEL still does not beat EWC on forgetting.
- Utility turnover is now real: clone and prune behavior can be activated by the right settings.
- Endogenous boundary pressure is observable in the traces, especially during pool fill and mid-stream regime shift.
- Two recent negative results were scientifically useful:
  - direct full-capacity replacement damaged retention
  - soft full-capacity competition rebalance mainly reallocated winner selection instead of resolving pressure

Best-known setting right now:

- `SURPRISE_THRESHOLD = 0.60`
- `UTILITY_DECAY = 0.005`
- `UTILITY_PRUNE = 0.08`
- compare using `--max-structures 12`

Best-known comparison right now:

- Fixed avg accuracy: `0.4844`
- EWC avg accuracy: `0.5020`
- Unified-SEL avg accuracy: `0.5176`
- Fixed forgetting: `0.8320`
- EWC forgetting: `0.0781`
- Unified-SEL forgetting: `0.1344`

So the project is currently at:

**boundary mechanism visible, but not yet controlled**

## Scientific Gaps

### 1. The main boundary claim is not closed yet

The current best run shows:

- Unified-SEL average accuracy > EWC
- Unified-SEL forgetting > EWC

This means there is evidence for better adaptation, but not yet for stable endogenous boundary management.

### 2. The no-boundary benchmark is still too weak for the intended claim

Current main experiment in [no_boundary.py](F:\unified-sel\experiments\continual\no_boundary.py) is still a lightweight synthetic stream:

- 4D Gaussian inputs
- two-rule labeling regime
- simple probabilistic drift

This is good enough for mechanism screening, but weak as final evidence for endogenous task-boundary formation.

Missing:

- a stronger continuous-shift benchmark
- a more explicit definition of "no boundary"
- a task family closer to the stronger continual-learning framing in `SEL-Lab`
- a cleaner separation between current-task adaptation and long-term retention
- a clearer operational definition of when an internal boundary has emerged

### 3. The endogenous-boundary story is still not mechanistically explained

Current tuning has shown:

- increasing `UTILITY_DECAY` helps
- moderate pruning helps
- increasing capacity hurts
- lowering tension threshold alone does not help

But it is still unclear how internal boundary pressure is being resolved, and why that resolution remains unstable.

Missing:

- lifecycle analysis of structures that are retained, replaced, or reused
- a clear diagnosis of when and why forgetting spikes
- a direct explanation of why clone becomes useful only in some turnover regimes
- a better answer to whether the system is truly reusing structures or only cycling them
- boundary metrics that distinguish:
  - emergence
  - persistence
  - collapse
  - pressure redistribution without true stabilization

### 4. Statistical evidence is still missing

The stated project goal requires significance, but the repository does not yet compute it.

Missing:

- seed-level statistical tests
- effect sizes or confidence intervals
- a clear pass/fail report against the project target

### 5. Baseline fairness is still somewhat fragile

The current EWC baseline is useful, but it is still evaluated in the same minimal synthetic environment.

Missing:

- validation that the benchmark is strong enough for the EWC comparison to be meaningful
- possibly another baseline or a stronger EWC protocol on a richer stream

## Engineering And Infrastructure Gaps

### 1. Configuration is still too implicit

Important knobs are still spread across:

- [pool.py](F:\unified-sel\core\pool.py)
- [learner.py](F:\unified-sel\core\learner.py)
- [no_boundary.py](F:\unified-sel\experiments\continual\no_boundary.py)

Recent tuning already exposed one wiring problem: changing a default in `learner.py` did not affect the main experiment until the experiment entrypoint was updated.

Missing:

- one canonical experiment config layer
- explicit serialization of all runtime hyperparameters into result files
- named presets for "baseline", "best known", and "diagnostic" runs

### 2. Result loading is brittle

[compare.py](F:\unified-sel\analysis\compare.py) currently loads the latest file in each result directory.

This is convenient, but it repeatedly created timing ambiguity during iterative runs.

Missing:

- explicit result selection by path or run id
- stable "best run" and "current run" references
- comparison inputs that are not implicitly time-dependent

### 3. Analysis is too shallow for the current tuning speed

`compare.py` is useful, but still minimal.

Missing:

- seed-level tables
- event and structure summaries in the report
- statistical tests
- direct target-oriented output like "passes avg accuracy / fails forgetting"

### 4. Tests are too narrow

[smoke_test.py](F:\unified-sel\tests\smoke_test.py) only verifies that major code paths do not crash.

Missing:

- regression tests for experiment wiring
- tests for result schema stability
- tests for comparison correctness
- focused tests on pool-evolution invariants

### 5. Experiment provenance is still manual

`STATUS.md` and `EXPERIMENT_LOG.md` are useful, but they are hand-maintained.

Missing:

- stable run manifests
- explicit storage of "best known setting"
- clearer distinction between current code defaults and evaluated experiment settings

## Weak Assumptions To Watch

These are the main assumptions the project currently leans on without enough proof:

- that the current synthetic drift benchmark is a good proxy for the intended no-boundary claim
- that visible boundary pressure in traces corresponds to meaningful endogenous boundary formation
- that faster utility decay helps because it improves reuse rather than merely increasing turnover
- that the current EWC implementation and benchmark pairing is already a fair enough target

These assumptions are not fatal, but they should be treated as open questions, not settled facts.

## Highest-Value Next Tasks

### Track A: Strengthen Boundary Evaluation

1. Add a stronger no-boundary benchmark variant.
2. Add richer metric collection to `no_boundary.py`.
3. Add direct forgetting diagnostics tied to structure lifecycle.

This track targets the biggest scientific gap: the system shows internal pressure and partial retention benefits, but boundary emergence and stabilization are not yet measured cleanly enough.

### Track B: Strengthen Compare And Statistics

1. Upgrade `compare.py` to compare explicit run paths or run ids.
2. Add seed-level tables and target-oriented summaries.
3. Add statistical tests for Unified-SEL vs EWC.

This track turns the project from descriptive output into defensible analysis.

### Track C: Strengthen Experiment Infrastructure

1. Introduce explicit experiment configuration.
2. Standardize result metadata and manifests.
3. Add regression tests for experiment wiring and result schemas.

This track reduces false iterations and makes the tuning loop trustworthy.

## Recommended Parallel Plan

These three tracks can proceed in parallel:

### Parallel Task 1: Boundary Diagnostics Upgrade

Goal:

- explain when an endogenous boundary appears, whether it holds, and whether it predicts later retention or transfer

Concrete outputs:

- per-window structure counts
- clone/create/branch/prune traces
- average surprise and tension traces
- a simple boundary-emergence and boundary-stability report

### Parallel Task 2: Compare And Statistics Upgrade

Goal:

- move from "latest result summary" to "explicit experiment report"

Concrete outputs:

- compare by run path or run id
- seed-level output tables
- statistical test output
- direct pass/fail summary against project goals

### Parallel Task 3: Configuration And Reproducibility Backbone

Goal:

- make tuning and replay precise instead of manual

Concrete outputs:

- explicit experiment config object or JSON config
- result manifests with all active hyperparameters
- regression tests for experiment wiring and compare logic

## Bottom Line

The project is not mainly missing code anymore.

It is mainly missing:

- stronger endogenous-boundary evidence
- cleaner boundary metrics
- stronger forgetting performance
- stronger measurement
- stronger experiment hygiene

That is a better position than a missing-implementation project, but it means progress from here should be more disciplined and less exploratory.
