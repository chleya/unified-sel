# PAPER_OUTLINE_UNIFIED_SEL

## Working Title

Shared Readout Drift as an Independent Source of Forgetting in Modular Continual Learners

## One-Sentence Thesis

In a modular continual learner with endogenous structure formation, shared readout drift is a real forgetting source, but simple output isolation is not a sufficient fix because forgetting also depends on routing and late-task specialization dynamics.

## Core Research Question

Does catastrophic forgetting in a modular continual learner come only from structure overwrite, or can a shared output readout act as an independent failure path even when structure isolation exists?

## Main Claims

1. Shared readout drift is a mechanistically meaningful source of forgetting.
2. Light `W_out` protection on the cleaned route is sufficient to beat EWC on means.
3. Output isolation is a real lever, but naive local-head designs do not solve the full problem.
4. The remaining failures are joint failures of readout interference and pool-level routing / specialization.

## Target Paper Shape

- length:
  - short paper / workshop paper
- style:
  - mechanism paper
  - not a broad capability paper
- contribution type:
  - positive mechanism result
  - negative ablations
  - failure-mode decomposition

## Section Outline

## 1. Introduction

Problem:

- modular continual learners are supposed to reduce interference by separating structure
- but in practice, forgetting can remain high even when structure-level isolation exists

Gap:

- prior intuition focuses on protecting structure weights
- less attention is given to the shared readout as a separate interference bottleneck

Our claim:

- in `Unified-SEL`, shared readout drift is an independent forgetting source
- but removing or bypassing the shared head is not enough by itself

## 2. System

Describe only what is needed:

- `StructurePool`
  - surprise-driven create / branch
  - tension-driven clone
  - utility-based persistence / pruning
- cleaned route
  - readout training uses the same pooled hidden route that inference uses
- shared readout
  - pooled hidden state mapped by `W_out`
- optional local readout probes
  - hybrid local residual
  - exclusive local

Keep this section compact.

## 3. Experimental Setup

Benchmark:

- two-rule continual stream
- no task-boundary labels
- probabilistic drift from task 0 to task 1

Metrics:

- task 0 final accuracy
- task 1 final accuracy
- forgetting on task 0
- average accuracy

Evaluation protocol:

- cleaned-route 5-seed comparison against EWC
- hard-seed `8/9` targeted probes for mechanism diagnosis

## 4. Main Result: Shared Readout Protection Improves Mean-Level Retention

Anchor result:

- cleaned-route `lambda=10`
  - [20260409_110330.json](F:/unified-sel/results/continual_no_boundary/20260409_110330.json)
- EWC baseline
  - [20260405_213122_multi_seed.json](F:/unified-sel/results/baseline_ewc/20260405_213122_multi_seed.json)

Key message:

- light `W_out` protection beats EWC on mean forgetting and mean average accuracy
- this establishes that the shared readout is not a side issue

Important restraint:

- statistical support is still weak at `n=5`
- variance becomes the next problem after route cleanup

## 5. Failure Decomposition After Route Cleanup

Anchor note:

- [CLEANED_ROUTE_VARIANCE_DIAGNOSIS_2026-04-09.md](F:/unified-sel/results/CLEANED_ROUTE_VARIANCE_DIAGNOSIS_2026-04-09.md)

Subsections:

- seed `8`: retention bias under repeated late stabilization
- seed `9`: mid-phase churn without stabilization recovery
- seed `11`: task-0 lock with task-1 sacrifice

Key message:

- after route cleanup, the next bottleneck is not Fisher strength alone
- the residual failures live in pool control policy and specialization dynamics

## 6. Output Isolation Ablations

Anchor note:

- [SELECTIVE_READOUT_PROBES_2026-04-09.md](F:/unified-sel/results/SELECTIVE_READOUT_PROBES_2026-04-09.md)

Organize as a ladder:

1. Constant-on hybrid local residual
2. Conservative gated hybrid local residual
3. Exclusive local readout
4. Strict event-window gate
5. Boundary-pressure-conditioned gate

Use them to show:

- local capacity helps adaptation
- too much local capacity destroys retention
- no shared head does not solve the full problem
- too strict a gate becomes nearly inert
- pressure-window gating is the cleanest middle regime so far, but still not a benchmark win

## 7. Discussion

Main interpretation:

- forgetting in this system is not one mechanism
- shared readout interference is necessary to model the behavior
- but routing / specialization remains a second failure path

What this means:

- modular storage isolation is not equivalent to knowledge protection
- output isolation must be studied jointly with route policy

What we do not claim:

- we do not claim current `StructurePool` is a final reasoning architecture
- we do not claim local heads solve continual learning in general

## 8. Limitations

- toy benchmark
- small sample count
- limited statistical support at `n=5`
- hard-seed ablations are mechanism probes, not benchmark finals
- capability claims beyond continual learning are out of scope

## 9. Conclusion

Short conclusion:

- shared readout drift is a real forgetting mechanism in modular continual learning
- cleaned-route `W_out` protection improves retention materially
- output isolation alone is not sufficient because routing and specialization still matter

## Must-Include Evidence

1. EWC vs cleaned-route `lambda=10`
2. route-cleanup explanation
3. seed-level variance diagnosis
4. exclusive-local negative result
5. strict-gate near-no-op control
6. pressure-window middle-regime probe

## Writing Rules

1. Do not sell this as a general small-model intelligence paper.
2. Keep the contribution narrow and mechanistic.
3. Use negative ablations as core evidence, not as appendix filler.
4. Distinguish mean improvement from variance explanation.
