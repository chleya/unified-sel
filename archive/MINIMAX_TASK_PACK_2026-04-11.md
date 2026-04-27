# MINIMAX_TASK_PACK_2026-04-11

**Date**: 2026-04-11
**Status**: Ready-to-assign task pack
**Parent Plan**: `MAINLINE_EXECUTION_PLAN_2026-04-11.md`
**Execution Model**: You assign; minimax implements; this pack defines concrete work contracts.

---

## 1. Usage Rules

1. Each task package must have one explicit owner (single accountability).
2. Each task package must ship:
- runnable command(s)
- result artifacts (JSON + summary)
- threshold-based pass/fail statement
3. No package is considered complete with narrative only.
4. `weight_graph` packages are sidecar and must not block A-track schedule.
5. If a task fails gates, return `HOLD` with blocker and fallback plan.

---

## 2. Assignment Board

| ID | Track | Priority | Owner (fill) | ETA | Depends On |
|---|---|---|---|---|---|
| A1 | A | P0 | minimax- | 1 day | - |
| A2 | A | P0 | minimax- | 1 day | A1 |
| A3 | A | P0 | minimax- | 2 days | A1 |
| A4 | A | P0 | minimax- | 1 day | A2,A3 |
| A5 | A | P0 | minimax- | 1 day | A4 |
| A6 | A | P0 | minimax- | 1 day | A5 |
| A7 | A | P1 | minimax- | 1 day | A5 |
| A8 | A | P1 | minimax- | 2 days | A6 |
| A9 | A | P1 | minimax- | 1 day | A8 |
| A10 | A | P1 | minimax- | 1 day | A8 |
| A11 | A | P1 | minimax- | 2 days | A10 |
| A12 | A | P1 | minimax- | 1 day | A11 |
| A13 | A | P0 | minimax- | 1 day | A12 |
| W1 | W | P1 | minimax- | 0.5 day | - |
| W2 | W | P1 | minimax- | 1 day | W1 |
| W3 | W | P1 | minimax- | 0.5 day | W1 |
| W4 | W | P1 | minimax- | 1 day | W2,W3 |
| W5 | W | P1 | minimax- | 1 day | W2,W3 |
| W6 | W | P1 | minimax- | 0.5 day | W4,W5 |
| W7 | W | P1 | minimax- | 1 day | W4,W5 |
| W8 | W | P1 | minimax- | 1 day | W7 |
| W9 | W | P1 | minimax- | 1 day | W7,W8 |
| W10 | W | P1 | minimax- | 1 day | W9 |
| W11 | W | P1 | minimax- | 0.5 day | W10 |
| W12 | W | P0 | minimax- | 0.5 day | W11 |

---

## 3. A-Track Task Packages (Mainline)

### A1. Protocol Hygiene Patch

- Objective: remove evaluation leakage in routing decision path and lock a clean protocol contract.
- Files:
- `core/capability_benchmark.py`
- `experiments/capability/benchmark.py`
- `tests/smoke_test.py`
- Input:
- existing protocol logic in `_run_protocol`
- Output:
- patched protocol behavior + explicit regression tests
- artifact note in `results/decision_notes/A1_protocol_hygiene.md`
- Acceptance:
- no branch should use verifier outcome before route decision for "accepted_without_verifier" path
- smoke tests pass
- command reproducibility documented
- Risks:
- behavior metrics may shift; must be explicitly reported as protocol correction

### A2. `stronger_paraphrase` Variant

- Objective: add stronger textual/task paraphrase perturbation beyond current light paraphrase.
- Files:
- `core/capability_benchmark.py`
- `experiments/capability/generalization.py`
- `tests/smoke_test.py`
- Output:
- new variant key `stronger_paraphrase`
- deterministic generation path (seed-stable)
- Acceptance:
- variant selectable from CLI
- run completes with saved JSON
- no breakage to existing `standard/paraphrase`

### A3. `naturalized` Code-Repair Slice

- Objective: add more natural task phrasing/problem statements not handcrafted to existing family template style.
- Files:
- `core/capability_benchmark.py`
- `experiments/capability/generalization.py`
- `tests/smoke_test.py`
- Output:
- variant key `naturalized`
- at least 20 tasks with hidden-test evaluation contract preserved
- Acceptance:
- end-to-end run saves JSON for `naturalized`
- task metadata includes `suite_variant=naturalized`

### A4. Transfer Matrix Summarizer

- Objective: produce canonical matrix report over domains x monitors x solvers.
- Files:
- new `experiments/capability/transfer_matrix.py`
- optional `analysis/` helper
- Output:
- `results/capability_generalization/*`
- `results/decision_notes/A4_transfer_matrix.md`
- matrix schema:
- Domain: `standard/paraphrase/stronger_paraphrase/naturalized`
- Monitor: `semantic/semantic+guard/counterfactual/confidence`
- Solver: `search/heuristic`
- Acceptance:
- single command generates matrix JSON + markdown summary

### A5. Guard-Band Sweep Runner

- Objective: sweep `low_signal_guard_band` for Pareto analysis.
- Files:
- new `experiments/capability/pareto_sweep.py`
- Output:
- `results/capability_pareto/*.json`
- Sweep spec:
- range `0.00..0.30`
- step `0.02`
- Acceptance:
- complete sweep over required domains and monitors
- each point contains success/cost/verifier/escalation stats

### A6. Pareto Report Generator

- Objective: convert sweep output into frontier report.
- Files:
- new `experiments/capability/pareto_report.py`
- Output:
- `results/capability_pareto/*_frontier.json`
- `results/decision_notes/A6_pareto_report.md`
- Acceptance:
- identifies frontier points and dominated points
- includes explicit comparison vs `counterfactual`

### A7. Multi-Seed Stability

- Objective: ensure conclusions are not single-seed artifacts.
- Files:
- `experiments/capability/generalization.py` or dedicated runner
- Output:
- multi-seed aggregate JSON (>=3 seeds)
- Acceptance:
- report mean/std for key metrics
- ordering direction (`semantic+guard` vs baselines) remains stable

### A8. Deterministic Hybrid Router v1

- Objective: implement minimal policy composition (no learning).
- Files:
- `core/capability_benchmark.py`
- possibly new `core/hybrid_policy.py`
- Output:
- policy using semantic prior + correction signals
- Acceptance:
- hybrid selectable via CLI / monitor-policy key
- reproducible benchmark output

### A9. Correction Path Concentration Analysis

- Objective: identify whether difficult cases concentrate in correction path.
- Files:
- new `analysis/correction_path_analysis.py`
- Output:
- `results/capability_hybrid/*_correction_path.json`
- Acceptance:
- clear breakdown by decision path and error class

### A10. Hybrid Ablation Matrix

- Objective: test component necessity in hybrid policy.
- Files:
- runner + analysis scripts
- Output:
- ablation table: remove one component at a time
- Acceptance:
- report deltas on success/cost vs full hybrid

### A11. Minimal Real Shell Adapter

- Objective: connect router to a minimal real request stream.
- Files:
- new `experiments/capability/shell_demo.py`
- Output:
- runnable shell demo and request log format
- Acceptance:
- at least one real request flow end-to-end

### A12. Shell Logging + Cost/Quality Stats

- Objective: standardize shell telemetry and evaluate trend consistency.
- Files:
- `experiments/capability/shell_demo.py`
- new `analysis/shell_metrics.py`
- Output:
- `results/capability_hybrid/*_shell_metrics.json`
- Acceptance:
- shell trend direction compared against benchmark trend

### A13. 30-Day Mainline Decision Note

- Objective: final Go/Hold/No-Go for A-track.
- Output:
- `results/decision_notes/MAINLINE_DECISION_2026-05-11.md`
- Must answer:
1. external validity
2. Pareto advantage
3. hybrid value
4. shell consistency
- Acceptance:
- explicit 4-question verdict with evidence links

---

## 4. W-Track Task Packages (`weight_graph` Sidecar)

### W1. Branch Reframe Note

- Objective: formally downgrade `Topo x capability` and freeze the fragility-prior framing.
- Files:
- `weight_graph/FRAGILITY_PRIOR_REALIGNMENT_2026-04-12.md`
- Acceptance:
- branch scope, downgrade reason, and new stop/go are documented

### W2. `#490` Activation Profile Design

- Objective: define the bounded follow-up for `#490` under the fragility frame.
- Files:
- new `weight_graph/experiments/exp10_activation_profile.py`
- Output:
- `results/weight_graph/exp10/activation_profile_design.json`
- Acceptance:
- design covers `#490`, same-layer high-PR control, and random control

### W3. Control Contract

- Objective: freeze control selection and reporting schema for the fragility follow-up.
- Files:
- new `weight_graph/CONTROL_CONTRACT_2026-04-12.md`
- Acceptance:
- controls, metrics, and pass/fail logic are explicit

### W4. `#490` Activation/Profile Run

- Objective: run the actual `#490` profile collection.
- Output:
- `results/weight_graph/exp10/activation_profile_490.json`
- Acceptance:
- layer/task-type summaries saved with reproducible command

### W5. Same-Layer High-PR Control Profile

- Objective: measure whether the profile is distinct from a same-layer hub-like control.
- Output:
- `results/weight_graph/exp10/activation_profile_high_pr_control.json`
- Acceptance:
- direct comparison against `#490` is present

### W6. Random Control Profile

- Objective: compare against a random neuron baseline.
- Output:
- `results/weight_graph/exp10/activation_profile_random_control.json`
- Acceptance:
- direct comparison against `#490` is present

### W7. Topology-Based Intervention Ranking

- Objective: rank candidate fragile sites using topology signals.
- Output:
- `results/weight_graph/exp11/intervention_ranking.json`
- Acceptance:
- candidate ranking schema is reproducible

### W8. Random-vs-Topology Ranking Check

- Objective: test whether topology-based ranking beats random intervention choice.
- Output:
- `results/weight_graph/exp11/ranking_vs_random.json`
- Acceptance:
- explicit uplift or no-uplift verdict

### W9. Monitoring/Fragility Summary

- Objective: summarize whether the branch has practical fragility value.
- Output:
- `results/weight_graph/exp11/fragility_summary.md`
- Acceptance:
- clear usable/not-usable verdict with evidence links

### W10. Observation-Only Fallback

- Objective: prepare fallback write-up if the fragility line fails.
- Output:
- `results/decision_notes/WEIGHT_GRAPH_OBSERVATION_FALLBACK_2026-05-11.md`
- Acceptance:
- observation-only wording is ready and evidence-linked

### W11. Fragility-Prior Note

- Objective: write the positive note if fragility value survives.
- Output:
- `results/decision_notes/WEIGHT_GRAPH_FRAGILITY_NOTE_2026-05-11.md`
- Acceptance:
- practical use cases are stated narrowly

### W12. Branch Decision Memo

- Objective: final sidecar decision (bounded follow-up / observation-only / usable fragility sidecar).
- Output:
- `results/decision_notes/WEIGHT_GRAPH_BRANCH_DECISION_2026-05-11.md`
- Acceptance:
- explicit promotion criteria checkboxes

---

## 5. Command Templates (fill and run)

A-track templates:

```powershell
python F:\unified-sel\experiments\capability\generalization.py --suite code --protocol monitor_repair_triage --variants standard paraphrase stronger_paraphrase naturalized --routing-monitors semantic counterfactual confidence --local-solver search --num-tasks 20 --seed 7 --routing-signal-threshold 0.5 --escalation-signal-threshold 0.9 --low-signal-guard-band 0.15
```

```powershell
python F:\unified-sel\experiments\capability\pareto_sweep.py --suite code --protocol monitor_repair_triage --variants standard paraphrase stronger_paraphrase naturalized --routing-monitors semantic counterfactual confidence --local-solvers search heuristic --num-tasks 20 --seeds 7 11 17 --guard-min 0.00 --guard-max 0.30 --guard-step 0.02
```

W-track templates:

```powershell
python F:\unified-sel\weight_graph\experiments\exp10_activation_profile.py --target 490 --control high_pr --control random
```

```powershell
python F:\unified-sel\weight_graph\experiments\exp11_fragility_ranking.py --model 1.5b --top-k 20
```

---

## 6. Global Acceptance Gates (copy into each handoff)

A-track global gate:

1. External validity: on `stronger_paraphrase + naturalized`, success gap vs `counterfactual` <= 3pp.
2. Pareto: `semantic+guard` must be frontier or near-frontier on at least 2 shifted domains.
3. Stability: >=3 seeds with consistent direction.
4. Hybrid: measurable success-cost gain over best single-signal baseline.

W-track global gate:

1. Topology-based ranking must beat random or naive intervention choice.
2. `#490` or related candidates must remain distinct from control profiles.
3. The branch must show practical fragility or monitoring value.

---

## 7. Daily Standup Reporting Format (for all minimax agents)

Use this exact schema in each update:

- Task ID:
- Status: `RUNNING / BLOCKED / DONE / HOLD`
- Command(s) run:
- Artifact path(s):
- Key metrics:
- Gate pass/fail:
- Blocker (if any):
- Next action:

---

## 8. Immediate Dispatch Order (today)

Dispatch this exact sequence:

1. A1 -> A2 -> A3 (parallel after A1 contract freeze)
2. A4 after A2/A3 complete
3. W1 and W2 in parallel with A2/A3
4. A5 starts once A4 available

This order maximizes critical-path speed without letting W-track block A-track.

