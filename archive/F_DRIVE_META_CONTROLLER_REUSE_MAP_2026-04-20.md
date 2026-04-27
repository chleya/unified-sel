# F Drive Reuse Map For Learned Meta-Controller Experiments - 2026-04-20

## Scope

This document maps local F-drive projects that are useful for the learned meta-controller / unified-control experiment described in:

- `META_CONTROLLER_EXPERIMENT_PROTOCOL_2026-04-20.md`

The question is not which project has the most ambitious narrative. The question is which existing code or design assets can help build a falsifiable experiment for:

- bounded shared state
- module arbitration
- habit vs deliberation switching
- memory read/write control
- long-horizon drift and recovery
- ablation and transfer evaluation

This is a reuse map, not a validation claim.

## Executive Ranking

| Rank | Project | Usefulness | Best Use |
|---|---|---:|---|
| 1 | `F:\cognitive-execution-engine` | Very high | Runtime, event log, policy boundary, WorldState / CommitmentEvent, audit/replay |
| 2 | `F:\sel-lab` | Very high | Learned router / selector benchmark patterns, ablation matrix, policy registry |
| 3 | `F:\SDAS` | High | Structure pool, action-driven update, world model loop, regime-shift signals |
| 4 | `F:\diff_world` | High | Minimal dynamic environment and prediction-error / delta-prediction baseline |
| 5 | `F:\FCRS-World` / `F:\fcrs-v5` | High | Predictive compression, fixed-capacity representation, lightweight multi-step planner |
| 6 | `F:\SDAS-LLM\sdas-structure-memory` | Medium-high | Lightweight JSON memory prototype with novelty / structure signals |
| 7 | `F:\highway-agent-kernel` | Medium | Governance/control-kernel pattern, policy explanation, invariant guard |
| 8 | `F:\system_stability` | Medium | Drift/reset/stability metrics and monitor concepts |
| 9 | `F:\diff_mdp` / `F:\dmdp_experiment` | Medium | Disagreement as error/uncertainty signal |
| 10 | `F:\SymbolFlow` | Medium-low | Conceptual architecture, visualization / internal simulation ideas |
| 11 | `F:\agent-evolution` | Low-medium | Narrative memory file practice; limited experimental code |
| 12 | `F:\physical-agi` | Low | Mostly workspace identity files; no obvious experimental substrate |

## Recommended Integration Plan

### Build The Experiment In `F:\unified-sel`

Do not move the main experiment into any single source project. Keep `unified-sel` as the integration repo because it already has:

- capability benchmark infrastructure
- TopoMem health-monitor work
- experiment logs and status discipline
- current pivot rules that prevent overclaiming

Use other projects as read-only references or copied minimal modules after review.

### Suggested Module Sources

| Experiment Need | Primary Source | Backup Source |
|---|---|---|
| Event log and replayable state transitions | `cognitive-execution-engine` | `highway-agent-kernel` |
| Bounded shared state / world state | `cognitive-execution-engine` | `SymbolFlow` architecture notes |
| Meta-controller policy registry | `sel-lab` Phase 3 registry | `unified-sel` capability policies |
| Learned router training/evaluation | `sel-lab` learned router benchmarks | `SDAS` meta-structure learning |
| Dynamic environment | `diff_world` | `SDAS` DigitalPetriDish |
| Predictor / surprise | `diff_world`, `FCRS-World` | `SDAS` world model |
| Planner | `FCRS-World` multi-step predictor | `SDAS` world model action policy |
| Habit policy | simple Q / cached policy from `SDAS` or new minimal policy | `FCRS` fixed selector |
| Episodic memory | `SDAS-LLM/sdas-structure-memory` | `cognitive-execution-engine` memory store/index |
| Drift / recovery metrics | `system_stability` concepts | `unified-sel` TopoMem-OBD metrics |
| Governance / invariant guard | `cognitive-execution-engine` policy | `highway-agent-kernel` policy engine |

## Project Notes

### `F:\cognitive-execution-engine`

Best assets:

- fixed safety core and explicit authority model
- typed state transitions
- event log and replay semantics
- `WorldState`, `CommitmentEvent`, `ModelRevisionEvent`
- `CommitmentPolicy`, `RevisionPolicy`, approval semantics
- confidence gate for belief/memory promotion
- memory store/index direction
- experiments comparing full-stack state machinery against stacked frameworks

How to use:

- Use it as the runtime/audit skeleton for meta-controller experiments.
- Reuse the idea that every macro-action should become an event.
- Treat memory writes as policy-mediated state transitions.
- Use its `WorldState` ideas to define the bounded `g_t`, but keep `g_t` smaller than a full world model.

Risks:

- Some memory/router components are marked by its own docs as insufficiently validated.
- Its architecture is larger than the minimal experiment needs.
- Do not inherit the entire WorldState layer at once; extract event/replay/policy boundaries first.

Verdict:

Use as the strongest engineering base, not as proof that the learned controller already exists.

### `F:\sel-lab`

Best assets:

- Phase 3 policy registry and family grouping
- many selector / router / regime-switch variants
- learned router benchmark over regime features
- selector generalization benchmark
- ablation/report-generation style
- explicit negative findings about selector limits

How to use:

- Reuse its policy registry pattern for meta-controller macro-actions.
- Reuse its benchmark matrix discipline: compare learned selector, fixed selector, oracle, ablations, transfer.
- Reuse its stress-map idea for held-out environment families.
- Adapt its learned-router training shape for arbitration over `habit`, `planner`, `memory_assisted_planner`, and `hold`.

Risks:

- It is about structural reuse in continual learning, not whole-system cognitive arbitration.
- Many selector variants are tightly coupled to Phase 3 task-specialist semantics.
- Prior results include saturated or negative selector directions; do not cherry-pick.

Verdict:

Best source for falsification protocol, policy registry, and learned-router benchmark mechanics.

### `F:\SDAS`

Best assets:

- structure pool with prototype, utility, surprise history, action values, decay, pruning
- action-driven updates
- simple world model predicting latent and reward
- action policy using active structures
- regime shift history inside the structure pool
- meta-structure learning module with structure memory and strategy adaptation
- DigitalPetriDish / long-term / transfer experiments

How to use:

- Reuse the structure-pool pattern for bounded working memory candidates.
- Use SDAS's action-driven update loop as an example of coupling memory/structure to action success.
- Borrow the DigitalPetriDish environment if `diff_world` is too small.
- Use `meta_structure_learning.py` only as inspiration, not as a finished meta-policy.

Risks:

- The code is prototype-heavy and uses many hand-tuned thresholds.
- Several claims are specific to custom grid environments.
- It can tempt the experiment back toward "structure pool is intelligence" rather than arbitration.

Verdict:

High-value source for habit/structure/memory mechanics and regime signals.

### `F:\diff_world`

Best assets:

- minimal continuous dynamic environments
- absolute-state predictor vs delta predictor contrast
- drift, noise, moving goals, obstacles
- simple models easy to rewrite into experiment-local modules

How to use:

- Use as the seed for the minimal POMDP-like environment.
- Use prediction error from delta model as `surprise`.
- Add hidden regimes and delayed memory facts on top.

Risks:

- Current tasks are not partial-observation memory tasks.
- Existing scripts are experiment prototypes, not clean reusable package code.

Verdict:

Best lightweight starting point for the synthetic environment and predictor/surprise signal.

### `F:\FCRS-World` And `F:\fcrs-v5`

Best assets:

- prediction-oriented compression
- fixed-capacity representation pool
- multi-step forward prediction
- forward decision binding
- memory/cost-aware lightweight design
- explicit boundary analysis about when it works and when it fails

How to use:

- Reuse multi-step predictor as the cheap planner or planner baseline.
- Use fixed-capacity representation ideas for bounded `g_t`.
- Use prediction-error metrics as part of uncertainty/surprise.

Risks:

- FCRS is a representation/planning mechanism, not a meta-controller.
- Some reported gains are environment-specific.

Verdict:

Strong source for predictor/planner internals and resource-constrained design.

### `F:\SDAS-LLM\sdas-structure-memory`

Best assets:

- small JSON-backed structure memory layer
- novelty scoring
- decay, merge, prune, active structures
- examples for agent-loop integration
- output fields like `active_structures`, `novelty`, `recommended_focus`, `state_summary`

How to use:

- Use as the first episodic/semantic memory prototype.
- Extend its read/write path with explicit meta-controller gates.
- Log read/write precision against hidden environment labels.

Risks:

- Tokenization is deliberately simple.
- It is a planner hint layer, not a reliable truth store.
- It does not natively measure causal memory contribution.

Verdict:

Good for fast memory-control prototype; should later be replaced or hardened.

### `F:\highway-agent-kernel`

Best assets:

- minimal control kernel
- task contract
- policy engine
- belief-style scores
- invariant guard
- approval gate
- self-review and drift check fields
- policy explanation tool

How to use:

- Borrow governance shape for experiment safety and invariant checks.
- Use its explicit `belief_state` style as a model for inspectable controller diagnostics.
- Use its self-review/drift-check concept for run reports.

Risks:

- It is a human-governed project-management kernel, not a learning experiment.
- Its policy is rule-based, not learned.

Verdict:

Useful for engineering discipline and report shape, not for the learning core.

### `F:\system_stability`

Best assets:

- reset/stability/drift metrics
- monitor experiments
- long-run stability thinking
- phase transition and instability threshold framing

How to use:

- Borrow drift metric style and recovery-after-reset framing.
- Use as conceptual source for `D_drift`, `R_recovery`, and stability threshold plots.

Risks:

- Hebbian/spectral reset work is far from agent arbitration.
- Reported claims are not directly transferable to symbolic/agent tasks.

Verdict:

Use for metrics and plots, not architecture.

### `F:\diff_mdp` And `F:\dmdp_experiment`

Best assets:

- dual-model disagreement as error detector
- structured output difference metrics
- AUROC/precision/recall evaluation of disagreement signal

How to use:

- Use model disagreement as one candidate `conflict_score`.
- Add signal-masking tests: remove conflict/disagreement and measure switch latency.

Risks:

- Current code is mostly mock/prototype.
- It does not include a full decision-control loop.

Verdict:

Useful for uncertainty/conflict feature engineering.

### `F:\SymbolFlow`

Best assets:

- architecture notes linking CEE, WorldState, commitment/revision events, memory, uncertainty routing
- visual symbolic graph interface
- internal simulation / candidate generation concept

How to use:

- Use as conceptual design reference for how to visualize controller state and transitions.
- Do not base the first experiment on its UI.

Risks:

- Mostly frontend and architecture document.
- It risks expanding into "internal universe" construction before experiment proof.

Verdict:

Useful for visualization and framing after the minimal experiment works.

### `F:\agent-evolution`

Best assets:

- file-based long-term memory practice
- unified memory JSON schema idea
- working notes around persistent assistant behavior

How to use:

- Use only for rough memory schema inspiration.

Risks:

- Mostly workspace/persona/process documents.
- Not a controlled experimental system.

Verdict:

Low direct reuse for this goal.

### `F:\physical-agi`

Observed assets:

- workspace identity files only in the inspected scope.

Verdict:

No meaningful reuse for the current experiment unless more code exists elsewhere.

## Projects To Avoid As Primary Substrate

Do not start the meta-controller experiment in:

- `F:\agent-evolution`: too process/persona-oriented.
- `F:\system_stability`: too far from agent arbitration.
- `F:\SymbolFlow`: too UI/architecture-heavy for first proof.
- `F:\highway-agent-kernel`: rule-governed, not learning-driven.
- `F:\SDAS-LLM`: useful memory layer, not full experiment stack.

## Minimal Implementation Stack

The smallest credible implementation can be assembled as:

1. `unified-sel` as integration repo.
2. `diff_world` style environment, rewritten as `experiments/meta_controller/`.
3. `FCRS-World` or `diff_world` predictor as `Predictor`.
4. Simple Q/habit policy from `SDAS` pattern.
5. Lightweight planner from `FCRS-World` multi-step lookahead.
6. JSON episodic memory adapted from `SDAS-LLM/sdas-structure-memory`.
7. Meta-controller registry and ablation matrix adapted from `sel-lab`.
8. Event logging / state transition schema inspired by `cognitive-execution-engine`.
9. Drift/recovery plots inspired by `system_stability`.

## Proposed First Code Layout

```text
unified-sel/
  experiments/
    meta_controller/
      README.md
      env.py
      predictor.py
      policies.py
      memory.py
      meta_controller.py
      baselines.py
      metrics.py
      run_experiment.py
      report.py
  tests/
    test_meta_controller_protocol.py
```

## First Milestone

Target a non-neural, fully logged version first:

- discrete macro-actions
- symbolic hidden regimes
- tabular/contextual-bandit meta-controller
- fixed-rule replacement baseline
- random dominance baseline
- oracle upper bound
- uncertainty/surprise/cost/memory-relevance/invariant signal masking

Acceptance for first milestone:

- metrics compute correctly
- controller logs every macro-action
- fixed-rule and random baselines run on the same seeds
- hidden regime is logged for evaluation but never exposed to the controller
- report includes negative conclusion if learned controller does not beat rules under transfer

## Main Engineering Warning

The biggest risk is not lack of code. The F drive has enough pieces.

The risk is mixing too many existing narratives:

- structure pool
- world model
- memory
- policy kernel
- symbolic graph
- stability reset
- predictive compression

The experiment should keep only what is needed to test one claim:

> Does a learned meta-controller causally improve when to switch, who dominates, when to read/write memory, and how to recover after shifts?

Everything else is support machinery.
