# Meta-Controller Route Audit - 2026-04-21

## Purpose

This note reviews the V0.0-V0.7 meta-controller experiment chain and checks whether important alternative routes were missed.

Verdict:

The current bottleneck is no longer memory control. It is safe dominance arbitration: when can planner use be reduced without losing success, recovery, or drift stability?

## Local Result Chain

### V0.0-V0.2

Findings:

- The experiment spine works.
- The bounded workspace state is sufficient for imitation/oracle macro-action learning.
- The flat contextual bandit is too weak.

Implication:

- The environment is learnable.
- The failure is not missing observability at the workspace level.

### V0.3-V0.5

Findings:

- Factored gates are the right direction.
- Warm-started factored control reduces memory over-read.
- Conservative fine-tuning can improve reward, but read gate can over-generalize.
- Read-disciplined fine-tuning solves most memory-read pathology.

Implication:

- Read/write control is no longer the main blocker.
- The read gate should stay frozen or updated only with explicit read-specific constraints.

### V0.6-V0.7

Findings:

- Same-step counterfactual dominance reduces planner calls but loses success.
- K-step rollout dominance reduces planner calls even more aggressively but loses more success.
- Risk-averse rollout still collapses toward too little planner use.

Implication:

- The objective is too cost-seeking.
- Longer horizon alone is not enough.
- Dominance learning must be constraint-first, not reward-difference-first.

## Internal Review Passes

### Review 1: Did We Overfit To Fixed Rule?

Partly yes.

The oracle/fixed rule is not a ground-truth optimal controller. It is a strong, transparent safety baseline. Beating it in reward while increasing drift or lowering success should not count as progress.

V0.4 briefly exceeded fixed reward but failed clean comparison because read precision deteriorated.

Audit conclusion:

- Fixed rule should remain the deployment safety anchor.
- Learned control should be evaluated as safe improvement over that anchor, not as unconstrained reward maximization.

### Review 2: Did We Confuse Planner Cost With Planner Harm?

Yes, in V0.6-V0.7.

The learned dominance variants treated planner calls as a cost to minimize. But in this environment planner also provides recovery and drift protection.

Audit conclusion:

- Planner cost should be optimized under safety constraints.
- Planner reduction is valid only if success, drift, and recovery remain within guard bands.

### Review 3: Is The Environment Too Simple?

Not yet a blocker.

The environment is simple, but it already exposes multiple failure modes:

- flat bandit over-reads memory
- unconstrained fine-tuning collapses
- read-disciplined control preserves memory but under-plans
- rollout dominance becomes too cost-seeking

Audit conclusion:

- Do not add environment complexity yet.
- First solve safe dominance arbitration in the current controlled setting.

### Review 4: Are We Missing A Key Signal In `g_t`?

Possibly.

The current `g_t` includes surprise, uncertainty, memory relevance, conflict, cost, drift, and recent failure. But it lacks explicit:

- planner safety value
- recovery debt
- regime-shift hazard estimate
- confidence that habit remains valid
- counterfactual uncertainty, not just counterfactual value

Audit conclusion:

- Add safety/risk estimates before adding more modules.
- The next useful state feature is not another memory score; it is a planner-necessity score.

## External Routes Checked

### Route A: Safe Policy Improvement / Baseline-Anchored RL

Relevant idea:

Safe policy improvement treats a known baseline as an anchor and allows policy changes only when they are likely not to degrade performance.

Sources:

- Safe Policy Improvement overview: https://www.emergentmind.com/topics/safe-policy-improvement-spi
- Offline constrained policy optimization with safe anchoring: https://www.sciencedirect.com/science/article/abs/pii/S0893608026003266

Fit to our problem:

Very high.

V0.8 should treat `fixed_rule_controller` or `imitation_controller` as the safe baseline. A learned dominance controller should only replace planner with habit when local evidence passes a safety-improvement test.

Concrete V0.8 direction:

- Start from fixed/imitation dominance.
- Candidate change: planner -> habit in a state region.
- Accept only if validation rollouts show:
  - no success drop
  - no drift increase
  - no recovery slowdown
  - compute reduction is real

### Route B: Shielded RL / Runtime Assurance

Relevant idea:

A shield monitors proposed learned actions and replaces unsafe actions with a backup policy. In our terms, the shield would override habit and restore planner when safety risk is high.

Sources:

- CACM Shields for Safe Reinforcement Learning: https://cacm.acm.org/research/shields-for-safe-reinforcement-learning/
- Adaptive robust model predictive shielding: https://www.sciencedirect.com/science/article/pii/S0098135425005241
- Shielded RL for real-time systems: https://link.springer.com/article/10.1007/s11241-025-09441-z

Fit to our problem:

Very high.

This directly matches the V0.6-V0.7 failure: the learner reduced planner too aggressively. A dominance shield can allow habit only when it is safe and otherwise fallback to planner.

Concrete V0.8 direction:

- Let learned dominance propose habit/planner.
- Add a shield:
  - if drift risk high -> planner
  - if recent failure -> planner
  - if regime hazard high -> planner
  - if memory query and memory not resolved -> planner/read
- Measure shield intervention rate.
- Success requires fewer interventions over training without increased violations.

### Route C: Rational Metareasoning / Expected Value Of Computation

Relevant idea:

Planner calls are computational actions. Their value is not raw task reward minus cost; it is expected value of computation under uncertainty and opportunity cost.

Sources:

- Principles of metareasoning: https://www.sciencedirect.com/science/article/abs/pii/000437029190015C
- Rational metareasoning and plasticity of cognitive control: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006043
- Time spent thinking reflects value of computation: https://pmc.ncbi.nlm.nih.gov/articles/PMC12553403/

Fit to our problem:

High.

Our V0.6-V0.7 rollout objective is too naive. It values planner mostly by immediate reward impact. EVOC-style scoring would ask whether planner reduces uncertainty enough to affect future action quality.

Concrete V0.8/V0.9 direction:

- Add `expected_planner_value` to `g_t`.
- Estimate it as:
  - predicted error reduction
  - drift risk reduction
  - recovery probability gain
  - minus planner cost
- Planner is selected when expected computational value exceeds cost and safety threshold.

### Route D: Option Initiation / Termination Learning

Relevant idea:

Habit and planner can be treated as options. The hard part is not only selecting an option, but learning where an option is safe to initiate and when to terminate it.

Sources:

- HRL survey / options framework: https://www.mdpi.com/2504-4990/4/1/9
- Effectively Learning Initiation Sets in HRL: https://openreview.net/forum?id=4JCVw8oMlf
- Option-Critic overview: https://www.emergentmind.com/topics/option-critic-framework

Fit to our problem:

High, but not immediate.

Our dominance problem can be reframed as:

- habit initiation set: states where habit is safe
- planner termination set: states where planner can hand control back

Concrete future direction:

- Learn `habit_safe_region(g_t)`.
- Use planner outside that region.
- Evaluate false-safe errors separately from false-planner errors.

This may be better than learning direct planner/habit action selection.

### Route E: Model Orchestration / SLM-LLM Routing

Relevant idea:

Recent model orchestration work frames small/large model selection as multi-objective RL over success, latency, and cost.

Source:

- Adaptive agentic meta-controller for SLM/LLM orchestration: https://www.sciencedirect.com/science/article/pii/S0925231226005898

Fit to our problem:

Medium.

It supports our cost-performance framing, but it is closer to model routing than cognitive dominance arbitration. It may help later for experiments that replace habit/planner with actual models.

Not recommended for immediate V0.8.

### Route F: Meta-Learning Memory Designs

Relevant idea:

Learn the memory design itself, not only memory content.

Sources:

- ALMA: https://arxiv.org/abs/2602.07755
- Agent memory survey: https://arxiv.org/abs/2603.07670
- Oblivion: https://arxiv.org/abs/2604.00131

Fit to our problem:

Medium-low for the current bottleneck.

Our memory read/write failure is mostly under control. Pushing memory design now would distract from dominance arbitration.

Not recommended for immediate V0.8.

## Overlooked Routes

### 1. Safety Shield Over Learned Dominance

This is the biggest missed route.

Instead of making the learned controller directly responsible for safe planner reduction, use:

- learned proposal policy
- hard or learned shield
- fixed planner fallback

This aligns with shielded RL and runtime assurance.

### 2. Safe Policy Improvement From Fixed Rule

We should stop comparing learned policies to fixed rule only after the fact. The fixed rule should become part of the learning algorithm as a baseline anchor.

### 3. Habit Safe-Set Learning

Dominance arbitration may be easier as a classification problem:

> In which states is habit safe enough?

This is closer to option initiation-set learning than direct policy optimization.

### 4. Planner-Value Estimator

The workspace needs a learned estimate of planner necessity:

- expected failure if habit continues
- expected drift if planner is skipped
- expected recovery benefit of planner

This is closer to EVOC than reward-only rollout.

### 5. Distributional / Tail-Risk Cost

Average drift is too weak. Planner exists to prevent tail failures, not just mean cost.

Use tail metrics:

- worst episode drift
- 95th percentile drift
- maximum consecutive failures
- recovery tail latency

This connects to CVaR / safe RL ideas.

## Routes To Pause

Do not prioritize:

- More memory architecture.
- More environment complexity.
- More flat contextual-bandit tuning.
- Direct F-drive project integration.
- Longer rollout without safety constraints.

These are unlikely to address the current bottleneck.

## Recommended V0.8

Implement `ShieldedDominanceController`.

Design:

1. Start from `imitation_controller` or `factored_warm_controller`.
2. Freeze read/write.
3. Learned dominance may propose habit.
4. Shield overrides habit to planner if any safety predicate fires:
   - `recent_failure_source > 0`
   - `invariant_violation > threshold`
   - `surprise > threshold`
   - `conflict_score > threshold`
   - memory query unresolved
5. Learn to reduce shield interventions over time, not to bypass the shield.

Metrics:

- task success
- drift
- planner calls
- shield intervention rate
- false-safe rate
- false-planner rate

Acceptance:

- `task_success >= 0.99`
- `drift <= fixed + 0.005`
- `memory_reads <= fixed + 1.0`
- `planner_calls < fixed`
- shield intervention rate decreases over training or across held-out evaluations

## Backup V0.8b

Implement `HabitSafeSetController`.

Design:

- Train a classifier for `habit_safe(g_t)`.
- Positive examples: habit succeeds without drift increase for K steps.
- Negative examples: habit causes failure, drift, or delayed recovery cost.
- Planner is used outside safe set.

This may be simpler and more stable than direct dominance RL.

## Bottom Line

The important missed idea is not another optimizer. It is baseline-anchored safe improvement.

The next credible step is:

> Learn where it is safe to remove planner, while a shield or baseline fallback prevents unsafe removals.

That is the route most aligned with our empirical failures and with the external literature on safe RL, shielding, metareasoning, and option initiation.
