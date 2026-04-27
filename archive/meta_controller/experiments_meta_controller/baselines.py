from __future__ import annotations

from typing import Dict

from experiments.meta_controller.meta_controller import (
    AlwaysController,
    BaseMetaController,
    ContextualBanditController,
    ConservativeFactoredController,
    CounterfactualDominanceController,
    DominanceTunedFactoredController,
    DriftAwarePlannerNecessityController,
    FactoredController,
    FixedRuleController,
    HabitSafeSetController,
    ImitationController,
    OracleMacroController,
    PlannerNecessityController,
    RandomController,
    ReadDisciplinedFactoredController,
    RiskAverseRolloutDominanceController,
    RolloutDominanceController,
    ShieldedDominanceController,
)


def build_controllers(seed: int = 0) -> Dict[str, BaseMetaController]:
    return {
        "habit_only": AlwaysController("habit"),
        "planner_always": AlwaysController("planner"),
        "memory_always": AlwaysController("planner_read_write"),
        "fixed_rule_controller": FixedRuleController(),
        "oracle_macro_controller": OracleMacroController(),
        "random_controller": RandomController(seed=seed),
        "learned_contextual_bandit": ContextualBanditController(seed=seed),
        "imitation_controller": ImitationController(seed=seed),
        "factored_controller": FactoredController(seed=seed),
        "factored_warm_controller": FactoredController(seed=seed + 17, epsilon=0.0),
        "conservative_factored_controller": ConservativeFactoredController(seed=seed + 23),
        "read_disciplined_factored_controller": ReadDisciplinedFactoredController(seed=seed + 29),
        "dominance_tuned_factored_controller": DominanceTunedFactoredController(seed=seed + 31),
        "counterfactual_dominance_controller": CounterfactualDominanceController(seed=seed + 37),
        "rollout_dominance_controller": RolloutDominanceController(seed=seed + 41, rollout_horizon=3),
        "risk_averse_rollout_controller": RiskAverseRolloutDominanceController(seed=seed + 43, rollout_horizon=3),
        "shielded_dominance_controller": ShieldedDominanceController(seed=seed + 47),
        "shielded_relaxed_dominance_controller": ShieldedDominanceController(
            seed=seed + 53,
            surprise_threshold=0.62,
            conflict_threshold=1.10,
            drift_threshold=0.08,
        ),
        "habit_safe_set_controller": HabitSafeSetController(seed=seed + 59),
        "habit_safe_set_h2_controller": HabitSafeSetController(seed=seed + 61, safe_bias=-0.05),
        "habit_safe_set_loose_controller": HabitSafeSetController(seed=seed + 67, safe_threshold=-0.08),
        "habit_safe_set_tight_controller": HabitSafeSetController(seed=seed + 71, safe_threshold=0.08),
        "planner_necessity_controller": PlannerNecessityController(seed=seed + 79, safe_threshold=-0.08),
        "planner_necessity_loose_controller": PlannerNecessityController(
            seed=seed + 83,
            safe_threshold=-0.08,
            necessity_threshold=0.08,
        ),
        "drift_aware_planner_necessity_controller": DriftAwarePlannerNecessityController(
            seed=seed + 89,
            safe_threshold=-0.08,
            necessity_threshold=0.08,
            drift_repair_threshold=0.08,
        ),
        "drift_aware_planner_necessity_loose_controller": DriftAwarePlannerNecessityController(
            seed=seed + 97,
            safe_threshold=-0.08,
            necessity_threshold=0.08,
            drift_repair_threshold=0.14,
        ),
    }
