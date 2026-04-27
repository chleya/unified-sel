from __future__ import annotations

from typing import Dict


DISPLAY_KEYS = (
    "task_success",
    "total_reward",
    "arbitration_regret",
    "switch_latency",
    "recovery_after_surprise",
    "compute_cost",
    "planner_calls",
    "memory_reads",
    "memory_writes",
    "memory_read_precision",
    "memory_read_recall",
    "memory_read_false_positive_rate",
    "memory_write_precision",
    "shield_interventions",
    "shield_intervention_rate",
    "drift_repairs",
    "drift_repair_rate",
    "drift_repair_delta_mean",
    "drift_repair_delta_positive_rate",
    "drift_residual_after_repair",
    "high_drift_no_repair_rate",
    "high_drift_no_repair_next_delta_mean",
    "repair_efficiency",
    "safe_set_positive_rate",
    "safe_score_mean",
    "safe_label_positive_rate",
    "necessity_positive_rate",
    "necessity_score_mean",
    "necessity_label_positive_rate",
    "action_habit_rate",
    "action_planner_rate",
    "action_read_rate",
    "action_write_rate",
    "drift_under_horizon",
)


def format_results_table(results: Dict[str, Dict[str, float]]) -> str:
    headers = ["controller", *DISPLAY_KEYS]
    rows = [" | ".join(headers), " | ".join(["---"] * len(headers))]
    for name, metrics in results.items():
        values = [name]
        for key in DISPLAY_KEYS:
            value = metrics.get(key, float("nan"))
            values.append("nan" if value != value else f"{value:.3f}")
        rows.append(" | ".join(values))
    return "\n".join(rows)


def summarize_acceptance_checks(results: Dict[str, Dict[str, float]]) -> str:
    learned = results.get("learned_contextual_bandit", {})
    habit = results.get("habit_only", {})
    planner = results.get("planner_always", {})
    memory = results.get("memory_always", {})
    random = results.get("random_controller", {})
    fixed = results.get("fixed_rule_controller", {})
    imitation = results.get("imitation_controller", {})
    oracle = results.get("oracle_macro_controller", {})
    factored = results.get("factored_controller", {})
    factored_warm = results.get("factored_warm_controller", {})
    conservative = results.get("conservative_factored_controller", {})
    disciplined = results.get("read_disciplined_factored_controller", {})
    dominance_tuned = results.get("dominance_tuned_factored_controller", {})
    counterfactual = results.get("counterfactual_dominance_controller", {})
    rollout = results.get("rollout_dominance_controller", {})
    risk_rollout = results.get("risk_averse_rollout_controller", {})
    shielded = results.get("shielded_dominance_controller", {})
    shielded_relaxed = results.get("shielded_relaxed_dominance_controller", {})
    habit_safe = results.get("habit_safe_set_controller", {})
    habit_safe_h2 = results.get("habit_safe_set_h2_controller", {})
    habit_safe_loose = results.get("habit_safe_set_loose_controller", {})
    habit_safe_tight = results.get("habit_safe_set_tight_controller", {})
    necessity = results.get("planner_necessity_controller", {})
    necessity_loose = results.get("planner_necessity_loose_controller", {})
    b3_mask_surprise = results.get("b3_mask_surprise", {})
    b3_mask_memory = results.get("b3_mask_memory", {})
    b3_mask_conflict = results.get("b3_mask_conflict", {})
    b3_mask_drift = results.get("b3_mask_drift", {})
    b3_mask_core = results.get("b3_mask_core_signals", {})

    checks = [
        (
            "learned_success_beats_habit",
            learned.get("task_success", 0.0) > habit.get("task_success", 1.0),
            learned.get("task_success", 0.0),
            habit.get("task_success", 0.0),
        ),
        (
            "learned_compute_below_planner_always",
            learned.get("compute_cost", float("inf")) < planner.get("compute_cost", 0.0),
            learned.get("compute_cost", 0.0),
            planner.get("compute_cost", 0.0),
        ),
        (
            "learned_reads_below_memory_always",
            learned.get("memory_reads", float("inf")) < memory.get("memory_reads", 0.0),
            learned.get("memory_reads", 0.0),
            memory.get("memory_reads", 0.0),
        ),
        (
            "learned_regret_below_random",
            learned.get("arbitration_regret", float("inf")) < random.get("arbitration_regret", 0.0),
            learned.get("arbitration_regret", 0.0),
            random.get("arbitration_regret", 0.0),
        ),
        (
            "learned_near_fixed_rule_reward",
            learned.get("total_reward", 0.0) >= fixed.get("total_reward", float("inf")) - 1.0,
            learned.get("total_reward", 0.0),
            fixed.get("total_reward", 0.0),
        ),
        (
            "imitation_near_oracle_reward",
            imitation.get("total_reward", 0.0) >= oracle.get("total_reward", float("inf")) - 1.0,
            imitation.get("total_reward", 0.0),
            oracle.get("total_reward", 0.0),
        ),
        (
            "learned_near_imitation_reward",
            learned.get("total_reward", 0.0) >= imitation.get("total_reward", float("inf")) - 1.0,
            learned.get("total_reward", 0.0),
            imitation.get("total_reward", 0.0),
        ),
        (
            "factored_near_imitation_reward",
            factored.get("total_reward", 0.0) >= imitation.get("total_reward", float("inf")) - 1.0,
            factored.get("total_reward", 0.0),
            imitation.get("total_reward", 0.0),
        ),
        (
            "factored_reads_below_flat_bandit",
            factored.get("memory_reads", float("inf")) < learned.get("memory_reads", 0.0),
            factored.get("memory_reads", 0.0),
            learned.get("memory_reads", 0.0),
        ),
        (
            "factored_warm_near_imitation_reward",
            factored_warm.get("total_reward", 0.0) >= imitation.get("total_reward", float("inf")) - 1.5,
            factored_warm.get("total_reward", 0.0),
            imitation.get("total_reward", 0.0),
        ),
        (
            "factored_warm_reads_below_flat_bandit",
            factored_warm.get("memory_reads", float("inf")) < learned.get("memory_reads", 0.0),
            factored_warm.get("memory_reads", 0.0),
            learned.get("memory_reads", 0.0),
        ),
        (
            "conservative_not_below_warm_reward",
            conservative.get("total_reward", 0.0) >= factored_warm.get("total_reward", float("inf")) - 0.5,
            conservative.get("total_reward", 0.0),
            factored_warm.get("total_reward", 0.0),
        ),
        (
            "conservative_reads_not_above_warm",
            conservative.get("memory_reads", float("inf")) <= factored_warm.get("memory_reads", 0.0) + 1.0,
            conservative.get("memory_reads", 0.0),
            factored_warm.get("memory_reads", 0.0),
        ),
        (
            "disciplined_reward_near_fixed",
            disciplined.get("total_reward", 0.0) >= fixed.get("total_reward", float("inf")) - 0.5,
            disciplined.get("total_reward", 0.0),
            fixed.get("total_reward", 0.0),
        ),
        (
            "disciplined_reads_near_fixed",
            disciplined.get("memory_reads", float("inf")) <= fixed.get("memory_reads", 0.0) + 1.0,
            disciplined.get("memory_reads", 0.0),
            fixed.get("memory_reads", 0.0),
        ),
        (
            "disciplined_false_positive_below_conservative",
            disciplined.get("memory_read_false_positive_rate", float("inf"))
            < conservative.get("memory_read_false_positive_rate", 0.0),
            disciplined.get("memory_read_false_positive_rate", 0.0),
            conservative.get("memory_read_false_positive_rate", 0.0),
        ),
        (
            "dominance_tuned_reward_near_fixed",
            dominance_tuned.get("total_reward", 0.0) >= fixed.get("total_reward", float("inf")) - 0.5,
            dominance_tuned.get("total_reward", 0.0),
            fixed.get("total_reward", 0.0),
        ),
        (
            "dominance_tuned_reads_near_fixed",
            dominance_tuned.get("memory_reads", float("inf")) <= fixed.get("memory_reads", 0.0) + 1.0,
            dominance_tuned.get("memory_reads", 0.0),
            fixed.get("memory_reads", 0.0),
        ),
        (
            "counterfactual_success_ge_099",
            counterfactual.get("task_success", 0.0) >= 0.99,
            counterfactual.get("task_success", 0.0),
            0.99,
        ),
        (
            "counterfactual_reads_near_fixed",
            counterfactual.get("memory_reads", float("inf")) <= fixed.get("memory_reads", 0.0) + 1.0,
            counterfactual.get("memory_reads", 0.0),
            fixed.get("memory_reads", 0.0),
        ),
        (
            "counterfactual_planner_below_fixed",
            counterfactual.get("planner_calls", float("inf")) < fixed.get("planner_calls", 0.0),
            counterfactual.get("planner_calls", 0.0),
            fixed.get("planner_calls", 0.0),
        ),
        (
            "rollout_success_ge_099",
            rollout.get("task_success", 0.0) >= 0.99,
            rollout.get("task_success", 0.0),
            0.99,
        ),
        (
            "rollout_reads_near_fixed",
            rollout.get("memory_reads", float("inf")) <= fixed.get("memory_reads", 0.0) + 1.0,
            rollout.get("memory_reads", 0.0),
            fixed.get("memory_reads", 0.0),
        ),
        (
            "rollout_planner_below_fixed",
            rollout.get("planner_calls", float("inf")) < fixed.get("planner_calls", 0.0),
            rollout.get("planner_calls", 0.0),
            fixed.get("planner_calls", 0.0),
        ),
        (
            "risk_rollout_success_ge_099",
            risk_rollout.get("task_success", 0.0) >= 0.99,
            risk_rollout.get("task_success", 0.0),
            0.99,
        ),
        (
            "risk_rollout_reads_near_fixed",
            risk_rollout.get("memory_reads", float("inf")) <= fixed.get("memory_reads", 0.0) + 1.0,
            risk_rollout.get("memory_reads", 0.0),
            fixed.get("memory_reads", 0.0),
        ),
        (
            "risk_rollout_planner_below_fixed",
            risk_rollout.get("planner_calls", float("inf")) < fixed.get("planner_calls", 0.0),
            risk_rollout.get("planner_calls", 0.0),
            fixed.get("planner_calls", 0.0),
        ),
        (
            "shielded_success_ge_099",
            shielded.get("task_success", 0.0) >= 0.99,
            shielded.get("task_success", 0.0),
            0.99,
        ),
        (
            "shielded_drift_guard",
            shielded.get("drift_under_horizon", float("inf")) <= fixed.get("drift_under_horizon", 0.0) + 0.005,
            shielded.get("drift_under_horizon", 0.0),
            fixed.get("drift_under_horizon", 0.0) + 0.005,
        ),
        (
            "shielded_reads_near_fixed",
            shielded.get("memory_reads", float("inf")) <= fixed.get("memory_reads", 0.0) + 1.0,
            shielded.get("memory_reads", 0.0),
            fixed.get("memory_reads", 0.0),
        ),
        (
            "shielded_planner_below_fixed",
            shielded.get("planner_calls", float("inf")) < fixed.get("planner_calls", 0.0),
            shielded.get("planner_calls", 0.0),
            fixed.get("planner_calls", 0.0),
        ),
        (
            "shielded_relaxed_success_ge_099",
            shielded_relaxed.get("task_success", 0.0) >= 0.99,
            shielded_relaxed.get("task_success", 0.0),
            0.99,
        ),
        (
            "shielded_relaxed_drift_guard",
            shielded_relaxed.get("drift_under_horizon", float("inf")) <= fixed.get("drift_under_horizon", 0.0) + 0.005,
            shielded_relaxed.get("drift_under_horizon", 0.0),
            fixed.get("drift_under_horizon", 0.0) + 0.005,
        ),
        (
            "shielded_relaxed_reads_near_fixed",
            shielded_relaxed.get("memory_reads", float("inf")) <= fixed.get("memory_reads", 0.0) + 1.0,
            shielded_relaxed.get("memory_reads", 0.0),
            fixed.get("memory_reads", 0.0),
        ),
        (
            "shielded_relaxed_planner_below_fixed",
            shielded_relaxed.get("planner_calls", float("inf")) < fixed.get("planner_calls", 0.0),
            shielded_relaxed.get("planner_calls", 0.0),
            fixed.get("planner_calls", 0.0),
        ),
        (
            "habit_safe_success_ge_099",
            habit_safe.get("task_success", 0.0) >= 0.99,
            habit_safe.get("task_success", 0.0),
            0.99,
        ),
        (
            "habit_safe_drift_guard",
            habit_safe.get("drift_under_horizon", float("inf")) <= fixed.get("drift_under_horizon", 0.0) + 0.005,
            habit_safe.get("drift_under_horizon", 0.0),
            fixed.get("drift_under_horizon", 0.0) + 0.005,
        ),
        (
            "habit_safe_reads_near_fixed",
            habit_safe.get("memory_reads", float("inf")) <= fixed.get("memory_reads", 0.0) + 1.0,
            habit_safe.get("memory_reads", 0.0),
            fixed.get("memory_reads", 0.0),
        ),
        (
            "habit_safe_planner_below_fixed",
            habit_safe.get("planner_calls", float("inf")) < fixed.get("planner_calls", 0.0),
            habit_safe.get("planner_calls", 0.0),
            fixed.get("planner_calls", 0.0),
        ),
        (
            "habit_safe_h2_success_ge_099",
            habit_safe_h2.get("task_success", 0.0) >= 0.99,
            habit_safe_h2.get("task_success", 0.0),
            0.99,
        ),
        (
            "habit_safe_h2_drift_guard",
            habit_safe_h2.get("drift_under_horizon", float("inf")) <= fixed.get("drift_under_horizon", 0.0) + 0.005,
            habit_safe_h2.get("drift_under_horizon", 0.0),
            fixed.get("drift_under_horizon", 0.0) + 0.005,
        ),
        (
            "habit_safe_h2_reads_near_fixed",
            habit_safe_h2.get("memory_reads", float("inf")) <= fixed.get("memory_reads", 0.0) + 1.0,
            habit_safe_h2.get("memory_reads", 0.0),
            fixed.get("memory_reads", 0.0),
        ),
        (
            "habit_safe_h2_planner_below_fixed",
            habit_safe_h2.get("planner_calls", float("inf")) < fixed.get("planner_calls", 0.0),
            habit_safe_h2.get("planner_calls", 0.0),
            fixed.get("planner_calls", 0.0),
        ),
        (
            "habit_safe_loose_success_ge_099",
            habit_safe_loose.get("task_success", 0.0) >= 0.99,
            habit_safe_loose.get("task_success", 0.0),
            0.99,
        ),
        (
            "habit_safe_loose_planner_below_fixed",
            habit_safe_loose.get("planner_calls", float("inf")) < fixed.get("planner_calls", 0.0),
            habit_safe_loose.get("planner_calls", 0.0),
            fixed.get("planner_calls", 0.0),
        ),
        (
            "habit_safe_tight_success_ge_099",
            habit_safe_tight.get("task_success", 0.0) >= 0.99,
            habit_safe_tight.get("task_success", 0.0),
            0.99,
        ),
        (
            "habit_safe_tight_planner_below_fixed",
            habit_safe_tight.get("planner_calls", float("inf")) < fixed.get("planner_calls", 0.0),
            habit_safe_tight.get("planner_calls", 0.0),
            fixed.get("planner_calls", 0.0),
        ),
        (
            "planner_necessity_success_ge_099",
            necessity.get("task_success", 0.0) >= 0.99,
            necessity.get("task_success", 0.0),
            0.99,
        ),
        (
            "planner_necessity_drift_guard",
            necessity.get("drift_under_horizon", float("inf")) <= fixed.get("drift_under_horizon", 0.0) + 0.005,
            necessity.get("drift_under_horizon", 0.0),
            fixed.get("drift_under_horizon", 0.0) + 0.005,
        ),
        (
            "planner_necessity_planner_below_fixed",
            necessity.get("planner_calls", float("inf")) < fixed.get("planner_calls", 0.0),
            necessity.get("planner_calls", 0.0),
            fixed.get("planner_calls", 0.0),
        ),
        (
            "planner_necessity_loose_success_ge_099",
            necessity_loose.get("task_success", 0.0) >= 0.99,
            necessity_loose.get("task_success", 0.0),
            0.99,
        ),
        (
            "planner_necessity_loose_planner_below_fixed",
            necessity_loose.get("planner_calls", float("inf")) < fixed.get("planner_calls", 0.0),
            necessity_loose.get("planner_calls", 0.0),
            fixed.get("planner_calls", 0.0),
        ),
        (
            "b3_mainline_success_ge_099",
            necessity_loose.get("task_success", 0.0) >= 0.99,
            necessity_loose.get("task_success", 0.0),
            0.99,
        ),
        (
            "b3_mainline_drift_guard",
            necessity_loose.get("drift_under_horizon", float("inf")) <= fixed.get("drift_under_horizon", 0.0) + 0.005,
            necessity_loose.get("drift_under_horizon", 0.0),
            fixed.get("drift_under_horizon", 0.0) + 0.005,
        ),
        (
            "b3_mainline_reads_near_fixed",
            necessity_loose.get("memory_reads", float("inf")) <= fixed.get("memory_reads", 0.0) + 1.0,
            necessity_loose.get("memory_reads", 0.0),
            fixed.get("memory_reads", 0.0),
        ),
        (
            "b3_mask_surprise_success_ge_099",
            b3_mask_surprise.get("task_success", 0.0) >= 0.99,
            b3_mask_surprise.get("task_success", 0.0),
            0.99,
        ),
        (
            "b3_mask_memory_success_ge_099",
            b3_mask_memory.get("task_success", 0.0) >= 0.99,
            b3_mask_memory.get("task_success", 0.0),
            0.99,
        ),
        (
            "b3_mask_conflict_success_ge_099",
            b3_mask_conflict.get("task_success", 0.0) >= 0.99,
            b3_mask_conflict.get("task_success", 0.0),
            0.99,
        ),
        (
            "b3_mask_drift_success_ge_099",
            b3_mask_drift.get("task_success", 0.0) >= 0.99,
            b3_mask_drift.get("task_success", 0.0),
            0.99,
        ),
        (
            "b3_mask_core_success_drop",
            b3_mask_core.get("task_success", 1.0) < necessity_loose.get("task_success", 0.0),
            b3_mask_core.get("task_success", 0.0),
            necessity_loose.get("task_success", 0.0),
        ),
    ]

    lines = ["", "Acceptance check | pass | learned | baseline", "--- | --- | --- | ---"]
    for name, passed, learned_value, baseline_value in checks:
        lines.append(f"{name} | {str(passed).lower()} | {learned_value:.3f} | {baseline_value:.3f}")
    return "\n".join(lines)
