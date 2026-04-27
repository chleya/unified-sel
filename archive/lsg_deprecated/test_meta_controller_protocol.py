from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.meta_controller.env import EnvConfig
from experiments.meta_controller.run_experiment import (
    run_b7_acceptance_artifact,
    run_b7_transfer_matrix,
    run_drift_threshold_sweep,
    run_episode,
    run_repair_benefit_analysis,
    run_suite,
    run_train_eval_suite,
)
from experiments.meta_controller.baselines import build_controllers


def test_meta_controller_suite_runs_all_baselines() -> None:
    results = run_suite(episodes=3, seed=0)
    expected = set(build_controllers(seed=0))
    assert set(results) == expected
    for metrics in results.values():
        assert "task_success" in metrics
        assert "arbitration_regret" in metrics
        assert "drift_under_horizon" in metrics
        assert "drift_repair_delta_mean" in metrics
        assert "high_drift_no_repair_rate" in metrics
        assert "action_planner_rate" in metrics
        assert "shield_intervention_rate" in metrics


def test_planner_beats_habit_after_regime_shifts() -> None:
    config = EnvConfig(seed=0)
    controllers = build_controllers(seed=0)
    habit = run_episode(config, controllers["habit_only"], seed=1)
    planner = run_episode(config, controllers["planner_always"], seed=1)
    assert planner["task_success"] >= habit["task_success"]
    assert planner["compute_cost"] > habit["compute_cost"]


def test_memory_always_reads_when_memory_is_required() -> None:
    config = EnvConfig(seed=2)
    controller = build_controllers(seed=2)["memory_always"]
    metrics = run_episode(config, controller, seed=2)
    assert metrics["memory_reads"] > 0
    assert metrics["memory_read_precision"] > 0


def test_train_eval_suite_includes_signal_ablations() -> None:
    results = run_train_eval_suite(train_episodes=3, eval_episodes=3, seed=3)
    assert "learned_contextual_bandit" in results
    assert "imitation_controller" in results
    assert "oracle_macro_controller" in results
    assert "factored_controller" in results
    assert "factored_warm_controller" in results
    assert "conservative_factored_controller" in results
    assert "read_disciplined_factored_controller" in results
    assert "dominance_tuned_factored_controller" in results
    assert "counterfactual_dominance_controller" in results
    assert "rollout_dominance_controller" in results
    assert "risk_averse_rollout_controller" in results
    assert "shielded_dominance_controller" in results
    assert "shielded_relaxed_dominance_controller" in results
    assert "habit_safe_set_controller" in results
    assert "habit_safe_set_h2_controller" in results
    assert "habit_safe_set_loose_controller" in results
    assert "habit_safe_set_tight_controller" in results
    assert "planner_necessity_controller" in results
    assert "planner_necessity_loose_controller" in results
    assert "drift_aware_planner_necessity_controller" in results
    assert "drift_aware_planner_necessity_loose_controller" in results
    assert "learned_mask_surprise" in results
    assert "learned_mask_memory" in results
    assert "learned_mask_core_signals" in results
    assert "b3_mask_surprise" in results
    assert "b3_mask_memory" in results
    assert "b3_mask_core_signals" in results
    assert "fixed_rule_controller" in results


def test_v01_profile_runs_with_multiaction_environment() -> None:
    results = run_train_eval_suite(train_episodes=3, eval_episodes=3, seed=4, profile="v01")
    assert "learned_contextual_bandit" in results
    assert "imitation_controller" in results
    assert "factored_controller" in results
    assert "factored_warm_controller" in results
    assert "conservative_factored_controller" in results
    assert "read_disciplined_factored_controller" in results
    assert "dominance_tuned_factored_controller" in results
    assert "counterfactual_dominance_controller" in results
    assert "rollout_dominance_controller" in results
    assert "risk_averse_rollout_controller" in results
    assert "shielded_dominance_controller" in results
    assert "shielded_relaxed_dominance_controller" in results
    assert "habit_safe_set_controller" in results
    assert "habit_safe_set_h2_controller" in results
    assert "habit_safe_set_loose_controller" in results
    assert "habit_safe_set_tight_controller" in results
    assert "planner_necessity_controller" in results
    assert "planner_necessity_loose_controller" in results
    assert "drift_aware_planner_necessity_controller" in results
    assert "drift_aware_planner_necessity_loose_controller" in results
    assert "shield_intervention_rate" in results["shielded_dominance_controller"]
    assert "safe_set_positive_rate" in results["habit_safe_set_controller"]
    assert "necessity_positive_rate" in results["planner_necessity_controller"]
    assert "necessity_positive_rate" in results["planner_necessity_loose_controller"]
    assert "b3_mask_conflict" in results
    assert "b3_mask_drift" in results
    assert "memory_read_recall" in results["read_disciplined_factored_controller"]
    assert results["memory_always"]["memory_reads"] > results["learned_contextual_bandit"]["memory_reads"]
    assert results["planner_always"]["planner_calls"] > 0


def test_v02_transfer_profile_runs() -> None:
    results = run_train_eval_suite(train_episodes=3, eval_episodes=3, seed=5, profile="v02")
    assert "fixed_rule_controller" in results
    assert "habit_safe_set_h2_controller" in results
    assert "planner_necessity_loose_controller" in results
    assert "drift_aware_planner_necessity_controller" in results
    assert "b3_mask_core_signals" in results
    assert results["planner_always"]["planner_calls"] > 0


def test_v03_long_horizon_drift_profile_runs() -> None:
    results = run_train_eval_suite(train_episodes=3, eval_episodes=3, seed=6, profile="v03")
    assert "fixed_rule_controller" in results
    assert "planner_necessity_loose_controller" in results
    assert "drift_aware_planner_necessity_controller" in results
    assert "b3_mask_drift" in results
    assert results["habit_only"]["drift_under_horizon"] > results["planner_always"]["drift_under_horizon"]
    assert results["planner_always"]["planner_calls"] > 0


def test_v03b_drift_variant_transfer_profile_runs() -> None:
    results = run_train_eval_suite(train_episodes=3, eval_episodes=3, seed=6, profile="v03b")
    assert "fixed_rule_controller" in results
    assert "planner_necessity_loose_controller" in results
    assert "drift_aware_planner_necessity_controller" in results
    assert "drift_aware_planner_necessity_loose_controller" in results
    assert "b3_mask_drift" in results
    assert results["habit_only"]["drift_under_horizon"] > results["planner_always"]["drift_under_horizon"]
    assert results["planner_always"]["planner_calls"] > 0


def test_drift_threshold_sweep_runs() -> None:
    results = run_drift_threshold_sweep(
        thresholds=[0.10, 0.14],
        train_episodes=2,
        eval_episodes=2,
        seed=7,
        profile="v03",
    )
    assert "fixed_rule_controller" in results
    assert "planner_necessity_loose_controller" in results
    assert "drift_threshold_0.10" in results
    assert "drift_threshold_0.14" in results
    assert "drift_repair_rate" in results["drift_threshold_0.10"]
    assert "drift_repair_delta_mean" in results["drift_threshold_0.10"]
    assert "drift_residual_after_repair" in results["drift_threshold_0.10"]
    assert "high_drift_no_repair_next_delta_mean" in results["drift_threshold_0.10"]


def test_repair_benefit_analysis_runs() -> None:
    results = run_repair_benefit_analysis(
        train_episodes=2,
        eval_episodes=2,
        seed=8,
        profile="v03",
        horizon=3,
    )
    assert "samples" in results
    assert "repair_drift_benefit_mean" in results
    assert "repair_reward_benefit_mean" in results
    assert "repair_terminal_drift_benefit_mean" in results
    assert "repair_cumulative_reward_benefit_mean" in results
    assert "actual_repair_rate_on_samples" in results


def test_b7_transfer_matrix_runs() -> None:
    results = run_b7_transfer_matrix(
        train_episodes=2,
        eval_episodes=2,
        seeds=[0],
        profiles=["v03"],
        repair_benefit_horizon=3,
    )
    run = results["v03:seed=0"]
    assert "b7_minus_fixed" in run
    assert "repair_benefit" in run
    assert "drift_aware_planner_necessity_controller" in run


def test_b7_acceptance_artifact_runs() -> None:
    artifact = run_b7_acceptance_artifact(
        train_episodes=2,
        eval_episodes=2,
        seeds=[0],
        profiles=["v03"],
        repair_benefit_horizon=3,
    )
    assert "matrix" in artifact
    assert "acceptance" in artifact
    assert "passed" in artifact
