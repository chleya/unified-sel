from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from experiments.meta_controller.adapters.sel_lab_benchmark import BenchmarkSuite, default_suites, evaluate_acceptance, suite_by_name
from experiments.meta_controller.baselines import build_controllers
from experiments.meta_controller.env import (
    EnvConfig,
    RegimeMemoryEnv,
    heldout_configs,
    v01_heldout_configs,
    v01_train_configs,
    v02_transfer_configs,
    v03b_drift_variant_configs,
    v03_heldout_configs,
    v03_train_configs,
)
from experiments.meta_controller.meta_controller import BaseMetaController
from experiments.meta_controller.meta_controller import (
    ContextualBanditController,
    ConservativeFactoredController,
    CounterfactualDominanceController,
    DominanceTunedFactoredController,
    DriftAwarePlannerNecessityController,
    FactoredController,
    HabitSafeSetController,
    ImitationController,
    PlannerNecessityController,
    ReadDisciplinedFactoredController,
    RiskAverseRolloutDominanceController,
    RolloutDominanceController,
    SignalMaskingController,
    ShieldedDominanceController,
    decode_action,
    oracle_action_name,
)
from experiments.meta_controller.metrics import EpisodeMetrics, aggregate_summaries
from experiments.meta_controller.modules import (
    EpisodicMemory,
    HabitPolicy,
    PlannerPolicy,
    Predictor,
    build_workspace,
)
from experiments.meta_controller.report import format_results_table, summarize_acceptance_checks


PLANNER_COST = 0.20
MEMORY_READ_COST = 0.05
MEMORY_WRITE_COST = 0.03
SWITCH_COST = 0.02
DRIFT_REPAIR_OBSERVATION_THRESHOLD = 0.08


def _action_value(config: EnvConfig, action: int, correct_action: int, drift: float, cost: float) -> float:
    success = int(action) == int(correct_action)
    next_drift = max(0.0, drift - config.drift_decay) if success else min(1.0, drift + config.drift_increase)
    reward = config.success_reward if success else config.failure_penalty
    reward -= config.drift_penalty * next_drift
    return reward - cost


def run_episode(
    config: EnvConfig,
    controller: BaseMetaController,
    seed: int = 0,
    train: bool = True,
    memory_read_cost: float = MEMORY_READ_COST,
) -> Dict[str, float]:
    env = RegimeMemoryEnv(config)
    obs = env.reset(seed=seed)
    memory = EpisodicMemory()
    habit = HabitPolicy()
    planner = PlannerPolicy()
    predictor = Predictor(action_dim=config.n_actions)
    metrics = EpisodeMetrics()
    last_surprise = 0.0
    last_decision_name = None

    for _ in range(config.horizon):
        if obs.step in config.regime_shift_steps:
            metrics.mark_shift(obs.step)

        habit_action = habit.act(obs)
        planner_action = planner.act(env, memory.values.get("secret"))
        state = build_workspace(
            obs=obs,
            env=env,
            habit_action=habit_action,
            planner_action=planner_action,
            predictor=predictor,
            memory_has_secret="secret" in memory.values,
            last_surprise=last_surprise,
        )
        decision = controller.select(state)
        pre_step_drift = obs.drift
        drift_repaired = bool(getattr(controller, "last_drift_repair", False))
        if bool(getattr(controller, "last_shield_intervened", False)):
            metrics.mark_shield_intervention()
        if drift_repaired:
            metrics.mark_drift_repair()
        if hasattr(controller, "last_habit_safe"):
            metrics.record_safe_set(
                is_safe=bool(getattr(controller, "last_habit_safe")),
                score=float(getattr(controller, "last_safe_score", 0.0)),
            )
            metrics.set_safe_set_training_stats(
                label_positive_rate=float(getattr(controller, "safe_label_positive_rate", 0.0)),
                score_mean=float(getattr(controller, "safe_score_mean", 0.0)),
            )
        if hasattr(controller, "last_planner_necessary"):
            metrics.record_necessity(
                is_necessary=bool(getattr(controller, "last_planner_necessary")),
                score=float(getattr(controller, "last_necessity_score", 0.0)),
            )
            metrics.set_necessity_training_stats(
                label_positive_rate=float(getattr(controller, "necessity_label_positive_rate", 0.0)),
                score_mean=float(getattr(controller, "necessity_score_mean", 0.0)),
            )

        cost = 0.0
        if last_decision_name is not None and decision.name != last_decision_name:
            cost += SWITCH_COST
        last_decision_name = decision.name

        wrote_useful_memory = False
        if decision.memory_write:
            wrote_useful_memory = memory.write_from_observation(obs)
            cost += MEMORY_WRITE_COST

        memory_value = None
        useful_read_before = memory.useful_reads
        if decision.memory_read:
            memory_value = memory.read_secret(required=obs.phase == "memory")
            cost += memory_read_cost
        made_useful_read = memory.useful_reads > useful_read_before

        if decision.dominant_module == "planner":
            action = planner.act(env, memory_value)
            cost += PLANNER_COST
        else:
            action = habit_action

        result = env.step(action, control_mode=decision.dominant_module)
        metrics.record_drift_transition(
            pre_drift=pre_step_drift,
            post_drift=result.observation.drift,
            repaired=drift_repaired,
            high_drift_threshold=DRIFT_REPAIR_OBSERVATION_THRESHOLD,
        )
        success = bool(result.info["success"])
        correct_action = int(result.info["correct_action"])
        habit.update(action=action, success=success, correct_action=correct_action)
        last_surprise = predictor.update(action=action, success=success)

        oracle_reward = env.config.success_reward
        effective_reward = result.reward - cost
        control_reward = effective_reward
        if wrote_useful_memory:
            control_reward += 0.4
        if made_useful_read:
            control_reward += 0.2
        if train:
            controller.update(state, decision, control_reward)

        metrics.memory_reads = memory.reads
        metrics.memory_writes = memory.writes
        metrics.useful_reads = memory.useful_reads
        metrics.useful_writes = memory.useful_writes
        metrics.record(
            step=obs.step,
            decision=decision.name,
            reward=effective_reward,
            success=success,
            drift=result.observation.drift,
            memory_required=bool(result.info["memory_required"]),
            oracle_reward=oracle_reward,
            cost=cost,
        )

        obs = result.observation
        if result.done:
            break

    return metrics.summary()


def _profile_configs(profile: str, seed: int = 0, read_cost_override: float | None = None) -> tuple[List[EnvConfig], List[EnvConfig], float]:
    if profile == "v03b":
        return (
            v03_train_configs(seed=seed),
            v03b_drift_variant_configs(seed=seed + 600),
            0.10 if read_cost_override is None else read_cost_override,
        )
    if profile == "v03":
        return v03_train_configs(seed=seed), v03_heldout_configs(seed=seed + 300), 0.10 if read_cost_override is None else read_cost_override
    if profile == "v02":
        return v01_train_configs(seed=seed), v02_transfer_configs(seed=seed + 200), 0.10 if read_cost_override is None else read_cost_override
    if profile == "v01":
        return v01_train_configs(seed=seed), v01_heldout_configs(seed=seed + 100), 0.10 if read_cost_override is None else read_cost_override
    return (
        [
            EnvConfig(seed=seed, regime_shift_steps=(20, 46), memory_query_steps=(34, 35, 62, 63)),
            EnvConfig(seed=seed + 1, regime_shift_steps=(16, 44, 64), memory_query_steps=(28, 29, 56, 57)),
            EnvConfig(seed=seed + 2, regime_shift_steps=(25, 50), memory_query_steps=(38, 39, 68, 69)),
        ],
        heldout_configs(),
        MEMORY_READ_COST if read_cost_override is None else read_cost_override,
    )


def run_suite(
    episodes: int = 12,
    seed: int = 0,
    profile: str = "v0",
    read_cost_override: float | None = None,
) -> Dict[str, Dict[str, float]]:
    controllers = build_controllers(seed=seed)
    results: Dict[str, Dict[str, float]] = {}
    _, configs, memory_read_cost = _profile_configs(profile, seed=seed, read_cost_override=read_cost_override)

    for name, controller in controllers.items():
        rows: List[Dict[str, float]] = []
        for i in range(episodes):
            config = configs[i % len(configs)]
            rows.append(
                run_episode(
                    config=config,
                    controller=controller,
                    seed=seed + i,
                    train=True,
                    memory_read_cost=memory_read_cost,
                )
            )
        results[name] = aggregate_summaries(rows)
    return results


def train_controller(
    controller: BaseMetaController,
    episodes: int,
    seed: int = 0,
    profile: str = "v0",
    read_cost_override: float | None = None,
) -> BaseMetaController:
    train_configs, _, memory_read_cost = _profile_configs(profile, seed=seed, read_cost_override=read_cost_override)
    for i in range(episodes):
        config = train_configs[i % len(train_configs)]
        run_episode(config=config, controller=controller, seed=seed + i, train=True, memory_read_cost=memory_read_cost)
    if isinstance(controller, ContextualBanditController):
        controller.set_epsilon(0.0)
    return controller


def collect_oracle_examples(profile: str, episodes: int, seed: int = 0, read_cost_override: float | None = None) -> list[tuple[object, str]]:
    train_configs, _, memory_read_cost = _profile_configs(profile, seed=seed, read_cost_override=read_cost_override)
    examples: list[tuple[object, str]] = []
    for i in range(episodes):
        controller = build_controllers(seed=seed + i)["oracle_macro_controller"]
        env = RegimeMemoryEnv(train_configs[i % len(train_configs)])
        obs = env.reset(seed=seed + i)
        memory = EpisodicMemory()
        habit = HabitPolicy()
        planner = PlannerPolicy()
        predictor = Predictor(action_dim=env.config.n_actions)
        last_surprise = 0.0
        for _ in range(env.config.horizon):
            habit_action = habit.act(obs)
            planner_action = planner.act(env, memory.values.get("secret"))
            state = build_workspace(
                obs=obs,
                env=env,
                habit_action=habit_action,
                planner_action=planner_action,
                predictor=predictor,
                memory_has_secret="secret" in memory.values,
                last_surprise=last_surprise,
            )
            label = oracle_action_name(state)
            examples.append((state, label))
            decision = controller.select(state)
            if decision.memory_write:
                memory.write_from_observation(obs)
            memory_value = memory.read_secret(required=obs.phase == "memory") if decision.memory_read else None
            action = planner.act(env, memory_value) if decision.dominant_module == "planner" else habit_action
            result = env.step(action, control_mode=decision.dominant_module)
            success = bool(result.info["success"])
            habit.update(action=action, success=success, correct_action=int(result.info["correct_action"]))
            last_surprise = predictor.update(action=action, success=success)
            obs = result.observation
            if result.done:
                break
        _ = memory_read_cost
    return examples


def train_counterfactual_dominance(
    controller: CounterfactualDominanceController,
    episodes: int,
    seed: int = 0,
    profile: str = "v0",
    read_cost_override: float | None = None,
) -> CounterfactualDominanceController:
    train_configs, _, memory_read_cost = _profile_configs(profile, seed=seed, read_cost_override=read_cost_override)
    for i in range(episodes):
        env = RegimeMemoryEnv(train_configs[i % len(train_configs)])
        obs = env.reset(seed=seed + i)
        memory = EpisodicMemory()
        habit = HabitPolicy()
        planner = PlannerPolicy()
        predictor = Predictor(action_dim=env.config.n_actions)
        last_surprise = 0.0
        last_decision_name = None

        for _ in range(env.config.horizon):
            habit_action = habit.act(obs)
            memory_value_for_planner = memory.values.get("secret") if obs.phase == "memory" else None
            planner_action = planner.act(env, memory_value_for_planner)
            state = build_workspace(
                obs=obs,
                env=env,
                habit_action=habit_action,
                planner_action=planner_action,
                predictor=predictor,
                memory_has_secret="secret" in memory.values,
                last_surprise=last_surprise,
            )

            decision = controller.select(state)
            if decision.memory_write:
                memory.write_from_observation(obs)
            memory_value = memory.read_secret(required=obs.phase == "memory") if decision.memory_read else None

            read_cost = memory_read_cost if decision.memory_read else 0.0
            switch_cost = SWITCH_COST if last_decision_name is not None and decision.name != last_decision_name else 0.0
            correct_action = env.correct_action()
            habit_value = _action_value(env.config, habit_action, correct_action, obs.drift, read_cost + switch_cost)
            planner_cf_action = planner.act(env, memory_value)
            planner_value = _action_value(env.config, planner_cf_action, correct_action, obs.drift, read_cost + switch_cost + PLANNER_COST)
            if isinstance(controller, RiskAverseRolloutDominanceController):
                controller.update_rollout(state, planner_value=planner_value, habit_value=habit_value)
            else:
                controller.update_counterfactual(state, planner_value=planner_value, habit_value=habit_value)

            action = planner_cf_action if decision.dominant_module == "planner" else habit_action
            result = env.step(action, control_mode=decision.dominant_module)
            success = bool(result.info["success"])
            habit.update(action=action, success=success, correct_action=int(result.info["correct_action"]))
            last_surprise = predictor.update(action=action, success=success)
            last_decision_name = decision.name
            obs = result.observation
            if result.done:
                break
    controller.set_epsilon(0.0)
    return controller


def _rollout_value(
    env: RegimeMemoryEnv,
    memory: EpisodicMemory,
    habit: HabitPolicy,
    predictor: Predictor,
    controller: BaseMetaController,
    first_module: str,
    horizon: int,
    memory_read_cost: float,
    last_surprise: float,
) -> float:
    value = 0.0
    obs = env._observation()
    for step in range(horizon):
        habit_action = habit.act(obs)
        planner = PlannerPolicy()
        planner_action = planner.act(env, memory.values.get("secret") if obs.phase == "memory" else None)
        state = build_workspace(
            obs=obs,
            env=env,
            habit_action=habit_action,
            planner_action=planner_action,
            predictor=predictor,
            memory_has_secret="secret" in memory.values,
            last_surprise=last_surprise,
        )
        decision = controller.select(state)
        if step == 0:
            if first_module == "planner":
                decision = type(decision)(
                    decision.name,
                    "planner",
                    decision.memory_read,
                    decision.memory_write,
                    decision.broadcast,
                    decision.deliberation_precision,
                )
            else:
                decision = type(decision)(
                    decision.name,
                    "habit",
                    decision.memory_read,
                    decision.memory_write,
                    decision.broadcast,
                    decision.deliberation_precision,
                )

        cost = 0.0
        if decision.memory_write:
            memory.write_from_observation(obs)
            cost += MEMORY_WRITE_COST
        memory_value = None
        if decision.memory_read:
            memory_value = memory.read_secret(required=obs.phase == "memory")
            cost += memory_read_cost
        if decision.dominant_module == "planner":
            action = planner.act(env, memory_value)
            cost += PLANNER_COST
        else:
            action = habit_action
        result = env.step(action, control_mode=decision.dominant_module)
        success = bool(result.info["success"])
        habit.update(action=action, success=success, correct_action=int(result.info["correct_action"]))
        last_surprise = predictor.update(action=action, success=success)
        value += result.reward - cost
        obs = result.observation
        if result.done:
            break
    return value


def train_rollout_dominance(
    controller: RolloutDominanceController,
    episodes: int,
    seed: int = 0,
    profile: str = "v0",
    read_cost_override: float | None = None,
) -> RolloutDominanceController:
    train_configs, _, memory_read_cost = _profile_configs(profile, seed=seed, read_cost_override=read_cost_override)
    for i in range(episodes):
        env = RegimeMemoryEnv(train_configs[i % len(train_configs)])
        obs = env.reset(seed=seed + i)
        memory = EpisodicMemory()
        habit = HabitPolicy()
        planner = PlannerPolicy()
        predictor = Predictor(action_dim=env.config.n_actions)
        last_surprise = 0.0

        for _ in range(env.config.horizon):
            habit_action = habit.act(obs)
            planner_action = planner.act(env, memory.values.get("secret") if obs.phase == "memory" else None)
            state = build_workspace(
                obs=obs,
                env=env,
                habit_action=habit_action,
                planner_action=planner_action,
                predictor=predictor,
                memory_has_secret="secret" in memory.values,
                last_surprise=last_surprise,
            )

            planner_env = env.clone()
            habit_env = env.clone()
            planner_memory = EpisodicMemory(values=dict(memory.values))
            habit_memory = EpisodicMemory(values=dict(memory.values))
            planner_habit = HabitPolicy()
            planner_habit.action = habit.action
            habit_habit = HabitPolicy()
            habit_habit.action = habit.action
            planner_predictor = Predictor(action_dim=env.config.n_actions)
            planner_predictor.success_counts = predictor.success_counts.copy()
            planner_predictor.total_counts = predictor.total_counts.copy()
            habit_predictor = Predictor(action_dim=env.config.n_actions)
            habit_predictor.success_counts = predictor.success_counts.copy()
            habit_predictor.total_counts = predictor.total_counts.copy()

            planner_value = _rollout_value(
                planner_env,
                planner_memory,
                planner_habit,
                planner_predictor,
                controller,
                first_module="planner",
                horizon=controller.rollout_horizon,
                memory_read_cost=memory_read_cost,
                last_surprise=last_surprise,
            )
            habit_value = _rollout_value(
                habit_env,
                habit_memory,
                habit_habit,
                habit_predictor,
                controller,
                first_module="habit",
                horizon=controller.rollout_horizon,
                memory_read_cost=memory_read_cost,
                last_surprise=last_surprise,
            )
            controller.update_counterfactual(state, planner_value=planner_value, habit_value=habit_value)

            decision = controller.select(state)
            if decision.memory_write:
                memory.write_from_observation(obs)
            memory_value = memory.read_secret(required=obs.phase == "memory") if decision.memory_read else None
            action = planner.act(env, memory_value) if decision.dominant_module == "planner" else habit_action
            result = env.step(action, control_mode=decision.dominant_module)
            success = bool(result.info["success"])
            habit.update(action=action, success=success, correct_action=int(result.info["correct_action"]))
            last_surprise = predictor.update(action=action, success=success)
            obs = result.observation
            if result.done:
                break
    controller.set_epsilon(0.0)
    return controller


def collect_habit_safe_examples(
    profile: str,
    episodes: int,
    seed: int = 0,
    read_cost_override: float | None = None,
    horizon: int = 3,
) -> list[tuple[object, bool]]:
    train_configs, _, _ = _profile_configs(profile, seed=seed, read_cost_override=read_cost_override)
    examples: list[tuple[object, bool]] = []
    for i in range(episodes):
        controller = build_controllers(seed=seed + i)["oracle_macro_controller"]
        env = RegimeMemoryEnv(train_configs[i % len(train_configs)])
        obs = env.reset(seed=seed + i)
        memory = EpisodicMemory()
        habit = HabitPolicy()
        planner = PlannerPolicy()
        predictor = Predictor(action_dim=env.config.n_actions)
        last_surprise = 0.0

        for _ in range(env.config.horizon):
            habit_action = habit.act(obs)
            planner_action = planner.act(env, memory.values.get("secret") if obs.phase == "memory" else None)
            state = build_workspace(
                obs=obs,
                env=env,
                habit_action=habit_action,
                planner_action=planner_action,
                predictor=predictor,
                memory_has_secret="secret" in memory.values,
                last_surprise=last_surprise,
            )
            examples.append((state, _habit_is_safe(env, habit, horizon=horizon)))

            decision = controller.select(state)
            if decision.memory_write:
                memory.write_from_observation(obs)
            memory_value = memory.read_secret(required=obs.phase == "memory") if decision.memory_read else None
            action = planner.act(env, memory_value) if decision.dominant_module == "planner" else habit_action
            result = env.step(action, control_mode=decision.dominant_module)
            success = bool(result.info["success"])
            habit.update(action=action, success=success, correct_action=int(result.info["correct_action"]))
            last_surprise = predictor.update(action=action, success=success)
            obs = result.observation
            if result.done:
                break
    return examples


def _habit_is_safe(env: RegimeMemoryEnv, habit: HabitPolicy, horizon: int) -> bool:
    probe_env = env.clone()
    probe_habit = HabitPolicy()
    probe_habit.action = habit.action
    start_drift = probe_env.drift
    obs = probe_env._observation()
    for _ in range(horizon):
        action = probe_habit.act(obs)
        result = probe_env.step(action, control_mode="habit")
        success = bool(result.info["success"])
        if not success:
            return False
        if result.observation.drift > start_drift + 0.005:
            return False
        probe_habit.update(action=action, success=success, correct_action=int(result.info["correct_action"]))
        obs = result.observation
        if result.done:
            break
    return True


def collect_planner_necessity_examples(
    profile: str,
    episodes: int,
    seed: int = 0,
    read_cost_override: float | None = None,
    horizon: int = 3,
    value_margin: float = 0.05,
) -> list[tuple[object, bool]]:
    train_configs, _, memory_read_cost = _profile_configs(profile, seed=seed, read_cost_override=read_cost_override)
    examples: list[tuple[object, bool]] = []
    for i in range(episodes):
        controller = build_controllers(seed=seed + i)["oracle_macro_controller"]
        env = RegimeMemoryEnv(train_configs[i % len(train_configs)])
        obs = env.reset(seed=seed + i)
        memory = EpisodicMemory()
        habit = HabitPolicy()
        planner = PlannerPolicy()
        predictor = Predictor(action_dim=env.config.n_actions)
        last_surprise = 0.0

        for _ in range(env.config.horizon):
            habit_action = habit.act(obs)
            planner_action = planner.act(env, memory.values.get("secret") if obs.phase == "memory" else None)
            state = build_workspace(
                obs=obs,
                env=env,
                habit_action=habit_action,
                planner_action=planner_action,
                predictor=predictor,
                memory_has_secret="secret" in memory.values,
                last_surprise=last_surprise,
            )
            planner_value = _dominance_rollout_value(
                env=env,
                memory=memory,
                habit=habit,
                predictor=predictor,
                first_module="planner",
                horizon=horizon,
                memory_read_cost=memory_read_cost,
                last_surprise=last_surprise,
            )
            habit_value = _dominance_rollout_value(
                env=env,
                memory=memory,
                habit=habit,
                predictor=predictor,
                first_module="habit",
                horizon=horizon,
                memory_read_cost=memory_read_cost,
                last_surprise=last_surprise,
            )
            examples.append((state, planner_value - habit_value > value_margin))

            decision = controller.select(state)
            if decision.memory_write:
                memory.write_from_observation(obs)
            memory_value = memory.read_secret(required=obs.phase == "memory") if decision.memory_read else None
            action = planner.act(env, memory_value) if decision.dominant_module == "planner" else habit_action
            result = env.step(action, control_mode=decision.dominant_module)
            success = bool(result.info["success"])
            habit.update(action=action, success=success, correct_action=int(result.info["correct_action"]))
            last_surprise = predictor.update(action=action, success=success)
            obs = result.observation
            if result.done:
                break
    return examples


def _dominance_rollout_value(
    env: RegimeMemoryEnv,
    memory: EpisodicMemory,
    habit: HabitPolicy,
    predictor: Predictor,
    first_module: str,
    horizon: int,
    memory_read_cost: float,
    last_surprise: float,
) -> float:
    probe_env = env.clone()
    probe_memory = EpisodicMemory(values=dict(memory.values))
    probe_habit = HabitPolicy()
    probe_habit.action = habit.action
    probe_predictor = Predictor(action_dim=env.config.n_actions)
    probe_predictor.success_counts = predictor.success_counts.copy()
    probe_predictor.total_counts = predictor.total_counts.copy()
    planner = PlannerPolicy()
    value = 0.0

    for step in range(horizon):
        obs = probe_env._observation()
        habit_action = probe_habit.act(obs)
        planner_action = planner.act(probe_env, probe_memory.values.get("secret") if obs.phase == "memory" else None)
        state = build_workspace(
            obs=obs,
            env=probe_env,
            habit_action=habit_action,
            planner_action=planner_action,
            predictor=probe_predictor,
            memory_has_secret="secret" in probe_memory.values,
            last_surprise=last_surprise,
        )
        decision = decode_action(oracle_action_name(state))
        dominant_module = first_module if step == 0 else decision.dominant_module

        cost = 0.0
        if decision.memory_write:
            probe_memory.write_from_observation(obs)
            cost += MEMORY_WRITE_COST
        memory_value = None
        if decision.memory_read:
            memory_value = probe_memory.read_secret(required=obs.phase == "memory")
            cost += memory_read_cost
        if dominant_module == "planner":
            action = planner.act(probe_env, memory_value)
            cost += PLANNER_COST
        else:
            action = habit_action
        result = probe_env.step(action, control_mode=dominant_module)
        success = bool(result.info["success"])
        probe_habit.update(action=action, success=success, correct_action=int(result.info["correct_action"]))
        last_surprise = probe_predictor.update(action=action, success=success)
        value += result.reward - cost
        if result.done:
            break
    return value


def _one_step_module_outcome(
    env: RegimeMemoryEnv,
    memory: EpisodicMemory,
    habit: HabitPolicy,
    module: str,
    memory_read_cost: float,
) -> Dict[str, float]:
    probe_env = env.clone()
    probe_habit = HabitPolicy()
    probe_habit.action = habit.action
    planner = PlannerPolicy()
    obs = probe_env._observation()
    cost = 0.0
    memory_value = None
    if module == "planner" and obs.phase == "memory":
        memory_value = memory.values.get("secret")
        if memory_value is not None:
            cost += memory_read_cost
    if module == "planner":
        action = planner.act(probe_env, memory_value)
        cost += PLANNER_COST
    else:
        action = probe_habit.act(obs)
    result = probe_env.step(action, control_mode=module)
    return {
        "post_drift": float(result.observation.drift),
        "reward": float(result.reward - cost),
        "success": float(bool(result.info["success"])),
    }


def _horizon_module_outcome(
    env: RegimeMemoryEnv,
    memory: EpisodicMemory,
    habit: HabitPolicy,
    predictor: Predictor,
    first_module: str,
    horizon: int,
    memory_read_cost: float,
    last_surprise: float,
) -> Dict[str, float]:
    probe_env = env.clone()
    probe_memory = EpisodicMemory(values=dict(memory.values))
    probe_habit = HabitPolicy()
    probe_habit.action = habit.action
    probe_predictor = Predictor(action_dim=env.config.n_actions)
    probe_predictor.success_counts = predictor.success_counts.copy()
    probe_predictor.total_counts = predictor.total_counts.copy()
    planner = PlannerPolicy()
    cumulative_reward = 0.0
    cumulative_drift = 0.0
    successes = 0
    steps = 0

    for step in range(horizon):
        obs = probe_env._observation()
        habit_action = probe_habit.act(obs)
        planner_action = planner.act(probe_env, probe_memory.values.get("secret") if obs.phase == "memory" else None)
        state = build_workspace(
            obs=obs,
            env=probe_env,
            habit_action=habit_action,
            planner_action=planner_action,
            predictor=probe_predictor,
            memory_has_secret="secret" in probe_memory.values,
            last_surprise=last_surprise,
        )
        decision = decode_action(oracle_action_name(state))
        dominant_module = first_module if step == 0 else decision.dominant_module

        cost = 0.0
        if decision.memory_write:
            probe_memory.write_from_observation(obs)
            cost += MEMORY_WRITE_COST
        memory_value = None
        if decision.memory_read:
            memory_value = probe_memory.read_secret(required=obs.phase == "memory")
            cost += memory_read_cost
        if dominant_module == "planner":
            action = planner.act(probe_env, memory_value)
            cost += PLANNER_COST
        else:
            action = habit_action

        result = probe_env.step(action, control_mode=dominant_module)
        success = bool(result.info["success"])
        probe_habit.update(action=action, success=success, correct_action=int(result.info["correct_action"]))
        last_surprise = probe_predictor.update(action=action, success=success)
        cumulative_reward += result.reward - cost
        cumulative_drift += result.observation.drift
        successes += int(success)
        steps += 1
        if result.done:
            break

    return {
        "terminal_drift": float(probe_env.drift),
        "cumulative_drift": float(cumulative_drift),
        "mean_drift": float(cumulative_drift / max(1, steps)),
        "cumulative_reward": float(cumulative_reward),
        "success_rate": float(successes / max(1, steps)),
        "steps": float(steps),
    }


def run_repair_benefit_analysis(
    train_episodes: int = 120,
    eval_episodes: int = 30,
    seed: int = 0,
    profile: str = "v03",
    drift_threshold: float = DRIFT_REPAIR_OBSERVATION_THRESHOLD,
    horizon: int = 1,
    read_cost_override: float | None = None,
) -> Dict[str, float]:
    controller = DriftAwarePlannerNecessityController(
        seed=seed + 1_000,
        safe_threshold=-0.08,
        necessity_threshold=0.08,
        drift_repair_threshold=drift_threshold,
    )
    _fit_planner_necessity_like_controller(
        controller,
        train_episodes=train_episodes,
        seed=seed,
        profile=profile,
        read_cost_override=read_cost_override,
    )
    _, configs, memory_read_cost = _profile_configs(profile, seed=seed, read_cost_override=read_cost_override)
    drift_benefits: List[float] = []
    reward_benefits: List[float] = []
    success_benefits: List[float] = []
    planner_post_drifts: List[float] = []
    habit_post_drifts: List[float] = []
    planner_successes: List[float] = []
    habit_successes: List[float] = []
    terminal_drift_benefits: List[float] = []
    cumulative_drift_benefits: List[float] = []
    cumulative_reward_benefits: List[float] = []
    horizon_success_benefits: List[float] = []
    planner_terminal_drifts: List[float] = []
    habit_terminal_drifts: List[float] = []
    actual_repairs = 0

    for i in range(eval_episodes):
        env = RegimeMemoryEnv(configs[i % len(configs)])
        obs = env.reset(seed=seed + 10_000 + i)
        memory = EpisodicMemory()
        habit = HabitPolicy()
        planner = PlannerPolicy()
        predictor = Predictor(action_dim=env.config.n_actions)
        last_surprise = 0.0

        for _ in range(env.config.horizon):
            habit_action = habit.act(obs)
            planner_action = planner.act(env, memory.values.get("secret") if obs.phase == "memory" else None)
            state = build_workspace(
                obs=obs,
                env=env,
                habit_action=habit_action,
                planner_action=planner_action,
                predictor=predictor,
                memory_has_secret="secret" in memory.values,
                last_surprise=last_surprise,
            )
            decision = controller.select(state)
            if obs.drift >= drift_threshold:
                planner_outcome = _one_step_module_outcome(
                    env=env,
                    memory=memory,
                    habit=habit,
                    module="planner",
                    memory_read_cost=memory_read_cost,
                )
                habit_outcome = _one_step_module_outcome(
                    env=env,
                    memory=memory,
                    habit=habit,
                    module="habit",
                    memory_read_cost=memory_read_cost,
                )
                drift_benefits.append(habit_outcome["post_drift"] - planner_outcome["post_drift"])
                reward_benefits.append(planner_outcome["reward"] - habit_outcome["reward"])
                success_benefits.append(planner_outcome["success"] - habit_outcome["success"])
                planner_post_drifts.append(planner_outcome["post_drift"])
                habit_post_drifts.append(habit_outcome["post_drift"])
                planner_successes.append(planner_outcome["success"])
                habit_successes.append(habit_outcome["success"])
                if horizon > 1:
                    planner_horizon = _horizon_module_outcome(
                        env=env,
                        memory=memory,
                        habit=habit,
                        predictor=predictor,
                        first_module="planner",
                        horizon=horizon,
                        memory_read_cost=memory_read_cost,
                        last_surprise=last_surprise,
                    )
                    habit_horizon = _horizon_module_outcome(
                        env=env,
                        memory=memory,
                        habit=habit,
                        predictor=predictor,
                        first_module="habit",
                        horizon=horizon,
                        memory_read_cost=memory_read_cost,
                        last_surprise=last_surprise,
                    )
                    terminal_drift_benefits.append(habit_horizon["terminal_drift"] - planner_horizon["terminal_drift"])
                    cumulative_drift_benefits.append(habit_horizon["cumulative_drift"] - planner_horizon["cumulative_drift"])
                    cumulative_reward_benefits.append(planner_horizon["cumulative_reward"] - habit_horizon["cumulative_reward"])
                    horizon_success_benefits.append(planner_horizon["success_rate"] - habit_horizon["success_rate"])
                    planner_terminal_drifts.append(planner_horizon["terminal_drift"])
                    habit_terminal_drifts.append(habit_horizon["terminal_drift"])
                actual_repairs += int(bool(getattr(controller, "last_drift_repair", False)))

            if decision.memory_write:
                memory.write_from_observation(obs)
            memory_value = memory.read_secret(required=obs.phase == "memory") if decision.memory_read else None
            action = planner.act(env, memory_value) if decision.dominant_module == "planner" else habit_action
            result = env.step(action, control_mode=decision.dominant_module)
            success = bool(result.info["success"])
            habit.update(action=action, success=success, correct_action=int(result.info["correct_action"]))
            last_surprise = predictor.update(action=action, success=success)
            obs = result.observation
            if result.done:
                break

    samples = len(drift_benefits)
    return {
        "samples": float(samples),
        "actual_repair_rate_on_samples": float(actual_repairs / samples) if samples else 0.0,
        "repair_drift_benefit_mean": float(np.mean(drift_benefits)) if drift_benefits else 0.0,
        "repair_drift_benefit_positive_rate": float(np.mean([value > 0.0 for value in drift_benefits])) if drift_benefits else 0.0,
        "repair_reward_benefit_mean": float(np.mean(reward_benefits)) if reward_benefits else 0.0,
        "repair_reward_benefit_positive_rate": float(np.mean([value > 0.0 for value in reward_benefits])) if reward_benefits else 0.0,
        "repair_success_benefit_mean": float(np.mean(success_benefits)) if success_benefits else 0.0,
        "planner_post_drift_mean": float(np.mean(planner_post_drifts)) if planner_post_drifts else 0.0,
        "habit_post_drift_mean": float(np.mean(habit_post_drifts)) if habit_post_drifts else 0.0,
        "planner_success_rate": float(np.mean(planner_successes)) if planner_successes else 0.0,
        "habit_success_rate": float(np.mean(habit_successes)) if habit_successes else 0.0,
        "horizon": float(horizon),
        "repair_terminal_drift_benefit_mean": float(np.mean(terminal_drift_benefits)) if terminal_drift_benefits else 0.0,
        "repair_terminal_drift_benefit_positive_rate": (
            float(np.mean([value > 0.0 for value in terminal_drift_benefits])) if terminal_drift_benefits else 0.0
        ),
        "repair_cumulative_drift_benefit_mean": float(np.mean(cumulative_drift_benefits)) if cumulative_drift_benefits else 0.0,
        "repair_cumulative_reward_benefit_mean": float(np.mean(cumulative_reward_benefits)) if cumulative_reward_benefits else 0.0,
        "repair_cumulative_reward_benefit_positive_rate": (
            float(np.mean([value > 0.0 for value in cumulative_reward_benefits])) if cumulative_reward_benefits else 0.0
        ),
        "repair_horizon_success_benefit_mean": float(np.mean(horizon_success_benefits)) if horizon_success_benefits else 0.0,
        "planner_terminal_drift_mean": float(np.mean(planner_terminal_drifts)) if planner_terminal_drifts else 0.0,
        "habit_terminal_drift_mean": float(np.mean(habit_terminal_drifts)) if habit_terminal_drifts else 0.0,
    }


def evaluate_controller(
    controller: BaseMetaController,
    episodes: int,
    seed: int = 1000,
    train: bool = False,
    profile: str = "v0",
    read_cost_override: float | None = None,
) -> Dict[str, float]:
    rows: List[Dict[str, float]] = []
    _, configs, memory_read_cost = _profile_configs(profile, seed=seed, read_cost_override=read_cost_override)
    for i in range(episodes):
        config = configs[i % len(configs)]
        rows.append(
            run_episode(
                config=config,
                controller=controller,
                seed=seed + i,
                train=train,
                memory_read_cost=memory_read_cost,
            )
        )
    return aggregate_summaries(rows)


def run_train_eval_suite(
    train_episodes: int = 120,
    eval_episodes: int = 30,
    seed: int = 0,
    profile: str = "v0",
    read_cost_override: float | None = None,
) -> Dict[str, Dict[str, float]]:
    controllers = build_controllers(seed=seed)
    learned = controllers["learned_contextual_bandit"]
    train_controller(learned, episodes=train_episodes, seed=seed, profile=profile, read_cost_override=read_cost_override)
    oracle_examples = collect_oracle_examples(
        profile=profile,
        episodes=train_episodes,
        seed=seed + 30_000,
        read_cost_override=read_cost_override,
    )
    imitation = controllers.get("imitation_controller")
    if isinstance(imitation, ImitationController):
        imitation.fit(oracle_examples)
    factored = controllers.get("factored_controller")
    if isinstance(factored, FactoredController):
        factored.fit(oracle_examples)
        factored.set_epsilon(0.02)
        train_controller(
            factored,
            episodes=max(1, train_episodes // 4),
            seed=seed + 40_000,
            profile=profile,
            read_cost_override=read_cost_override,
        )
    factored_warm = controllers.get("factored_warm_controller")
    if isinstance(factored_warm, FactoredController):
        factored_warm.fit(oracle_examples)
    conservative = controllers.get("conservative_factored_controller")
    if isinstance(conservative, ConservativeFactoredController):
        conservative.fit(oracle_examples)
        conservative.set_epsilon(0.01)
        train_controller(
            conservative,
            episodes=max(1, train_episodes // 4),
            seed=seed + 50_000,
            profile=profile,
            read_cost_override=read_cost_override,
        )
    read_disciplined = controllers.get("read_disciplined_factored_controller")
    if isinstance(read_disciplined, ReadDisciplinedFactoredController):
        read_disciplined.fit(oracle_examples)
        read_disciplined.set_epsilon(0.005)
        train_controller(
            read_disciplined,
            episodes=max(1, train_episodes // 4),
            seed=seed + 60_000,
            profile=profile,
            read_cost_override=read_cost_override,
        )
    dominance_tuned = controllers.get("dominance_tuned_factored_controller")
    if isinstance(dominance_tuned, DominanceTunedFactoredController):
        dominance_tuned.fit(oracle_examples)
        dominance_tuned.set_epsilon(0.005)
        train_controller(
            dominance_tuned,
            episodes=max(1, train_episodes // 4),
            seed=seed + 70_000,
            profile=profile,
            read_cost_override=read_cost_override,
        )
    counterfactual = controllers.get("counterfactual_dominance_controller")
    if isinstance(counterfactual, CounterfactualDominanceController):
        counterfactual.fit(oracle_examples)
        train_counterfactual_dominance(
            counterfactual,
            episodes=max(1, train_episodes // 2),
            seed=seed + 80_000,
            profile=profile,
            read_cost_override=read_cost_override,
        )
    rollout = controllers.get("rollout_dominance_controller")
    if isinstance(rollout, RolloutDominanceController):
        rollout.fit(oracle_examples)
        train_rollout_dominance(
            rollout,
            episodes=max(1, train_episodes // 2),
            seed=seed + 90_000,
            profile=profile,
            read_cost_override=read_cost_override,
        )
    risk_rollout = controllers.get("risk_averse_rollout_controller")
    if isinstance(risk_rollout, RiskAverseRolloutDominanceController):
        risk_rollout.fit(oracle_examples)
        train_rollout_dominance(
            risk_rollout,
            episodes=max(1, train_episodes // 2),
            seed=seed + 100_000,
            profile=profile,
            read_cost_override=read_cost_override,
        )
    shielded = controllers.get("shielded_dominance_controller")
    if isinstance(shielded, ShieldedDominanceController):
        shielded.fit(oracle_examples)
        train_counterfactual_dominance(
            shielded,
            episodes=max(1, train_episodes // 2),
            seed=seed + 110_000,
            profile=profile,
            read_cost_override=read_cost_override,
        )
    shielded_relaxed = controllers.get("shielded_relaxed_dominance_controller")
    if isinstance(shielded_relaxed, ShieldedDominanceController):
        shielded_relaxed.fit(oracle_examples)
        train_counterfactual_dominance(
            shielded_relaxed,
            episodes=max(1, train_episodes // 2),
            seed=seed + 120_000,
            profile=profile,
            read_cost_override=read_cost_override,
        )
    habit_safe = controllers.get("habit_safe_set_controller")
    if isinstance(habit_safe, HabitSafeSetController):
        habit_safe.fit(oracle_examples)
        habit_safe_examples = collect_habit_safe_examples(
            profile=profile,
            episodes=train_episodes,
            seed=seed + 130_000,
            read_cost_override=read_cost_override,
            horizon=3,
        )
        habit_safe.fit_safe_set(habit_safe_examples)
    habit_safe_h2 = controllers.get("habit_safe_set_h2_controller")
    if isinstance(habit_safe_h2, HabitSafeSetController):
        habit_safe_h2.fit(oracle_examples)
        habit_safe_h2_examples = collect_habit_safe_examples(
            profile=profile,
            episodes=train_episodes,
            seed=seed + 140_000,
            read_cost_override=read_cost_override,
            horizon=2,
        )
        habit_safe_h2.fit_safe_set(habit_safe_h2_examples)
    for name, horizon in (
        ("habit_safe_set_loose_controller", 3),
        ("habit_safe_set_tight_controller", 3),
    ):
        calibrated = controllers.get(name)
        if isinstance(calibrated, HabitSafeSetController):
            calibrated.fit(oracle_examples)
            calibrated_examples = collect_habit_safe_examples(
                profile=profile,
                episodes=train_episodes,
                seed=seed + 150_000 + len(name),
                read_cost_override=read_cost_override,
                horizon=horizon,
            )
            calibrated.fit_safe_set(calibrated_examples)
    for name in (
        "planner_necessity_controller",
        "planner_necessity_loose_controller",
        "drift_aware_planner_necessity_controller",
        "drift_aware_planner_necessity_loose_controller",
    ):
        necessity = controllers.get(name)
        if isinstance(necessity, PlannerNecessityController):
            necessity.fit(oracle_examples)
            safe_examples = collect_habit_safe_examples(
                profile=profile,
                episodes=train_episodes,
                seed=seed + 160_000 + len(name),
                read_cost_override=read_cost_override,
                horizon=3,
            )
            necessity.fit_safe_set(safe_examples)
            necessity_examples = collect_planner_necessity_examples(
                profile=profile,
                episodes=train_episodes,
                seed=seed + 170_000 + len(name),
                read_cost_override=read_cost_override,
                horizon=3,
                value_margin=0.05,
            )
            necessity.fit_necessity(necessity_examples)

    results: Dict[str, Dict[str, float]] = {}
    for name, controller in controllers.items():
        if isinstance(controller, ContextualBanditController):
            controller.set_epsilon(0.0)
        if isinstance(controller, FactoredController):
            controller.set_epsilon(0.0)
        results[name] = evaluate_controller(
            controller,
            episodes=eval_episodes,
            seed=seed + 10_000,
            profile=profile,
            read_cost_override=read_cost_override,
        )

    ablations = {
        "learned_mask_surprise": ("surprise",),
        "learned_mask_memory": ("memory",),
        "learned_mask_cost": ("cost",),
        "learned_mask_conflict": ("conflict",),
        "learned_mask_drift": ("drift",),
        "learned_mask_core_signals": ("surprise", "memory", "cost", "conflict", "drift"),
    }
    for name, mask in ablations.items():
        results[name] = evaluate_controller(
            SignalMaskingController(learned, mask=mask),
            episodes=eval_episodes,
            seed=seed + 20_000,
            profile=profile,
            read_cost_override=read_cost_override,
        )
    b3_mainline = controllers.get("planner_necessity_loose_controller")
    if b3_mainline is not None:
        b3_ablations = {
            "b3_mask_surprise": ("surprise",),
            "b3_mask_memory": ("memory",),
            "b3_mask_cost": ("cost",),
            "b3_mask_conflict": ("conflict",),
            "b3_mask_drift": ("drift",),
            "b3_mask_core_signals": ("surprise", "memory", "cost", "conflict", "drift"),
        }
        for name, mask in b3_ablations.items():
            results[name] = evaluate_controller(
                SignalMaskingController(b3_mainline, mask=mask),
                episodes=eval_episodes,
                seed=seed + 25_000,
                profile=profile,
                read_cost_override=read_cost_override,
            )
    return results


def run_read_cost_sweep(
    read_costs: List[float],
    train_episodes: int = 120,
    eval_episodes: int = 30,
    seed: int = 0,
    profile: str = "v01",
) -> Dict[str, Dict[str, Dict[str, float]]]:
    return {
        f"read_cost_{cost:.2f}": run_train_eval_suite(
            train_episodes=train_episodes,
            eval_episodes=eval_episodes,
            seed=seed,
            profile=profile,
            read_cost_override=cost,
        )
        for cost in read_costs
    }


def _fit_planner_necessity_like_controller(
    controller: PlannerNecessityController,
    train_episodes: int,
    seed: int,
    profile: str,
    read_cost_override: float | None = None,
) -> PlannerNecessityController:
    oracle_examples = collect_oracle_examples(
        profile=profile,
        episodes=train_episodes,
        seed=seed + 30_000,
        read_cost_override=read_cost_override,
    )
    controller.fit(oracle_examples)
    safe_examples = collect_habit_safe_examples(
        profile=profile,
        episodes=train_episodes,
        seed=seed + 160_000,
        read_cost_override=read_cost_override,
        horizon=3,
    )
    controller.fit_safe_set(safe_examples)
    necessity_examples = collect_planner_necessity_examples(
        profile=profile,
        episodes=train_episodes,
        seed=seed + 170_000,
        read_cost_override=read_cost_override,
        horizon=3,
        value_margin=0.05,
    )
    controller.fit_necessity(necessity_examples)
    return controller


def run_drift_threshold_sweep(
    thresholds: List[float],
    train_episodes: int = 120,
    eval_episodes: int = 30,
    seed: int = 0,
    profile: str = "v03",
    read_cost_override: float | None = None,
) -> Dict[str, Dict[str, float]]:
    controllers = build_controllers(seed=seed)
    results: Dict[str, Dict[str, float]] = {
        "fixed_rule_controller": evaluate_controller(
            controllers["fixed_rule_controller"],
            episodes=eval_episodes,
            seed=seed + 10_000,
            profile=profile,
            read_cost_override=read_cost_override,
        ),
    }

    b3 = controllers["planner_necessity_loose_controller"]
    if isinstance(b3, PlannerNecessityController):
        _fit_planner_necessity_like_controller(
            b3,
            train_episodes=train_episodes,
            seed=seed,
            profile=profile,
            read_cost_override=read_cost_override,
        )
    results["planner_necessity_loose_controller"] = evaluate_controller(
        b3,
        episodes=eval_episodes,
        seed=seed + 10_000,
        profile=profile,
        read_cost_override=read_cost_override,
    )

    for idx, threshold in enumerate(thresholds):
        controller = DriftAwarePlannerNecessityController(
            seed=seed + 1_000 + idx,
            safe_threshold=-0.08,
            necessity_threshold=0.08,
            drift_repair_threshold=threshold,
        )
        _fit_planner_necessity_like_controller(
            controller,
            train_episodes=train_episodes,
            seed=seed,
            profile=profile,
            read_cost_override=read_cost_override,
        )
        results[f"drift_threshold_{threshold:.2f}"] = evaluate_controller(
            controller,
            episodes=eval_episodes,
            seed=seed + 10_000,
            profile=profile,
            read_cost_override=read_cost_override,
        )
    return results


def _select_b7_matrix_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    keys = (
        "task_success",
        "total_reward",
        "planner_calls",
        "drift_under_horizon",
        "drift_repair_rate",
        "high_drift_no_repair_rate",
        "drift_repair_delta_mean",
        "drift_residual_after_repair",
    )
    return {key: float(metrics.get(key, 0.0)) for key in keys}


def run_b7_transfer_matrix(
    train_episodes: int = 120,
    eval_episodes: int = 30,
    seeds: List[int] | None = None,
    profiles: List[str] | None = None,
    repair_benefit_horizon: int = 3,
    read_cost_override: float | None = None,
) -> Dict[str, Dict[str, object]]:
    selected_seeds = [0, 1, 2] if seeds is None else seeds
    selected_profiles = ["v01", "v02", "v03", "v03b"] if profiles is None else profiles
    matrix: Dict[str, Dict[str, object]] = {}
    for profile in selected_profiles:
        for seed in selected_seeds:
            controllers = build_controllers(seed=seed)
            fixed = controllers["fixed_rule_controller"]
            b3 = controllers["planner_necessity_loose_controller"]
            b7 = controllers["drift_aware_planner_necessity_controller"]
            if isinstance(b3, PlannerNecessityController):
                _fit_planner_necessity_like_controller(
                    b3,
                    train_episodes=train_episodes,
                    seed=seed,
                    profile=profile,
                    read_cost_override=read_cost_override,
                )
            if isinstance(b7, PlannerNecessityController):
                _fit_planner_necessity_like_controller(
                    b7,
                    train_episodes=train_episodes,
                    seed=seed,
                    profile=profile,
                    read_cost_override=read_cost_override,
                )
            fixed_metrics = evaluate_controller(
                fixed,
                episodes=eval_episodes,
                seed=seed + 10_000,
                profile=profile,
                read_cost_override=read_cost_override,
            )
            b3_metrics = evaluate_controller(
                b3,
                episodes=eval_episodes,
                seed=seed + 10_000,
                profile=profile,
                read_cost_override=read_cost_override,
            )
            b7_metrics = evaluate_controller(
                b7,
                episodes=eval_episodes,
                seed=seed + 10_000,
                profile=profile,
                read_cost_override=read_cost_override,
            )
            benefit = run_repair_benefit_analysis(
                train_episodes=train_episodes,
                eval_episodes=eval_episodes,
                seed=seed,
                profile=profile,
                horizon=repair_benefit_horizon,
                read_cost_override=read_cost_override,
            )
            matrix[f"{profile}:seed={seed}"] = {
                "profile": profile,
                "seed": seed,
                "fixed_rule_controller": _select_b7_matrix_metrics(fixed_metrics),
                "planner_necessity_loose_controller": _select_b7_matrix_metrics(b3_metrics),
                "drift_aware_planner_necessity_controller": _select_b7_matrix_metrics(b7_metrics),
                "b7_minus_fixed": {
                    "success_delta": float(b7_metrics["task_success"] - fixed_metrics["task_success"]),
                    "planner_delta": float(b7_metrics["planner_calls"] - fixed_metrics["planner_calls"]),
                    "drift_delta": float(b7_metrics["drift_under_horizon"] - fixed_metrics["drift_under_horizon"]),
                    "reward_delta": float(b7_metrics["total_reward"] - fixed_metrics["total_reward"]),
                    "high_drift_no_repair_delta": float(
                        b7_metrics["high_drift_no_repair_rate"] - fixed_metrics["high_drift_no_repair_rate"]
                    ),
                },
                "repair_benefit_horizon": repair_benefit_horizon,
                "repair_benefit": {
                    "samples": benefit["samples"],
                    "terminal_drift_benefit": benefit["repair_terminal_drift_benefit_mean"],
                    "cumulative_drift_benefit": benefit["repair_cumulative_drift_benefit_mean"],
                    "cumulative_reward_benefit": benefit["repair_cumulative_reward_benefit_mean"],
                    "terminal_drift_benefit_positive_rate": benefit["repair_terminal_drift_benefit_positive_rate"],
                },
            }
    return matrix


def evaluate_b7_acceptance(matrix: Dict[str, Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    checks: Dict[str, Dict[str, object]] = {}
    pressure_profiles = {"v03", "v03b"}
    regression_profiles = {"v01", "v02"}
    for run_name, run_payload in matrix.items():
        profile = str(run_payload["profile"])
        deltas = run_payload["b7_minus_fixed"]  # type: ignore[index]
        benefit = run_payload["repair_benefit"]  # type: ignore[index]
        success_delta = float(deltas["success_delta"])  # type: ignore[index]
        drift_delta = float(deltas["drift_delta"])  # type: ignore[index]
        high_drift_delta = float(deltas["high_drift_no_repair_delta"])  # type: ignore[index]
        planner_delta = float(deltas["planner_delta"])  # type: ignore[index]
        reward_delta = float(deltas["reward_delta"])  # type: ignore[index]
        samples = float(benefit["samples"])  # type: ignore[index]
        terminal_benefit = float(benefit["terminal_drift_benefit"])  # type: ignore[index]
        cumulative_benefit = float(benefit["cumulative_drift_benefit"])  # type: ignore[index]
        cumulative_reward = float(benefit["cumulative_reward_benefit"])  # type: ignore[index]

        if profile in pressure_profiles:
            checks[f"{run_name}.pressure_drift_improves"] = {
                "pass": drift_delta < -0.05,
                "value": drift_delta,
                "threshold": -0.05,
            }
            checks[f"{run_name}.pressure_high_drift_exposure_improves"] = {
                "pass": high_drift_delta < -0.10,
                "value": high_drift_delta,
                "threshold": -0.10,
            }
            checks[f"{run_name}.pressure_terminal_benefit_positive"] = {
                "pass": samples > 0.0 and terminal_benefit > 0.0,
                "value": terminal_benefit,
                "threshold": 0.0,
            }
            checks[f"{run_name}.pressure_cumulative_benefit_positive"] = {
                "pass": samples > 0.0 and cumulative_benefit > 0.0,
                "value": cumulative_benefit,
                "threshold": 0.0,
            }
            checks[f"{run_name}.pressure_success_bounded"] = {
                "pass": success_delta >= -0.01,
                "value": success_delta,
                "threshold": -0.01,
            }
            checks[f"{run_name}.pressure_reward_tradeoff_reported"] = {
                "pass": True,
                "value": reward_delta,
                "threshold": 0.0,
                "note": "Reward delta is diagnostic, not an acceptance target.",
            }
            checks[f"{run_name}.pressure_cumulative_reward_tradeoff_reported"] = {
                "pass": True,
                "value": cumulative_reward,
                "threshold": 0.0,
                "note": "Counterfactual reward benefit is diagnostic, not an acceptance target.",
            }
        elif profile in regression_profiles:
            checks[f"{run_name}.regression_success_bounded"] = {
                "pass": success_delta >= -0.02,
                "value": success_delta,
                "threshold": -0.02,
            }
            checks[f"{run_name}.regression_planner_bounded"] = {
                "pass": planner_delta <= 2.0,
                "value": planner_delta,
                "threshold": 2.0,
            }
            checks[f"{run_name}.regression_drift_not_worse_beyond_tolerance"] = {
                "pass": drift_delta <= 0.02,
                "value": drift_delta,
                "threshold": 0.02,
            }
            checks[f"{run_name}.regression_reward_bounded"] = {
                "pass": reward_delta >= -3.0,
                "value": reward_delta,
                "threshold": -3.0,
            }
            checks[f"{run_name}.regression_reward_tradeoff_reported"] = {
                "pass": True,
                "value": reward_delta,
                "threshold": 0.0,
                "note": "Reward delta is diagnostic and separately bounded.",
            }
    return checks


def run_b7_acceptance_artifact(
    train_episodes: int = 120,
    eval_episodes: int = 30,
    seeds: List[int] | None = None,
    profiles: List[str] | None = None,
    repair_benefit_horizon: int = 3,
    read_cost_override: float | None = None,
) -> Dict[str, object]:
    matrix = run_b7_transfer_matrix(
        train_episodes=train_episodes,
        eval_episodes=eval_episodes,
        seeds=seeds,
        profiles=profiles,
        repair_benefit_horizon=repair_benefit_horizon,
        read_cost_override=read_cost_override,
    )
    acceptance = evaluate_b7_acceptance(matrix)
    return {
        "matrix": matrix,
        "acceptance": acceptance,
        "passed": all(bool(check["pass"]) for check in acceptance.values()),
    }


def _suite_result_names(suite: BenchmarkSuite) -> List[str]:
    names = list(suite.controllers)
    for mask in suite.masks:
        names.append(f"b3_mask_{mask}")
    return names


def _filter_suite_results(
    results: Dict[str, Dict[str, float]],
    suite: BenchmarkSuite,
) -> Dict[str, Dict[str, float]]:
    selected: Dict[str, Dict[str, float]] = {}
    for name in _suite_result_names(suite):
        if name in results:
            selected[name] = results[name]
    return selected


def run_benchmark_suite(
    suite: BenchmarkSuite,
) -> Dict[str, Dict[str, object]]:
    run_results: Dict[str, Dict[str, object]] = {}
    for spec in suite.expand_runs():
        results = run_train_eval_suite(
            train_episodes=spec.train_episodes,
            eval_episodes=spec.eval_episodes,
            seed=spec.seed,
            profile=spec.profile,
            read_cost_override=spec.read_cost,
        )
        filtered = _filter_suite_results(results, suite)
        run_results[f"{spec.suite}:seed={spec.seed}"] = {
            "spec": asdict(spec),
            "results": filtered,
            "acceptance": evaluate_acceptance(suite, filtered),
        }
    return run_results


def run_benchmark_suites(
    suite_name: str,
) -> Dict[str, Dict[str, object]]:
    suites = default_suites() if suite_name == "all" else (suite_by_name(suite_name),)
    combined: Dict[str, Dict[str, object]] = {}
    for suite in suites:
        combined.update(run_benchmark_suite(suite))
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Run minimal meta-controller V0 experiment.")
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--train-episodes", type=int, default=120)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--table", action="store_true")
    parser.add_argument("--mode", choices=("online", "train-eval"), default="online")
    parser.add_argument("--profile", choices=("v0", "v01", "v02", "v03", "v03b"), default="v0")
    parser.add_argument("--read-cost", type=float, default=None)
    parser.add_argument("--read-cost-sweep", default=None, help="Comma-separated read costs, e.g. 0.10,0.15,0.20")
    parser.add_argument(
        "--drift-threshold-sweep",
        default=None,
        help="Comma-separated B7 drift repair thresholds, e.g. 0.08,0.10,0.12,0.14,0.16",
    )
    parser.add_argument(
        "--repair-benefit-analysis",
        action="store_true",
        help="Estimate planner-repair vs habit/no-repair counterfactual benefit on high-drift states.",
    )
    parser.add_argument("--repair-benefit-threshold", type=float, default=DRIFT_REPAIR_OBSERVATION_THRESHOLD)
    parser.add_argument("--repair-benefit-horizon", type=int, default=1)
    parser.add_argument(
        "--b7-transfer-matrix",
        action="store_true",
        help="Run compact B7 transfer matrix across profiles and seeds.",
    )
    parser.add_argument(
        "--b7-acceptance-artifact",
        action="store_true",
        help="Run B7 transfer matrix plus frozen-claim acceptance checks.",
    )
    parser.add_argument("--matrix-profiles", default="v01,v02,v03,v03b")
    parser.add_argument("--matrix-seeds", default="0,1,2")
    parser.add_argument(
        "--suite",
        choices=(
            "all",
            "mainline_acceptance",
            "transfer_v02",
            "read_cost_stress",
            "b3_signal_masking",
            "long_horizon_drift_v03",
            "v03b_drift_variant_transfer",
            "b7_cross_profile_v01",
            "b7_cross_profile_v02",
        ),
        default=None,
        help="Run a benchmark suite defined by the local sel-lab-style adapter.",
    )
    args = parser.parse_args()

    if args.b7_acceptance_artifact:
        profiles = [part.strip() for part in args.matrix_profiles.split(",") if part.strip()]
        seeds = [int(part.strip()) for part in args.matrix_seeds.split(",") if part.strip()]
        results = run_b7_acceptance_artifact(
            train_episodes=args.train_episodes,
            eval_episodes=args.eval_episodes,
            seeds=seeds,
            profiles=profiles,
            repair_benefit_horizon=args.repair_benefit_horizon,
            read_cost_override=args.read_cost,
        )
        payload = {
            "mode": "b7-acceptance-artifact",
            "train_episodes": args.train_episodes,
            "eval_episodes": args.eval_episodes,
            "profiles": profiles,
            "seeds": seeds,
            "read_cost": args.read_cost,
            "repair_benefit_horizon": args.repair_benefit_horizon,
            "results": results,
        }
    elif args.b7_transfer_matrix:
        profiles = [part.strip() for part in args.matrix_profiles.split(",") if part.strip()]
        seeds = [int(part.strip()) for part in args.matrix_seeds.split(",") if part.strip()]
        results = run_b7_transfer_matrix(
            train_episodes=args.train_episodes,
            eval_episodes=args.eval_episodes,
            seeds=seeds,
            profiles=profiles,
            repair_benefit_horizon=args.repair_benefit_horizon,
            read_cost_override=args.read_cost,
        )
        payload = {
            "mode": "b7-transfer-matrix",
            "train_episodes": args.train_episodes,
            "eval_episodes": args.eval_episodes,
            "profiles": profiles,
            "seeds": seeds,
            "read_cost": args.read_cost,
            "repair_benefit_horizon": args.repair_benefit_horizon,
            "results": results,
        }
    elif args.suite:
        suite_results = run_benchmark_suites(args.suite)
        payload = {
            "mode": "benchmark-suite",
            "suite": args.suite,
            "results": suite_results,
        }
    elif args.repair_benefit_analysis:
        results = run_repair_benefit_analysis(
            train_episodes=args.train_episodes,
            eval_episodes=args.eval_episodes,
            seed=args.seed,
            profile=args.profile,
            drift_threshold=args.repair_benefit_threshold,
            horizon=args.repair_benefit_horizon,
            read_cost_override=args.read_cost,
        )
        payload = {
            "mode": "repair-benefit-analysis",
            "train_episodes": args.train_episodes,
            "eval_episodes": args.eval_episodes,
            "seed": args.seed,
            "profile": args.profile,
            "read_cost": args.read_cost,
            "drift_threshold": args.repair_benefit_threshold,
            "horizon": args.repair_benefit_horizon,
            "results": results,
        }
    elif args.drift_threshold_sweep:
        thresholds = [float(part.strip()) for part in args.drift_threshold_sweep.split(",") if part.strip()]
        results = run_drift_threshold_sweep(
            thresholds=thresholds,
            train_episodes=args.train_episodes,
            eval_episodes=args.eval_episodes,
            seed=args.seed,
            profile=args.profile,
            read_cost_override=args.read_cost,
        )
        payload = {
            "mode": "drift-threshold-sweep",
            "thresholds": thresholds,
            "train_episodes": args.train_episodes,
            "eval_episodes": args.eval_episodes,
            "seed": args.seed,
            "profile": args.profile,
            "read_cost": args.read_cost,
            "results": results,
        }
    elif args.read_cost_sweep:
        costs = [float(part.strip()) for part in args.read_cost_sweep.split(",") if part.strip()]
        results = run_read_cost_sweep(
            read_costs=costs,
            train_episodes=args.train_episodes,
            eval_episodes=args.eval_episodes,
            seed=args.seed,
            profile=args.profile,
        )
        payload = {
            "mode": "read-cost-sweep",
            "train_episodes": args.train_episodes,
            "eval_episodes": args.eval_episodes,
            "seed": args.seed,
            "profile": args.profile,
            "results": results,
        }
    elif args.mode == "train-eval":
        results = run_train_eval_suite(
            train_episodes=args.train_episodes,
            eval_episodes=args.eval_episodes,
            seed=args.seed,
            profile=args.profile,
            read_cost_override=args.read_cost,
        )
        payload = {
            "mode": args.mode,
            "train_episodes": args.train_episodes,
            "eval_episodes": args.eval_episodes,
            "seed": args.seed,
            "profile": args.profile,
            "read_cost": args.read_cost,
            "results": results,
        }
    else:
        results = run_suite(episodes=args.episodes, seed=args.seed, profile=args.profile, read_cost_override=args.read_cost)
        payload = {
            "mode": args.mode,
            "episodes": args.episodes,
            "seed": args.seed,
            "profile": args.profile,
            "read_cost": args.read_cost,
            "results": results,
        }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if args.table and args.b7_acceptance_artifact:
        artifact = results
        print("check | pass | value | threshold")
        print("--- | --- | --- | ---")
        for check_name, check in artifact["acceptance"].items():  # type: ignore[index]
            print(
                " | ".join(
                    [
                        check_name,
                        "PASS" if check["pass"] else "FAIL",
                        f"{check['value']:.3f}",
                        f"{check['threshold']:.3f}",
                    ]
                )
            )
        print(f"\noverall | {'PASS' if artifact['passed'] else 'FAIL'}")
    elif args.table and args.b7_transfer_matrix:
        headers = [
            "run",
            "success_delta",
            "planner_delta",
            "drift_delta",
            "reward_delta",
            "high_drift_no_repair_delta",
            "benefit_samples",
            "terminal_drift_benefit",
            "cumulative_drift_benefit",
            "cumulative_reward_benefit",
        ]
        print(" | ".join(headers))
        print(" | ".join(["---"] * len(headers)))
        for run_name, run_payload in results.items():
            deltas = run_payload["b7_minus_fixed"]  # type: ignore[index]
            benefit = run_payload["repair_benefit"]  # type: ignore[index]
            print(
                " | ".join(
                    [
                        run_name,
                        f"{deltas['success_delta']:.3f}",
                        f"{deltas['planner_delta']:.3f}",
                        f"{deltas['drift_delta']:.3f}",
                        f"{deltas['reward_delta']:.3f}",
                        f"{deltas['high_drift_no_repair_delta']:.3f}",
                        f"{benefit['samples']:.0f}",
                        f"{benefit['terminal_drift_benefit']:.3f}",
                        f"{benefit['cumulative_drift_benefit']:.3f}",
                        f"{benefit['cumulative_reward_benefit']:.3f}",
                    ]
                )
            )
    elif args.table and args.suite:
        for run_name, run_payload in suite_results.items():
            print(f"\n## {run_name}")
            print(format_results_table(run_payload["results"]))  # type: ignore[arg-type]
            acceptance = run_payload["acceptance"]
            if acceptance:
                print("\nacceptance")
                for check, passed in acceptance.items():
                    print(f"- {check}: {'PASS' if passed else 'FAIL'}")
    elif args.table and args.repair_benefit_analysis:
        print("metric | value")
        print("--- | ---")
        for key, value in results.items():
            print(f"{key} | {value:.3f}")
    elif args.table and not args.read_cost_sweep:
        print(format_results_table(results))
        if args.mode == "train-eval" and not args.drift_threshold_sweep:
            print(summarize_acceptance_checks(results))
    else:
        print(text)


if __name__ == "__main__":
    main()
