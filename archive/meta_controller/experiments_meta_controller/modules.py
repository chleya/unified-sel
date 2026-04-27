from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from experiments.meta_controller.env import Observation, RegimeMemoryEnv


@dataclass
class WorkspaceState:
    task_progress: float
    surprise: float
    uncertainty: float
    control_cost_estimate: float
    memory_relevance: float
    conflict_score: float
    invariant_violation: float
    recent_failure_source: float

    def vector(self) -> np.ndarray:
        return np.array(
            [
                self.task_progress,
                self.surprise,
                self.uncertainty,
                self.control_cost_estimate,
                self.memory_relevance,
                self.conflict_score,
                self.invariant_violation,
                self.recent_failure_source,
            ],
            dtype=float,
        )


@dataclass
class EpisodicMemory:
    values: Dict[str, int] = field(default_factory=dict)
    reads: int = 0
    writes: int = 0
    useful_reads: int = 0
    useful_writes: int = 0

    def read_secret(self, required: bool) -> Optional[int]:
        self.reads += 1
        value = self.values.get("secret")
        if required and value is not None:
            self.useful_reads += 1
        return value

    def write_from_observation(self, obs: Observation) -> bool:
        if obs.clue is None:
            return False
        self.writes += 1
        self.values["secret"] = int(obs.clue)
        self.useful_writes += 1
        return True


class HabitPolicy:
    def __init__(self):
        self.action = 0

    def act(self, obs: Observation) -> int:
        if obs.visible_regime_hint is not None and obs.previous_success is False:
            self.action = int(obs.visible_regime_hint)
        return self.action

    def update(self, action: int, success: bool, correct_action: int) -> None:
        if not success:
            self.action = int(correct_action)


class PlannerPolicy:
    def act(self, env: RegimeMemoryEnv, memory_value: Optional[int]) -> int:
        return env.oracle_action(memory_value=memory_value)


class Predictor:
    def __init__(self, action_dim: int = 2):
        self.success_counts = np.ones(action_dim, dtype=float)
        self.total_counts = np.ones(action_dim, dtype=float) * 2.0
        self.last_expected_success = 0.5
        self.last_surprise = 0.0

    def expected_success(self, action: int) -> float:
        return float(self.success_counts[action] / self.total_counts[action])

    def update(self, action: int, success: bool) -> float:
        expected = self.expected_success(action)
        actual = 1.0 if success else 0.0
        surprise = abs(actual - expected)
        self.success_counts[action] += actual
        self.total_counts[action] += 1.0
        self.last_expected_success = expected
        self.last_surprise = surprise
        return surprise


def build_workspace(
    obs: Observation,
    env: RegimeMemoryEnv,
    habit_action: int,
    planner_action: int,
    predictor: Predictor,
    memory_has_secret: bool,
    last_surprise: float,
) -> WorkspaceState:
    uncertainty = 1.0 - max(predictor.expected_success(0), predictor.expected_success(1))
    memory_relevance = 1.0 if obs.phase == "memory" or obs.clue is not None else 0.0
    conflict = 1.0 if habit_action != planner_action else 0.0
    recent_failure = 1.0 if obs.previous_success is False else 0.0
    return WorkspaceState(
        task_progress=obs.step / max(1, env.config.horizon),
        surprise=last_surprise,
        uncertainty=uncertainty,
        control_cost_estimate=0.25 + 0.5 * conflict + 0.25 * memory_relevance,
        memory_relevance=memory_relevance,
        conflict_score=conflict,
        invariant_violation=obs.drift,
        recent_failure_source=recent_failure,
    )
