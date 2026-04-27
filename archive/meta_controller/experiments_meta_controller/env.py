from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class EnvConfig:
    horizon: int = 80
    regime_shift_steps: Tuple[int, ...] = (24, 52)
    memory_clue_steps: Tuple[int, ...] = (3,)
    memory_query_steps: Tuple[int, ...] = (37, 38, 39, 66, 67, 68)
    n_actions: int = 2
    noisy_clue_steps: Tuple[int, ...] = ()
    invariant_guard_steps: Tuple[int, ...] = ()
    invariant_guard_threshold: float = 0.0
    success_reward: float = 1.0
    failure_penalty: float = -1.0
    drift_penalty: float = -0.15
    drift_increase: float = 0.2
    drift_decay: float = 0.04
    unguarded_drift_increase: float = 0.0
    guarded_drift_decay: float = 0.0
    seed: int = 0


@dataclass
class Observation:
    step: int
    phase: str
    visible_regime_hint: Optional[int]
    clue: Optional[int]
    previous_success: Optional[bool]
    drift: float

    def vector(self) -> np.ndarray:
        phase_id = {"stable": 0.0, "memory": 1.0}.get(self.phase, 0.0)
        hint = -1.0 if self.visible_regime_hint is None else float(self.visible_regime_hint)
        clue = -1.0 if self.clue is None else float(self.clue)
        prev = -1.0 if self.previous_success is None else float(self.previous_success)
        return np.array([self.step / 100.0, phase_id, hint, clue, prev, self.drift], dtype=float)


@dataclass
class StepResult:
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, object] = field(default_factory=dict)


class RegimeMemoryEnv:
    """Small POMDP for testing arbitration, memory control, and recovery.

    Most steps require action == current hidden regime. A few later memory-query
    steps require the early secret clue instead. The clue is visible only near
    the beginning, so agents must decide whether to write and later read memory.
    """

    def __init__(self, config: EnvConfig | None = None):
        self.config = config or EnvConfig()
        self.action_space = tuple(range(self.config.n_actions))
        self.rng = np.random.default_rng(self.config.seed)
        self.secret = 0
        self.step_index = 0
        self.regime = 0
        self.drift = 0.0
        self.previous_success: Optional[bool] = None

    def reset(self, seed: int | None = None) -> Observation:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.secret = int(self.rng.integers(0, self.config.n_actions))
        self.step_index = 0
        self.regime = 0
        self.drift = 0.0
        self.previous_success = None
        return self._observation()

    def clone(self) -> "RegimeMemoryEnv":
        other = RegimeMemoryEnv(self.config)
        other.rng = self.rng
        other.secret = self.secret
        other.step_index = self.step_index
        other.regime = self.regime
        other.drift = self.drift
        other.previous_success = self.previous_success
        return other

    def oracle_action(self, memory_value: Optional[int] = None) -> int:
        if self._is_memory_query():
            return int(memory_value) if memory_value is not None else self.regime
        return self.regime

    def correct_action(self) -> int:
        return self.secret if self._is_memory_query() else self.regime

    def step(self, action: int, control_mode: str | None = None) -> StepResult:
        if action not in self.action_space:
            raise ValueError(f"Unsupported action: {action}")

        old_step = self.step_index
        correct = self.correct_action()
        success = int(action) == correct
        reward = self.config.success_reward if success else self.config.failure_penalty

        if success:
            self.drift = max(0.0, self.drift - self.config.drift_decay)
        else:
            self.drift = min(1.0, self.drift + self.config.drift_increase)

        invariant_guard_required = old_step in self.config.invariant_guard_steps or (
            self.config.invariant_guard_threshold > 0.0 and self.drift >= self.config.invariant_guard_threshold
        )
        if invariant_guard_required:
            if control_mode == "planner":
                self.drift = max(0.0, self.drift - self.config.guarded_drift_decay)
            else:
                self.drift = min(1.0, self.drift + self.config.unguarded_drift_increase)
        reward -= self.config.drift_penalty * self.drift

        self.previous_success = success
        self.step_index += 1
        if self.step_index in self.config.regime_shift_steps:
            self.regime = (self.regime + 1) % self.config.n_actions

        done = self.step_index >= self.config.horizon
        obs = self._observation()
        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            info={
                "correct_action": correct,
                "success": success,
                "regime": self.regime,
                "secret": self.secret,
                "memory_required": self._was_memory_query(self.step_index - 1),
                "invariant_guard_required": invariant_guard_required,
            },
        )

    def _observation(self) -> Observation:
        phase = "memory" if self._is_memory_query() else "stable"
        if self.step_index in self.config.memory_clue_steps:
            clue = self.secret
        elif self.step_index in self.config.noisy_clue_steps:
            clue = int((self.secret + 1) % self.config.n_actions)
        else:
            clue = None
        visible_hint = self.regime if self.previous_success is False else None
        return Observation(
            step=self.step_index,
            phase=phase,
            visible_regime_hint=visible_hint,
            clue=clue,
            previous_success=self.previous_success,
            drift=self.drift,
        )

    def _is_memory_query(self) -> bool:
        return self._was_memory_query(self.step_index)

    def _was_memory_query(self, step: int) -> bool:
        return step in self.config.memory_query_steps


def heldout_configs() -> List[EnvConfig]:
    return [
        EnvConfig(seed=0),
        EnvConfig(seed=1, regime_shift_steps=(18, 47, 63), memory_query_steps=(31, 32, 57, 58, 74)),
        EnvConfig(seed=2, regime_shift_steps=(12, 35, 59), memory_query_steps=(22, 23, 49, 50, 70, 71)),
    ]


def v01_train_configs(seed: int = 0) -> List[EnvConfig]:
    return [
        EnvConfig(
            seed=seed,
            n_actions=3,
            regime_shift_steps=(12, 29, 46, 63),
            memory_clue_steps=(4,),
            noisy_clue_steps=(8, 9),
            memory_query_steps=(33, 34, 35, 68, 69),
        ),
        EnvConfig(
            seed=seed + 1,
            n_actions=3,
            regime_shift_steps=(17, 31, 48, 61),
            memory_clue_steps=(5,),
            noisy_clue_steps=(10, 11),
            memory_query_steps=(27, 28, 52, 53, 72),
        ),
        EnvConfig(
            seed=seed + 2,
            n_actions=3,
            regime_shift_steps=(10, 25, 41, 58, 70),
            memory_clue_steps=(3,),
            noisy_clue_steps=(7, 13),
            memory_query_steps=(22, 23, 44, 45, 66, 67),
        ),
    ]


def v01_heldout_configs(seed: int = 100) -> List[EnvConfig]:
    return [
        EnvConfig(
            seed=seed,
            n_actions=3,
            regime_shift_steps=(14, 34, 55, 73),
            memory_clue_steps=(6,),
            noisy_clue_steps=(9, 15),
            memory_query_steps=(30, 31, 59, 60, 76),
        ),
        EnvConfig(
            seed=seed + 1,
            n_actions=3,
            regime_shift_steps=(8, 24, 39, 56, 71),
            memory_clue_steps=(4,),
            noisy_clue_steps=(12, 18),
            memory_query_steps=(20, 21, 47, 48, 69, 70),
        ),
        EnvConfig(
            seed=seed + 2,
            n_actions=3,
            regime_shift_steps=(19, 37, 51, 65),
            memory_clue_steps=(5,),
            noisy_clue_steps=(11, 16),
            memory_query_steps=(26, 27, 54, 55, 74, 75),
        ),
    ]


def v02_transfer_configs(seed: int = 200) -> List[EnvConfig]:
    return [
        EnvConfig(
            seed=seed,
            horizon=100,
            n_actions=3,
            regime_shift_steps=(9, 21, 36, 52, 67, 83),
            memory_clue_steps=(4,),
            noisy_clue_steps=(8, 14, 23),
            memory_query_steps=(26, 27, 28, 55, 56, 57, 88, 89),
        ),
        EnvConfig(
            seed=seed + 1,
            horizon=100,
            n_actions=3,
            regime_shift_steps=(13, 28, 43, 59, 74, 91),
            memory_clue_steps=(6,),
            noisy_clue_steps=(10, 17, 31),
            memory_query_steps=(34, 35, 61, 62, 63, 92, 93),
        ),
        EnvConfig(
            seed=seed + 2,
            horizon=100,
            n_actions=3,
            regime_shift_steps=(7, 19, 33, 48, 64, 78, 94),
            memory_clue_steps=(5,),
            noisy_clue_steps=(11, 16, 24),
            memory_query_steps=(29, 30, 49, 50, 70, 71, 95, 96),
        ),
    ]


def v03_train_configs(seed: int = 0) -> List[EnvConfig]:
    return [
        EnvConfig(
            seed=seed,
            horizon=120,
            n_actions=3,
            regime_shift_steps=(14, 33, 51, 76, 98),
            memory_clue_steps=(5,),
            noisy_clue_steps=(9, 17, 28),
            memory_query_steps=(39, 40, 73, 74, 105, 106),
            invariant_guard_steps=(24, 25, 58, 59, 90, 91),
            invariant_guard_threshold=0.10,
            unguarded_drift_increase=0.16,
            guarded_drift_decay=0.12,
        ),
        EnvConfig(
            seed=seed + 1,
            horizon=120,
            n_actions=3,
            regime_shift_steps=(11, 29, 47, 70, 94, 111),
            memory_clue_steps=(4,),
            noisy_clue_steps=(8, 16, 30),
            memory_query_steps=(35, 36, 64, 65, 99, 100),
            invariant_guard_steps=(21, 22, 55, 56, 86, 87),
            invariant_guard_threshold=0.10,
            unguarded_drift_increase=0.16,
            guarded_drift_decay=0.12,
        ),
        EnvConfig(
            seed=seed + 2,
            horizon=120,
            n_actions=3,
            regime_shift_steps=(16, 37, 55, 79, 101),
            memory_clue_steps=(6,),
            noisy_clue_steps=(12, 20, 33),
            memory_query_steps=(42, 43, 68, 69, 109, 110),
            invariant_guard_steps=(27, 28, 62, 63, 93, 94),
            invariant_guard_threshold=0.10,
            unguarded_drift_increase=0.16,
            guarded_drift_decay=0.12,
        ),
    ]


def v03_heldout_configs(seed: int = 300) -> List[EnvConfig]:
    return [
        EnvConfig(
            seed=seed,
            horizon=120,
            n_actions=3,
            regime_shift_steps=(10, 26, 44, 67, 89, 108),
            memory_clue_steps=(5,),
            noisy_clue_steps=(11, 19, 31),
            memory_query_steps=(37, 38, 71, 72, 103, 104),
            invariant_guard_steps=(23, 24, 53, 54, 83, 84, 112),
            invariant_guard_threshold=0.10,
            unguarded_drift_increase=0.18,
            guarded_drift_decay=0.12,
        ),
        EnvConfig(
            seed=seed + 1,
            horizon=120,
            n_actions=3,
            regime_shift_steps=(13, 32, 49, 75, 96, 114),
            memory_clue_steps=(4,),
            noisy_clue_steps=(9, 18, 27),
            memory_query_steps=(41, 42, 66, 67, 101, 102),
            invariant_guard_steps=(20, 21, 57, 58, 88, 89, 116),
            invariant_guard_threshold=0.10,
            unguarded_drift_increase=0.18,
            guarded_drift_decay=0.12,
        ),
        EnvConfig(
            seed=seed + 2,
            horizon=120,
            n_actions=3,
            regime_shift_steps=(8, 24, 43, 63, 82, 107),
            memory_clue_steps=(6,),
            noisy_clue_steps=(13, 22, 35),
            memory_query_steps=(33, 34, 60, 61, 96, 97),
            invariant_guard_steps=(18, 19, 50, 51, 80, 81, 110),
            invariant_guard_threshold=0.10,
            unguarded_drift_increase=0.18,
            guarded_drift_decay=0.12,
        ),
    ]


def v03b_drift_variant_configs(seed: int = 600) -> List[EnvConfig]:
    return [
        EnvConfig(
            seed=seed,
            horizon=120,
            n_actions=3,
            regime_shift_steps=(10, 26, 44, 67, 89, 108),
            memory_clue_steps=(5,),
            noisy_clue_steps=(11, 19, 31),
            memory_query_steps=(37, 38, 71, 72, 103, 104),
            invariant_guard_steps=(19, 20, 47, 48, 79, 80, 109),
            invariant_guard_threshold=0.08,
            unguarded_drift_increase=0.22,
            guarded_drift_decay=0.08,
        ),
        EnvConfig(
            seed=seed + 1,
            horizon=120,
            n_actions=3,
            regime_shift_steps=(13, 32, 49, 75, 96, 114),
            memory_clue_steps=(4,),
            noisy_clue_steps=(9, 18, 27),
            memory_query_steps=(41, 42, 66, 67, 101, 102),
            invariant_guard_steps=(24, 25, 61, 62, 91, 92, 117),
            invariant_guard_threshold=0.12,
            unguarded_drift_increase=0.14,
            guarded_drift_decay=0.16,
        ),
        EnvConfig(
            seed=seed + 2,
            horizon=120,
            n_actions=3,
            regime_shift_steps=(8, 24, 43, 63, 82, 107),
            memory_clue_steps=(6,),
            noisy_clue_steps=(13, 22, 35),
            memory_query_steps=(33, 34, 60, 61, 96, 97),
            invariant_guard_steps=(16, 17, 45, 46, 74, 75, 106, 107),
            invariant_guard_threshold=0.09,
            unguarded_drift_increase=0.20,
            guarded_drift_decay=0.10,
        ),
    ]
