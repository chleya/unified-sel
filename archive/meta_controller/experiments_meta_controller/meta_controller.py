from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

from experiments.meta_controller.modules import WorkspaceState


META_ACTIONS = (
    "habit",
    "planner",
    "habit_read",
    "planner_read",
    "planner_read_write",
    "write_then_habit",
    "write_then_planner",
)


@dataclass(frozen=True)
class ControlDecision:
    name: str
    dominant_module: str
    memory_read: bool
    memory_write: bool
    broadcast: bool
    deliberation_precision: float


def decode_action(name: str) -> ControlDecision:
    if name == "habit":
        return ControlDecision(name, "habit", False, False, False, 0.0)
    if name == "planner":
        return ControlDecision(name, "planner", False, False, True, 1.0)
    if name == "habit_read":
        return ControlDecision(name, "habit", True, False, True, 0.2)
    if name == "planner_read":
        return ControlDecision(name, "planner", True, False, True, 1.0)
    if name == "planner_read_write":
        return ControlDecision(name, "planner", True, True, True, 1.0)
    if name == "write_then_habit":
        return ControlDecision(name, "habit", False, True, True, 0.0)
    if name == "write_then_planner":
        return ControlDecision(name, "planner", False, True, True, 1.0)
    raise ValueError(f"Unknown meta-action: {name}")


class BaseMetaController:
    def select(self, state: WorkspaceState) -> ControlDecision:
        raise NotImplementedError

    def update(self, state: WorkspaceState, decision: ControlDecision, reward: float) -> None:
        return None


def mask_workspace_state(state: WorkspaceState, mask: Iterable[str]) -> WorkspaceState:
    masked = set(mask)
    return WorkspaceState(
        task_progress=0.0 if "task_progress" in masked else state.task_progress,
        surprise=0.0 if "surprise" in masked else state.surprise,
        uncertainty=0.0 if "uncertainty" in masked else state.uncertainty,
        control_cost_estimate=0.0 if "cost" in masked or "control_cost_estimate" in masked else state.control_cost_estimate,
        memory_relevance=0.0 if "memory" in masked or "memory_relevance" in masked else state.memory_relevance,
        conflict_score=0.0 if "conflict" in masked or "conflict_score" in masked else state.conflict_score,
        invariant_violation=0.0 if "drift" in masked or "invariant_violation" in masked else state.invariant_violation,
        recent_failure_source=0.0 if "failure" in masked or "recent_failure_source" in masked else state.recent_failure_source,
    )


class SignalMaskingController(BaseMetaController):
    def __init__(self, wrapped: BaseMetaController, mask: Iterable[str]):
        self.wrapped = wrapped
        self.mask = tuple(mask)

    def select(self, state: WorkspaceState) -> ControlDecision:
        return self.wrapped.select(mask_workspace_state(state, self.mask))

    def update(self, state: WorkspaceState, decision: ControlDecision, reward: float) -> None:
        self.wrapped.update(mask_workspace_state(state, self.mask), decision, reward)


class FixedRuleController(BaseMetaController):
    def select(self, state: WorkspaceState) -> ControlDecision:
        if state.task_progress < 0.1 and state.memory_relevance >= 0.75:
            return decode_action("write_then_habit")
        if state.memory_relevance >= 0.75:
            return decode_action("planner_read")
        if state.surprise >= 0.45 or state.conflict_score >= 0.75:
            return decode_action("planner")
        return decode_action("habit")


class OracleMacroController(BaseMetaController):
    def select(self, state: WorkspaceState) -> ControlDecision:
        return decode_action(oracle_action_name(state))


def oracle_action_name(state: WorkspaceState) -> str:
    if state.task_progress < 0.1 and state.memory_relevance >= 0.75:
        return "write_then_habit"
    if state.memory_relevance >= 0.75:
        return "planner_read"
    if state.surprise >= 0.45 or state.conflict_score >= 0.75:
        return "planner"
    return "habit"


class AlwaysController(BaseMetaController):
    def __init__(self, action_name: str):
        self.action_name = action_name

    def select(self, state: WorkspaceState) -> ControlDecision:
        return decode_action(self.action_name)


class RandomController(BaseMetaController):
    def __init__(self, seed: int = 0, actions: Iterable[str] = META_ACTIONS):
        self.rng = np.random.default_rng(seed)
        self.actions = tuple(actions)

    def select(self, state: WorkspaceState) -> ControlDecision:
        return decode_action(str(self.rng.choice(self.actions)))


class ContextualBanditController(BaseMetaController):
    def __init__(
        self,
        seed: int = 0,
        actions: Iterable[str] = META_ACTIONS,
        lr: float = 0.08,
        epsilon: float = 0.15,
    ):
        self.rng = np.random.default_rng(seed)
        self.actions: List[str] = list(actions)
        self.lr = lr
        self.epsilon = epsilon
        self.weights: Dict[str, np.ndarray] = {a: np.zeros(9, dtype=float) for a in self.actions}

    def set_epsilon(self, epsilon: float) -> None:
        self.epsilon = epsilon

    def select(self, state: WorkspaceState) -> ControlDecision:
        if self.rng.random() < self.epsilon:
            return decode_action(str(self.rng.choice(self.actions)))
        features = self._features(state)
        scores = {a: float(self.weights[a] @ features) for a in self.actions}
        best = max(self.actions, key=lambda a: (scores[a], -self.actions.index(a)))
        return decode_action(best)

    def update(self, state: WorkspaceState, decision: ControlDecision, reward: float) -> None:
        features = self._features(state)
        pred = float(self.weights[decision.name] @ features)
        error = reward - pred
        self.weights[decision.name] += self.lr * error * features

    def _features(self, state: WorkspaceState) -> np.ndarray:
        return np.concatenate([np.array([1.0], dtype=float), state.vector()])


class ImitationController(BaseMetaController):
    def __init__(
        self,
        seed: int = 0,
        actions: Iterable[str] = META_ACTIONS,
        lr: float = 0.12,
        epochs: int = 8,
    ):
        self.rng = np.random.default_rng(seed)
        self.actions: List[str] = list(actions)
        self.action_to_idx = {action: i for i, action in enumerate(self.actions)}
        self.lr = lr
        self.epochs = epochs
        self.weights = np.zeros((len(self.actions), 9), dtype=float)

    def fit(self, examples: List[Tuple[WorkspaceState, str]]) -> None:
        if not examples:
            return
        for _ in range(self.epochs):
            order = self.rng.permutation(len(examples))
            for idx in order:
                state, label = examples[int(idx)]
                if label not in self.action_to_idx:
                    continue
                features = self._features(state)
                gold = self.action_to_idx[label]
                pred = int(np.argmax(self.weights @ features))
                if pred != gold:
                    self.weights[gold] += self.lr * features
                    self.weights[pred] -= self.lr * features

    def select(self, state: WorkspaceState) -> ControlDecision:
        features = self._features(state)
        idx = int(np.argmax(self.weights @ features))
        return decode_action(self.actions[idx])

    def _features(self, state: WorkspaceState) -> np.ndarray:
        return np.concatenate([np.array([1.0], dtype=float), state.vector()])


class FactoredController(BaseMetaController):
    """Factored learned controller for dominance, read, and write gates."""

    def __init__(
        self,
        seed: int = 0,
        lr: float = 0.08,
        epsilon: float = 0.08,
        imitation_epochs: int = 8,
    ):
        self.rng = np.random.default_rng(seed)
        self.lr = lr
        self.epsilon = epsilon
        self.imitation_epochs = imitation_epochs
        self.dominance_weights = np.zeros(9, dtype=float)
        self.read_weights = np.zeros(9, dtype=float)
        self.write_weights = np.zeros(9, dtype=float)
        self.baseline = 0.0
        self.explore_write = True

    def fit(self, examples: List[Tuple[WorkspaceState, str]]) -> None:
        if not examples:
            return
        for _ in range(self.imitation_epochs):
            order = self.rng.permutation(len(examples))
            for idx in order:
                state, label = examples[int(idx)]
                decision = decode_action(label)
                features = self._features(state)
                self._perceptron_update(self.dominance_weights, features, 1 if decision.dominant_module == "planner" else 0)
                self._perceptron_update(self.read_weights, features, 1 if decision.memory_read else 0)
                self._perceptron_update(self.write_weights, features, 1 if decision.memory_write else 0)

    def set_epsilon(self, epsilon: float) -> None:
        self.epsilon = epsilon

    def select(self, state: WorkspaceState) -> ControlDecision:
        features = self._features(state)
        planner = self._binary_decision(self.dominance_weights, features)
        read = self._binary_decision(self.read_weights, features)
        write = self._binary_decision(self.write_weights, features)

        if self.rng.random() < self.epsilon:
            planner = bool(self.rng.integers(0, 2))
        if self.rng.random() < self.epsilon:
            read = bool(self.rng.integers(0, 2))
        if self.explore_write and self.rng.random() < self.epsilon:
            write = bool(self.rng.integers(0, 2))

        if planner and read and write:
            return decode_action("planner_read_write")
        if planner and read:
            return decode_action("planner_read")
        if planner and write:
            return decode_action("write_then_planner")
        if planner:
            return decode_action("planner")
        if read:
            return decode_action("habit_read")
        if write:
            return decode_action("write_then_habit")
        return decode_action("habit")

    def update(self, state: WorkspaceState, decision: ControlDecision, reward: float) -> None:
        features = self._features(state)
        advantage = reward - self.baseline
        self.baseline = 0.95 * self.baseline + 0.05 * reward
        direction = 1.0 if advantage >= 0.0 else -1.0
        self._policy_update(self.dominance_weights, features, decision.dominant_module == "planner", direction)
        self._policy_update(self.read_weights, features, decision.memory_read, direction)
        self._policy_update(self.write_weights, features, decision.memory_write, direction)

    def _features(self, state: WorkspaceState) -> np.ndarray:
        return np.concatenate([np.array([1.0], dtype=float), state.vector()])

    def _binary_decision(self, weights: np.ndarray, features: np.ndarray) -> bool:
        return bool(weights @ features >= 0.0)

    def _perceptron_update(self, weights: np.ndarray, features: np.ndarray, label: int) -> None:
        pred = 1 if weights @ features >= 0.0 else 0
        if pred != label:
            weights += self.lr * (label - pred) * features

    def _policy_update(self, weights: np.ndarray, features: np.ndarray, chosen: bool, direction: float) -> None:
        sign = 1.0 if chosen else -1.0
        weights += self.lr * direction * sign * features


class ConservativeFactoredController(FactoredController):
    """Factored controller with guarded online updates.

    This variant is intentionally conservative: it preserves the imitation
    warm-start unless an online reward signal is strong enough to justify a
    small read/dominance adjustment. The write gate is frozen after warm-start.
    """

    def __init__(
        self,
        seed: int = 0,
        lr: float = 0.015,
        epsilon: float = 0.01,
        imitation_epochs: int = 8,
        margin: float = 0.35,
    ):
        super().__init__(seed=seed, lr=lr, epsilon=epsilon, imitation_epochs=imitation_epochs)
        self.margin = margin
        self.explore_write = False

    def update(self, state: WorkspaceState, decision: ControlDecision, reward: float) -> None:
        features = self._features(state)
        advantage = reward - self.baseline
        self.baseline = 0.98 * self.baseline + 0.02 * reward
        if abs(advantage) < self.margin:
            return

        direction = 1.0 if advantage > 0.0 else -1.0
        dominance_lr = self.lr
        read_lr = self.lr * 0.35

        self.dominance_weights += dominance_lr * direction * (1.0 if decision.dominant_module == "planner" else -1.0) * features

        # Only adjust read when the state is explicitly memory-relevant. This
        # avoids turning read into a generic response to low reward.
        if state.memory_relevance >= 0.5:
            self.read_weights += read_lr * direction * (1.0 if decision.memory_read else -1.0) * features

        # The write gate remains frozen after warm-start.


class ReadDisciplinedFactoredController(ConservativeFactoredController):
    """Conservative factored controller with stricter read-gate discipline."""

    def __init__(
        self,
        seed: int = 0,
        lr: float = 0.012,
        epsilon: float = 0.005,
        imitation_epochs: int = 8,
        margin: float = 0.35,
        read_margin: float = 0.45,
    ):
        super().__init__(seed=seed, lr=lr, epsilon=epsilon, imitation_epochs=imitation_epochs, margin=margin)
        self.read_margin = read_margin

    def update(self, state: WorkspaceState, decision: ControlDecision, reward: float) -> None:
        features = self._features(state)
        advantage = reward - self.baseline
        self.baseline = 0.98 * self.baseline + 0.02 * reward
        if abs(advantage) >= self.margin:
            direction = 1.0 if advantage > 0.0 else -1.0
            self.dominance_weights += self.lr * direction * (1.0 if decision.dominant_module == "planner" else -1.0) * features

        if state.memory_relevance >= 0.75 and not decision.memory_read and advantage < -self.read_margin:
            self.read_weights += self.lr * 0.35 * features
            return

        if not decision.memory_read:
            return

        if state.memory_relevance < 0.75:
            self.read_weights -= self.lr * 0.45 * features
            return

        if advantage < -self.read_margin:
            self.read_weights -= self.lr * 0.15 * features
        elif advantage > self.read_margin:
            self.read_weights += self.lr * 0.1 * features


class DominanceTunedFactoredController(ConservativeFactoredController):
    """Conservative controller that freezes read/write and tunes dominance only."""

    def __init__(
        self,
        seed: int = 0,
        lr: float = 0.01,
        epsilon: float = 0.005,
        imitation_epochs: int = 8,
        margin: float = 0.4,
    ):
        super().__init__(seed=seed, lr=lr, epsilon=epsilon, imitation_epochs=imitation_epochs, margin=margin)
        self.explore_write = False

    def select(self, state: WorkspaceState) -> ControlDecision:
        saved_epsilon = self.epsilon
        self.epsilon = 0.0
        decision = super().select(state)
        self.epsilon = saved_epsilon
        if self.rng.random() < self.epsilon:
            features = self._features(state)
            read = self._binary_decision(self.read_weights, features)
            write = self._binary_decision(self.write_weights, features)
            planner = bool(self.rng.integers(0, 2))
            if planner and read and write:
                return decode_action("planner_read_write")
            if planner and read:
                return decode_action("planner_read")
            if planner and write:
                return decode_action("write_then_planner")
            if planner:
                return decode_action("planner")
            if read:
                return decode_action("habit_read")
            if write:
                return decode_action("write_then_habit")
            return decode_action("habit")
        return decision

    def update(self, state: WorkspaceState, decision: ControlDecision, reward: float) -> None:
        features = self._features(state)
        advantage = reward - self.baseline
        self.baseline = 0.98 * self.baseline + 0.02 * reward
        if abs(advantage) < self.margin:
            return
        direction = 1.0 if advantage > 0.0 else -1.0
        self.dominance_weights += self.lr * direction * (1.0 if decision.dominant_module == "planner" else -1.0) * features


class CounterfactualDominanceController(DominanceTunedFactoredController):
    """Dominance-only controller trained from counterfactual module values."""

    def __init__(
        self,
        seed: int = 0,
        lr: float = 0.025,
        epsilon: float = 0.0,
        imitation_epochs: int = 8,
        margin: float = 0.08,
    ):
        super().__init__(seed=seed, lr=lr, epsilon=epsilon, imitation_epochs=imitation_epochs, margin=margin)

    def update_counterfactual(self, state: WorkspaceState, planner_value: float, habit_value: float) -> None:
        diff = planner_value - habit_value
        if abs(diff) < self.margin:
            return
        label = 1 if diff > 0.0 else 0
        features = self._features(state)
        self._perceptron_update(self.dominance_weights, features, label)


class RolloutDominanceController(CounterfactualDominanceController):
    """Dominance controller trained from short-horizon counterfactual rollouts."""

    def __init__(
        self,
        seed: int = 0,
        lr: float = 0.02,
        epsilon: float = 0.0,
        imitation_epochs: int = 8,
        margin: float = 0.12,
        rollout_horizon: int = 3,
    ):
        super().__init__(seed=seed, lr=lr, epsilon=epsilon, imitation_epochs=imitation_epochs, margin=margin)
        self.rollout_horizon = rollout_horizon


class RiskAverseRolloutDominanceController(RolloutDominanceController):
    """Rollout dominance with asymmetric margins favoring planner safety."""

    def __init__(
        self,
        seed: int = 0,
        lr: float = 0.015,
        epsilon: float = 0.0,
        imitation_epochs: int = 8,
        margin: float = 0.12,
        rollout_horizon: int = 3,
        habit_margin: float = 0.45,
    ):
        super().__init__(
            seed=seed,
            lr=lr,
            epsilon=epsilon,
            imitation_epochs=imitation_epochs,
            margin=margin,
            rollout_horizon=rollout_horizon,
        )
        self.habit_margin = habit_margin

    def update_rollout(self, state: WorkspaceState, planner_value: float, habit_value: float) -> None:
        diff = planner_value - habit_value
        if diff > self.margin:
            label = 1
        elif diff < -self.habit_margin:
            label = 0
        else:
            return
        self._perceptron_update(self.dominance_weights, self._features(state), label)


class ShieldedDominanceController(CounterfactualDominanceController):
    """Counterfactual dominance with a runtime planner fallback shield."""

    def __init__(
        self,
        seed: int = 0,
        lr: float = 0.025,
        epsilon: float = 0.0,
        imitation_epochs: int = 8,
        margin: float = 0.08,
        surprise_threshold: float = 0.45,
        conflict_threshold: float = 0.75,
        drift_threshold: float = 0.08,
    ):
        super().__init__(seed=seed, lr=lr, epsilon=epsilon, imitation_epochs=imitation_epochs, margin=margin)
        self.surprise_threshold = surprise_threshold
        self.conflict_threshold = conflict_threshold
        self.drift_threshold = drift_threshold
        self.last_shield_intervened = False

    def select(self, state: WorkspaceState) -> ControlDecision:
        proposed = super().select(state)
        self.last_shield_intervened = False
        if proposed.dominant_module == "planner":
            return proposed
        if not self._shield_requires_planner(state):
            return proposed
        self.last_shield_intervened = True
        return self._with_planner_dominance(proposed)

    def _shield_requires_planner(self, state: WorkspaceState) -> bool:
        if state.recent_failure_source > 0.0:
            return True
        if state.invariant_violation > self.drift_threshold:
            return True
        if state.surprise >= self.surprise_threshold:
            return True
        if state.conflict_score >= self.conflict_threshold:
            return True
        if state.memory_relevance >= 0.75 and state.task_progress >= 0.1:
            return True
        return False

    def _with_planner_dominance(self, decision: ControlDecision) -> ControlDecision:
        if decision.memory_read and decision.memory_write:
            return decode_action("planner_read_write")
        if decision.memory_read:
            return decode_action("planner_read")
        if decision.memory_write:
            return decode_action("write_then_planner")
        return decode_action("planner")


class HabitSafeSetController(DominanceTunedFactoredController):
    """Dominance controller that chooses habit only inside a learned safe set."""

    def __init__(
        self,
        seed: int = 0,
        lr: float = 0.035,
        epsilon: float = 0.0,
        imitation_epochs: int = 8,
        margin: float = 0.4,
        safe_epochs: int = 6,
        safe_bias: float = -0.2,
        safe_threshold: float = 0.0,
    ):
        super().__init__(seed=seed, lr=lr, epsilon=epsilon, imitation_epochs=imitation_epochs, margin=margin)
        self.safe_epochs = safe_epochs
        self.safe_threshold = safe_threshold
        self.safe_weights = np.zeros(9, dtype=float)
        self.safe_weights[0] = safe_bias
        self.safe_label_positive_rate = 0.0
        self.safe_score_mean = 0.0
        self.safe_score_min = 0.0
        self.safe_score_max = 0.0
        self.last_habit_safe = False
        self.last_safe_score = 0.0

    def fit_safe_set(self, examples: List[Tuple[WorkspaceState, bool]]) -> None:
        if not examples:
            return
        self.safe_label_positive_rate = float(sum(1 for _, safe in examples if safe) / len(examples))
        for _ in range(self.safe_epochs):
            order = self.rng.permutation(len(examples))
            for idx in order:
                state, safe = examples[int(idx)]
                self._perceptron_update(self.safe_weights, self._features(state), 1 if safe else 0)
        scores = np.array([self.safe_score(state) for state, _ in examples], dtype=float)
        self.safe_score_mean = float(np.mean(scores))
        self.safe_score_min = float(np.min(scores))
        self.safe_score_max = float(np.max(scores))

    def safe_score(self, state: WorkspaceState) -> float:
        return float(self.safe_weights @ self._features(state))

    def habit_is_safe(self, state: WorkspaceState) -> bool:
        return self.safe_score(state) >= self.safe_threshold

    def select(self, state: WorkspaceState) -> ControlDecision:
        features = self._features(state)
        read = self._binary_decision(self.read_weights, features)
        write = self._binary_decision(self.write_weights, features)
        self.last_safe_score = self.safe_score(state)
        habit_safe = self.last_safe_score >= self.safe_threshold
        self.last_habit_safe = habit_safe
        if habit_safe:
            if read:
                return decode_action("habit_read")
            if write:
                return decode_action("write_then_habit")
            return decode_action("habit")
        if read and write:
            return decode_action("planner_read_write")
        if read:
            return decode_action("planner_read")
        if write:
            return decode_action("write_then_planner")
        return decode_action("planner")


class PlannerNecessityController(HabitSafeSetController):
    """Safe-set controller with an additional learned planner-necessity gate."""

    def __init__(
        self,
        seed: int = 0,
        lr: float = 0.035,
        epsilon: float = 0.0,
        imitation_epochs: int = 8,
        margin: float = 0.4,
        safe_epochs: int = 6,
        safe_bias: float = -0.05,
        safe_threshold: float = 0.0,
        necessity_epochs: int = 6,
        necessity_threshold: float = 0.0,
        necessity_bias: float = 0.1,
    ):
        super().__init__(
            seed=seed,
            lr=lr,
            epsilon=epsilon,
            imitation_epochs=imitation_epochs,
            margin=margin,
            safe_epochs=safe_epochs,
            safe_bias=safe_bias,
            safe_threshold=safe_threshold,
        )
        self.necessity_epochs = necessity_epochs
        self.necessity_threshold = necessity_threshold
        self.necessity_weights = np.zeros(9, dtype=float)
        self.necessity_weights[0] = necessity_bias
        self.necessity_label_positive_rate = 0.0
        self.necessity_score_mean = 0.0
        self.last_planner_necessary = False
        self.last_necessity_score = 0.0

    def fit_necessity(self, examples: List[Tuple[WorkspaceState, bool]]) -> None:
        if not examples:
            return
        self.necessity_label_positive_rate = float(sum(1 for _, needed in examples if needed) / len(examples))
        for _ in range(self.necessity_epochs):
            order = self.rng.permutation(len(examples))
            for idx in order:
                state, needed = examples[int(idx)]
                self._perceptron_update(self.necessity_weights, self._features(state), 1 if needed else 0)
        scores = np.array([self.necessity_score(state) for state, _ in examples], dtype=float)
        self.necessity_score_mean = float(np.mean(scores))

    def necessity_score(self, state: WorkspaceState) -> float:
        return float(self.necessity_weights @ self._features(state))

    def planner_is_necessary(self, state: WorkspaceState) -> bool:
        return self.necessity_score(state) >= self.necessity_threshold

    def select(self, state: WorkspaceState) -> ControlDecision:
        features = self._features(state)
        read = self._binary_decision(self.read_weights, features)
        write = self._binary_decision(self.write_weights, features)
        self.last_safe_score = self.safe_score(state)
        self.last_habit_safe = self.last_safe_score >= self.safe_threshold
        self.last_necessity_score = self.necessity_score(state)
        self.last_planner_necessary = self.last_necessity_score >= self.necessity_threshold

        if self.last_habit_safe and not self.last_planner_necessary:
            if read:
                return decode_action("habit_read")
            if write:
                return decode_action("write_then_habit")
            return decode_action("habit")
        if read and write:
            return decode_action("planner_read_write")
        if read:
            return decode_action("planner_read")
        if write:
            return decode_action("write_then_planner")
        return decode_action("planner")


class DriftAwarePlannerNecessityController(PlannerNecessityController):
    """B3 controller with a narrow drift repair gate."""

    def __init__(self, *args, drift_repair_threshold: float = 0.10, **kwargs):
        super().__init__(*args, **kwargs)
        self.drift_repair_threshold = drift_repair_threshold
        self.last_drift_repair = False

    def select(self, state: WorkspaceState) -> ControlDecision:
        decision = super().select(state)
        self.last_drift_repair = state.invariant_violation >= self.drift_repair_threshold
        if not self.last_drift_repair or decision.dominant_module == "planner":
            return decision
        return ControlDecision(
            name="planner_read" if decision.memory_read else "planner",
            dominant_module="planner",
            memory_read=decision.memory_read,
            memory_write=False,
            broadcast=True,
            deliberation_precision=1.0,
        )
