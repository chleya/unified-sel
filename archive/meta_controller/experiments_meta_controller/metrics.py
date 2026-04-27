from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class EpisodeMetrics:
    rewards: List[float] = field(default_factory=list)
    successes: List[bool] = field(default_factory=list)
    decisions: List[str] = field(default_factory=list)
    memory_required: List[bool] = field(default_factory=list)
    memory_reads: int = 0
    memory_writes: int = 0
    useful_reads: int = 0
    useful_writes: int = 0
    planner_calls: int = 0
    switches: int = 0
    switch_latencies: List[int] = field(default_factory=list)
    recovery_steps: List[int] = field(default_factory=list)
    drift_values: List[float] = field(default_factory=list)
    arbitration_regret: List[float] = field(default_factory=list)
    compute_cost: float = 0.0
    shield_interventions: int = 0
    drift_repairs: int = 0
    drift_repair_pre_values: List[float] = field(default_factory=list)
    drift_repair_post_values: List[float] = field(default_factory=list)
    drift_repair_deltas: List[float] = field(default_factory=list)
    high_drift_no_repair_deltas: List[float] = field(default_factory=list)
    safe_set_checks: int = 0
    safe_set_positive: int = 0
    safe_set_scores: List[float] = field(default_factory=list)
    safe_label_positive_rate: Optional[float] = None
    safe_train_score_mean: Optional[float] = None
    necessity_checks: int = 0
    necessity_positive: int = 0
    necessity_scores: List[float] = field(default_factory=list)
    necessity_label_positive_rate: Optional[float] = None
    necessity_train_score_mean: Optional[float] = None
    _last_decision: Optional[str] = None
    _shift_pending_step: Optional[int] = None
    _surprise_pending_step: Optional[int] = None

    def mark_shift(self, step: int) -> None:
        self._shift_pending_step = step
        self._surprise_pending_step = step

    def mark_shield_intervention(self) -> None:
        self.shield_interventions += 1

    def mark_drift_repair(self) -> None:
        self.drift_repairs += 1

    def record_drift_transition(
        self,
        pre_drift: float,
        post_drift: float,
        repaired: bool,
        high_drift_threshold: float = 0.08,
    ) -> None:
        pre = float(pre_drift)
        post = float(post_drift)
        if repaired:
            self.drift_repair_pre_values.append(pre)
            self.drift_repair_post_values.append(post)
            self.drift_repair_deltas.append(pre - post)
        elif pre >= high_drift_threshold:
            self.high_drift_no_repair_deltas.append(post - pre)

    def record_safe_set(self, is_safe: bool, score: float) -> None:
        self.safe_set_checks += 1
        self.safe_set_positive += int(is_safe)
        self.safe_set_scores.append(float(score))

    def set_safe_set_training_stats(self, label_positive_rate: float, score_mean: float) -> None:
        self.safe_label_positive_rate = float(label_positive_rate)
        self.safe_train_score_mean = float(score_mean)

    def record_necessity(self, is_necessary: bool, score: float) -> None:
        self.necessity_checks += 1
        self.necessity_positive += int(is_necessary)
        self.necessity_scores.append(float(score))

    def set_necessity_training_stats(self, label_positive_rate: float, score_mean: float) -> None:
        self.necessity_label_positive_rate = float(label_positive_rate)
        self.necessity_train_score_mean = float(score_mean)

    def record(
        self,
        step: int,
        decision: str,
        reward: float,
        success: bool,
        drift: float,
        memory_required: bool,
        oracle_reward: float,
        cost: float,
    ) -> None:
        self.rewards.append(reward)
        self.successes.append(bool(success))
        self.decisions.append(decision)
        self.memory_required.append(bool(memory_required))
        self.drift_values.append(drift)
        self.compute_cost += cost
        self.arbitration_regret.append(max(0.0, oracle_reward - reward))

        if decision != self._last_decision and self._last_decision is not None:
            self.switches += 1
        self._last_decision = decision

        if self._shift_pending_step is not None and "planner" in decision:
            self.switch_latencies.append(step - self._shift_pending_step)
            self._shift_pending_step = None
        if self._surprise_pending_step is not None and success:
            self.recovery_steps.append(step - self._surprise_pending_step)
            self._surprise_pending_step = None

        if "planner" in decision:
            self.planner_calls += 1

    def summary(self) -> Dict[str, float]:
        n = max(1, len(self.rewards))
        return {
            "total_reward": float(np.sum(self.rewards)),
            "task_success": float(np.mean(self.successes)) if self.successes else 0.0,
            "compute_cost": float(self.compute_cost),
            "planner_calls": float(self.planner_calls),
            "memory_reads": float(self.memory_reads),
            "memory_writes": float(self.memory_writes),
            "switch_count": float(self.switches),
            "switch_latency": float(np.mean(self.switch_latencies)) if self.switch_latencies else float("nan"),
            "arbitration_regret": float(np.mean(self.arbitration_regret)) if self.arbitration_regret else 0.0,
            "memory_read_precision": float(self.useful_reads / self.memory_reads) if self.memory_reads else 0.0,
            "memory_read_recall": self._memory_read_recall(),
            "memory_read_false_positive_rate": self._memory_read_false_positive_rate(),
            "memory_write_precision": float(self.useful_writes / self.memory_writes) if self.memory_writes else 0.0,
            "drift_under_horizon": float(np.mean(self.drift_values)) if self.drift_values else 0.0,
            "recovery_after_surprise": float(np.mean(self.recovery_steps)) if self.recovery_steps else float("nan"),
            "shield_interventions": float(self.shield_interventions),
            "shield_intervention_rate": float(self.shield_interventions / n),
            "drift_repairs": float(self.drift_repairs),
            "drift_repair_rate": float(self.drift_repairs / n),
            "drift_repair_pre_mean": float(np.mean(self.drift_repair_pre_values)) if self.drift_repair_pre_values else 0.0,
            "drift_repair_post_mean": float(np.mean(self.drift_repair_post_values)) if self.drift_repair_post_values else 0.0,
            "drift_repair_delta_mean": float(np.mean(self.drift_repair_deltas)) if self.drift_repair_deltas else 0.0,
            "drift_repair_delta_positive_rate": self._positive_rate(self.drift_repair_deltas),
            "high_drift_no_repair_rate": float(len(self.high_drift_no_repair_deltas) / n),
            "high_drift_no_repair_next_delta_mean": (
                float(np.mean(self.high_drift_no_repair_deltas)) if self.high_drift_no_repair_deltas else 0.0
            ),
            "drift_residual_after_repair": (
                float(np.mean(self.drift_repair_post_values)) if self.drift_repair_post_values else 0.0
            ),
            "repair_efficiency": (
                float(np.mean(self.drift_repair_deltas) / max(self.drift_repairs / n, 1e-9))
                if self.drift_repair_deltas
                else 0.0
            ),
            "safe_set_positive_rate": float(self.safe_set_positive / self.safe_set_checks) if self.safe_set_checks else 0.0,
            "safe_score_mean": float(np.mean(self.safe_set_scores)) if self.safe_set_scores else 0.0,
            "safe_score_min": float(np.min(self.safe_set_scores)) if self.safe_set_scores else 0.0,
            "safe_score_max": float(np.max(self.safe_set_scores)) if self.safe_set_scores else 0.0,
            "safe_label_positive_rate": self.safe_label_positive_rate if self.safe_label_positive_rate is not None else 0.0,
            "safe_train_score_mean": self.safe_train_score_mean if self.safe_train_score_mean is not None else 0.0,
            "necessity_positive_rate": float(self.necessity_positive / self.necessity_checks) if self.necessity_checks else 0.0,
            "necessity_score_mean": float(np.mean(self.necessity_scores)) if self.necessity_scores else 0.0,
            "necessity_label_positive_rate": self.necessity_label_positive_rate if self.necessity_label_positive_rate is not None else 0.0,
            "necessity_train_score_mean": self.necessity_train_score_mean if self.necessity_train_score_mean is not None else 0.0,
            "action_habit_rate": self._decision_rate("habit"),
            "action_planner_rate": self._decision_rate("planner"),
            "action_read_rate": self._decision_rate("read"),
            "action_write_rate": self._decision_rate("write"),
            "steps": float(n),
        }

    def _decision_rate(self, token: str) -> float:
        if not self.decisions:
            return 0.0
        return float(sum(token in decision for decision in self.decisions) / len(self.decisions))

    def _positive_rate(self, values: List[float]) -> float:
        if not values:
            return 0.0
        return float(sum(value > 0.0 for value in values) / len(values))

    def _memory_read_recall(self) -> float:
        required = sum(self.memory_required)
        return float(self.useful_reads / required) if required else 0.0

    def _memory_read_false_positive_rate(self) -> float:
        false_reads = sum(("read" in decision) and not required for decision, required in zip(self.decisions, self.memory_required))
        non_required = len(self.memory_required) - sum(self.memory_required)
        return float(false_reads / non_required) if non_required else 0.0


def aggregate_summaries(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    summary: Dict[str, float] = {}
    for key in keys:
        values = np.array([row[key] for row in rows], dtype=float)
        summary[key] = float("nan") if np.all(np.isnan(values)) else float(np.nanmean(values))
    return summary
