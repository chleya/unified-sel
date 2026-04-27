"""
core/pool.py — Archived StructurePool lifecycle manager

This module is kept for backward compatibility (smoke tests).
It manages a collection of DFA structure units with input-novelty-driven
creation/branching and loss-plateau-driven cloning.

NOTE: The "surprise", "tension", and "utility" terminology is legacy
nomenclature from the original SEL-Lab project.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from core.structure import Structure, make_structure


SURPRISE_THRESHOLD = 0.60
TENSION_THRESHOLD = 0.08
UTILITY_DECAY = 0.005
UTILITY_PRUNE = 0.08
REINFORCE_AMOUNT = 0.05
CLONE_PERTURBATION = 0.05
MATURE_AGE = 80
MATURE_DECAY_SCALE = 0.35
_PHASE_STEP_THRESHOLD = 200
_SURPRISE_PRESSURE_RATIO = 0.7
_TENSION_PRESSURE_THRESHOLD = 0.3
_STABILIZATION_PRESSURE_STREAK = 3
_STABILIZATION_COOLDOWN_STEPS = 8
_STABILIZATION_LATE_OFFSET_STEPS = 200
_STABILIZATION_LATE_WINDOW_STEPS = 50
_STABILIZATION_LATE_MAX_PER_WINDOW = 4
_STABILIZATION_NEAR_FULL_GAP = 1
_STABILIZATION_MATURE_SURPRISE_GAP = 0.12
_STABILIZATION_MATURE_REINFORCE_SCALE = 0.35
_STABILIZATION_ACTIVE_REINFORCE_SCALE = 0.45
_STABILIZATION_YOUNG_AGE_THRESHOLD = 40
_STABILIZATION_YOUNG_ACTIVE_BONUS = 0.15
_PRESSURE_RELIEF_MIN_AGE = 25
_PRESSURE_RELIEF_UTILITY_MARGIN = 0.0


class StructurePool:
    def __init__(
        self,
        in_size: int,
        out_size: int,
        max_structures: int = 12,
        initial_structures: int = 1,
        surprise_threshold: float = SURPRISE_THRESHOLD,
        tension_threshold: float = TENSION_THRESHOLD,
        utility_decay: float = UTILITY_DECAY,
        utility_prune: float = UTILITY_PRUNE,
        reinforce_amount: float = REINFORCE_AMOUNT,
        clone_perturbation: float = CLONE_PERTURBATION,
        mature_age: int = MATURE_AGE,
        mature_decay_scale: float = MATURE_DECAY_SCALE,
        phase_step_threshold: int = _PHASE_STEP_THRESHOLD,
        surprise_pressure_ratio: float = _SURPRISE_PRESSURE_RATIO,
        tension_pressure_threshold: float = _TENSION_PRESSURE_THRESHOLD,
        stabilization_pressure_streak: int = _STABILIZATION_PRESSURE_STREAK,
        stabilization_cooldown_steps: int = _STABILIZATION_COOLDOWN_STEPS,
        stabilization_late_offset_steps: int = _STABILIZATION_LATE_OFFSET_STEPS,
        stabilization_late_window_steps: int = _STABILIZATION_LATE_WINDOW_STEPS,
        stabilization_late_max_per_window: int = _STABILIZATION_LATE_MAX_PER_WINDOW,
        stabilization_near_full_gap: int = _STABILIZATION_NEAR_FULL_GAP,
        stabilization_mature_surprise_gap: float = _STABILIZATION_MATURE_SURPRISE_GAP,
        stabilization_mature_reinforce_scale: float = _STABILIZATION_MATURE_REINFORCE_SCALE,
        stabilization_active_reinforce_scale: float = _STABILIZATION_ACTIVE_REINFORCE_SCALE,
        stabilization_young_age_threshold: int = _STABILIZATION_YOUNG_AGE_THRESHOLD,
        stabilization_young_active_bonus: float = _STABILIZATION_YOUNG_ACTIVE_BONUS,
        pressure_relief_min_age: int = _PRESSURE_RELIEF_MIN_AGE,
        pressure_relief_utility_margin: float = _PRESSURE_RELIEF_UTILITY_MARGIN,
        seed: Optional[int] = None,
    ):
        self.in_size = in_size
        self.out_size = out_size
        self.max_structures = max_structures
        self.surprise_threshold = surprise_threshold
        self.tension_threshold = tension_threshold
        self.utility_decay = utility_decay
        self.utility_prune = utility_prune
        self.reinforce_amount = reinforce_amount
        self.clone_perturbation = clone_perturbation
        self.mature_age = mature_age
        self.mature_decay_scale = mature_decay_scale
        self.phase_step_threshold = phase_step_threshold
        self.surprise_pressure_ratio = surprise_pressure_ratio
        self.tension_pressure_threshold = tension_pressure_threshold
        self.stabilization_pressure_streak = stabilization_pressure_streak
        self.stabilization_cooldown_steps = stabilization_cooldown_steps
        self.stabilization_late_offset_steps = stabilization_late_offset_steps
        self.stabilization_late_window_steps = stabilization_late_window_steps
        self.stabilization_late_max_per_window = stabilization_late_max_per_window
        self.stabilization_near_full_gap = stabilization_near_full_gap
        self.stabilization_mature_surprise_gap = stabilization_mature_surprise_gap
        self.stabilization_mature_reinforce_scale = stabilization_mature_reinforce_scale
        self.stabilization_active_reinforce_scale = stabilization_active_reinforce_scale
        self.stabilization_young_age_threshold = stabilization_young_age_threshold
        self.stabilization_young_active_bonus = stabilization_young_active_bonus
        self.pressure_relief_min_age = pressure_relief_min_age
        self.pressure_relief_utility_margin = pressure_relief_utility_margin
        self.rng = np.random.default_rng(seed)
        self._next_id = 0

        self.structures: List[Structure] = []
        for _ in range(initial_structures):
            self.structures.append(self._new_structure())

        self.total_creates = 0
        self.total_clones = 0
        self.total_prunes = 0
        self.step_count = 0
        self._pressure_streak = 0
        self._last_stabilize_step = -self.stabilization_cooldown_steps
        self._late_stabilize_steps: List[int] = []
        self._pending_reinforces: List[tuple[Structure, float]] = []
        self.total_stabilize_guard_blocks = 0
        self.total_stabilize_age_guard_blocks = 0
        self.total_stabilize_post_commits = 0
        self.total_pressure_relief_swaps = 0
        self.frozen = False
        self.frozen_prune_only = False  # 只冻结剪枝，允许创建新结构

    def observe(self, x: np.ndarray, y: int = None, out_size: int = 2) -> Dict:
        self.step_count += 1

        if not self.structures:
            structure = self._new_structure()
            self.structures.append(structure)
            self.total_creates += 1
            return self._build_result(
                "create",
                structure,
                1.0,
                pressure_active=False,
                persistent_pressure=False,
                high_pressure=False,
                can_boundary_stabilize=False,
            )

        surprises = [structure.current_surprise(x, y, out_size) for structure in self.structures]
        best_idx = int(np.argmin(surprises))
        best_surprise = float(surprises[best_idx])
        best_structure = self.structures[best_idx]

        best_structure.surprise_history.append(best_surprise)
        if len(best_structure.surprise_history) > 20:
            best_structure.surprise_history.pop(0)

        is_mid_late_phase = self.step_count > self.phase_step_threshold
        is_late_phase = self.step_count > (self.phase_step_threshold + self.stabilization_late_offset_steps)
        avg_tension = float(np.mean([s.tension for s in self.structures])) if self.structures else 0.0
        is_high_pressure = (
            best_surprise > self.surprise_threshold * self.surprise_pressure_ratio
            and avg_tension > self.tension_pressure_threshold
        )

        low_threshold = self.surprise_threshold * 0.5
        near_full = len(self.structures) >= max(1, self.max_structures - self.stabilization_near_full_gap)
        pressure_event = (
            is_mid_late_phase
            and near_full
            and best_surprise >= low_threshold
            and avg_tension >= self.tension_pressure_threshold
        )
        if pressure_event:
            self._pressure_streak += 1
        else:
            self._pressure_streak = 0

        mature_candidates = [
            (structure, float(structure.current_surprise(x, y, out_size)))
            for structure in self.structures
            if structure.age >= self.mature_age
        ]
        mature_structures = [row[0] for row in mature_candidates]
        persistent_pressure = self._pressure_streak >= self.stabilization_pressure_streak
        base_can_stabilize = (
            is_mid_late_phase
            and bool(mature_structures)
            and (is_high_pressure or persistent_pressure)
            and self._can_boundary_stabilize(is_late_phase)
        )
        mature_structure = None
        mature_structure_surprise = None
        stabilization_match_ok = False
        stabilization_age_ok = False
        if mature_candidates:
            mature_structure, mature_structure_surprise = min(
                mature_candidates,
                key=lambda row: (row[1], -row[0].utility, -row[0].age),
            )
            stabilization_match_ok = (
                mature_structure_surprise <= best_surprise + self.stabilization_mature_surprise_gap
            )
            stabilization_age_ok = (
                mature_structure is best_structure
                or best_structure.age >= self.stabilization_young_age_threshold
            )
        can_stabilize = base_can_stabilize and stabilization_match_ok and stabilization_age_ok
        if base_can_stabilize and not stabilization_match_ok:
            self.total_stabilize_guard_blocks += 1
        if base_can_stabilize and stabilization_match_ok and not stabilization_age_ok:
            self.total_stabilize_age_guard_blocks += 1

        if best_surprise < low_threshold:
            if best_structure.frozen:
                unfrozen = [s for s in self.structures if not s.frozen]
                if unfrozen:
                    unfrozen_surprises = [s.current_surprise(x, y, out_size) for s in unfrozen]
                    best_unfrozen_idx = int(np.argmin(unfrozen_surprises))
                    active = unfrozen[best_unfrozen_idx]
                    active.reinforce(self.reinforce_amount)
                    event = "reinforce"
                else:
                    if not self.frozen:
                        active = self._new_structure()
                        active.label = f"create_frozen_{self._next_id}"
                        self.structures.append(active)
                        self._next_id += 1
                        self.total_creates += 1
                        event = "create"
                    else:
                        active = best_structure
                        event = "reinforce"
            else:
                best_structure.reinforce(self.reinforce_amount)
                event = "reinforce"
                active = best_structure
        elif best_surprise < self.surprise_threshold:
            if not self.frozen and len(self.structures) < self.max_structures:
                active = best_structure.clone(
                    self._next_id,
                    self.clone_perturbation,
                    self.rng,
                )
                active.label = f"branch_{self._next_id}"
                self.structures.append(active)
                self._next_id += 1
                self.total_creates += 1
                event = "branch"
            elif not self.frozen:
                relief_candidate = None
                if best_structure.age < self.stabilization_young_age_threshold:
                    relief_candidate = self._select_pressure_relief_candidate(best_structure)
                if relief_candidate is not None:
                    self._replace_structure(relief_candidate)
                    active = best_structure.clone(
                        self._next_id,
                        self.clone_perturbation,
                        self.rng,
                    )
                    active.label = f"branch_{self._next_id}"
                    self.structures.append(active)
                    self._next_id += 1
                    self.total_creates += 1
                    self.total_pressure_relief_swaps += 1
                    event = "branch"
                elif can_stabilize:
                    self._apply_boundary_stabilize(
                        active_structure=best_structure,
                        mature_structure=mature_structure,
                        is_late_phase=is_late_phase,
                    )
                    self._record_boundary_stabilize(is_late_phase)
                    active = best_structure
                    event = "boundary_stabilize"
                else:
                    best_structure.reinforce(self.reinforce_amount * 0.5)
                    event = "reinforce"
                    active = best_structure
            else:
                best_structure.reinforce(self.reinforce_amount * 0.5)
                event = "reinforce"
                active = best_structure
        else:
            if (not self.frozen or self.frozen_prune_only) and len(self.structures) < self.max_structures:
                active = self._new_structure(label=f"new_{self._next_id}")
                self.structures.append(active)
                self.total_creates += 1
                event = "create"
            elif not self.frozen:
                relief_candidate = self._select_pressure_relief_candidate(best_structure)
                if relief_candidate is not None:
                    self._replace_structure(relief_candidate)
                    active = self._new_structure(label=f"new_{self._next_id}")
                    self.structures.append(active)
                    self.total_creates += 1
                    self.total_pressure_relief_swaps += 1
                    event = "create"
                elif can_stabilize:
                    self._apply_boundary_stabilize(
                        active_structure=best_structure,
                        mature_structure=mature_structure,
                        is_late_phase=is_late_phase,
                    )
                    self._record_boundary_stabilize(is_late_phase)
                    active = best_structure
                    event = "boundary_stabilize"
                else:
                    best_structure.reinforce(self.reinforce_amount * 0.3)
                    event = "reinforce"
                    active = best_structure
            else:
                best_structure.reinforce(self.reinforce_amount * 0.3)
                event = "reinforce"
                active = best_structure

        self._decay_all()
        # 剪枝在 frozen 或 frozen_prune_only 模式下都被冻结
        if not self.frozen and not self.frozen_prune_only and self.step_count % 50 == 0:
            self.prune()

        return self._build_result(
            event,
            active,
            best_surprise,
            pressure_active=bool(is_mid_late_phase and (is_high_pressure or persistent_pressure)),
            persistent_pressure=persistent_pressure,
            high_pressure=is_high_pressure,
            can_boundary_stabilize=can_stabilize,
        )

    def learn_active(
        self,
        active_structure: Structure,
        x: np.ndarray,
        output_error: np.ndarray,
        lr: float = 0.05,
        anchor_lambda: float = 0.0,
    ) -> float:
        return active_structure.learn(x, output_error, lr=lr, anchor_lambda=anchor_lambda)

    def set_anchors(self, X: np.ndarray = None, y: np.ndarray = None, out_size: int = 2, min_age: int = 50, freeze: bool = True) -> int:
        """为所有成熟结构设置锚点并冻结。来源：SEL-Lab phase3_model.py:1354-1367"""
        count = 0
        for s in self.structures:
            if s.age >= min_age and not s.anchor_set:
                s.set_anchor()
                if X is not None and y is not None:
                    s.estimate_anchor_fisher(X, y, out_size)
                if freeze:
                    s.frozen = True
                count += 1
        return count

    def evolve(self) -> List[str]:
        if self.frozen or self.frozen_prune_only:
            return []
        changes: List[str] = []
        if len(self.structures) >= self.max_structures:
            return changes

        high_tension = [
            structure
            for structure in self.structures
            if structure.tension > self.tension_threshold and structure.age > 10
        ]
        if not high_tension:
            return changes

        source = max(high_tension, key=lambda structure: structure.tension)
        clone = source.clone(self._next_id, self.clone_perturbation, self.rng)
        clone.label = f"clone_of_{source.id}"
        self.structures.append(clone)
        self._next_id += 1
        self.total_clones += 1
        changes.append(f"clone(source={source.id}, tension={source.tension:.3f})")
        return changes

    def prune(self) -> int:
        before = len(self.structures)
        if before <= 1:
            return 0

        self.structures = [
            structure
            for structure in self.structures
            if structure.utility >= self.utility_prune or structure.age < 20
        ]
        if not self.structures:
            self.structures.append(self._new_structure())

        pruned = before - len(self.structures)
        self.total_prunes += pruned
        return pruned

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.structures:
            return np.zeros(self.out_size)

        outputs = []
        weights = []
        for structure in self.structures:
            outputs.append(structure.forward(x))
            weights.append(structure.utility)

        weight_array = np.asarray(weights, dtype=float)
        weight_array = weight_array / (weight_array.sum() + 1e-8)
        return np.sum([w * o for w, o in zip(weight_array, outputs)], axis=0)

    def select_best_structure(self, x: np.ndarray) -> Optional[Structure]:
        if not self.structures:
            return None
        surprises = [structure.current_surprise(x) for structure in self.structures]
        best_idx = int(np.argmin(surprises))
        return self.structures[best_idx]

    def get_stats(self) -> Dict:
        tensions = [structure.tension for structure in self.structures]
        utilities = [structure.utility for structure in self.structures]
        return {
            "n_structures": len(self.structures),
            "avg_tension": float(np.mean(tensions)) if tensions else 0.0,
            "avg_utility": float(np.mean(utilities)) if utilities else 0.0,
            "total_creates": self.total_creates,
            "total_clones": self.total_clones,
            "total_prunes": self.total_prunes,
            "total_stabilize_guard_blocks": self.total_stabilize_guard_blocks,
            "total_stabilize_age_guard_blocks": self.total_stabilize_age_guard_blocks,
            "total_stabilize_post_commits": self.total_stabilize_post_commits,
            "total_pressure_relief_swaps": self.total_pressure_relief_swaps,
            "structure_ids": [structure.id for structure in self.structures],
            "config": {
                "max_structures": self.max_structures,
                "surprise_threshold": self.surprise_threshold,
                "tension_threshold": self.tension_threshold,
                "utility_decay": self.utility_decay,
                "utility_prune": self.utility_prune,
                "reinforce_amount": self.reinforce_amount,
                "clone_perturbation": self.clone_perturbation,
                "mature_age": self.mature_age,
                "mature_decay_scale": self.mature_decay_scale,
                "phase_step_threshold": self.phase_step_threshold,
                "surprise_pressure_ratio": self.surprise_pressure_ratio,
                "tension_pressure_threshold": self.tension_pressure_threshold,
                "stabilization_pressure_streak": self.stabilization_pressure_streak,
                "stabilization_cooldown_steps": self.stabilization_cooldown_steps,
                "stabilization_late_offset_steps": self.stabilization_late_offset_steps,
                "stabilization_late_window_steps": self.stabilization_late_window_steps,
                "stabilization_late_max_per_window": self.stabilization_late_max_per_window,
                "stabilization_near_full_gap": self.stabilization_near_full_gap,
                "stabilization_mature_surprise_gap": self.stabilization_mature_surprise_gap,
                "stabilization_mature_reinforce_scale": self.stabilization_mature_reinforce_scale,
                "stabilization_active_reinforce_scale": self.stabilization_active_reinforce_scale,
                "stabilization_young_age_threshold": self.stabilization_young_age_threshold,
                "stabilization_young_active_bonus": self.stabilization_young_active_bonus,
                "pressure_relief_min_age": self.pressure_relief_min_age,
                "pressure_relief_utility_margin": self.pressure_relief_utility_margin,
            },
        }

    def _new_structure(self, label: str = "") -> Structure:
        structure = make_structure(
            self._next_id,
            self.in_size,
            self.out_size,
            label=label or f"struct_{self._next_id}",
            rng=self.rng,
        )
        self._next_id += 1
        return structure

    def _decay_all(self) -> None:
        for structure in self.structures:
            decay_rate = self.utility_decay
            if structure.age >= self.mature_age:
                decay_rate *= self.mature_decay_scale
            structure.decay_utility(decay_rate)

    def _can_boundary_stabilize(self, is_late_phase: bool) -> bool:
        if self.step_count - self._last_stabilize_step < self.stabilization_cooldown_steps:
            return False
        if not is_late_phase:
            return True

        window_start = self.step_count - self.stabilization_late_window_steps
        self._late_stabilize_steps = [
            step for step in self._late_stabilize_steps if step > window_start
        ]
        return len(self._late_stabilize_steps) < self.stabilization_late_max_per_window

    def _record_boundary_stabilize(self, is_late_phase: bool) -> None:
        self._last_stabilize_step = self.step_count
        self._pressure_streak = 0
        if is_late_phase:
            self._late_stabilize_steps.append(self.step_count)

    def _apply_boundary_stabilize(
        self,
        active_structure: Structure,
        mature_structure: Structure,
        is_late_phase: bool,
    ) -> None:
        del is_late_phase
        mature_boost = self.reinforce_amount * self.stabilization_mature_reinforce_scale
        active_boost = self.reinforce_amount * self.stabilization_active_reinforce_scale

        if mature_structure is active_structure:
            active_structure.reinforce(max(active_boost, mature_boost))
            return

        if active_structure.age < self.stabilization_young_age_threshold:
            active_boost += self.reinforce_amount * self.stabilization_young_active_bonus
            mature_boost *= 0.5

        active_structure.reinforce(active_boost)
        self._pending_reinforces.append((mature_structure, mature_boost))

    def commit_pending_reinforcements(self) -> None:
        if not self._pending_reinforces:
            return

        pending = self._pending_reinforces
        self._pending_reinforces = []
        for structure, amount in pending:
            if any(candidate is structure for candidate in self.structures):
                structure.reinforce(amount)
                self.total_stabilize_post_commits += 1

    def _select_pressure_relief_candidate(self, active_structure: Structure) -> Optional[Structure]:
        if self.pressure_relief_utility_margin <= 0.0:
            return None
        utility_limit = self.utility_prune + self.pressure_relief_utility_margin
        candidates = [
            structure
            for structure in self.structures
            if structure is not active_structure
            and structure.age >= self.pressure_relief_min_age
            and structure.age < self.mature_age
            and structure.utility <= utility_limit
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda structure: (structure.utility, structure.age, structure.id))

    def _replace_structure(self, structure: Structure) -> None:
        self.structures = [candidate for candidate in self.structures if candidate is not structure]
        self.total_prunes += 1

    def _build_result(
        self,
        event: str,
        active: Structure,
        surprise: float,
        pressure_active: bool,
        persistent_pressure: bool,
        high_pressure: bool,
        can_boundary_stabilize: bool,
    ) -> Dict:
        return {
            "event": event,
            "active_structure": active,
            "surprise": round(float(surprise), 4),
            "n_structures": len(self.structures),
            "pressure_active": bool(pressure_active),
            "persistent_pressure": bool(persistent_pressure),
            "high_pressure": bool(high_pressure),
            "can_boundary_stabilize": bool(can_boundary_stabilize),
        }

    def __repr__(self) -> str:
        return f"StructurePool(n={len(self.structures)}, max={self.max_structures})"
