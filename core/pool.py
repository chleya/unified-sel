"""
Structure pool with surprise-driven create/branch and tension-driven clone.
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
        self.rng = np.random.default_rng(seed)
        self._next_id = 0

        self.structures: List[Structure] = []
        for _ in range(initial_structures):
            self.structures.append(self._new_structure())

        self.total_creates = 0
        self.total_clones = 0
        self.total_prunes = 0
        self.step_count = 0

    def observe(self, x: np.ndarray) -> Dict:
        self.step_count += 1

        if not self.structures:
            structure = self._new_structure()
            self.structures.append(structure)
            self.total_creates += 1
            return self._build_result("create", structure, 1.0)

        surprises = [structure.current_surprise(x) for structure in self.structures]
        best_idx = int(np.argmin(surprises))
        best_surprise = float(surprises[best_idx])
        best_structure = self.structures[best_idx]

        best_structure.surprise_history.append(best_surprise)
        if len(best_structure.surprise_history) > 20:
            best_structure.surprise_history.pop(0)

        low_threshold = self.surprise_threshold * 0.5
        if best_surprise < low_threshold:
            best_structure.reinforce(self.reinforce_amount)
            event = "reinforce"
            active = best_structure
        elif best_surprise < self.surprise_threshold:
            if len(self.structures) < self.max_structures:
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
            else:
                best_structure.reinforce(self.reinforce_amount * 0.5)
                event = "reinforce"
                active = best_structure
        else:
            if len(self.structures) < self.max_structures:
                active = self._new_structure(label=f"new_{self._next_id}")
                self.structures.append(active)
                self.total_creates += 1
                event = "create"
            else:
                best_structure.reinforce(self.reinforce_amount * 0.3)
                event = "reinforce"
                active = best_structure

        self._decay_all()
        if self.step_count % 50 == 0:
            self.prune()

        return self._build_result(event, active, best_surprise)

    def learn_active(
        self,
        active_structure: Structure,
        x: np.ndarray,
        output_error: np.ndarray,
        lr: float = 0.05,
    ) -> float:
        return active_structure.learn(x, output_error, lr=lr)

    def evolve(self) -> List[str]:
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

    def _build_result(self, event: str, active: Structure, surprise: float) -> Dict:
        return {
            "event": event,
            "active_structure": active,
            "surprise": round(float(surprise), 4),
            "n_structures": len(self.structures),
        }

    def __repr__(self) -> str:
        return f"StructurePool(n={len(self.structures)}, max={self.max_structures})"
