"""
core/topo_fusion.py — Phase 4: TopoMem ECU + Unified-SEL Pool Fusion

Purpose:
- TopoMem ECU health signals modulate StructurePool lifecycle events
- This is NOT retrieval fusion (retrieval was shown to degrade performance)
- Instead, topological health informs create/branch/clone/reinforce decisions

Key insight: TopoMem operates in high-D text embedding space; Unified-SEL in low-D.
The fusion uses health_status signals (not retrieval) to bridge the two systems.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from core.pool import StructurePool
from core.structure import Structure


# ------------------------------------------------------------------
# Health modulation
# ------------------------------------------------------------------

@dataclass
class HealthModulatedParams:
    """Pool parameters modulated by TopoMem ECU health status."""
    surprise_threshold: float = 0.60
    tension_threshold: float = 0.08
    utility_prune: float = 0.08
    reinforce_scale: float = 1.0
    phase_step_threshold: int = 200
    suppress_creation: bool = False


def modulate_by_health(
    health_score: float,
    prune_aggressiveness: float,
    cluster_filter_enabled: bool,
    should_early_intervene: bool,
    trend_direction: str,
    current_step: int,
    base_params: Optional[HealthModulatedParams] = None,
) -> HealthModulatedParams:
    """Translate TopoMem ECU HealthStatus into pool parameter adjustments."""
    if base_params is None:
        base_params = HealthModulatedParams()

    params = HealthModulatedParams(
        surprise_threshold=base_params.surprise_threshold,
        tension_threshold=base_params.tension_threshold,
        utility_prune=base_params.utility_prune,
        reinforce_scale=base_params.reinforce_scale,
        phase_step_threshold=base_params.phase_step_threshold,
        suppress_creation=False,
    )

    # 1. Health score [0,1] -> reinforce_scale [0.5, 1.0]
    # Low health = reduce reinforcement (structures are unstable)
    params.reinforce_scale = 0.5 + 0.5 * health_score

    # 2. Prune aggressiveness overrides utility_prune
    # ECU outputs [0, 0.5], map to utility_prune range [0.05, 0.15]
    params.utility_prune = 0.05 + prune_aggressiveness * 0.2

    # 3. Trend direction modulates surprise_threshold
    # GREEN: normal, YELLOW: slightly higher (more tolerant),
    # ORANGE/RED: lower (more sensitive to novelty)
    trend_multipliers = {
        "green": 1.0,
        "yellow": 1.1,
        "orange": 0.85,
        "red": 0.7,
    }
    mult = trend_multipliers.get(trend_direction, 1.0)
    params.surprise_threshold = base_params.surprise_threshold * mult

    # 4. Early intervention -> suppress creation events
    if should_early_intervene:
        params.suppress_creation = True
        params.phase_step_threshold = int(base_params.phase_step_threshold * 1.5)

    # 5. Cluster filter enabled -> favor older structures
    if cluster_filter_enabled:
        params.phase_step_threshold = int(params.phase_step_threshold * 0.8)

    return params


# ------------------------------------------------------------------
# Health-aware StructurePool
# ------------------------------------------------------------------

class HealthAwareStructurePool(StructurePool):
    """StructurePool extended with health-aware parameter modulation.

    This subclass overrides observe() to incorporate health_status signals
    from TopoMem ECU into pool lifecycle decisions.
    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        max_structures: int = 12,
        initial_structures: int = 1,
        surprise_threshold: float = 0.60,
        tension_threshold: float = 0.08,
        utility_decay: float = 0.005,
        utility_prune: float = 0.08,
        reinforce_amount: float = 0.05,
        clone_perturbation: float = 0.05,
        mature_age: int = 80,
        mature_decay_scale: float = 0.35,
        phase_step_threshold: int = 200,
        surprise_pressure_ratio: float = 0.7,
        tension_pressure_threshold: float = 0.3,
        stabilization_pressure_streak: int = 3,
        stabilization_cooldown_steps: int = 8,
        stabilization_late_offset_steps: int = 200,
        stabilization_late_window_steps: int = 50,
        stabilization_late_max_per_window: int = 4,
        stabilization_near_full_gap: int = 1,
        stabilization_mature_surprise_gap: float = 0.12,
        stabilization_mature_reinforce_scale: float = 0.35,
        stabilization_active_reinforce_scale: float = 0.45,
        stabilization_young_age_threshold: int = 40,
        stabilization_young_active_bonus: float = 0.15,
        pressure_relief_min_age: int = 25,
        pressure_relief_utility_margin: float = 0.0,
        seed: Optional[int] = None,
    ):
        super().__init__(
            in_size=in_size,
            out_size=out_size,
            max_structures=max_structures,
            initial_structures=initial_structures,
            surprise_threshold=surprise_threshold,
            tension_threshold=tension_threshold,
            utility_decay=utility_decay,
            utility_prune=utility_prune,
            reinforce_amount=reinforce_amount,
            clone_perturbation=clone_perturbation,
            mature_age=mature_age,
            mature_decay_scale=mature_decay_scale,
            phase_step_threshold=phase_step_threshold,
            surprise_pressure_ratio=surprise_pressure_ratio,
            tension_pressure_threshold=tension_pressure_threshold,
            stabilization_pressure_streak=stabilization_pressure_streak,
            stabilization_cooldown_steps=stabilization_cooldown_steps,
            stabilization_late_offset_steps=stabilization_late_offset_steps,
            stabilization_late_window_steps=stabilization_late_window_steps,
            stabilization_late_max_per_window=stabilization_late_max_per_window,
            stabilization_near_full_gap=stabilization_near_full_gap,
            stabilization_mature_surprise_gap=stabilization_mature_surprise_gap,
            stabilization_mature_reinforce_scale=stabilization_mature_reinforce_scale,
            stabilization_active_reinforce_scale=stabilization_active_reinforce_scale,
            stabilization_young_age_threshold=stabilization_young_age_threshold,
            stabilization_young_active_bonus=stabilization_young_active_bonus,
            pressure_relief_min_age=pressure_relief_min_age,
            pressure_relief_utility_margin=pressure_relief_utility_margin,
            seed=seed,
        )
        self._health_params = HealthModulatedParams()
        self._consolidation_pending = False

    def set_health_status(
        self,
        health_score: float,
        prune_aggressiveness: float,
        consolidate_threshold: float,
        cluster_filter_enabled: bool,
        should_early_intervene: bool,
        trend_direction: str,
    ) -> None:
        """Update pool parameters based on TopoMem ECU HealthStatus."""
        self._health_params = modulate_by_health(
            health_score=health_score,
            prune_aggressiveness=prune_aggressiveness,
            cluster_filter_enabled=cluster_filter_enabled,
            should_early_intervene=should_early_intervene,
            trend_direction=trend_direction,
            current_step=self.step_count,
            base_params=HealthModulatedParams(
                surprise_threshold=self.surprise_threshold,
                tension_threshold=self.tension_threshold,
                utility_prune=self.utility_prune,
                phase_step_threshold=self.phase_step_threshold,
            ),
        )
        self.surprise_threshold = self._health_params.surprise_threshold
        self.utility_prune = self._health_params.utility_prune
        self.phase_step_threshold = self._health_params.phase_step_threshold
        self._consolidation_pending = should_early_intervene

    def get_effective_reinforce_amount(self) -> float:
        """Get health-adjusted reinforce amount."""
        return self.reinforce_amount * self._health_params.reinforce_scale

    def is_creation_suppressed(self) -> bool:
        """Check if new structure creation should be suppressed."""
        return self._health_params.suppress_creation

    def observe(self, x: np.ndarray, y: int = None, out_size: int = 2) -> Dict:
        """Observe input with health-adjusted parameters."""
        self.step_count += 1

        if not self.structures:
            structure = self._new_structure()
            self.structures.append(structure)
            self.total_creates += 1
            return self._build_result("create", structure, 1.0)

        surprises = [structure.current_surprise(x, y, out_size) for structure in self.structures]
        best_idx = int(np.argmin(surprises))
        best_surprise = float(surprises[best_idx])
        best_structure = self.structures[best_idx]

        best_structure.surprise_history.append(best_surprise)
        if len(best_structure.surprise_history) > 20:
            best_structure.surprise_history.pop(0)

        is_mid_late_phase = self.step_count > self.phase_step_threshold
        avg_tension = float(np.mean([s.tension for s in self.structures])) if self.structures else 0.0
        is_high_pressure = (
            best_surprise > self.surprise_threshold * self.surprise_pressure_ratio
            and avg_tension > self.tension_pressure_threshold
        )

        effective_reinforce = self.get_effective_reinforce_amount()
        low_threshold = self.surprise_threshold * 0.5

        if best_surprise < low_threshold:
            best_structure.reinforce(effective_reinforce)
            event = "reinforce"
            active = best_structure

        elif best_surprise < self.surprise_threshold:
            if self.is_creation_suppressed() and len(self.structures) >= self.max_structures:
                best_structure.reinforce(effective_reinforce * 0.5)
                event = "reinforce_suppressed"
                active = best_structure
            elif len(self.structures) < self.max_structures:
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
                if is_mid_late_phase and is_high_pressure:
                    mature_structures = [s for s in self.structures if s.age >= self.mature_age]
                    if mature_structures:
                        mature_structure = max(mature_structures, key=lambda s: s.age)
                        mature_structure.reinforce(effective_reinforce * 0.8)
                        active = mature_structure
                        event = "boundary_stabilize"
                    else:
                        best_structure.reinforce(effective_reinforce * 0.7)
                        active = best_structure
                        event = "reinforce"
                else:
                    best_structure.reinforce(effective_reinforce * 0.5)
                    event = "reinforce"
                    active = best_structure

        else:
            if self.is_creation_suppressed():
                mature_structures = [s for s in self.structures if s.age >= self.mature_age]
                if mature_structures:
                    mature_structure = max(mature_structures, key=lambda s: s.age)
                    mature_structure.reinforce(effective_reinforce * 0.6)
                    active = mature_structure
                    event = "boundary_stabilize"
                else:
                    best_structure.reinforce(effective_reinforce * 0.4)
                    active = best_structure
                    event = "reinforce"
            elif len(self.structures) < self.max_structures:
                active = self._new_structure(label=f"new_{self._next_id}")
                self.structures.append(active)
                self.total_creates += 1
                event = "create"
            else:
                if is_mid_late_phase and is_high_pressure:
                    mature_structures = [s for s in self.structures if s.age >= self.mature_age]
                    if mature_structures:
                        mature_structure = max(mature_structures, key=lambda s: s.age)
                        mature_structure.reinforce(effective_reinforce * 0.6)
                        active = mature_structure
                        event = "boundary_stabilize"
                    else:
                        best_structure.reinforce(effective_reinforce * 0.4)
                        active = best_structure
                        event = "reinforce"
                else:
                    best_structure.reinforce(effective_reinforce * 0.3)
                    event = "reinforce"
                    active = best_structure

        self._decay_all()
        if self.step_count % 50 == 0:
            self.prune()

        return self._build_result(event, active, best_surprise)

    def prune(self) -> int:
        """Prune with health-adjusted utility threshold."""
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

    def get_health_modulation_info(self) -> Dict:
        """Return current health modulation state for diagnostics."""
        return {
            "reinforce_scale": round(self._health_params.reinforce_scale, 3),
            "effective_utility_prune": round(self._health_params.utility_prune, 3),
            "suppress_creation": self._health_params.suppress_creation,
            "consolidation_pending": self._consolidation_pending,
            "phase_step_threshold": self.phase_step_threshold,
        }
