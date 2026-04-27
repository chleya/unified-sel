"""
Structured experiment configuration for Unified-SEL.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class PoolConfig:
    surprise_threshold: float = 0.60
    tension_threshold: float = 0.08
    utility_decay: float = 0.005
    utility_prune: float = 0.08
    reinforce_amount: float = 0.05
    clone_perturbation: float = 0.05
    mature_age: int = 80
    mature_decay_scale: float = 0.35
    max_structures: int = 12
    phase_step_threshold: int = 200
    surprise_pressure_ratio: float = 0.7
    tension_pressure_threshold: float = 0.3
    stabilization_pressure_streak: int = 3
    stabilization_cooldown_steps: int = 8
    stabilization_late_offset_steps: int = 200
    stabilization_late_window_steps: int = 50
    stabilization_late_max_per_window: int = 4
    stabilization_near_full_gap: int = 1
    stabilization_mature_surprise_gap: float = 0.12
    stabilization_mature_reinforce_scale: float = 0.35
    stabilization_active_reinforce_scale: float = 0.45
    stabilization_young_age_threshold: int = 40
    stabilization_young_active_bonus: float = 0.15
    pressure_relief_min_age: int = 25
    pressure_relief_utility_margin: float = 0.0

    def to_pool_kwargs(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NoBoundaryConfig:
    steps: int = 600
    checkpoint_step: int = 200
    eval_samples_per_task: int = 256
    seeds: List[int] = field(default_factory=lambda: [7, 8, 9, 10, 11])
    in_size: int = 4
    out_size: int = 2
    lr: float = 0.05
    evolve_every: int = 20
    readout_mode: str = "shared"
    shared_readout_scale: float = 1.0
    shared_readout_post_checkpoint_scale: float = 1.0
    local_readout_lr_scale: float = 1.0
    local_readout_start_step: int = 0
    local_readout_surprise_threshold: float | None = None
    local_readout_young_age_max: int | None = None
    local_readout_training_events: List[str] | None = None
    local_readout_inference_surprise_threshold: float | None = None
    local_readout_episode_events: List[str] | None = None
    local_readout_episode_window_steps: int = 0
    local_readout_pressure_window_steps: int = 0
    pool: PoolConfig = field(default_factory=PoolConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CompareConfig:
    fixed_result_path: str | None = None
    ewc_result_path: str | None = None
    unified_result_path: str | None = None
    baseline_label_fixed: str = "fixed"
    baseline_label_ewc: str = "ewc"
    unified_label: str = "unified_sel"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
