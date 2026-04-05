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
