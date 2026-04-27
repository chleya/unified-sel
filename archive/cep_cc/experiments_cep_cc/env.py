from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EnvConfig:
    n_objects: int = 4
    object_dim: int = 5
    goal_dim: int = 4
    listener_object_dim: int = 2
    position_scale: float = 1.0
    velocity_scale: float = 0.25
    rule_mode: str = "scalar"
    nuisance_mode: str = "none"


@dataclass
class ObjectSelectionBatch:
    speaker_obs: torch.Tensor
    listener_obs: torch.Tensor
    target_index: torch.Tensor
    latent_factors: torch.Tensor
    object_states: torch.Tensor
    goal: torch.Tensor


class ContinuousObjectSelectionEnv:
    """Continuous object-selection task for emergent communication experiments."""

    def __init__(self, config: EnvConfig | None = None, device: str | torch.device = "cpu") -> None:
        self.config = config or EnvConfig()
        self.device = torch.device(device)

    @property
    def speaker_obs_dim(self) -> int:
        return self.config.n_objects * self.config.object_dim + self.config.goal_dim

    @property
    def listener_obs_dim(self) -> int:
        return self.config.n_objects * self.config.listener_object_dim + 2

    def sample_batch(self, batch_size: int, seed: int | None = None) -> ObjectSelectionBatch:
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)

        cfg = self.config
        positions = (
            torch.rand(batch_size, cfg.n_objects, 2, generator=generator, device=self.device) * 2.0 - 1.0
        ) * cfg.position_scale
        velocities = (
            torch.rand(batch_size, cfg.n_objects, 2, generator=generator, device=self.device) * 2.0 - 1.0
        ) * cfg.velocity_scale
        q = torch.rand(batch_size, cfg.n_objects, 1, generator=generator, device=self.device) * 2.0 - 1.0
        goal = torch.rand(batch_size, cfg.goal_dim, generator=generator, device=self.device) * 2.0 - 1.0
        positions, velocities, goal = self._apply_nuisance(positions, velocities, goal)

        object_states = torch.cat([positions, velocities, q], dim=-1)
        scores, latent_factors = self._score_objects(object_states, goal)
        target_index = scores.argmax(dim=1)

        speaker_obs = torch.cat([object_states.flatten(start_dim=1), goal], dim=1)
        # Listener sees positions and the positional part of the goal only.
        listener_obs = torch.cat([positions.flatten(start_dim=1), goal[:, :2]], dim=1)
        return ObjectSelectionBatch(
            speaker_obs=speaker_obs,
            listener_obs=listener_obs,
            target_index=target_index,
            latent_factors=latent_factors,
            object_states=object_states,
            goal=goal,
        )

    def _apply_nuisance(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        goal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mode = self.config.nuisance_mode
        if mode == "none":
            return positions, velocities, goal
        if mode == "mirror_x":
            positions = positions.clone()
            velocities = velocities.clone()
            goal = goal.clone()
            positions[:, :, 0] = -positions[:, :, 0]
            velocities[:, :, 0] = -velocities[:, :, 0]
            goal[:, 0] = -goal[:, 0]
            return positions, velocities, goal
        if mode == "rotate90":
            positions = torch.stack([-positions[:, :, 1], positions[:, :, 0]], dim=-1)
            velocities = torch.stack([-velocities[:, :, 1], velocities[:, :, 0]], dim=-1)
            goal = goal.clone()
            goal_xy = goal[:, 0:2]
            goal[:, 0:2] = torch.stack([-goal_xy[:, 1], goal_xy[:, 0]], dim=-1)
            return positions, velocities, goal
        if mode == "velocity_scale":
            return positions, velocities * 1.5, goal
        raise ValueError(f"unknown nuisance_mode: {mode}")

    def _score_objects(self, object_states: torch.Tensor, goal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        positions = object_states[:, :, 0:2]
        velocities = object_states[:, :, 2:4]
        q = object_states[:, :, 4]

        goal_pos = torch.nn.functional.normalize(goal[:, 0:2], dim=-1)
        goal_q = goal[:, 2].unsqueeze(1)
        goal_motion = goal[:, 3].unsqueeze(1)

        pos_score = (positions * goal_pos.unsqueeze(1)).sum(dim=-1)
        q_score = q * goal_q

        relative = positions.unsqueeze(2) - positions.unsqueeze(1)
        distances = torch.linalg.norm(relative, dim=-1)
        masked = distances + torch.eye(positions.shape[1], device=positions.device).unsqueeze(0) * 100.0
        nearest = masked.argmin(dim=-1)
        nearest_velocity = velocities.gather(1, nearest.unsqueeze(-1).expand(-1, -1, 2))
        velocity_alignment = (velocities * nearest_velocity).sum(dim=-1) * goal_motion

        anchor = positions.mean(dim=1, keepdim=True)
        if self.config.rule_mode in {
            "compositional",
            "temporal_memory",
            "motion_pressure",
            "motion_decoupled",
        }:
            relation_sign = torch.sign(goal[:, 0]).unsqueeze(1)
        elif self.config.rule_mode == "scalar":
            relation_sign = torch.sign(goal_motion)
        else:
            raise ValueError(f"unknown rule_mode: {self.config.rule_mode}")
        relation_to_anchor = -torch.linalg.norm(positions - anchor, dim=-1) * relation_sign

        if self.config.rule_mode == "motion_decoupled":
            motion_axis = torch.nn.functional.normalize(
                torch.stack([goal[:, 3], torch.ones_like(goal[:, 3])], dim=-1),
                dim=-1,
            )
            velocity_alignment = (velocities * motion_axis.unsqueeze(1)).sum(dim=-1)

        latent_factors = torch.stack([pos_score, q_score, velocity_alignment, relation_to_anchor], dim=-1)
        if self.config.rule_mode in {
            "compositional",
            "temporal_memory",
            "motion_pressure",
            "motion_decoupled",
        }:
            components = latent_factors[:, :, 1:4]
            centered = components - components.mean(dim=1, keepdim=True)
            normalized = centered / components.std(dim=1, keepdim=True).clamp_min(1e-4)
            q_component = normalized[:, :, 0]
            motion_component = normalized[:, :, 1]
            relation_component = normalized[:, :, 2]
            if self.config.rule_mode == "motion_decoupled":
                scores = (
                    0.45 * q_component
                    + 0.80 * motion_component
                    + 0.30 * relation_component
                    + 0.55 * q_component * motion_component
                    + 0.25 * motion_component * relation_component
                    + 0.03 * pos_score
                )
            elif self.config.rule_mode == "motion_pressure":
                late_plan = 0.75 * motion_component + 0.25 * relation_component
                scores = (
                    0.55 * q_component
                    + 0.70 * motion_component
                    + 0.20 * relation_component
                    + 0.65 * q_component * motion_component
                    + 0.45 * motion_component * relation_component
                    + 0.03 * pos_score
                )
            elif self.config.rule_mode == "temporal_memory":
                late_plan = 0.55 * motion_component + 0.45 * relation_component
                scores = (
                    0.70 * q_component
                    + 0.35 * late_plan
                    + 0.55 * q_component * late_plan
                    + 0.20 * motion_component * relation_component
                    + 0.03 * pos_score
                )
            else:
                scores = (
                    0.45 * q_component
                    + 0.35 * motion_component
                    + 0.35 * relation_component
                    + 0.30 * q_component * motion_component
                    + 0.20 * motion_component * relation_component
                    + 0.05 * pos_score
                )
        else:
            scores = 0.25 * pos_score + 1.15 * q_score + 0.90 * velocity_alignment + 0.15 * relation_to_anchor
        return scores, latent_factors
