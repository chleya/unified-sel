from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    speaker_obs_dim: int
    listener_obs_dim: int
    n_objects: int = 4
    comm_steps: int = 8
    comm_dim: int = 6
    hidden_dim: int = 64
    state_dim: int = 32
    object_comm_dim: int = 8


class Speaker(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Linear(config.speaker_obs_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.state_dim),
            nn.Tanh(),
        )
        self.comm_head = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.comm_steps * config.comm_dim),
        )

    def forward(self, speaker_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.encoder(speaker_obs)
        comm = torch.tanh(self.comm_head(state)).view(
            speaker_obs.shape[0],
            self.config.comm_steps,
            self.config.comm_dim,
        )
        return comm, state


class Listener(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.global_obs_encoder = nn.Sequential(
            nn.Linear(2, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.state_dim),
            nn.Tanh(),
        )
        self.comm_gru = nn.GRU(config.comm_dim, config.state_dim, batch_first=True)
        self.object_comm_decoder = nn.Linear(config.state_dim, config.n_objects * config.object_comm_dim)
        self.object_head = nn.Sequential(
            nn.Linear(config.state_dim * 2 + config.object_comm_dim + 4, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, listener_obs: torch.Tensor, comm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        positions = listener_obs[:, : self.config.n_objects * 2].view(-1, self.config.n_objects, 2)
        goal_pos = listener_obs[:, self.config.n_objects * 2 : self.config.n_objects * 2 + 2]
        obs_state = self.global_obs_encoder(goal_pos)
        _, hidden = self.comm_gru(comm)
        comm_state = hidden[-1]
        state = torch.cat([obs_state, comm_state], dim=-1)
        repeated_state = state.unsqueeze(1).expand(-1, self.config.n_objects, -1)
        repeated_goal = goal_pos.unsqueeze(1).expand(-1, self.config.n_objects, -1)
        object_comm = self.object_comm_decoder(comm_state).view(
            -1,
            self.config.n_objects,
            self.config.object_comm_dim,
        )
        object_features = torch.cat([positions, repeated_goal, repeated_state, object_comm], dim=-1)
        logits = self.object_head(object_features).squeeze(-1)
        return logits, state


class RandomProjectionSpeaker(nn.Module):
    def __init__(self, config: ModelConfig, seed: int = 0) -> None:
        super().__init__()
        generator = torch.Generator()
        generator.manual_seed(seed)
        weight = torch.randn(
            config.speaker_obs_dim,
            config.comm_steps * config.comm_dim,
            generator=generator,
        ) / (config.speaker_obs_dim ** 0.5)
        self.register_buffer("weight", weight)
        self.config = config

    def forward(self, speaker_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat = torch.tanh(speaker_obs @ self.weight)
        comm = flat.view(speaker_obs.shape[0], self.config.comm_steps, self.config.comm_dim)
        state = flat[:, : self.config.state_dim] if flat.shape[1] >= self.config.state_dim else flat
        return comm, state


class TeacherSignalSpeaker(nn.Module):
    """Continuous positive-control speaker derived from hidden continuous factors."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, speaker_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        objects_end = self.config.n_objects * 5
        object_states = speaker_obs[:, :objects_end].view(-1, self.config.n_objects, 5)
        goal = speaker_obs[:, objects_end : objects_end + 4]
        scores = self._score_objects(object_states, goal)
        centered = scores - scores.mean(dim=1, keepdim=True)
        normalized = centered / centered.std(dim=1, keepdim=True).clamp_min(1e-4)

        comm = torch.zeros(
            speaker_obs.shape[0],
            self.config.comm_steps,
            self.config.comm_dim,
            device=speaker_obs.device,
        )
        comm[:, :, : self.config.n_objects] = normalized.unsqueeze(1)
        comm[:, :, self.config.n_objects] = goal[:, 2].unsqueeze(1)
        comm[:, :, self.config.n_objects + 1] = goal[:, 3].unsqueeze(1)
        comm = torch.tanh(comm)

        flat = comm.flatten(start_dim=1)
        if flat.shape[1] >= self.config.state_dim:
            state = flat[:, : self.config.state_dim]
        else:
            pad = torch.zeros(flat.shape[0], self.config.state_dim - flat.shape[1], device=flat.device)
            state = torch.cat([flat, pad], dim=1)
        return comm, state

    def _score_objects(self, object_states: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
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
        relation_to_anchor = -torch.linalg.norm(positions - anchor, dim=-1) * torch.sign(goal_motion)
        return 0.25 * pos_score + 1.15 * q_score + 0.90 * velocity_alignment + 0.15 * relation_to_anchor


def teacher_signal_from_speaker_obs(config: ModelConfig, speaker_obs: torch.Tensor) -> torch.Tensor:
    teacher = TeacherSignalSpeaker(config).to(speaker_obs.device)
    comm, _ = teacher(speaker_obs)
    return comm


class SpeakerListenerSystem(nn.Module):
    def __init__(self, config: ModelConfig, baseline: str = "learned", seed: int = 0) -> None:
        super().__init__()
        self.config = config
        self.baseline = baseline
        if baseline == "random-projection":
            self.speaker: nn.Module = RandomProjectionSpeaker(config, seed=seed)
        elif baseline == "teacher-signal":
            self.speaker = TeacherSignalSpeaker(config)
        else:
            self.speaker = Speaker(config)
        self.listener = Listener(config)

    def forward(self, speaker_obs: torch.Tensor, listener_obs: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.baseline == "no-communication":
            batch = speaker_obs.shape[0]
            comm = torch.zeros(batch, self.config.comm_steps, self.config.comm_dim, device=speaker_obs.device)
            speaker_state = torch.zeros(batch, self.config.state_dim, device=speaker_obs.device)
        else:
            comm, speaker_state = self.speaker(speaker_obs)
        logits, listener_state = self.listener(listener_obs, comm)
        return {
            "logits": logits,
            "comm": comm,
            "speaker_state": speaker_state,
            "listener_state": listener_state,
        }
