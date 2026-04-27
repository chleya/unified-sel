from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass

import torch

from .env import ContinuousObjectSelectionEnv, EnvConfig
from .losses import total_loss
from .metrics import (
    accuracy,
    format_table,
    geometry_audit_metrics,
    paired_nuisance_alignment_metrics,
    segment_audit_metrics,
    summarize_batch,
)
from .models import ModelConfig, SpeakerListenerSystem, teacher_signal_from_speaker_obs


@dataclass(frozen=True)
class TrainConfig:
    episodes: int = 200
    batch_size: int = 64
    lr: float = 2e-3
    lambda_energy: float = 0.0
    lambda_sparse: float = 0.0
    lambda_smooth: float = 0.001
    lambda_dimeff: float = 0.0
    lambda_consistency: float = 0.0
    lambda_factor_consistency: float = 0.0
    factor_name: str = "q"
    lambda_state: float = 0.0
    lambda_comm_distill: float = 0.0
    seed: int = 0
    baseline: str = "learned"
    eval_batches: int = 4
    rule_mode: str = "scalar"
    segment_dropout_prob: float = 0.0


def _make_env(rule_mode: str = "scalar", nuisance_mode: str = "none") -> ContinuousObjectSelectionEnv:
    return ContinuousObjectSelectionEnv(EnvConfig(rule_mode=rule_mode, nuisance_mode=nuisance_mode))


def _build_system(env: ContinuousObjectSelectionEnv, seed: int, baseline: str) -> SpeakerListenerSystem:
    torch.manual_seed(seed)
    model_config = ModelConfig(
        speaker_obs_dim=env.speaker_obs_dim,
        listener_obs_dim=env.listener_obs_dim,
        n_objects=env.config.n_objects,
    )
    return SpeakerListenerSystem(model_config, baseline=baseline, seed=seed)


def _apply_segment_dropout(comm: torch.Tensor, probability: float, seed: int) -> torch.Tensor:
    if probability <= 0.0 or comm.shape[1] < 3:
        return comm
    generator = torch.Generator(device=comm.device)
    generator.manual_seed(seed)
    if float(torch.rand((), generator=generator, device=comm.device).detach().cpu()) >= probability:
        return comm
    chunks = list(torch.chunk(comm, chunks=3, dim=1))
    segment_idx = int(torch.randint(0, len(chunks), (), generator=generator, device=comm.device).detach().cpu())
    chunks[segment_idx] = torch.zeros_like(chunks[segment_idx])
    return torch.cat(chunks, dim=1)


def train_once(config: TrainConfig) -> tuple[SpeakerListenerSystem, dict[str, float], dict[str, float]]:
    env = _make_env(config.rule_mode)
    system = _build_system(env, seed=config.seed, baseline=config.baseline)
    losses, train_metrics = train_existing(system, config, start_step=0)
    return system, losses, train_metrics


def train_existing(
    system: SpeakerListenerSystem,
    config: TrainConfig,
    start_step: int = 0,
    train_speaker: bool = True,
    train_listener: bool = True,
) -> tuple[dict[str, float], dict[str, float]]:
    env = _make_env(config.rule_mode)
    original_requires_grad = {name: param.requires_grad for name, param in system.named_parameters()}
    for name, param in system.named_parameters():
        if name.startswith("speaker."):
            param.requires_grad = train_speaker
        elif name.startswith("listener."):
            param.requires_grad = train_listener
    parameters = [param for param in system.parameters() if param.requires_grad]
    if not parameters:
        raise ValueError("train_existing requires at least one trainable parameter")
    optimizer = torch.optim.Adam(parameters, lr=config.lr)
    last_losses: dict[str, float] = {}
    train_metrics: dict[str, float] = {}
    for step in range(config.episodes):
        batch = env.sample_batch(config.batch_size, seed=config.seed * 100_000 + start_step + step)
        out = system(batch.speaker_obs, batch.listener_obs)
        logits_for_loss = out["logits"]
        listener_state_for_loss = out["listener_state"]
        dropped_comm = _apply_segment_dropout(
            out["comm"],
            probability=config.segment_dropout_prob,
            seed=config.seed * 1_000_000 + start_step + step,
        )
        if dropped_comm is not out["comm"]:
            logits_for_loss, listener_state_for_loss = system.listener(batch.listener_obs, dropped_comm)
        target_comm = None
        if config.lambda_comm_distill > 0.0 and config.baseline == "learned":
            target_comm = teacher_signal_from_speaker_obs(system.config, batch.speaker_obs)
        factor_values = None
        if config.lambda_factor_consistency > 0.0:
            factor_values = _target_factor_values(batch.latent_factors, batch.object_states, batch.target_index, config.factor_name)
        loss, parts = total_loss(
            logits_for_loss,
            batch.target_index,
            out["comm"],
            out["speaker_state"],
            listener_state_for_loss,
            lambda_energy=config.lambda_energy,
            lambda_sparse=config.lambda_sparse,
            lambda_smooth=config.lambda_smooth,
            lambda_dimeff=config.lambda_dimeff,
            lambda_consistency=config.lambda_consistency,
            lambda_state=config.lambda_state,
            target_comm=target_comm,
            lambda_comm_distill=config.lambda_comm_distill,
            factor_values=factor_values,
            lambda_factor_consistency=config.lambda_factor_consistency,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last_losses = {"loss": float(loss.detach().cpu()), **parts}
        if step == config.episodes - 1:
            train_metrics = {
                "train_task_accuracy": accuracy(logits_for_loss, batch.target_index),
            }
            if config.segment_dropout_prob > 0.0:
                train_metrics["train_segment_dropout_prob"] = config.segment_dropout_prob
    for name, param in system.named_parameters():
        param.requires_grad = original_requires_grad[name]
    return last_losses, train_metrics


@torch.no_grad()
def evaluate(
    system: SpeakerListenerSystem,
    seed: int = 1000,
    batches: int = 4,
    batch_size: int = 128,
    rule_mode: str = "scalar",
    nuisance_mode: str = "none",
) -> dict[str, float]:
    env = _make_env(rule_mode, nuisance_mode=nuisance_mode)
    logits_list = []
    target_list = []
    comm_list = []
    for idx in range(batches):
        batch = env.sample_batch(batch_size, seed=seed + idx)
        out = system(batch.speaker_obs, batch.listener_obs)
        logits_list.append(out["logits"])
        target_list.append(batch.target_index)
        comm_list.append(out["comm"])
    logits = torch.cat(logits_list, dim=0)
    target = torch.cat(target_list, dim=0)
    comm = torch.cat(comm_list, dim=0)
    return summarize_batch(logits, target, comm)


def run_train_eval(
    episodes: int = 200,
    batch_size: int = 64,
    seed: int = 0,
    lambda_comm: float = 0.0,
    baseline: str = "learned",
    lambda_comm_distill: float = 0.0,
    lr: float = 2e-3,
    rule_mode: str = "scalar",
) -> dict[str, float]:
    train_config = TrainConfig(
        episodes=episodes,
        batch_size=batch_size,
        seed=seed,
        lr=lr,
        baseline=baseline,
        lambda_energy=lambda_comm,
        lambda_sparse=lambda_comm,
        lambda_dimeff=lambda_comm * 0.01,
        lambda_comm_distill=lambda_comm_distill,
        rule_mode=rule_mode,
    )
    system, losses, train_metrics = train_once(train_config)
    metrics = evaluate(
        system,
        seed=seed + 10_000,
        batches=train_config.eval_batches,
        batch_size=batch_size,
        rule_mode=rule_mode,
    )
    metrics.update(train_metrics)
    metrics.update({f"train_{name}": value for name, value in losses.items()})
    return metrics


def run_lambda_sweep(
    lambdas: list[float],
    episodes: int = 200,
    batch_size: int = 64,
    seed: int = 0,
    lr: float = 2e-3,
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    results["no_communication"] = run_train_eval(
        episodes=episodes,
        batch_size=batch_size,
        seed=seed,
        lr=lr,
        lambda_comm=0.0,
        baseline="no-communication",
    )
    for value in lambdas:
        results[f"lambda_{value:g}"] = run_train_eval(
            episodes=episodes,
            batch_size=batch_size,
            seed=seed,
            lr=lr,
            lambda_comm=value,
            baseline="learned",
        )
    return results


def run_multiseed_compression_sweep(
    seeds: list[int],
    lambdas: list[float],
    episodes: int = 300,
    batch_size: int = 128,
    lr: float = 0.005,
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for seed in seeds:
        per_seed = run_lambda_sweep(
            lambdas=lambdas,
            episodes=episodes,
            batch_size=batch_size,
            seed=seed,
            lr=lr,
        )
        for name, metrics in per_seed.items():
            results[f"seed_{seed}:{name}"] = metrics
    return results


def _with_prefixed_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _target_factor_values(
    latent_factors: torch.Tensor,
    object_states: torch.Tensor,
    target_index: torch.Tensor,
    factor_name: str,
) -> torch.Tensor:
    factor_names = [item.strip() for item in factor_name.replace(",", "+").split("+") if item.strip()]
    if len(factor_names) > 1:
        return torch.stack(
            [
                _target_factor_values(latent_factors, object_states, target_index, item)
                for item in factor_names
            ],
            dim=1,
        )
    if len(factor_names) == 1:
        factor_name = factor_names[0]
    batch_indices = torch.arange(target_index.shape[0], device=target_index.device)
    if factor_name == "q":
        return object_states[batch_indices, target_index, 4]
    factor_to_idx = {
        "pos": 0,
        "hidden_q_score": 1,
        "motion": 2,
        "relation": 3,
    }
    if factor_name not in factor_to_idx:
        raise ValueError(f"unknown factor_name: {factor_name}")
    return latent_factors[batch_indices, target_index, factor_to_idx[factor_name]]


def run_two_stage_compression(
    seeds: list[int],
    lambdas: list[float],
    stage1_episodes: int = 300,
    stage2_episodes: int = 120,
    batch_size: int = 128,
    lr: float = 0.005,
    stage2_lr: float | None = None,
    freeze_listener_stage2: bool = False,
) -> dict[str, dict[str, float]]:
    env = ContinuousObjectSelectionEnv(EnvConfig())
    results: dict[str, dict[str, float]] = {}
    for seed in seeds:
        stage1_config = TrainConfig(
            episodes=stage1_episodes,
            batch_size=batch_size,
            seed=seed,
            lr=lr,
            baseline="learned",
        )
        base_system = _build_system(env, seed=seed, baseline="learned")
        stage1_losses, stage1_train = train_existing(base_system, stage1_config, start_step=0)
        base_metrics = evaluate(base_system, seed=seed + 10_000, batches=stage1_config.eval_batches, batch_size=batch_size)
        base_metrics.update(stage1_train)
        base_metrics.update({f"train_{name}": value for name, value in stage1_losses.items()})
        results[f"seed_{seed}:stage1_high_bandwidth"] = base_metrics

        for value in lambdas:
            tuned = copy.deepcopy(base_system)
            stage2_config = TrainConfig(
                episodes=stage2_episodes,
                batch_size=batch_size,
                seed=seed,
                lr=stage2_lr if stage2_lr is not None else lr,
                baseline="learned",
                lambda_energy=value,
                lambda_sparse=value,
                lambda_dimeff=value * 0.01,
            )
            losses, train_metrics = train_existing(
                tuned,
                stage2_config,
                start_step=stage1_episodes,
                train_speaker=True,
                train_listener=not freeze_listener_stage2,
            )
            metrics = evaluate(tuned, seed=seed + 10_000, batches=stage2_config.eval_batches, batch_size=batch_size)
            metrics.update(train_metrics)
            metrics.update({f"train_{name}": value_loss for name, value_loss in losses.items()})
            metrics.update(_with_prefixed_metrics("stage1", base_metrics))
            metrics["delta_task_accuracy"] = metrics["task_accuracy"] - base_metrics["task_accuracy"]
            metrics["delta_comm_energy"] = metrics["comm_energy"] - base_metrics["comm_energy"]
            metrics["delta_comm_l1"] = metrics["comm_l1"] - base_metrics["comm_l1"]
            metrics["delta_comm_effective_dim"] = metrics["comm_effective_dim"] - base_metrics["comm_effective_dim"]
            metrics["delta_target_cluster_alignment"] = (
                metrics["target_cluster_alignment"] - base_metrics["target_cluster_alignment"]
            )
            results[f"seed_{seed}:stage2_lambda_{value:g}"] = metrics
    return results


def run_c1f_ablation(
    seeds: list[int],
    lambdas: list[float],
    stage1_episodes: int = 300,
    stage2_episodes: int = 120,
    batch_size: int = 128,
    lr: float = 0.005,
    stage2_lr: float = 0.001,
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for mode_name, freeze_listener in [
        ("joint", False),
        ("freeze_listener", True),
    ]:
        per_mode = run_two_stage_compression(
            seeds=seeds,
            lambdas=lambdas,
            stage1_episodes=stage1_episodes,
            stage2_episodes=stage2_episodes,
            batch_size=batch_size,
            lr=lr,
            stage2_lr=stage2_lr,
            freeze_listener_stage2=freeze_listener,
        )
        for name, metrics in per_mode.items():
            results[f"{mode_name}:{name}"] = metrics
    return results


@torch.no_grad()
def _collect_audit_batch(
    system: SpeakerListenerSystem,
    seed: int,
    batches: int,
    batch_size: int,
    rule_mode: str = "scalar",
    nuisance_mode: str = "none",
) -> dict[str, torch.Tensor]:
    env = _make_env(rule_mode, nuisance_mode=nuisance_mode)
    logits_list = []
    target_list = []
    comm_list = []
    latent_list = []
    object_list = []
    listener_obs_list = []
    for idx in range(batches):
        batch = env.sample_batch(batch_size, seed=seed + idx)
        out = system(batch.speaker_obs, batch.listener_obs)
        logits_list.append(out["logits"])
        target_list.append(batch.target_index)
        comm_list.append(out["comm"])
        latent_list.append(batch.latent_factors)
        object_list.append(batch.object_states)
        listener_obs_list.append(batch.listener_obs)
    return {
        "logits": torch.cat(logits_list, dim=0),
        "target": torch.cat(target_list, dim=0),
        "comm": torch.cat(comm_list, dim=0),
        "latent": torch.cat(latent_list, dim=0),
        "objects": torch.cat(object_list, dim=0),
        "listener_obs": torch.cat(listener_obs_list, dim=0),
    }


def _segment_swap_metrics(system: SpeakerListenerSystem, collected: dict[str, torch.Tensor]) -> dict[str, float]:
    comm = collected["comm"]
    listener_obs = collected["listener_obs"]
    target = collected["target"]
    base_logits = collected["logits"]
    base_pred = base_logits.argmax(dim=1)
    batch_indices = torch.arange(target.shape[0], device=target.device)
    base_target_logit = base_logits[batch_indices, target]
    metrics: dict[str, float] = {}
    chunks = list(torch.chunk(comm, chunks=3, dim=1))
    names = ["early", "middle", "late"]
    for idx, name in enumerate(names):
        swapped_chunks = [chunk.clone() for chunk in chunks]
        swapped_chunks[idx] = swapped_chunks[idx].roll(shifts=1, dims=0)
        swapped_comm = torch.cat(swapped_chunks, dim=1)
        swapped_logits, _ = system.listener(listener_obs, swapped_comm)
        swapped_pred = swapped_logits.argmax(dim=1)
        swapped_target_logit = swapped_logits[batch_indices, target]
        metrics[f"segment_{name}_swap_action_change_rate"] = float(
            (swapped_pred != base_pred).float().mean().detach().cpu()
        )
        metrics[f"segment_{name}_swap_target_logit_drop"] = float(
            (base_target_logit - swapped_target_logit).mean().detach().cpu()
        )
        ablated_chunks = [chunk.clone() for chunk in chunks]
        ablated_chunks[idx] = torch.zeros_like(ablated_chunks[idx])
        ablated_comm = torch.cat(ablated_chunks, dim=1)
        ablated_logits, _ = system.listener(listener_obs, ablated_comm)
        ablated_pred = ablated_logits.argmax(dim=1)
        ablated_target_logit = ablated_logits[batch_indices, target]
        metrics[f"segment_{name}_ablation_action_change_rate"] = float(
            (ablated_pred != base_pred).float().mean().detach().cpu()
        )
        metrics[f"segment_{name}_ablation_target_logit_drop"] = float(
            (base_target_logit - ablated_target_logit).mean().detach().cpu()
        )
    return metrics


def _targeted_segment_intervention_metrics(
    system: SpeakerListenerSystem,
    collected: dict[str, torch.Tensor],
) -> dict[str, float]:
    comm = collected["comm"]
    listener_obs = collected["listener_obs"]
    latent = collected["latent"]
    if comm.shape[1] < 3 or comm.shape[0] < 4:
        return {}
    names = ["early", "middle", "late"]
    factor_indices = {"q": 1, "motion": 2, "relation": 3}
    chunks = list(torch.chunk(comm, chunks=3, dim=1))
    batch_indices = torch.arange(comm.shape[0], device=comm.device)
    metrics: dict[str, float] = {}
    best_ratio = -1.0
    best_targeted = 0.0
    best_offtarget = 0.0

    for factor_name, factor_idx in factor_indices.items():
        factor_values = latent[batch_indices, collected["target"], factor_idx]
        sorted_indices = factor_values.argsort()
        donor = sorted_indices.flip(0)
        factor_best_ratio = -1.0
        factor_best_segment = 0
        for segment_idx, segment_name in enumerate(names):
            swapped_chunks = [chunk.clone() for chunk in chunks]
            swapped_chunks[segment_idx] = chunks[segment_idx][donor]
            swapped_comm = torch.cat(swapped_chunks, dim=1)
            swapped_logits, _ = system.listener(listener_obs, swapped_comm)
            swapped_target = swapped_logits.argmax(dim=1)
            swapped_latents = latent[batch_indices, swapped_target]
            base_latents = latent[batch_indices, collected["target"]]
            deltas = (swapped_latents - base_latents).abs().mean(dim=0)
            targeted = float(deltas[factor_idx].detach().cpu())
            off_indices = [idx for idx in factor_indices.values() if idx != factor_idx]
            off_target = float(deltas[off_indices].mean().detach().cpu())
            ratio = targeted / max(off_target, 1e-8)
            metrics[f"intervention_{factor_name}_{segment_name}_targeted_effect"] = targeted
            metrics[f"intervention_{factor_name}_{segment_name}_offtarget_effect"] = off_target
            metrics[f"intervention_{factor_name}_{segment_name}_ratio"] = ratio
            if ratio > factor_best_ratio:
                factor_best_ratio = ratio
                factor_best_segment = segment_idx
            if ratio > best_ratio:
                best_ratio = ratio
                best_targeted = targeted
                best_offtarget = off_target
        metrics[f"intervention_{factor_name}_best_ratio"] = factor_best_ratio
        metrics[f"intervention_{factor_name}_best_segment"] = float(factor_best_segment)

    metrics["intervention_best_targeted_effect"] = best_targeted
    metrics["intervention_best_offtarget_effect"] = best_offtarget
    metrics["intervention_best_targeted_ratio"] = best_ratio
    return metrics


def _summarize_audit(
    system: SpeakerListenerSystem,
    seed: int,
    batches: int,
    batch_size: int,
    rule_mode: str = "scalar",
    nuisance_mode: str = "none",
    include_segment_swap: bool = False,
) -> dict[str, float]:
    collected = _collect_audit_batch(
        system,
        seed=seed,
        batches=batches,
        batch_size=batch_size,
        rule_mode=rule_mode,
        nuisance_mode=nuisance_mode,
    )
    metrics = summarize_batch(collected["logits"], collected["target"], collected["comm"])
    metrics.update(
        geometry_audit_metrics(
            collected["comm"],
            collected["target"],
            collected["latent"],
            collected["objects"],
        )
    )
    metrics.update(
        segment_audit_metrics(
            collected["comm"],
            collected["target"],
            collected["latent"],
            collected["objects"],
        )
    )
    if include_segment_swap:
        metrics.update(_segment_swap_metrics(system, collected))
    return metrics


def _summarize_collected_audit(
    system: SpeakerListenerSystem,
    collected: dict[str, torch.Tensor],
    include_segment_swap: bool = False,
    include_targeted_intervention: bool = False,
) -> dict[str, float]:
    metrics = summarize_batch(collected["logits"], collected["target"], collected["comm"])
    metrics.update(
        geometry_audit_metrics(
            collected["comm"],
            collected["target"],
            collected["latent"],
            collected["objects"],
        )
    )
    metrics.update(
        segment_audit_metrics(
            collected["comm"],
            collected["target"],
            collected["latent"],
            collected["objects"],
        )
    )
    if include_segment_swap:
        metrics.update(_segment_swap_metrics(system, collected))
    if include_targeted_intervention:
        metrics.update(_targeted_segment_intervention_metrics(system, collected))
    return metrics


def run_c1g_geometry_audit(
    seeds: list[int],
    stage1_episodes: int = 300,
    stage2_episodes: int = 120,
    batch_size: int = 128,
    lr: float = 0.005,
    stage2_lr: float = 0.001,
    lambda_comm: float = 0.10,
    audit_batches: int = 8,
) -> dict[str, dict[str, float]]:
    env = ContinuousObjectSelectionEnv(EnvConfig())
    results: dict[str, dict[str, float]] = {}
    for seed in seeds:
        base_system = _build_system(env, seed=seed, baseline="learned")
        stage1_config = TrainConfig(
            episodes=stage1_episodes,
            batch_size=batch_size,
            seed=seed,
            lr=lr,
            baseline="learned",
        )
        train_existing(base_system, stage1_config, start_step=0)
        base_metrics = _summarize_audit(
            base_system,
            seed=seed + 20_000,
            batches=audit_batches,
            batch_size=batch_size,
        )
        results[f"seed_{seed}:stage1_high_bandwidth"] = base_metrics

        compressed = copy.deepcopy(base_system)
        stage2_config = TrainConfig(
            episodes=stage2_episodes,
            batch_size=batch_size,
            seed=seed,
            lr=stage2_lr,
            baseline="learned",
            lambda_energy=lambda_comm,
            lambda_sparse=lambda_comm,
            lambda_dimeff=lambda_comm * 0.01,
        )
        train_existing(
            compressed,
            stage2_config,
            start_step=stage1_episodes,
            train_speaker=True,
            train_listener=False,
        )
        compressed_metrics = _summarize_audit(
            compressed,
            seed=seed + 20_000,
            batches=audit_batches,
            batch_size=batch_size,
        )
        compressed_metrics.update(_with_prefixed_metrics("stage1", base_metrics))
        compressed_metrics["delta_task_accuracy"] = (
            compressed_metrics["task_accuracy"] - base_metrics["task_accuracy"]
        )
        compressed_metrics["delta_comm_energy"] = compressed_metrics["comm_energy"] - base_metrics["comm_energy"]
        compressed_metrics["delta_comm_l1"] = compressed_metrics["comm_l1"] - base_metrics["comm_l1"]
        compressed_metrics["delta_comm_effective_dim"] = (
            compressed_metrics["comm_effective_dim"] - base_metrics["comm_effective_dim"]
        )
        compressed_metrics["delta_target_cluster_alignment"] = (
            compressed_metrics["target_cluster_alignment"] - base_metrics["target_cluster_alignment"]
        )
        results[f"seed_{seed}:compressed_freeze_listener_lambda_{lambda_comm:g}"] = compressed_metrics
    return results


def run_c1h_consistency_partitioning(
    seeds: list[int],
    consistency_values: list[float],
    stage1_episodes: int = 300,
    stage2_episodes: int = 120,
    batch_size: int = 128,
    lr: float = 0.005,
    stage2_lr: float = 0.001,
    lambda_comm: float = 0.10,
    audit_batches: int = 8,
) -> dict[str, dict[str, float]]:
    env = ContinuousObjectSelectionEnv(EnvConfig())
    results: dict[str, dict[str, float]] = {}
    for seed in seeds:
        base_system = _build_system(env, seed=seed, baseline="learned")
        stage1_config = TrainConfig(
            episodes=stage1_episodes,
            batch_size=batch_size,
            seed=seed,
            lr=lr,
            baseline="learned",
        )
        train_existing(base_system, stage1_config, start_step=0)
        base_metrics = _summarize_audit(
            base_system,
            seed=seed + 20_000,
            batches=audit_batches,
            batch_size=batch_size,
        )
        results[f"seed_{seed}:stage1_high_bandwidth"] = base_metrics

        for value in consistency_values:
            compressed = copy.deepcopy(base_system)
            stage2_config = TrainConfig(
                episodes=stage2_episodes,
                batch_size=batch_size,
                seed=seed,
                lr=stage2_lr,
                baseline="learned",
                lambda_energy=lambda_comm,
                lambda_sparse=lambda_comm,
                lambda_dimeff=lambda_comm * 0.01,
                lambda_consistency=value,
            )
            losses, train_metrics = train_existing(
                compressed,
                stage2_config,
                start_step=stage1_episodes,
                train_speaker=True,
                train_listener=False,
            )
            metrics = _summarize_audit(
                compressed,
                seed=seed + 20_000,
                batches=audit_batches,
                batch_size=batch_size,
            )
            metrics.update(train_metrics)
            metrics.update({f"train_{name}": loss_value for name, loss_value in losses.items()})
            metrics.update(_with_prefixed_metrics("stage1", base_metrics))
            metrics["delta_task_accuracy"] = metrics["task_accuracy"] - base_metrics["task_accuracy"]
            metrics["delta_comm_energy"] = metrics["comm_energy"] - base_metrics["comm_energy"]
            metrics["delta_comm_l1"] = metrics["comm_l1"] - base_metrics["comm_l1"]
            metrics["delta_comm_effective_dim"] = (
                metrics["comm_effective_dim"] - base_metrics["comm_effective_dim"]
            )
            metrics["delta_target_cluster_alignment"] = (
                metrics["target_cluster_alignment"] - base_metrics["target_cluster_alignment"]
            )
            metrics["consistency_lambda"] = value
            results[f"seed_{seed}:consistency_{value:g}"] = metrics
    return results


def run_c1i_factor_segment_audit(
    seeds: list[int],
    factor_names: list[str],
    factor_consistency_values: list[float],
    stage1_episodes: int = 300,
    stage2_episodes: int = 120,
    batch_size: int = 128,
    lr: float = 0.005,
    stage2_lr: float = 0.001,
    lambda_comm: float = 0.10,
    audit_batches: int = 8,
) -> dict[str, dict[str, float]]:
    env = ContinuousObjectSelectionEnv(EnvConfig())
    results: dict[str, dict[str, float]] = {}
    for seed in seeds:
        base_system = _build_system(env, seed=seed, baseline="learned")
        stage1_config = TrainConfig(
            episodes=stage1_episodes,
            batch_size=batch_size,
            seed=seed,
            lr=lr,
            baseline="learned",
        )
        train_existing(base_system, stage1_config, start_step=0)
        base_metrics = _summarize_audit(
            base_system,
            seed=seed + 20_000,
            batches=audit_batches,
            batch_size=batch_size,
        )
        results[f"seed_{seed}:stage1_high_bandwidth"] = base_metrics
        for factor_name in factor_names:
            for value in factor_consistency_values:
                compressed = copy.deepcopy(base_system)
                stage2_config = TrainConfig(
                    episodes=stage2_episodes,
                    batch_size=batch_size,
                    seed=seed,
                    lr=stage2_lr,
                    baseline="learned",
                    lambda_energy=lambda_comm,
                    lambda_sparse=lambda_comm,
                    lambda_dimeff=lambda_comm * 0.01,
                    lambda_factor_consistency=value,
                    factor_name=factor_name,
                )
                losses, train_metrics = train_existing(
                    compressed,
                    stage2_config,
                    start_step=stage1_episodes,
                    train_speaker=True,
                    train_listener=False,
                )
                metrics = _summarize_audit(
                    compressed,
                    seed=seed + 20_000,
                    batches=audit_batches,
                    batch_size=batch_size,
                )
                metrics.update(train_metrics)
                metrics.update({f"train_{name}": loss_value for name, loss_value in losses.items()})
                metrics.update(_with_prefixed_metrics("stage1", base_metrics))
                metrics["delta_task_accuracy"] = metrics["task_accuracy"] - base_metrics["task_accuracy"]
                metrics["delta_comm_energy"] = metrics["comm_energy"] - base_metrics["comm_energy"]
                metrics["delta_comm_l1"] = metrics["comm_l1"] - base_metrics["comm_l1"]
                metrics["delta_comm_effective_dim"] = (
                    metrics["comm_effective_dim"] - base_metrics["comm_effective_dim"]
                )
                metrics["delta_target_cluster_alignment"] = (
                    metrics["target_cluster_alignment"] - base_metrics["target_cluster_alignment"]
                )
                metrics["factor_consistency_lambda"] = value
                results[f"seed_{seed}:factor_{factor_name}_{value:g}"] = metrics
    return results


def run_c1j_compositional_variant(
    seeds: list[int],
    factor_names: list[str],
    factor_consistency_values: list[float],
    stage1_episodes: int = 300,
    stage2_episodes: int = 120,
    batch_size: int = 128,
    lr: float = 0.005,
    stage2_lr: float = 0.001,
    lambda_comm: float = 0.10,
    audit_batches: int = 8,
) -> dict[str, dict[str, float]]:
    rule_mode = "compositional"
    env = _make_env(rule_mode)
    results: dict[str, dict[str, float]] = {}
    for seed in seeds:
        results[f"seed_{seed}:no_communication"] = run_train_eval(
            episodes=stage1_episodes,
            batch_size=batch_size,
            seed=seed,
            lr=lr,
            baseline="no-communication",
            rule_mode=rule_mode,
        )

        base_system = _build_system(env, seed=seed, baseline="learned")
        stage1_config = TrainConfig(
            episodes=stage1_episodes,
            batch_size=batch_size,
            seed=seed,
            lr=lr,
            baseline="learned",
            rule_mode=rule_mode,
        )
        train_existing(base_system, stage1_config, start_step=0)
        base_metrics = _summarize_audit(
            base_system,
            seed=seed + 20_000,
            batches=audit_batches,
            batch_size=batch_size,
            rule_mode=rule_mode,
            include_segment_swap=True,
        )
        results[f"seed_{seed}:stage1_high_bandwidth_compositional"] = base_metrics

        for factor_name in factor_names:
            for value in factor_consistency_values:
                compressed = copy.deepcopy(base_system)
                stage2_config = TrainConfig(
                    episodes=stage2_episodes,
                    batch_size=batch_size,
                    seed=seed,
                    lr=stage2_lr,
                    baseline="learned",
                    lambda_energy=lambda_comm,
                    lambda_sparse=lambda_comm,
                    lambda_dimeff=lambda_comm * 0.01,
                    lambda_factor_consistency=value,
                    factor_name=factor_name,
                    rule_mode=rule_mode,
                )
                losses, train_metrics = train_existing(
                    compressed,
                    stage2_config,
                    start_step=stage1_episodes,
                    train_speaker=True,
                    train_listener=False,
                )
                metrics = _summarize_audit(
                    compressed,
                    seed=seed + 20_000,
                    batches=audit_batches,
                    batch_size=batch_size,
                    rule_mode=rule_mode,
                    include_segment_swap=True,
                )
                metrics.update(train_metrics)
                metrics.update({f"train_{name}": loss_value for name, loss_value in losses.items()})
                metrics.update(_with_prefixed_metrics("stage1", base_metrics))
                metrics["delta_task_accuracy"] = metrics["task_accuracy"] - base_metrics["task_accuracy"]
                metrics["delta_comm_energy"] = metrics["comm_energy"] - base_metrics["comm_energy"]
                metrics["delta_comm_l1"] = metrics["comm_l1"] - base_metrics["comm_l1"]
                metrics["delta_comm_effective_dim"] = (
                    metrics["comm_effective_dim"] - base_metrics["comm_effective_dim"]
                )
                metrics["delta_target_cluster_alignment"] = (
                    metrics["target_cluster_alignment"] - base_metrics["target_cluster_alignment"]
                )
                metrics["factor_consistency_lambda"] = value
                metrics["rule_mode_compositional"] = 1.0
                results[f"seed_{seed}:c1j_factor_{factor_name}_{value:g}"] = metrics
    return results


def run_c1k_temporal_memory_variant(
    seeds: list[int],
    factor_names: list[str],
    factor_consistency_values: list[float],
    stage1_episodes: int = 600,
    stage2_episodes: int = 120,
    batch_size: int = 128,
    lr: float = 0.005,
    stage2_lr: float = 0.001,
    lambda_comm: float = 0.10,
    audit_batches: int = 8,
) -> dict[str, dict[str, float]]:
    rule_mode = "temporal_memory"
    env = _make_env(rule_mode)
    results: dict[str, dict[str, float]] = {}
    for seed in seeds:
        results[f"seed_{seed}:no_communication"] = run_train_eval(
            episodes=stage1_episodes,
            batch_size=batch_size,
            seed=seed,
            lr=lr,
            baseline="no-communication",
            rule_mode=rule_mode,
        )

        base_system = _build_system(env, seed=seed, baseline="learned")
        stage1_config = TrainConfig(
            episodes=stage1_episodes,
            batch_size=batch_size,
            seed=seed,
            lr=lr,
            baseline="learned",
            rule_mode=rule_mode,
        )
        train_existing(base_system, stage1_config, start_step=0)
        base_metrics = _summarize_audit(
            base_system,
            seed=seed + 20_000,
            batches=audit_batches,
            batch_size=batch_size,
            rule_mode=rule_mode,
            include_segment_swap=True,
        )
        base_metrics["rule_mode_temporal_memory"] = 1.0
        results[f"seed_{seed}:stage1_high_bandwidth_temporal_memory"] = base_metrics

        for factor_name in factor_names:
            for value in factor_consistency_values:
                compressed = copy.deepcopy(base_system)
                stage2_config = TrainConfig(
                    episodes=stage2_episodes,
                    batch_size=batch_size,
                    seed=seed,
                    lr=stage2_lr,
                    baseline="learned",
                    lambda_energy=lambda_comm,
                    lambda_sparse=lambda_comm,
                    lambda_dimeff=lambda_comm * 0.01,
                    lambda_factor_consistency=value,
                    factor_name=factor_name,
                    rule_mode=rule_mode,
                )
                losses, train_metrics = train_existing(
                    compressed,
                    stage2_config,
                    start_step=stage1_episodes,
                    train_speaker=True,
                    train_listener=False,
                )
                metrics = _summarize_audit(
                    compressed,
                    seed=seed + 20_000,
                    batches=audit_batches,
                    batch_size=batch_size,
                    rule_mode=rule_mode,
                    include_segment_swap=True,
                )
                metrics.update(train_metrics)
                metrics.update({f"train_{name}": loss_value for name, loss_value in losses.items()})
                metrics.update(_with_prefixed_metrics("stage1", base_metrics))
                metrics["delta_task_accuracy"] = metrics["task_accuracy"] - base_metrics["task_accuracy"]
                metrics["delta_comm_energy"] = metrics["comm_energy"] - base_metrics["comm_energy"]
                metrics["delta_comm_l1"] = metrics["comm_l1"] - base_metrics["comm_l1"]
                metrics["delta_comm_effective_dim"] = (
                    metrics["comm_effective_dim"] - base_metrics["comm_effective_dim"]
                )
                metrics["delta_target_cluster_alignment"] = (
                    metrics["target_cluster_alignment"] - base_metrics["target_cluster_alignment"]
                )
                metrics["factor_consistency_lambda"] = value
                metrics["rule_mode_temporal_memory"] = 1.0
                results[f"seed_{seed}:c1k_factor_{factor_name}_{value:g}"] = metrics
    return results


def run_c1l_temporal_bootstrap(
    seeds: list[int],
    bootstrap_modes: list[str],
    factor_names: list[str],
    factor_consistency_values: list[float],
    stage1_episodes: int = 600,
    curriculum_episodes: int = 300,
    stage2_episodes: int = 120,
    batch_size: int = 128,
    lr: float = 0.005,
    stage2_lr: float = 0.001,
    lambda_comm: float = 0.10,
    segment_dropout_prob: float = 0.25,
    audit_batches: int = 8,
) -> dict[str, dict[str, float]]:
    rule_mode = "temporal_memory"
    env = _make_env(rule_mode)
    results: dict[str, dict[str, float]] = {}
    for seed in seeds:
        results[f"seed_{seed}:no_communication"] = run_train_eval(
            episodes=stage1_episodes,
            batch_size=batch_size,
            seed=seed,
            lr=lr,
            baseline="no-communication",
            rule_mode=rule_mode,
        )

        for mode in bootstrap_modes:
            mode = mode.strip()
            if mode not in {"direct", "curriculum", "segment_dropout"}:
                raise ValueError(f"unknown bootstrap mode: {mode}")
            base_system = _build_system(env, seed=seed, baseline="learned")
            start_step = 0
            if mode == "curriculum":
                curriculum_config = TrainConfig(
                    episodes=curriculum_episodes,
                    batch_size=batch_size,
                    seed=seed,
                    lr=lr,
                    baseline="learned",
                    rule_mode="compositional",
                )
                train_existing(base_system, curriculum_config, start_step=start_step)
                start_step += curriculum_episodes

            stage1_config = TrainConfig(
                episodes=stage1_episodes,
                batch_size=batch_size,
                seed=seed,
                lr=lr,
                baseline="learned",
                rule_mode=rule_mode,
                segment_dropout_prob=segment_dropout_prob if mode == "segment_dropout" else 0.0,
            )
            train_existing(base_system, stage1_config, start_step=start_step)
            base_metrics = _summarize_audit(
                base_system,
                seed=seed + 20_000,
                batches=audit_batches,
                batch_size=batch_size,
                rule_mode=rule_mode,
                include_segment_swap=True,
            )
            base_metrics["rule_mode_temporal_memory"] = 1.0
            base_metrics["bootstrap_curriculum"] = 1.0 if mode == "curriculum" else 0.0
            base_metrics["bootstrap_segment_dropout"] = 1.0 if mode == "segment_dropout" else 0.0
            base_metrics["segment_dropout_prob"] = segment_dropout_prob if mode == "segment_dropout" else 0.0
            results[f"seed_{seed}:{mode}_stage1_temporal_memory"] = base_metrics

            for factor_name in factor_names:
                for value in factor_consistency_values:
                    compressed = copy.deepcopy(base_system)
                    stage2_config = TrainConfig(
                        episodes=stage2_episodes,
                        batch_size=batch_size,
                        seed=seed,
                        lr=stage2_lr,
                        baseline="learned",
                        lambda_energy=lambda_comm,
                        lambda_sparse=lambda_comm,
                        lambda_dimeff=lambda_comm * 0.01,
                        lambda_factor_consistency=value,
                        factor_name=factor_name,
                        rule_mode=rule_mode,
                        segment_dropout_prob=segment_dropout_prob if mode == "segment_dropout" else 0.0,
                    )
                    losses, train_metrics = train_existing(
                        compressed,
                        stage2_config,
                        start_step=start_step + stage1_episodes,
                        train_speaker=True,
                        train_listener=False,
                    )
                    metrics = _summarize_audit(
                        compressed,
                        seed=seed + 20_000,
                        batches=audit_batches,
                        batch_size=batch_size,
                        rule_mode=rule_mode,
                        include_segment_swap=True,
                    )
                    metrics.update(train_metrics)
                    metrics.update({f"train_{name}": loss_value for name, loss_value in losses.items()})
                    metrics.update(_with_prefixed_metrics("stage1", base_metrics))
                    metrics["delta_task_accuracy"] = metrics["task_accuracy"] - base_metrics["task_accuracy"]
                    metrics["delta_comm_energy"] = metrics["comm_energy"] - base_metrics["comm_energy"]
                    metrics["delta_comm_l1"] = metrics["comm_l1"] - base_metrics["comm_l1"]
                    metrics["delta_comm_effective_dim"] = (
                        metrics["comm_effective_dim"] - base_metrics["comm_effective_dim"]
                    )
                    metrics["delta_target_cluster_alignment"] = (
                        metrics["target_cluster_alignment"] - base_metrics["target_cluster_alignment"]
                    )
                    metrics["factor_consistency_lambda"] = value
                    metrics["rule_mode_temporal_memory"] = 1.0
                    metrics["bootstrap_curriculum"] = 1.0 if mode == "curriculum" else 0.0
                    metrics["bootstrap_segment_dropout"] = 1.0 if mode == "segment_dropout" else 0.0
                    metrics["segment_dropout_prob"] = segment_dropout_prob if mode == "segment_dropout" else 0.0
                    results[f"seed_{seed}:{mode}_c1l_factor_{factor_name}_{value:g}"] = metrics
    return results


def run_c2a_semantic_stability(
    seeds: list[int],
    nuisance_modes: list[str],
    stage1_episodes: int = 600,
    curriculum_episodes: int = 300,
    stage2_episodes: int = 120,
    batch_size: int = 128,
    lr: float = 0.005,
    stage2_lr: float = 0.001,
    lambda_comm: float = 0.10,
    factor_name: str = "q",
    factor_consistency: float = 0.03,
    audit_batches: int = 8,
) -> dict[str, dict[str, float]]:
    rule_mode = "temporal_memory"
    env = _make_env(rule_mode)
    results: dict[str, dict[str, float]] = {}
    for seed in seeds:
        system = _build_system(env, seed=seed, baseline="learned")
        curriculum_config = TrainConfig(
            episodes=curriculum_episodes,
            batch_size=batch_size,
            seed=seed,
            lr=lr,
            baseline="learned",
            rule_mode="compositional",
        )
        train_existing(system, curriculum_config, start_step=0)
        stage1_config = TrainConfig(
            episodes=stage1_episodes,
            batch_size=batch_size,
            seed=seed,
            lr=lr,
            baseline="learned",
            rule_mode=rule_mode,
        )
        train_existing(system, stage1_config, start_step=curriculum_episodes)
        stage2_config = TrainConfig(
            episodes=stage2_episodes,
            batch_size=batch_size,
            seed=seed,
            lr=stage2_lr,
            baseline="learned",
            lambda_energy=lambda_comm,
            lambda_sparse=lambda_comm,
            lambda_dimeff=lambda_comm * 0.01,
            lambda_factor_consistency=factor_consistency,
            factor_name=factor_name,
            rule_mode=rule_mode,
        )
        losses, train_metrics = train_existing(
            system,
            stage2_config,
            start_step=curriculum_episodes + stage1_episodes,
            train_speaker=True,
            train_listener=False,
        )

        clean_metrics: dict[str, float] | None = None
        for nuisance_mode in nuisance_modes:
            metrics = _summarize_audit(
                system,
                seed=seed + 20_000,
                batches=audit_batches,
                batch_size=batch_size,
                rule_mode=rule_mode,
                nuisance_mode=nuisance_mode,
                include_segment_swap=True,
            )
            metrics.update(train_metrics)
            metrics.update({f"train_{name}": loss_value for name, loss_value in losses.items()})
            metrics["rule_mode_temporal_memory"] = 1.0
            metrics["bootstrap_curriculum"] = 1.0
            metrics["nuisance_is_mirror_x"] = 1.0 if nuisance_mode == "mirror_x" else 0.0
            metrics["nuisance_is_rotate90"] = 1.0 if nuisance_mode == "rotate90" else 0.0
            metrics["nuisance_is_velocity_scale"] = 1.0 if nuisance_mode == "velocity_scale" else 0.0
            if nuisance_mode == "none":
                clean_metrics = metrics.copy()
            elif clean_metrics is not None:
                for key in [
                    "task_accuracy",
                    "audit_hidden_q_probe_r2",
                    "segment_late_swap_action_change_rate",
                    "segment_late_ablation_action_change_rate",
                    "segment_late_ablation_target_logit_drop",
                ]:
                    metrics[f"delta_clean_{key}"] = metrics[key] - clean_metrics[key]
            results[f"seed_{seed}:c2a_{nuisance_mode}"] = metrics
    return results


def _train_c2_curriculum_system(
    seed: int,
    stage1_episodes: int,
    curriculum_episodes: int,
    stage2_episodes: int,
    batch_size: int,
    lr: float,
    stage2_lr: float,
    lambda_comm: float,
    factor_name: str,
    factor_consistency: float,
    rule_mode: str = "temporal_memory",
) -> tuple[SpeakerListenerSystem, dict[str, float], dict[str, float]]:
    env = _make_env(rule_mode)
    system = _build_system(env, seed=seed, baseline="learned")
    train_existing(
        system,
        TrainConfig(
            episodes=curriculum_episodes,
            batch_size=batch_size,
            seed=seed,
            lr=lr,
            baseline="learned",
            rule_mode="compositional",
        ),
        start_step=0,
    )
    train_existing(
        system,
        TrainConfig(
            episodes=stage1_episodes,
            batch_size=batch_size,
            seed=seed,
            lr=lr,
            baseline="learned",
            rule_mode=rule_mode,
        ),
        start_step=curriculum_episodes,
    )
    losses, train_metrics = train_existing(
        system,
        TrainConfig(
            episodes=stage2_episodes,
            batch_size=batch_size,
            seed=seed,
            lr=stage2_lr,
            baseline="learned",
            lambda_energy=lambda_comm,
            lambda_sparse=lambda_comm,
            lambda_dimeff=lambda_comm * 0.01,
            lambda_factor_consistency=factor_consistency,
            factor_name=factor_name,
            rule_mode=rule_mode,
        ),
        start_step=curriculum_episodes + stage1_episodes,
        train_speaker=True,
        train_listener=False,
    )
    return system, losses, train_metrics


def run_c2b_cluster_invariant_alignment(
    seeds: list[int],
    nuisance_modes: list[str],
    stage1_episodes: int = 600,
    curriculum_episodes: int = 300,
    stage2_episodes: int = 120,
    batch_size: int = 128,
    lr: float = 0.005,
    stage2_lr: float = 0.001,
    lambda_comm: float = 0.10,
    factor_name: str = "q",
    factor_consistency: float = 0.03,
    audit_batches: int = 8,
) -> dict[str, dict[str, float]]:
    rule_mode = "temporal_memory"
    results: dict[str, dict[str, float]] = {}
    nuisance_modes = [mode for mode in nuisance_modes if mode != "none"]
    for seed in seeds:
        system, losses, train_metrics = _train_c2_curriculum_system(
            seed=seed,
            stage1_episodes=stage1_episodes,
            curriculum_episodes=curriculum_episodes,
            stage2_episodes=stage2_episodes,
            batch_size=batch_size,
            lr=lr,
            stage2_lr=stage2_lr,
            lambda_comm=lambda_comm,
            factor_name=factor_name,
            factor_consistency=factor_consistency,
        )
        clean = _collect_audit_batch(
            system,
            seed=seed + 20_000,
            batches=audit_batches,
            batch_size=batch_size,
            rule_mode=rule_mode,
            nuisance_mode="none",
        )
        clean_metrics = _summarize_collected_audit(system, clean, include_segment_swap=True)
        clean_metrics.update(train_metrics)
        clean_metrics.update({f"train_{name}": loss_value for name, loss_value in losses.items()})
        clean_metrics["rule_mode_temporal_memory"] = 1.0
        clean_metrics["bootstrap_curriculum"] = 1.0
        results[f"seed_{seed}:c2b_clean"] = clean_metrics

        for nuisance_mode in nuisance_modes:
            nuisance = _collect_audit_batch(
                system,
                seed=seed + 20_000,
                batches=audit_batches,
                batch_size=batch_size,
                rule_mode=rule_mode,
                nuisance_mode=nuisance_mode,
            )
            metrics = _summarize_collected_audit(system, nuisance, include_segment_swap=True)
            metrics.update(
                paired_nuisance_alignment_metrics(
                    clean["comm"],
                    nuisance["comm"],
                    clean["target"],
                    nuisance["target"],
                    clean["latent"],
                    nuisance["latent"],
                    clean["objects"],
                    nuisance["objects"],
                )
            )
            metrics["rule_mode_temporal_memory"] = 1.0
            metrics["bootstrap_curriculum"] = 1.0
            metrics["nuisance_is_mirror_x"] = 1.0 if nuisance_mode == "mirror_x" else 0.0
            metrics["nuisance_is_rotate90"] = 1.0 if nuisance_mode == "rotate90" else 0.0
            metrics["nuisance_is_velocity_scale"] = 1.0 if nuisance_mode == "velocity_scale" else 0.0
            metrics["delta_clean_task_accuracy"] = metrics["task_accuracy"] - clean_metrics["task_accuracy"]
            metrics["delta_clean_audit_hidden_q_probe_r2"] = (
                metrics["audit_hidden_q_probe_r2"] - clean_metrics["audit_hidden_q_probe_r2"]
            )
            metrics["delta_clean_segment_late_swap_action_change_rate"] = (
                metrics["segment_late_swap_action_change_rate"]
                - clean_metrics["segment_late_swap_action_change_rate"]
            )
            results[f"seed_{seed}:c2b_{nuisance_mode}"] = metrics
    return results


def run_c2c_factor_separated_invariance(
    seeds: list[int],
    nuisance_modes: list[str],
    stage1_episodes: int = 600,
    curriculum_episodes: int = 300,
    stage2_episodes: int = 120,
    batch_size: int = 128,
    lr: float = 0.005,
    stage2_lr: float = 0.001,
    lambda_comm: float = 0.10,
    factor_name: str = "q",
    factor_consistency: float = 0.03,
    audit_batches: int = 4,
) -> dict[str, dict[str, float]]:
    results = run_c2b_cluster_invariant_alignment(
        seeds=seeds,
        nuisance_modes=nuisance_modes,
        stage1_episodes=stage1_episodes,
        curriculum_episodes=curriculum_episodes,
        stage2_episodes=stage2_episodes,
        batch_size=batch_size,
        lr=lr,
        stage2_lr=stage2_lr,
        lambda_comm=lambda_comm,
        factor_name=factor_name,
        factor_consistency=factor_consistency,
        audit_batches=audit_batches,
    )
    renamed: dict[str, dict[str, float]] = {}
    for name, metrics in results.items():
        renamed[name.replace(":c2b_", ":c2c_")] = metrics
    return renamed


def run_c3a_targeted_segment_intervention(
    seeds: list[int],
    stage1_episodes: int = 600,
    curriculum_episodes: int = 300,
    stage2_episodes: int = 120,
    batch_size: int = 128,
    lr: float = 0.005,
    stage2_lr: float = 0.001,
    lambda_comm: float = 0.10,
    factor_name: str = "q",
    factor_consistency: float = 0.03,
    audit_batches: int = 8,
) -> dict[str, dict[str, float]]:
    rule_mode = "temporal_memory"
    results: dict[str, dict[str, float]] = {}
    for seed in seeds:
        system, losses, train_metrics = _train_c2_curriculum_system(
            seed=seed,
            stage1_episodes=stage1_episodes,
            curriculum_episodes=curriculum_episodes,
            stage2_episodes=stage2_episodes,
            batch_size=batch_size,
            lr=lr,
            stage2_lr=stage2_lr,
            lambda_comm=lambda_comm,
            factor_name=factor_name,
            factor_consistency=factor_consistency,
        )
        collected = _collect_audit_batch(
            system,
            seed=seed + 20_000,
            batches=audit_batches,
            batch_size=batch_size,
            rule_mode=rule_mode,
            nuisance_mode="none",
        )
        metrics = _summarize_collected_audit(
            system,
            collected,
            include_segment_swap=True,
            include_targeted_intervention=True,
        )
        metrics.update(train_metrics)
        metrics.update({f"train_{name}": loss_value for name, loss_value in losses.items()})
        metrics["rule_mode_temporal_memory"] = 1.0
        metrics["bootstrap_curriculum"] = 1.0
        results[f"seed_{seed}:c3a_targeted_intervention"] = metrics
    return results


def run_c3b_targeted_intervention_robustness(
    seeds: list[int],
    nuisance_modes: list[str],
    stage1_episodes: int = 600,
    curriculum_episodes: int = 300,
    stage2_episodes: int = 120,
    batch_size: int = 128,
    lr: float = 0.005,
    stage2_lr: float = 0.001,
    lambda_comm: float = 0.10,
    factor_name: str = "q",
    factor_consistency: float = 0.03,
    audit_batches: int = 12,
) -> dict[str, dict[str, float]]:
    rule_mode = "temporal_memory"
    results: dict[str, dict[str, float]] = {}
    for seed in seeds:
        system, losses, train_metrics = _train_c2_curriculum_system(
            seed=seed,
            stage1_episodes=stage1_episodes,
            curriculum_episodes=curriculum_episodes,
            stage2_episodes=stage2_episodes,
            batch_size=batch_size,
            lr=lr,
            stage2_lr=stage2_lr,
            lambda_comm=lambda_comm,
            factor_name=factor_name,
            factor_consistency=factor_consistency,
        )
        for nuisance_mode in nuisance_modes:
            collected = _collect_audit_batch(
                system,
                seed=seed + 20_000,
                batches=audit_batches,
                batch_size=batch_size,
                rule_mode=rule_mode,
                nuisance_mode=nuisance_mode,
            )
            metrics = _summarize_collected_audit(
                system,
                collected,
                include_segment_swap=True,
                include_targeted_intervention=True,
            )
            metrics.update(train_metrics)
            metrics.update({f"train_{name}": loss_value for name, loss_value in losses.items()})
            metrics["rule_mode_temporal_memory"] = 1.0
            metrics["bootstrap_curriculum"] = 1.0
            metrics["nuisance_is_mirror_x"] = 1.0 if nuisance_mode == "mirror_x" else 0.0
            metrics["nuisance_is_rotate90"] = 1.0 if nuisance_mode == "rotate90" else 0.0
            metrics["nuisance_is_velocity_scale"] = 1.0 if nuisance_mode == "velocity_scale" else 0.0
            results[f"seed_{seed}:c3b_{nuisance_mode}"] = metrics
    return results


def run_c3c_motion_pressure_variant(
    seeds: list[int],
    nuisance_modes: list[str],
    stage1_episodes: int = 600,
    curriculum_episodes: int = 300,
    stage2_episodes: int = 120,
    batch_size: int = 128,
    lr: float = 0.005,
    stage2_lr: float = 0.001,
    lambda_comm: float = 0.10,
    factor_name: str = "motion",
    factor_consistency: float = 0.03,
    audit_batches: int = 12,
) -> dict[str, dict[str, float]]:
    rule_mode = "motion_pressure"
    results: dict[str, dict[str, float]] = {}
    for seed in seeds:
        system, losses, train_metrics = _train_c2_curriculum_system(
            seed=seed,
            stage1_episodes=stage1_episodes,
            curriculum_episodes=curriculum_episodes,
            stage2_episodes=stage2_episodes,
            batch_size=batch_size,
            lr=lr,
            stage2_lr=stage2_lr,
            lambda_comm=lambda_comm,
            factor_name=factor_name,
            factor_consistency=factor_consistency,
            rule_mode=rule_mode,
        )
        for nuisance_mode in nuisance_modes:
            collected = _collect_audit_batch(
                system,
                seed=seed + 20_000,
                batches=audit_batches,
                batch_size=batch_size,
                rule_mode=rule_mode,
                nuisance_mode=nuisance_mode,
            )
            metrics = _summarize_collected_audit(
                system,
                collected,
                include_segment_swap=True,
                include_targeted_intervention=True,
            )
            metrics.update(train_metrics)
            metrics.update({f"train_{name}": loss_value for name, loss_value in losses.items()})
            metrics["rule_mode_motion_pressure"] = 1.0
            metrics["bootstrap_curriculum"] = 1.0
            metrics["nuisance_is_mirror_x"] = 1.0 if nuisance_mode == "mirror_x" else 0.0
            metrics["nuisance_is_rotate90"] = 1.0 if nuisance_mode == "rotate90" else 0.0
            metrics["nuisance_is_velocity_scale"] = 1.0 if nuisance_mode == "velocity_scale" else 0.0
            results[f"seed_{seed}:c3c_{nuisance_mode}"] = metrics
    return results


def run_c3d_motion_decoupling_variant(
    seeds: list[int],
    nuisance_modes: list[str],
    stage1_episodes: int = 600,
    curriculum_episodes: int = 300,
    stage2_episodes: int = 120,
    batch_size: int = 128,
    lr: float = 0.005,
    stage2_lr: float = 0.001,
    lambda_comm: float = 0.10,
    factor_name: str = "motion",
    factor_consistency: float = 0.03,
    audit_batches: int = 12,
) -> dict[str, dict[str, float]]:
    rule_mode = "motion_decoupled"
    results: dict[str, dict[str, float]] = {}
    for seed in seeds:
        system, losses, train_metrics = _train_c2_curriculum_system(
            seed=seed,
            stage1_episodes=stage1_episodes,
            curriculum_episodes=curriculum_episodes,
            stage2_episodes=stage2_episodes,
            batch_size=batch_size,
            lr=lr,
            stage2_lr=stage2_lr,
            lambda_comm=lambda_comm,
            factor_name=factor_name,
            factor_consistency=factor_consistency,
            rule_mode=rule_mode,
        )
        for nuisance_mode in nuisance_modes:
            collected = _collect_audit_batch(
                system,
                seed=seed + 24_000,
                batches=audit_batches,
                batch_size=batch_size,
                rule_mode=rule_mode,
                nuisance_mode=nuisance_mode,
            )
            metrics = _summarize_collected_audit(
                system,
                collected,
                include_segment_swap=True,
                include_targeted_intervention=True,
            )
            metrics.update(train_metrics)
            metrics.update({f"train_{name}": loss_value for name, loss_value in losses.items()})
            metrics["rule_mode_motion_decoupled"] = 1.0
            metrics["bootstrap_curriculum"] = 1.0
            metrics["nuisance_is_mirror_x"] = 1.0 if nuisance_mode == "mirror_x" else 0.0
            metrics["nuisance_is_rotate90"] = 1.0 if nuisance_mode == "rotate90" else 0.0
            metrics["nuisance_is_velocity_scale"] = 1.0 if nuisance_mode == "velocity_scale" else 0.0
            results[f"seed_{seed}:c3d_{nuisance_mode}"] = metrics
    return results


def run_c3e_motion_readout_audit(
    seeds: list[int],
    nuisance_modes: list[str],
    stage1_episodes: int = 600,
    curriculum_episodes: int = 300,
    stage2_episodes: int = 120,
    batch_size: int = 128,
    lr: float = 0.005,
    stage2_lr: float = 0.001,
    lambda_comm: float = 0.10,
    factor_name: str = "motion",
    factor_consistency: float = 0.03,
    audit_batches: int = 12,
) -> dict[str, dict[str, float]]:
    rule_mode = "motion_decoupled"
    results: dict[str, dict[str, float]] = {}
    for seed in seeds:
        system, losses, train_metrics = _train_c2_curriculum_system(
            seed=seed,
            stage1_episodes=stage1_episodes,
            curriculum_episodes=curriculum_episodes,
            stage2_episodes=stage2_episodes,
            batch_size=batch_size,
            lr=lr,
            stage2_lr=stage2_lr,
            lambda_comm=lambda_comm,
            factor_name=factor_name,
            factor_consistency=factor_consistency,
            rule_mode=rule_mode,
        )
        for nuisance_mode in nuisance_modes:
            collected = _collect_audit_batch(
                system,
                seed=seed + 28_000,
                batches=audit_batches,
                batch_size=batch_size,
                rule_mode=rule_mode,
                nuisance_mode=nuisance_mode,
            )
            metrics = _summarize_collected_audit(
                system,
                collected,
                include_segment_swap=True,
                include_targeted_intervention=True,
            )
            metrics.update(train_metrics)
            metrics.update({f"train_{name}": loss_value for name, loss_value in losses.items()})
            metrics["rule_mode_motion_decoupled"] = 1.0
            metrics["bootstrap_curriculum"] = 1.0
            metrics["nuisance_is_mirror_x"] = 1.0 if nuisance_mode == "mirror_x" else 0.0
            metrics["nuisance_is_rotate90"] = 1.0 if nuisance_mode == "rotate90" else 0.0
            metrics["nuisance_is_velocity_scale"] = 1.0 if nuisance_mode == "velocity_scale" else 0.0
            results[f"seed_{seed}:c3e_{nuisance_mode}"] = metrics
    return results


def run_c3f_factorized_pressure_objective(
    seeds: list[int],
    factor_sets: list[str],
    stage1_episodes: int = 600,
    curriculum_episodes: int = 300,
    stage2_episodes: int = 120,
    batch_size: int = 128,
    lr: float = 0.005,
    stage2_lr: float = 0.001,
    lambda_comm: float = 0.10,
    factor_consistency: float = 0.03,
    audit_batches: int = 12,
) -> dict[str, dict[str, float]]:
    rule_mode = "motion_decoupled"
    results: dict[str, dict[str, float]] = {}
    for seed in seeds:
        for factor_set in factor_sets:
            normalized_factor_set = "+".join(
                item.strip() for item in factor_set.replace(",", "+").split("+") if item.strip()
            )
            system, losses, train_metrics = _train_c2_curriculum_system(
                seed=seed,
                stage1_episodes=stage1_episodes,
                curriculum_episodes=curriculum_episodes,
                stage2_episodes=stage2_episodes,
                batch_size=batch_size,
                lr=lr,
                stage2_lr=stage2_lr,
                lambda_comm=lambda_comm,
                factor_name=normalized_factor_set,
                factor_consistency=factor_consistency,
                rule_mode=rule_mode,
            )
            collected = _collect_audit_batch(
                system,
                seed=seed + 32_000,
                batches=audit_batches,
                batch_size=batch_size,
                rule_mode=rule_mode,
                nuisance_mode="none",
            )
            metrics = _summarize_collected_audit(
                system,
                collected,
                include_segment_swap=True,
                include_targeted_intervention=True,
            )
            metrics.update(train_metrics)
            metrics.update({f"train_{name}": loss_value for name, loss_value in losses.items()})
            factor_items = [item for item in normalized_factor_set.split("+") if item]
            metrics["rule_mode_motion_decoupled"] = 1.0
            metrics["bootstrap_curriculum"] = 1.0
            metrics["factor_set_size"] = float(len(factor_items))
            metrics["factor_set_has_q"] = 1.0 if "q" in factor_items else 0.0
            metrics["factor_set_has_motion"] = 1.0 if "motion" in factor_items else 0.0
            metrics["factor_set_has_relation"] = 1.0 if "relation" in factor_items else 0.0
            metrics["factor_consistency_lambda"] = factor_consistency
            results[f"seed_{seed}:c3f_{normalized_factor_set}"] = metrics
    return results


def run_c1c_calibration(
    episodes: int = 120,
    batch_size: int = 64,
    seed: int = 0,
    compressed_lambda: float = 0.001,
    lr: float = 2e-3,
) -> dict[str, dict[str, float]]:
    return {
        "no_communication": run_train_eval(
            episodes=episodes,
            batch_size=batch_size,
            seed=seed,
            lr=lr,
            baseline="no-communication",
        ),
        "teacher_signal": run_train_eval(
            episodes=episodes,
            batch_size=batch_size,
            seed=seed,
            lr=lr,
            baseline="teacher-signal",
        ),
        "learned_high_bandwidth": run_train_eval(
            episodes=episodes,
            batch_size=batch_size,
            seed=seed,
            lr=lr,
            baseline="learned",
            lambda_comm=0.0,
        ),
        f"learned_compressed_{compressed_lambda:g}": run_train_eval(
            episodes=episodes,
            batch_size=batch_size,
            seed=seed,
            lr=lr,
            baseline="learned",
            lambda_comm=compressed_lambda,
        ),
        "learned_distilled": run_train_eval(
            episodes=episodes,
            batch_size=batch_size,
            seed=seed,
            lr=lr,
            baseline="learned",
            lambda_comm=0.0,
            lambda_comm_distill=1.0,
        ),
    }


def _parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="CEP-CC continuous communication experiment runner")
    parser.add_argument("--mode", choices=["train-eval"], default="train-eval")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--lambda-comm", type=float, default=0.0)
    parser.add_argument("--lambda-comm-distill", type=float, default=0.0)
    parser.add_argument("--lambda-sweep", type=str, default="")
    parser.add_argument("--c1c-calibration", action="store_true")
    parser.add_argument("--multiseed-compression-sweep", action="store_true")
    parser.add_argument("--two-stage-compression", action="store_true")
    parser.add_argument("--c1f-ablation", action="store_true")
    parser.add_argument("--c1g-geometry-audit", action="store_true")
    parser.add_argument("--c1h-consistency", action="store_true")
    parser.add_argument("--c1i-factor-segment", action="store_true")
    parser.add_argument("--c1j-compositional", action="store_true")
    parser.add_argument("--c1k-temporal-memory", action="store_true")
    parser.add_argument("--c1l-bootstrap", action="store_true")
    parser.add_argument("--c2a-semantic-stability", action="store_true")
    parser.add_argument("--c2b-cluster-alignment", action="store_true")
    parser.add_argument("--c2c-factor-invariance", action="store_true")
    parser.add_argument("--c3a-targeted-intervention", action="store_true")
    parser.add_argument("--c3b-targeted-robustness", action="store_true")
    parser.add_argument("--c3c-motion-pressure", action="store_true")
    parser.add_argument("--c3d-motion-decoupling", action="store_true")
    parser.add_argument("--c3e-motion-readout", action="store_true")
    parser.add_argument("--c3f-factorized-pressure", action="store_true")
    parser.add_argument(
        "--rule-mode",
        choices=["scalar", "compositional", "temporal_memory", "motion_pressure", "motion_decoupled"],
        default="scalar",
    )
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--stage1-episodes", type=int, default=300)
    parser.add_argument("--curriculum-episodes", type=int, default=300)
    parser.add_argument("--stage2-episodes", type=int, default=120)
    parser.add_argument("--stage2-lr", type=float, default=None)
    parser.add_argument("--freeze-listener-stage2", action="store_true")
    parser.add_argument("--audit-batches", type=int, default=8)
    parser.add_argument("--consistency-sweep", type=str, default="0.01,0.03,0.1")
    parser.add_argument("--factor-names", type=str, default="q,motion,relation")
    parser.add_argument("--factor-sets", type=str, default="q;motion;q+motion;q+motion+relation")
    parser.add_argument("--factor-consistency-sweep", type=str, default="0.01,0.03")
    parser.add_argument("--bootstrap-modes", type=str, default="direct,curriculum,segment_dropout")
    parser.add_argument("--segment-dropout-prob", type=float, default=0.25)
    parser.add_argument("--nuisance-modes", type=str, default="none,mirror_x,rotate90,velocity_scale")
    parser.add_argument(
        "--baseline",
        choices=["learned", "no-communication", "high-bandwidth", "random-projection", "teacher-signal"],
        default="learned",
    )
    parser.add_argument("--table", action="store_true")
    args = parser.parse_args()

    if args.c3f_factorized_pressure:
        factor_sets = [item.strip() for item in args.factor_sets.split(";") if item.strip()]
        results = run_c3f_factorized_pressure_objective(
            seeds=_parse_int_list(args.seeds),
            factor_sets=factor_sets,
            stage1_episodes=args.stage1_episodes,
            curriculum_episodes=args.curriculum_episodes,
            stage2_episodes=args.stage2_episodes,
            batch_size=args.batch_size,
            lr=args.lr,
            stage2_lr=args.stage2_lr if args.stage2_lr is not None else 0.001,
            lambda_comm=args.lambda_comm if args.lambda_comm > 0 else 0.10,
            factor_consistency=_parse_float_list(args.factor_consistency_sweep)[0],
            audit_batches=args.audit_batches,
        )
    elif args.c3e_motion_readout:
        factor_names = [item.strip() for item in args.factor_names.split(",") if item.strip()]
        results = run_c3e_motion_readout_audit(
            seeds=_parse_int_list(args.seeds),
            nuisance_modes=[item.strip() for item in args.nuisance_modes.split(",") if item.strip()],
            stage1_episodes=args.stage1_episodes,
            curriculum_episodes=args.curriculum_episodes,
            stage2_episodes=args.stage2_episodes,
            batch_size=args.batch_size,
            lr=args.lr,
            stage2_lr=args.stage2_lr if args.stage2_lr is not None else 0.001,
            lambda_comm=args.lambda_comm if args.lambda_comm > 0 else 0.10,
            factor_name=factor_names[0] if factor_names else "motion",
            factor_consistency=_parse_float_list(args.factor_consistency_sweep)[0],
            audit_batches=args.audit_batches,
        )
    elif args.c3d_motion_decoupling:
        factor_names = [item.strip() for item in args.factor_names.split(",") if item.strip()]
        results = run_c3d_motion_decoupling_variant(
            seeds=_parse_int_list(args.seeds),
            nuisance_modes=[item.strip() for item in args.nuisance_modes.split(",") if item.strip()],
            stage1_episodes=args.stage1_episodes,
            curriculum_episodes=args.curriculum_episodes,
            stage2_episodes=args.stage2_episodes,
            batch_size=args.batch_size,
            lr=args.lr,
            stage2_lr=args.stage2_lr if args.stage2_lr is not None else 0.001,
            lambda_comm=args.lambda_comm if args.lambda_comm > 0 else 0.10,
            factor_name=factor_names[0] if factor_names else "motion",
            factor_consistency=_parse_float_list(args.factor_consistency_sweep)[0],
            audit_batches=args.audit_batches,
        )
    elif args.c3c_motion_pressure:
        factor_names = [item.strip() for item in args.factor_names.split(",") if item.strip()]
        results = run_c3c_motion_pressure_variant(
            seeds=_parse_int_list(args.seeds),
            nuisance_modes=[item.strip() for item in args.nuisance_modes.split(",") if item.strip()],
            stage1_episodes=args.stage1_episodes,
            curriculum_episodes=args.curriculum_episodes,
            stage2_episodes=args.stage2_episodes,
            batch_size=args.batch_size,
            lr=args.lr,
            stage2_lr=args.stage2_lr if args.stage2_lr is not None else 0.001,
            lambda_comm=args.lambda_comm if args.lambda_comm > 0 else 0.10,
            factor_name=factor_names[0] if factor_names else "motion",
            factor_consistency=_parse_float_list(args.factor_consistency_sweep)[0],
            audit_batches=args.audit_batches,
        )
    elif args.c3b_targeted_robustness:
        factor_names = [item.strip() for item in args.factor_names.split(",") if item.strip()]
        results = run_c3b_targeted_intervention_robustness(
            seeds=_parse_int_list(args.seeds),
            nuisance_modes=[item.strip() for item in args.nuisance_modes.split(",") if item.strip()],
            stage1_episodes=args.stage1_episodes,
            curriculum_episodes=args.curriculum_episodes,
            stage2_episodes=args.stage2_episodes,
            batch_size=args.batch_size,
            lr=args.lr,
            stage2_lr=args.stage2_lr if args.stage2_lr is not None else 0.001,
            lambda_comm=args.lambda_comm if args.lambda_comm > 0 else 0.10,
            factor_name=factor_names[0] if factor_names else "q",
            factor_consistency=_parse_float_list(args.factor_consistency_sweep)[0],
            audit_batches=args.audit_batches,
        )
    elif args.c3a_targeted_intervention:
        factor_names = [item.strip() for item in args.factor_names.split(",") if item.strip()]
        results = run_c3a_targeted_segment_intervention(
            seeds=_parse_int_list(args.seeds),
            stage1_episodes=args.stage1_episodes,
            curriculum_episodes=args.curriculum_episodes,
            stage2_episodes=args.stage2_episodes,
            batch_size=args.batch_size,
            lr=args.lr,
            stage2_lr=args.stage2_lr if args.stage2_lr is not None else 0.001,
            lambda_comm=args.lambda_comm if args.lambda_comm > 0 else 0.10,
            factor_name=factor_names[0] if factor_names else "q",
            factor_consistency=_parse_float_list(args.factor_consistency_sweep)[0],
            audit_batches=args.audit_batches,
        )
    elif args.c2c_factor_invariance:
        factor_names = [item.strip() for item in args.factor_names.split(",") if item.strip()]
        results = run_c2c_factor_separated_invariance(
            seeds=_parse_int_list(args.seeds),
            nuisance_modes=[item.strip() for item in args.nuisance_modes.split(",") if item.strip()],
            stage1_episodes=args.stage1_episodes,
            curriculum_episodes=args.curriculum_episodes,
            stage2_episodes=args.stage2_episodes,
            batch_size=args.batch_size,
            lr=args.lr,
            stage2_lr=args.stage2_lr if args.stage2_lr is not None else 0.001,
            lambda_comm=args.lambda_comm if args.lambda_comm > 0 else 0.10,
            factor_name=factor_names[0] if factor_names else "q",
            factor_consistency=_parse_float_list(args.factor_consistency_sweep)[0],
            audit_batches=args.audit_batches,
        )
    elif args.c2b_cluster_alignment:
        factor_names = [item.strip() for item in args.factor_names.split(",") if item.strip()]
        results = run_c2b_cluster_invariant_alignment(
            seeds=_parse_int_list(args.seeds),
            nuisance_modes=[item.strip() for item in args.nuisance_modes.split(",") if item.strip()],
            stage1_episodes=args.stage1_episodes,
            curriculum_episodes=args.curriculum_episodes,
            stage2_episodes=args.stage2_episodes,
            batch_size=args.batch_size,
            lr=args.lr,
            stage2_lr=args.stage2_lr if args.stage2_lr is not None else 0.001,
            lambda_comm=args.lambda_comm if args.lambda_comm > 0 else 0.10,
            factor_name=factor_names[0] if factor_names else "q",
            factor_consistency=_parse_float_list(args.factor_consistency_sweep)[0],
            audit_batches=args.audit_batches,
        )
    elif args.c2a_semantic_stability:
        factor_names = [item.strip() for item in args.factor_names.split(",") if item.strip()]
        results = run_c2a_semantic_stability(
            seeds=_parse_int_list(args.seeds),
            nuisance_modes=[item.strip() for item in args.nuisance_modes.split(",") if item.strip()],
            stage1_episodes=args.stage1_episodes,
            curriculum_episodes=args.curriculum_episodes,
            stage2_episodes=args.stage2_episodes,
            batch_size=args.batch_size,
            lr=args.lr,
            stage2_lr=args.stage2_lr if args.stage2_lr is not None else 0.001,
            lambda_comm=args.lambda_comm if args.lambda_comm > 0 else 0.10,
            factor_name=factor_names[0] if factor_names else "q",
            factor_consistency=_parse_float_list(args.factor_consistency_sweep)[0],
            audit_batches=args.audit_batches,
        )
    elif args.c1l_bootstrap:
        results = run_c1l_temporal_bootstrap(
            seeds=_parse_int_list(args.seeds),
            bootstrap_modes=[item.strip() for item in args.bootstrap_modes.split(",") if item.strip()],
            factor_names=[item.strip() for item in args.factor_names.split(",") if item.strip()],
            factor_consistency_values=_parse_float_list(args.factor_consistency_sweep),
            stage1_episodes=args.stage1_episodes,
            curriculum_episodes=args.curriculum_episodes,
            stage2_episodes=args.stage2_episodes,
            batch_size=args.batch_size,
            lr=args.lr,
            stage2_lr=args.stage2_lr if args.stage2_lr is not None else 0.001,
            lambda_comm=args.lambda_comm if args.lambda_comm > 0 else 0.10,
            segment_dropout_prob=args.segment_dropout_prob,
            audit_batches=args.audit_batches,
        )
    elif args.c1k_temporal_memory:
        results = run_c1k_temporal_memory_variant(
            seeds=_parse_int_list(args.seeds),
            factor_names=[item.strip() for item in args.factor_names.split(",") if item.strip()],
            factor_consistency_values=_parse_float_list(args.factor_consistency_sweep),
            stage1_episodes=args.stage1_episodes,
            stage2_episodes=args.stage2_episodes,
            batch_size=args.batch_size,
            lr=args.lr,
            stage2_lr=args.stage2_lr if args.stage2_lr is not None else 0.001,
            lambda_comm=args.lambda_comm if args.lambda_comm > 0 else 0.10,
            audit_batches=args.audit_batches,
        )
    elif args.c1j_compositional:
        results = run_c1j_compositional_variant(
            seeds=_parse_int_list(args.seeds),
            factor_names=[item.strip() for item in args.factor_names.split(",") if item.strip()],
            factor_consistency_values=_parse_float_list(args.factor_consistency_sweep),
            stage1_episodes=args.stage1_episodes,
            stage2_episodes=args.stage2_episodes,
            batch_size=args.batch_size,
            lr=args.lr,
            stage2_lr=args.stage2_lr if args.stage2_lr is not None else 0.001,
            lambda_comm=args.lambda_comm if args.lambda_comm > 0 else 0.10,
            audit_batches=args.audit_batches,
        )
    elif args.c1i_factor_segment:
        results = run_c1i_factor_segment_audit(
            seeds=_parse_int_list(args.seeds),
            factor_names=[item.strip() for item in args.factor_names.split(",") if item.strip()],
            factor_consistency_values=_parse_float_list(args.factor_consistency_sweep),
            stage1_episodes=args.stage1_episodes,
            stage2_episodes=args.stage2_episodes,
            batch_size=args.batch_size,
            lr=args.lr,
            stage2_lr=args.stage2_lr if args.stage2_lr is not None else 0.001,
            lambda_comm=args.lambda_comm if args.lambda_comm > 0 else 0.10,
            audit_batches=args.audit_batches,
        )
    elif args.c1h_consistency:
        results = run_c1h_consistency_partitioning(
            seeds=_parse_int_list(args.seeds),
            consistency_values=_parse_float_list(args.consistency_sweep),
            stage1_episodes=args.stage1_episodes,
            stage2_episodes=args.stage2_episodes,
            batch_size=args.batch_size,
            lr=args.lr,
            stage2_lr=args.stage2_lr if args.stage2_lr is not None else 0.001,
            lambda_comm=args.lambda_comm if args.lambda_comm > 0 else 0.10,
            audit_batches=args.audit_batches,
        )
    elif args.c1g_geometry_audit:
        results = run_c1g_geometry_audit(
            seeds=_parse_int_list(args.seeds),
            stage1_episodes=args.stage1_episodes,
            stage2_episodes=args.stage2_episodes,
            batch_size=args.batch_size,
            lr=args.lr,
            stage2_lr=args.stage2_lr if args.stage2_lr is not None else 0.001,
            lambda_comm=args.lambda_comm if args.lambda_comm > 0 else 0.10,
            audit_batches=args.audit_batches,
        )
    elif args.c1f_ablation:
        lambdas = _parse_float_list(args.lambda_sweep or "0.03,0.06,0.1")
        results = run_c1f_ablation(
            seeds=_parse_int_list(args.seeds),
            lambdas=lambdas,
            stage1_episodes=args.stage1_episodes,
            stage2_episodes=args.stage2_episodes,
            batch_size=args.batch_size,
            lr=args.lr,
            stage2_lr=args.stage2_lr if args.stage2_lr is not None else 0.001,
        )
    elif args.two_stage_compression:
        lambdas = _parse_float_list(args.lambda_sweep or "0.001,0.003,0.006,0.01")
        results = run_two_stage_compression(
            seeds=_parse_int_list(args.seeds),
            lambdas=lambdas,
            stage1_episodes=args.stage1_episodes,
            stage2_episodes=args.stage2_episodes,
            batch_size=args.batch_size,
            lr=args.lr,
            stage2_lr=args.stage2_lr,
            freeze_listener_stage2=args.freeze_listener_stage2,
        )
    elif args.multiseed_compression_sweep:
        lambdas = _parse_float_list(args.lambda_sweep or "0.0,0.001,0.003,0.006,0.01")
        results = run_multiseed_compression_sweep(
            seeds=_parse_int_list(args.seeds),
            lambdas=lambdas,
            episodes=args.episodes,
            batch_size=args.batch_size,
            lr=args.lr,
        )
    elif args.c1c_calibration:
        results = run_c1c_calibration(
            episodes=args.episodes,
            batch_size=args.batch_size,
            seed=args.seed,
            lr=args.lr,
            compressed_lambda=args.lambda_comm if args.lambda_comm > 0 else 0.001,
        )
    elif args.lambda_sweep:
        results = run_lambda_sweep(
            _parse_float_list(args.lambda_sweep),
            episodes=args.episodes,
            batch_size=args.batch_size,
            seed=args.seed,
            lr=args.lr,
        )
    else:
        baseline = "learned" if args.baseline == "high-bandwidth" else args.baseline
        lambda_comm = 0.0 if args.baseline == "high-bandwidth" else args.lambda_comm
        results = {
            args.baseline: run_train_eval(
                episodes=args.episodes,
                batch_size=args.batch_size,
                seed=args.seed,
                lr=args.lr,
                lambda_comm=lambda_comm,
                baseline=baseline,
                lambda_comm_distill=args.lambda_comm_distill,
                rule_mode=args.rule_mode,
            )
        }

    if args.table:
        print(format_table(results))
    else:
        for name, metrics in results.items():
            print(name, metrics)


if __name__ == "__main__":
    main()
