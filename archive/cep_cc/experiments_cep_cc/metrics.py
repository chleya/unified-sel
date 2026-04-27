from __future__ import annotations

import math

import torch

from .losses import communication_energy, communication_sparsity, effective_dimension


def accuracy(logits: torch.Tensor, target_index: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return float((pred == target_index).float().mean().detach().cpu())


def _kmeans(data: torch.Tensor, k: int = 8, steps: int = 12) -> tuple[torch.Tensor, torch.Tensor]:
    if data.shape[0] < k:
        k = max(1, data.shape[0])
    centers = data[:k].clone()
    labels = torch.zeros(data.shape[0], dtype=torch.long, device=data.device)
    for _ in range(steps):
        distances = torch.cdist(data, centers)
        labels = distances.argmin(dim=1)
        new_centers = []
        for idx in range(k):
            mask = labels == idx
            if mask.any():
                new_centers.append(data[mask].mean(dim=0))
            else:
                new_centers.append(centers[idx])
        centers = torch.stack(new_centers, dim=0)
    return labels, centers


def prototype_metrics(comm: torch.Tensor, target_index: torch.Tensor | None = None, k: int = 8) -> dict[str, float]:
    flat = comm.flatten(start_dim=1).detach()
    if flat.shape[0] < 2:
        return {"prototype_reuse_rate": 0.0, "cluster_compactness": 0.0, "target_cluster_alignment": 0.0}
    if float(flat.var().detach().cpu()) <= 1e-10:
        return {"prototype_reuse_rate": 1.0, "cluster_compactness": 0.0, "target_cluster_alignment": 0.0}
    labels, centers = _kmeans(flat, k=min(k, flat.shape[0]))
    counts = torch.bincount(labels, minlength=centers.shape[0]).float()
    reuse_rate = float((counts.max() / counts.sum().clamp_min(1.0)).cpu())
    assigned = centers[labels]
    within = torch.linalg.norm(flat - assigned, dim=1).mean()
    center_dist = torch.cdist(centers, centers)
    between = center_dist[center_dist > 0].mean() if (center_dist > 0).any() else torch.ones((), device=flat.device)
    compactness = float((between / within.clamp_min(1e-6)).detach().cpu())

    alignment = 0.0
    if target_index is not None:
        matches = []
        for label in labels.unique():
            mask = labels == label
            majority = torch.mode(target_index[mask]).values
            matches.append((target_index[mask] == majority).float().mean())
        alignment = float(torch.stack(matches).mean().detach().cpu()) if matches else 0.0
    return {
        "prototype_reuse_rate": reuse_rate,
        "cluster_compactness": compactness,
        "target_cluster_alignment": alignment,
    }


def summarize_batch(logits: torch.Tensor, target_index: torch.Tensor, comm: torch.Tensor) -> dict[str, float]:
    summary = {
        "task_accuracy": accuracy(logits, target_index),
        "comm_energy": float(communication_energy(comm).detach().cpu()),
        "comm_l1": float(communication_sparsity(comm).detach().cpu()),
        "comm_effective_dim": float(effective_dimension(comm).detach().cpu()),
    }
    summary.update(prototype_metrics(comm, target_index=target_index))
    return summary


def _ridge_predict_score(features: torch.Tensor, targets: torch.Tensor, ridge: float = 1e-3) -> float:
    x = features.detach()
    y = targets.detach()
    x = torch.cat([x, torch.ones(x.shape[0], 1, device=x.device)], dim=1)
    train_count = max(4, int(x.shape[0] * 0.7))
    x_train = x[:train_count]
    y_train = y[:train_count]
    x_test = x[train_count:]
    y_test = y[train_count:]
    if x_test.numel() == 0:
        return 0.0
    eye = torch.eye(x_train.shape[1], device=x.device)
    weights = torch.linalg.solve(x_train.T @ x_train + ridge * eye, x_train.T @ y_train)
    pred = x_test @ weights
    mse = (pred - y_test).pow(2).mean()
    baseline = (y_test - y_train.mean(dim=0, keepdim=True)).pow(2).mean().clamp_min(1e-8)
    return float((1.0 - mse / baseline).detach().cpu())


def geometry_audit_metrics(
    comm: torch.Tensor,
    target_index: torch.Tensor,
    latent_factors: torch.Tensor,
    object_states: torch.Tensor,
    k: int = 8,
) -> dict[str, float]:
    flat = comm.flatten(start_dim=1).detach()
    if flat.shape[0] < 4 or float(flat.var().detach().cpu()) <= 1e-10:
        return {
            "audit_target_purity": 0.0,
            "audit_within_between_ratio": 0.0,
            "audit_nearest_proto_stability": 0.0,
            "audit_latent_probe_r2": 0.0,
            "audit_pos_probe_r2": 0.0,
            "audit_q_probe_r2": 0.0,
            "audit_motion_probe_r2": 0.0,
            "audit_relation_probe_r2": 0.0,
            "audit_hidden_q_probe_r2": 0.0,
        }

    labels, centers = _kmeans(flat, k=min(k, flat.shape[0]))
    purities = []
    for label in labels.unique():
        mask = labels == label
        majority = torch.mode(target_index[mask]).values
        purities.append((target_index[mask] == majority).float().mean())
    target_purity = float(torch.stack(purities).mean().detach().cpu()) if purities else 0.0

    pairwise = torch.cdist(flat, flat)
    same = target_index.unsqueeze(0) == target_index.unsqueeze(1)
    eye = torch.eye(flat.shape[0], dtype=torch.bool, device=flat.device)
    within_mask = same & ~eye
    between_mask = ~same
    within = pairwise[within_mask].mean() if within_mask.any() else torch.zeros((), device=flat.device)
    between = pairwise[between_mask].mean() if between_mask.any() else torch.ones((), device=flat.device)
    within_between_ratio = float((within / between.clamp_min(1e-8)).detach().cpu())

    noisy = flat + 0.03 * flat.std().clamp_min(1e-6) * torch.randn_like(flat)
    labels_noisy = torch.cdist(noisy, centers).argmin(dim=1)
    nearest_stability = float((labels_noisy == labels).float().mean().detach().cpu())

    batch_indices = torch.arange(target_index.shape[0], device=target_index.device)
    target_latents = latent_factors[batch_indices, target_index]
    latent_probe = _ridge_predict_score(flat, target_latents)
    target_q = object_states[batch_indices, target_index, 4:5]
    q_probe = _ridge_predict_score(flat, target_q)
    return {
        "audit_target_purity": target_purity,
        "audit_within_between_ratio": within_between_ratio,
        "audit_nearest_proto_stability": nearest_stability,
        "audit_latent_probe_r2": latent_probe,
        "audit_pos_probe_r2": _ridge_predict_score(flat, target_latents[:, 0:1]),
        "audit_q_probe_r2": _ridge_predict_score(flat, target_latents[:, 1:2]),
        "audit_motion_probe_r2": _ridge_predict_score(flat, target_latents[:, 2:3]),
        "audit_relation_probe_r2": _ridge_predict_score(flat, target_latents[:, 3:4]),
        "audit_hidden_q_probe_r2": q_probe,
    }


def segment_audit_metrics(
    comm: torch.Tensor,
    target_index: torch.Tensor,
    latent_factors: torch.Tensor,
    object_states: torch.Tensor,
) -> dict[str, float]:
    if comm.shape[1] < 3:
        return {}
    batch_indices = torch.arange(target_index.shape[0], device=target_index.device)
    target_latents = latent_factors[batch_indices, target_index]
    target_q = object_states[batch_indices, target_index, 4:5]
    names = ["early", "middle", "late"]
    chunks = torch.chunk(comm, chunks=3, dim=1)
    metrics: dict[str, float] = {}
    for name, chunk in zip(names, chunks):
        flat = chunk.flatten(start_dim=1)
        metrics[f"segment_{name}_latent_probe_r2"] = _ridge_predict_score(flat, target_latents)
        metrics[f"segment_{name}_pos_probe_r2"] = _ridge_predict_score(flat, target_latents[:, 0:1])
        metrics[f"segment_{name}_q_score_probe_r2"] = _ridge_predict_score(flat, target_latents[:, 1:2])
        metrics[f"segment_{name}_motion_probe_r2"] = _ridge_predict_score(flat, target_latents[:, 2:3])
        metrics[f"segment_{name}_relation_probe_r2"] = _ridge_predict_score(flat, target_latents[:, 3:4])
        metrics[f"segment_{name}_q_probe_r2"] = _ridge_predict_score(flat, target_q)
        metrics[f"segment_{name}_target_purity"] = prototype_metrics(chunk, target_index=target_index)[
            "target_cluster_alignment"
        ]
    q_scores = [metrics[f"segment_{name}_q_probe_r2"] for name in names]
    metrics["segment_q_specialization_gap"] = max(q_scores) - min(q_scores)
    latent_scores = [metrics[f"segment_{name}_latent_probe_r2"] for name in names]
    metrics["segment_latent_specialization_gap"] = max(latent_scores) - min(latent_scores)
    for factor in ["pos", "q_score", "motion", "relation"]:
        factor_scores = [metrics[f"segment_{name}_{factor}_probe_r2"] for name in names]
        metrics[f"segment_{factor}_specialization_gap"] = max(factor_scores) - min(factor_scores)
    return metrics


def paired_nuisance_alignment_metrics(
    clean_comm: torch.Tensor,
    nuisance_comm: torch.Tensor,
    clean_target: torch.Tensor,
    nuisance_target: torch.Tensor,
    clean_latent_factors: torch.Tensor,
    nuisance_latent_factors: torch.Tensor,
    clean_object_states: torch.Tensor,
    nuisance_object_states: torch.Tensor,
    k: int = 8,
) -> dict[str, float]:
    clean_flat = clean_comm.flatten(start_dim=1).detach()
    nuisance_flat = nuisance_comm.flatten(start_dim=1).detach()
    if clean_flat.shape != nuisance_flat.shape or clean_flat.shape[0] < 4:
        return {
            "paired_target_agreement": 0.0,
            "paired_proto_assignment_stability": 0.0,
            "paired_comm_distance_ratio": 0.0,
            "paired_hidden_q_corr": 0.0,
            "paired_latent_corr": 0.0,
            "paired_pos_factor_corr": 0.0,
            "paired_q_score_factor_corr": 0.0,
            "paired_motion_factor_corr": 0.0,
            "paired_relation_factor_corr": 0.0,
            "paired_invariant_factor_corr": 0.0,
            "paired_equivariant_factor_corr": 0.0,
        }

    labels, centers = _kmeans(clean_flat, k=min(k, clean_flat.shape[0]))
    nuisance_labels = torch.cdist(nuisance_flat, centers).argmin(dim=1)
    label_stability = float((labels == nuisance_labels).float().mean().detach().cpu())

    paired_distance = torch.linalg.norm(clean_flat - nuisance_flat, dim=1).mean()
    clean_pairwise = torch.cdist(clean_flat, clean_flat)
    eye = torch.eye(clean_flat.shape[0], dtype=torch.bool, device=clean_flat.device)
    baseline_distance = clean_pairwise[~eye].mean() if (~eye).any() else clean_flat.new_ones(())
    distance_ratio = float((paired_distance / baseline_distance.clamp_min(1e-8)).detach().cpu())

    target_agreement = float((clean_target == nuisance_target).float().mean().detach().cpu())
    batch_indices = torch.arange(clean_target.shape[0], device=clean_target.device)
    clean_q = clean_object_states[batch_indices, clean_target, 4]
    nuisance_q = nuisance_object_states[batch_indices, nuisance_target, 4]
    clean_latents = clean_latent_factors[batch_indices, clean_target]
    nuisance_latents = nuisance_latent_factors[batch_indices, nuisance_target]

    def _corr(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left = left.flatten()
        right = right.flatten()
        left = left - left.mean()
        right = right - right.mean()
        denom = left.norm() * right.norm()
        if float(denom.detach().cpu()) <= 1e-8:
            return left.new_zeros(())
        return (left * right).sum() / denom

    latent_corrs = [_corr(clean_latents[:, idx], nuisance_latents[:, idx]) for idx in range(clean_latents.shape[1])]
    invariant_factor_corr = latent_corrs[1]
    equivariant_factor_corr = torch.stack([latent_corrs[0], latent_corrs[2], latent_corrs[3]]).mean()
    return {
        "paired_target_agreement": target_agreement,
        "paired_proto_assignment_stability": label_stability,
        "paired_comm_distance_ratio": distance_ratio,
        "paired_hidden_q_corr": float(_corr(clean_q, nuisance_q).detach().cpu()),
        "paired_latent_corr": float(torch.stack(latent_corrs).mean().detach().cpu()),
        "paired_pos_factor_corr": float(latent_corrs[0].detach().cpu()),
        "paired_q_score_factor_corr": float(latent_corrs[1].detach().cpu()),
        "paired_motion_factor_corr": float(latent_corrs[2].detach().cpu()),
        "paired_relation_factor_corr": float(latent_corrs[3].detach().cpu()),
        "paired_invariant_factor_corr": float(invariant_factor_corr.detach().cpu()),
        "paired_equivariant_factor_corr": float(equivariant_factor_corr.detach().cpu()),
    }


def format_table(results: dict[str, dict[str, float]]) -> str:
    keys = [
        "task_accuracy",
        "train_task_accuracy",
        "comm_energy",
        "comm_l1",
        "comm_effective_dim",
        "prototype_reuse_rate",
        "cluster_compactness",
        "target_cluster_alignment",
        "delta_task_accuracy",
        "delta_comm_energy",
        "delta_comm_l1",
        "delta_comm_effective_dim",
        "delta_target_cluster_alignment",
        "audit_target_purity",
        "audit_within_between_ratio",
        "audit_nearest_proto_stability",
        "audit_latent_probe_r2",
        "audit_pos_probe_r2",
        "audit_q_probe_r2",
        "audit_motion_probe_r2",
        "audit_relation_probe_r2",
        "audit_hidden_q_probe_r2",
        "consistency_lambda",
        "factor_consistency_lambda",
        "factor_set_size",
        "factor_set_has_q",
        "factor_set_has_motion",
        "factor_set_has_relation",
        "segment_early_latent_probe_r2",
        "segment_middle_latent_probe_r2",
        "segment_late_latent_probe_r2",
        "segment_early_pos_probe_r2",
        "segment_middle_pos_probe_r2",
        "segment_late_pos_probe_r2",
        "segment_early_q_score_probe_r2",
        "segment_middle_q_score_probe_r2",
        "segment_late_q_score_probe_r2",
        "segment_early_motion_probe_r2",
        "segment_middle_motion_probe_r2",
        "segment_late_motion_probe_r2",
        "segment_early_relation_probe_r2",
        "segment_middle_relation_probe_r2",
        "segment_late_relation_probe_r2",
        "segment_early_q_probe_r2",
        "segment_middle_q_probe_r2",
        "segment_late_q_probe_r2",
        "segment_q_specialization_gap",
        "segment_latent_specialization_gap",
        "segment_pos_specialization_gap",
        "segment_q_score_specialization_gap",
        "segment_motion_specialization_gap",
        "segment_relation_specialization_gap",
        "segment_early_target_purity",
        "segment_middle_target_purity",
        "segment_late_target_purity",
        "segment_early_swap_action_change_rate",
        "segment_middle_swap_action_change_rate",
        "segment_late_swap_action_change_rate",
        "segment_early_swap_target_logit_drop",
        "segment_middle_swap_target_logit_drop",
        "segment_late_swap_target_logit_drop",
        "segment_early_ablation_action_change_rate",
        "segment_middle_ablation_action_change_rate",
        "segment_late_ablation_action_change_rate",
        "segment_early_ablation_target_logit_drop",
        "segment_middle_ablation_target_logit_drop",
        "segment_late_ablation_target_logit_drop",
        "rule_mode_compositional",
        "rule_mode_temporal_memory",
        "rule_mode_motion_pressure",
        "rule_mode_motion_decoupled",
        "bootstrap_curriculum",
        "bootstrap_segment_dropout",
        "segment_dropout_prob",
        "train_segment_dropout_prob",
        "nuisance_is_mirror_x",
        "nuisance_is_rotate90",
        "nuisance_is_velocity_scale",
        "delta_clean_task_accuracy",
        "delta_clean_audit_hidden_q_probe_r2",
        "delta_clean_segment_late_swap_action_change_rate",
        "delta_clean_segment_late_ablation_action_change_rate",
        "delta_clean_segment_late_ablation_target_logit_drop",
        "paired_target_agreement",
        "paired_proto_assignment_stability",
        "paired_comm_distance_ratio",
        "paired_hidden_q_corr",
        "paired_latent_corr",
        "paired_pos_factor_corr",
        "paired_q_score_factor_corr",
        "paired_motion_factor_corr",
        "paired_relation_factor_corr",
        "paired_invariant_factor_corr",
        "paired_equivariant_factor_corr",
        "intervention_best_targeted_effect",
        "intervention_best_offtarget_effect",
        "intervention_best_targeted_ratio",
        "intervention_q_best_ratio",
        "intervention_motion_best_ratio",
        "intervention_relation_best_ratio",
        "intervention_q_best_segment",
        "intervention_motion_best_segment",
        "intervention_relation_best_segment",
    ]
    header = ["run", *keys]
    rows = [" | ".join(header), " | ".join(["---", *["---:" for _ in keys]])]
    for name, metrics in results.items():
        row = [name]
        for key in keys:
            value = metrics.get(key, math.nan)
            row.append(f"{value:.3f}")
        rows.append(" | ".join(row))
    return "\n".join(rows)
