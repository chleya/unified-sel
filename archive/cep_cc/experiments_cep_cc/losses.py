from __future__ import annotations

import torch
from torch.nn import functional as F


def task_loss(logits: torch.Tensor, target_index: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, target_index)


def communication_energy(comm: torch.Tensor) -> torch.Tensor:
    return comm.pow(2).mean()


def communication_sparsity(comm: torch.Tensor) -> torch.Tensor:
    return comm.abs().mean()


def communication_smoothness(comm: torch.Tensor) -> torch.Tensor:
    if comm.shape[1] <= 1:
        return torch.zeros((), device=comm.device)
    return (comm[:, 1:, :] - comm[:, :-1, :]).pow(2).mean()


def effective_dimension(values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    flat = values.flatten(start_dim=1)
    centered = flat - flat.mean(dim=0, keepdim=True)
    cov = centered.T @ centered / max(flat.shape[0] - 1, 1)
    eigvals = torch.linalg.eigvalsh(cov).clamp_min(0.0)
    total = eigvals.sum()
    if float(total.detach().cpu()) <= eps:
        return values.new_zeros(())
    return total.pow(2) / eigvals.pow(2).sum().clamp_min(eps)


def state_l1(*states: torch.Tensor) -> torch.Tensor:
    if not states:
        raise ValueError("state_l1 requires at least one state tensor")
    return sum(state.abs().mean() for state in states) / len(states)


def communication_consistency(comm: torch.Tensor, target_index: torch.Tensor) -> torch.Tensor:
    flat = comm.flatten(start_dim=1)
    losses = []
    for target in target_index.unique():
        mask = target_index == target
        if int(mask.sum().detach().cpu()) < 2:
            continue
        group = flat[mask]
        center = group.mean(dim=0, keepdim=True)
        losses.append((group - center).pow(2).mean())
    if not losses:
        return comm.new_zeros(())
    return torch.stack(losses).mean()


def factor_bin_consistency(
    comm: torch.Tensor,
    factor_values: torch.Tensor | None,
    n_bins: int = 4,
) -> torch.Tensor:
    if factor_values is None:
        return comm.new_zeros(())
    if factor_values.ndim > 1:
        losses = [
            factor_bin_consistency(comm, factor_values[:, idx], n_bins=n_bins)
            for idx in range(factor_values.shape[1])
        ]
        if not losses:
            return comm.new_zeros(())
        return torch.stack(losses).mean()
    flat = comm.flatten(start_dim=1)
    values = factor_values.detach().flatten()
    if values.numel() != flat.shape[0] or flat.shape[0] < n_bins:
        return comm.new_zeros(())
    quantiles = torch.linspace(0.0, 1.0, n_bins + 1, device=values.device)
    edges = torch.quantile(values, quantiles)
    losses = []
    for idx in range(n_bins):
        if idx == n_bins - 1:
            mask = (values >= edges[idx]) & (values <= edges[idx + 1])
        else:
            mask = (values >= edges[idx]) & (values < edges[idx + 1])
        if int(mask.sum().detach().cpu()) < 2:
            continue
        group = flat[mask]
        center = group.mean(dim=0, keepdim=True)
        losses.append((group - center).pow(2).mean())
    if not losses:
        return comm.new_zeros(())
    return torch.stack(losses).mean()


def total_loss(
    logits: torch.Tensor,
    target_index: torch.Tensor,
    comm: torch.Tensor,
    speaker_state: torch.Tensor,
    listener_state: torch.Tensor,
    lambda_energy: float = 0.0,
    lambda_sparse: float = 0.0,
    lambda_smooth: float = 0.0,
    lambda_dimeff: float = 0.0,
    lambda_state: float = 0.0,
    target_comm: torch.Tensor | None = None,
    lambda_comm_distill: float = 0.0,
    lambda_consistency: float = 0.0,
    factor_values: torch.Tensor | None = None,
    lambda_factor_consistency: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    losses = {
        "task_loss": task_loss(logits, target_index),
        "comm_energy": communication_energy(comm),
        "comm_sparse": communication_sparsity(comm),
        "comm_smooth": communication_smoothness(comm),
        "comm_dimeff": effective_dimension(comm),
        "comm_consistency": communication_consistency(comm, target_index),
        "factor_consistency": factor_bin_consistency(comm, factor_values),
        "state_l1": state_l1(speaker_state, listener_state),
    }
    if target_comm is not None:
        losses["comm_distill"] = F.mse_loss(comm, target_comm)
    else:
        losses["comm_distill"] = comm.new_zeros(())
    total = (
        losses["task_loss"]
        + lambda_energy * losses["comm_energy"]
        + lambda_sparse * losses["comm_sparse"]
        + lambda_smooth * losses["comm_smooth"]
        + lambda_dimeff * losses["comm_dimeff"]
        + lambda_consistency * losses["comm_consistency"]
        + lambda_factor_consistency * losses["factor_consistency"]
        + lambda_state * losses["state_l1"]
        + lambda_comm_distill * losses["comm_distill"]
    )
    return total, {name: float(value.detach().cpu()) for name, value in losses.items()}
