"""
core/learner.py — Archived UnifiedSELClassifier

This module is kept for backward compatibility (smoke tests).
It wraps StructurePool with a unified classifier interface including
DFA readout, anchor-based regularization, and snapshot experts.

NOTE: The "surprise", "tension", and "utility" terminology is legacy
nomenclature from the original SEL-Lab project.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from core.pool import StructurePool


class UnifiedSELClassifier:
    def __init__(
        self,
        in_size: int,
        out_size: int,
        lr: float = 0.05,
        max_structures: int = 20,
        evolve_every: int = 20,
        pool_config: Optional[Dict] = None,
        seed: Optional[int] = None,
        ewc_lambda: float = 0.0,
        readout_mode: str = "shared",
        shared_readout_scale: float = 1.0,
        shared_readout_post_checkpoint_scale: float = 1.0,
        local_readout_lr_scale: float = 1.0,
        local_readout_start_step: int = 0,
        local_readout_surprise_threshold: Optional[float] = None,
        local_readout_young_age_max: Optional[int] = None,
        local_readout_training_events: Optional[List[str]] = None,
        local_readout_inference_surprise_threshold: Optional[float] = None,
        local_readout_episode_events: Optional[List[str]] = None,
        local_readout_episode_window_steps: int = 0,
        local_readout_pressure_window_steps: int = 0,
        anchor_lambda: float = 0.0,
    ):
        self.in_size = in_size
        self.out_size = out_size
        self.lr = lr
        self.evolve_every = evolve_every
        self.step = 0
        self.ewc_lambda = ewc_lambda
        self.anchor_lambda = anchor_lambda
        self.readout_mode = readout_mode
        self.shared_readout_scale = shared_readout_scale
        self.shared_readout_post_checkpoint_scale = shared_readout_post_checkpoint_scale
        self.local_readout_lr_scale = local_readout_lr_scale
        self.local_readout_start_step = local_readout_start_step
        self.local_readout_surprise_threshold = local_readout_surprise_threshold
        self.local_readout_young_age_max = local_readout_young_age_max
        self.local_readout_training_events = (
            set(local_readout_training_events) if local_readout_training_events else None
        )
        self.local_readout_inference_surprise_threshold = local_readout_inference_surprise_threshold
        self.local_readout_episode_events = (
            set(local_readout_episode_events) if local_readout_episode_events else None
        )
        self.local_readout_episode_window_steps = local_readout_episode_window_steps
        self.local_readout_pressure_window_steps = local_readout_pressure_window_steps
        self._local_pressure_steps_remaining = 0
        self._local_pressure_structure_id: Optional[int] = None
        if self.readout_mode not in {"shared", "hybrid_local", "exclusive_local"}:
            raise ValueError(f"Unsupported readout_mode: {self.readout_mode}")

        pool_kwargs = dict(pool_config or {})
        pool_kwargs["max_structures"] = max_structures
        self.pool = StructurePool(
            in_size=in_size,
            out_size=out_size,
            initial_structures=1,
            seed=seed,
            **pool_kwargs,
        )

        rng = np.random.default_rng(seed)
        self.W_out = rng.normal(0.0, 0.1, size=(out_size, out_size))
        self.W_out_fisher = np.zeros_like(self.W_out)
        self.W_out_anchor = self.W_out.copy()
        self.fisher_estimated = False
        self.W_out_memory: Optional[np.ndarray] = None
        self.dual_path_alpha: float = 0.5
        self.dual_path_active: bool = False
        self._snapshot_experts: List[Dict] = []
        self._snapshot_confidence_ratio_threshold: float = 0.5
        self._history: List[Dict] = []
        self._pool_frozen: bool = False

    def _local_readout_active(self) -> bool:
        if self.readout_mode == "exclusive_local":
            return True
        return self.readout_mode == "hybrid_local" and self.step >= self.local_readout_start_step

    def _compute_local_gate(
        self,
        routed_structure,
        surprise: Optional[float],
        stage: str,
        event: Optional[str] = None,
    ) -> tuple[bool, str]:
        if not self._local_readout_active() or routed_structure is None:
            return False, "inactive"

        reasons: List[str] = []
        if self.local_readout_pressure_window_steps > 0:
            if (
                self._local_pressure_steps_remaining <= 0
                or self._local_pressure_structure_id != int(routed_structure.id)
            ):
                return False, "pressure_blocked"
            reasons.append("pressure")
        if (
            self.local_readout_episode_window_steps > 0
            and routed_structure.local_readout_episode_steps_remaining <= 0
        ):
            return False, "episode_blocked"
        if (
            stage == "train"
            and self.local_readout_training_events is not None
            and event not in self.local_readout_training_events
        ):
            return False, "event_blocked"

        if (
            stage == "predict"
            and self.local_readout_inference_surprise_threshold is not None
        ):
            if surprise is None or surprise < self.local_readout_inference_surprise_threshold:
                return False, "predict_surprise_blocked"
            reasons.append("predict_surprise")

        if (
            self.local_readout_surprise_threshold is not None
            and surprise is not None
            and surprise >= self.local_readout_surprise_threshold
        ):
            reasons.append("surprise")
        if (
            self.local_readout_young_age_max is not None
            and routed_structure.age <= self.local_readout_young_age_max
        ):
            reasons.append("young")

        if not reasons and (
            self.local_readout_surprise_threshold is None
            and self.local_readout_young_age_max is None
        ):
            return True, "always"
        if not reasons:
            return False, "blocked"
        return True, "+".join(reasons)

    def _activate_pressure_local_window(self, routed_structure_id: int) -> None:
        if self.local_readout_pressure_window_steps <= 0:
            return
        self._local_pressure_structure_id = int(routed_structure_id)
        self._local_pressure_steps_remaining = max(
            self._local_pressure_steps_remaining,
            self.local_readout_pressure_window_steps,
        )

    def _compute_output(
        self,
        x: np.ndarray,
        pooled_hidden: np.ndarray,
        active_structure=None,
        surprise: Optional[float] = None,
        stage: str = "predict",
        event: Optional[str] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[int], bool, str]:
        use_exclusive_local = self.readout_mode == "exclusive_local"
        shared_output = np.zeros(self.out_size, dtype=float)
        if not use_exclusive_local:
            shared_output = self.shared_readout_scale * self._dual_path_output(pooled_hidden)
        local_output = np.zeros(self.out_size, dtype=float)
        routed_structure = active_structure
        local_gate_active = False
        local_gate_reason = "inactive"

        if routed_structure is None and self._local_readout_active():
            routed_structure = self.pool.select_best_structure(np.atleast_2d(x))
            if routed_structure is not None:
                surprise = float(routed_structure.current_surprise(x))

        if routed_structure is not None:
            if use_exclusive_local:
                local_gate_active, local_gate_reason = True, "exclusive"
            else:
                local_gate_active, local_gate_reason = self._compute_local_gate(
                    routed_structure=routed_structure,
                    surprise=surprise,
                    stage=stage,
                    event=event,
                )
            if local_gate_active:
                routed_hidden = routed_structure.forward(x).flatten()
                local_output = routed_structure.readout(routed_hidden).flatten()

        routed_structure_id = None if routed_structure is None else int(routed_structure.id)
        return (
            shared_output + local_output,
            shared_output,
            local_output,
            routed_structure_id,
            local_gate_active,
            local_gate_reason,
        )

    def activate_dual_path(self, alpha: float = 0.5) -> None:
        self.W_out_memory = self.W_out.copy()
        self.dual_path_alpha = alpha
        self.dual_path_active = True

    def snapshot_expert(self, confidence_ratio_threshold: float = 0.5) -> None:
        """Deep-copy current model state as a snapshot expert."""
        import copy
        self._snapshot_confidence_ratio_threshold = confidence_ratio_threshold
        snapshot = {
            "W_out": self.W_out.copy(),
            "structures": [],
        }
        for s in self.pool.structures:
            snapshot["structures"].append({
                "id": s.id,
                "weights": s.weights.copy(),
                "feedback": s.feedback.copy(),
                "local_readout": s.local_readout.copy() if s.local_readout is not None else None,
                "utility": s.utility,
            })
        self._snapshot_experts.append(snapshot)

    def freeze_pool(self) -> None:
        """Freeze the structure pool to prevent creation and pruning."""
        self._pool_frozen = True
        self.pool.frozen = True

    def freeze_pool_prune_only(self) -> None:
        """Freeze only pruning, allow new structure creation."""
        self._pool_frozen = True
        self.pool.frozen_prune_only = True

    def _predict_with_snapshot(self, x: np.ndarray, snapshot: Dict) -> np.ndarray:
        """Predict using a snapshot expert's frozen state. Returns probabilities."""
        x_flat = x.flatten()
        outputs = []
        utilities = []
        for s_data in snapshot["structures"]:
            w = s_data["weights"]
            h = np.tanh(x_flat @ w)
            outputs.append(h)
            utilities.append(s_data.get("utility", 1.0))
        if not outputs:
            probs = np.ones(self.out_size) / self.out_size
            return probs
        # 使用 utility 加权平均，与 pool.forward 一致
        weight_array = np.asarray(utilities, dtype=float)
        weight_array = weight_array / (weight_array.sum() + 1e-8)
        pooled = np.sum([w * o for w, o in zip(weight_array, outputs)], axis=0)
        output = (pooled @ snapshot["W_out"]).flatten()
        output -= output.max()
        exp_out = np.exp(output)
        probs = exp_out / (exp_out.sum() + 1e-8)
        return probs

    def _ensemble_predict(self, x: np.ndarray) -> np.ndarray:
        """Combine current model + snapshot experts via prediction disagreement routing.

        When the current model and snapshot expert disagree (predict different classes),
        pick the one with higher confidence. When they agree, use either.

        This is better than surprise-gated routing because:
        - Surprise signal has almost no task-discrimination power (gap = 0.006)
        - Prediction disagreement directly measures "which expert knows better"
        - If snapshot predicts class A with 0.9 confidence and current predicts class B
          with 0.6 confidence, snapshot is likely correct for this input
        """
        current_output = self.predict_proba_single(x)
        if not self._snapshot_experts:
            return current_output

        snap_outputs = [self._predict_with_snapshot(x, s) for s in self._snapshot_experts]
        best_snap = max(snap_outputs, key=lambda p: float(np.max(p)))

        current_class = np.argmax(current_output)
        snap_class = np.argmax(best_snap)

        if current_class == snap_class:
            return current_output if float(np.max(current_output)) >= float(np.max(best_snap)) else best_snap

        current_conf = float(np.max(current_output))
        snap_conf = float(np.max(best_snap))

        if snap_conf > current_conf * self._snapshot_confidence_ratio_threshold:
            return best_snap
        return current_output

    def _dual_path_output(self, pooled_hidden: np.ndarray) -> np.ndarray:
        """Combine memory path and current path outputs."""
        if not self.dual_path_active or self.W_out_memory is None:
            return (pooled_hidden @ self.W_out).flatten()
        memory_output = (pooled_hidden @ self.W_out_memory).flatten()
        current_output = (pooled_hidden @ self.W_out).flatten()
        alpha = self.dual_path_alpha
        return alpha * memory_output + (1 - alpha) * current_output

    def _compute_route_views(
        self,
        x: np.ndarray,
        active_structure,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Return active/pool hidden views plus route-gap diagnostics."""
        active_hidden = active_structure.forward(x).flatten()
        pooled_hidden = self.pool.forward(np.atleast_2d(x)).flatten()
        route_l2 = float(np.linalg.norm(active_hidden - pooled_hidden))

        denom = float(np.linalg.norm(active_hidden) * np.linalg.norm(pooled_hidden))
        if denom > 1e-8:
            route_cosine = float(np.dot(active_hidden, pooled_hidden) / denom)
        else:
            route_cosine = 1.0

        return active_hidden, pooled_hidden, route_l2, route_cosine

    def estimate_w_out_fisher(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.readout_mode == "exclusive_local":
            self.W_out_fisher = np.zeros_like(self.W_out)
            self.W_out_anchor = self.W_out.copy()
            self.fisher_estimated = False
            return
        fisher = np.zeros_like(self.W_out)
        for i in range(len(X)):
            x_i = X[i].flatten()
            hidden = self.pool.forward(np.atleast_2d(x_i))
            output, _, _, _, _, _ = self._compute_output(x_i, hidden, stage="predict")
            y_i = np.atleast_1d(y[i]).flatten()
            if y_i.shape[0] == 1:
                target = np.zeros(self.out_size)
                target[int(y_i[0])] = 1.0
            else:
                target = y_i.astype(float)
            error = target - output
            grad = self.shared_readout_scale * np.outer(hidden.flatten(), error)
            fisher += grad ** 2
        fisher /= max(len(X), 1)
        self.W_out_fisher = np.clip(fisher, 0.0, 5.0)
        self.W_out_anchor = self.W_out.copy()
        self.fisher_estimated = True

    def fit_one(self, x: np.ndarray, y: np.ndarray) -> float:
        x = x.flatten()
        y = np.atleast_1d(y).flatten()

        # 提取标签（假设是单标签分类）
        label = int(y[0]) if y.shape[0] == 1 else int(np.argmax(y))
        
        observation = self.pool.observe(x, label, self.out_size)
        active = observation["active_structure"]
        effective_lr = self.lr
        surprise = float(observation["surprise"])
        if (
            self.local_readout_pressure_window_steps > 0
            and (
                observation.get("pressure_active", False)
                or observation["event"] == "boundary_stabilize"
            )
        ):
            self._activate_pressure_local_window(active.id)
        if (
            self.local_readout_episode_events is not None
            and observation["event"] in self.local_readout_episode_events
            and self.local_readout_episode_window_steps > 0
        ):
            active.local_readout_episode_steps_remaining = max(
                active.local_readout_episode_steps_remaining,
                self.local_readout_episode_window_steps,
            )

        active_hidden, pooled_hidden, route_l2, route_cosine = self._compute_route_views(x, active)
        output, shared_output, local_output, routed_structure_id, local_gate_active, local_gate_reason = self._compute_output(
            x,
            pooled_hidden,
            active_structure=active,
            surprise=surprise,
            stage="train",
            event=observation["event"],
        )

        if y.shape[0] == 1:
            target = np.zeros(self.out_size)
            target[int(y[0])] = 1.0
        else:
            target = y

        error = target - output
        loss = self.pool.learn_active(
            active_structure=active,
            x=np.atleast_2d(x),
            output_error=np.atleast_2d(error),
            lr=effective_lr,
            anchor_lambda=self.anchor_lambda,
        )
        if local_gate_active:
            active.learn_readout(
                active_hidden,
                error,
                lr=effective_lr * self.local_readout_lr_scale,
            )

        shared_update_scale = 1.0
        if self.readout_mode == "exclusive_local":
            shared_update_scale = 0.0
        elif local_gate_active and self.step >= self.local_readout_start_step:
            shared_update_scale = self.shared_readout_post_checkpoint_scale

        w_out_grad = (
            effective_lr
            * shared_update_scale
            * self.shared_readout_scale
            * np.outer(pooled_hidden, error)
        )
        if self.fisher_estimated and self.ewc_lambda > 0:
            fisher_penalty = self.ewc_lambda * self.W_out_fisher * (self.W_out - self.W_out_anchor)
            w_out_grad -= fisher_penalty
        if shared_update_scale > 0.0 or np.any(w_out_grad):
            self.W_out += w_out_grad
            self.W_out = np.clip(self.W_out, -2.0, 2.0)
        self.pool.commit_pending_reinforcements()

        self.step += 1
        if not self._pool_frozen and self.step % self.evolve_every == 0:
            self.pool.evolve()
        if self.local_readout_episode_window_steps > 0:
            for structure in self.pool.structures:
                if structure.local_readout_episode_steps_remaining > 0:
                    structure.local_readout_episode_steps_remaining -= 1
        if self.local_readout_pressure_window_steps > 0 and self._local_pressure_steps_remaining > 0:
            self._local_pressure_steps_remaining -= 1
            if self._local_pressure_steps_remaining <= 0:
                self._local_pressure_structure_id = None

        self._history.append(
            {
                "step": self.step,
                "loss": float(loss),
                "event": observation["event"],
                "surprise": surprise,
                "n_structures": int(observation["n_structures"]),
                "pressure_active": bool(observation.get("pressure_active", False)),
                "persistent_pressure": bool(observation.get("persistent_pressure", False)),
                "high_pressure": bool(observation.get("high_pressure", False)),
                "can_boundary_stabilize": bool(observation.get("can_boundary_stabilize", False)),
                "route_l2": route_l2,
                "route_cosine": route_cosine,
                "shared_output_norm": float(np.linalg.norm(shared_output)),
                "local_output_norm": float(np.linalg.norm(local_output)),
                "routed_structure_id": routed_structure_id,
                "local_gate_active": local_gate_active,
                "local_gate_reason": local_gate_reason,
                "shared_update_scale": shared_update_scale,
                "local_pressure_steps_remaining": self._local_pressure_steps_remaining,
                "local_pressure_structure_id": self._local_pressure_structure_id,
            }
        )
        return float(loss)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self._snapshot_experts:
            return self._ensemble_predict(x)
        return self.predict_proba_single(x)

    def predict_proba_single(self, x: np.ndarray) -> np.ndarray:
        x = x.flatten()
        hidden = self.pool.forward(np.atleast_2d(x))
        best_structure = self.pool.select_best_structure(np.atleast_2d(x))
        surprise = None
        if best_structure is not None:
            surprise = float(best_structure.current_surprise(x))
        output, _, _, _, _, _ = self._compute_output(
            x,
            hidden,
            active_structure=best_structure,
            surprise=surprise,
            stage="predict",
        )
        output -= output.max()
        exp_out = np.exp(output)
        return exp_out / (exp_out.sum() + 1e-8)

    def predict(self, x: np.ndarray) -> int:
        return int(np.argmax(self.predict_proba(x)))

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        correct = sum(1 for idx in range(len(X)) if self.predict(X[idx]) == int(y[idx]))
        return correct / len(X)

    def get_stats(self) -> Dict:
        stats = self.pool.get_stats()
        stats["total_steps"] = self.step
        if self._history:
            recent = self._history[-20:]
            stats["recent_avg_loss"] = float(np.mean([row["loss"] for row in recent]))
            stats["recent_avg_surprise"] = float(np.mean([row["surprise"] for row in recent]))
            stats["recent_route_l2"] = float(np.mean([row["route_l2"] for row in recent]))
            stats["recent_route_cosine"] = float(np.mean([row["route_cosine"] for row in recent]))
            stats["recent_shared_output_norm"] = float(np.mean([row.get("shared_output_norm", 0.0) for row in recent]))
            stats["recent_local_output_norm"] = float(np.mean([row.get("local_output_norm", 0.0) for row in recent]))
            stats["recent_local_gate_rate"] = float(np.mean([float(row.get("local_gate_active", False)) for row in recent]))
            stats["recent_shared_update_scale"] = float(np.mean([row.get("shared_update_scale", 1.0) for row in recent]))
            stats["recent_pressure_active_rate"] = float(np.mean([float(row.get("pressure_active", False)) for row in recent]))
            stats["recent_boundary_stabilize_candidate_rate"] = float(np.mean([float(row.get("can_boundary_stabilize", False)) for row in recent]))
            stats["train_route_mode"] = "pooled_hidden"
        stats["readout_mode"] = self.readout_mode
        stats["shared_readout_scale"] = self.shared_readout_scale
        stats["shared_readout_post_checkpoint_scale"] = self.shared_readout_post_checkpoint_scale
        stats["local_readout_lr_scale"] = self.local_readout_lr_scale
        stats["local_readout_start_step"] = self.local_readout_start_step
        stats["local_readout_surprise_threshold"] = self.local_readout_surprise_threshold
        stats["local_readout_young_age_max"] = self.local_readout_young_age_max
        stats["local_readout_training_events"] = (
            sorted(self.local_readout_training_events) if self.local_readout_training_events else None
        )
        stats["local_readout_inference_surprise_threshold"] = self.local_readout_inference_surprise_threshold
        stats["local_readout_episode_events"] = (
            sorted(self.local_readout_episode_events) if self.local_readout_episode_events else None
        )
        stats["local_readout_episode_window_steps"] = self.local_readout_episode_window_steps
        stats["local_readout_pressure_window_steps"] = self.local_readout_pressure_window_steps
        stats["local_pressure_steps_remaining"] = self._local_pressure_steps_remaining
        stats["local_pressure_structure_id"] = self._local_pressure_structure_id
        stats["local_readout_active"] = self._local_readout_active()
        return stats

    def get_event_counts(self) -> Dict:
        counts = {
            "reinforce": 0,
            "branch": 0,
            "create": 0,
            "boundary_stabilize": 0,
        }
        for row in self._history:
            event = row.get("event", "")
            if event in counts:
                counts[event] += 1
        return counts


# ------------------------------------------------------------------
# TopoMem Fusion
# ------------------------------------------------------------------


class TopoMemUnifiedClassifier:
    """Fusion classifier combining Unified-SEL pool with TopoMem ECU health signals.

    This is NOT a retrieval fusion (retrieval was shown to degrade performance).
    Instead, TopoMem ECU health signals modulate pool lifecycle decisions.

    Architecture:
    - UnifiedSELClassifier: handles low-D stream (x0+x1>0 / x0+x1<0 tasks)
    - TopoMemSystem: monitors topological health from high-D embeddings
    - HealthStatus signals bridge the two systems
    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        lr: float = 0.05,
        max_structures: int = 20,
        evolve_every: int = 20,
        pool_config: Optional[Dict] = None,
        seed: Optional[int] = None,
        use_topo_health: bool = True,
        topo_health_update_interval: int = 50,
    ):
        self.in_size = in_size
        self.out_size = out_size
        self.lr = lr
        self.evolve_every = evolve_every
        self.step = 0
        self.use_topo_health = use_topo_health
        self.topo_health_update_interval = topo_health_update_interval
        self._last_topo_health_step = 0
        self._current_health_status: Optional[Dict] = None

        pool_kwargs = dict(pool_config or {})
        pool_kwargs["max_structures"] = max_structures

        if use_topo_health:
            from core.topo_fusion import HealthAwareStructurePool
            self.pool = HealthAwareStructurePool(
                in_size=in_size,
                out_size=out_size,
                initial_structures=1,
                seed=seed,
                **pool_kwargs,
            )
        else:
            self.pool = StructurePool(
                in_size=in_size,
                out_size=out_size,
                initial_structures=1,
                seed=seed,
                **pool_kwargs,
            )

        rng = np.random.default_rng(seed)
        self.W_out = rng.normal(0.0, 0.1, size=(out_size, out_size))
        self.W_out_memory: Optional[np.ndarray] = None
        self.dual_path_alpha: float = 0.5
        self.dual_path_active: bool = False
        self._history: List[Dict] = []

    def _compute_route_views(
        self,
        x: np.ndarray,
        active_structure,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Return active/pool hidden views plus route-gap diagnostics."""
        active_hidden = active_structure.forward(x).flatten()
        pooled_hidden = self.pool.forward(np.atleast_2d(x)).flatten()
        route_l2 = float(np.linalg.norm(active_hidden - pooled_hidden))

        denom = float(np.linalg.norm(active_hidden) * np.linalg.norm(pooled_hidden))
        if denom > 1e-8:
            route_cosine = float(np.dot(active_hidden, pooled_hidden) / denom)
        else:
            route_cosine = 1.0

        return active_hidden, pooled_hidden, route_l2, route_cosine

    def set_topo_health_status(
        self,
        health_score: float,
        prune_aggressiveness: float,
        consolidate_threshold: float,
        cluster_filter_enabled: bool,
        should_early_intervene: bool,
        trend_direction: str,
    ) -> None:
        """Set health status from external TopoMem ECU source."""
        if not self.use_topo_health:
            return

        self._current_health_status = {
            "health_score": health_score,
            "prune_aggressiveness": prune_aggressiveness,
            "consolidate_threshold": consolidate_threshold,
            "cluster_filter_enabled": cluster_filter_enabled,
            "should_early_intervene": should_early_intervene,
            "trend_direction": trend_direction,
        }

        if hasattr(self.pool, "set_health_status"):
            self.pool.set_health_status(
                health_score=health_score,
                prune_aggressiveness=prune_aggressiveness,
                consolidate_threshold=consolidate_threshold,
                cluster_filter_enabled=cluster_filter_enabled,
                should_early_intervene=should_early_intervene,
                trend_direction=trend_direction,
            )

    def fit_one(self, x: np.ndarray, y: np.ndarray, topo_system=None) -> float:
        """Fit one sample with optional TopoMem integration."""
        x = x.flatten()
        y = np.atleast_1d(y).flatten()

        if topo_system is not None and self.use_topo_health:
            if self.step - self._last_topo_health_step >= self.topo_health_update_interval:
                self._update_topo_health(topo_system)
                self._last_topo_health_step = self.step

        # 提取标签（假设是单标签分类）
        label = int(y[0]) if y.shape[0] == 1 else int(np.argmax(y))
        
        observation = self.pool.observe(x, label, self.out_size)
        active = observation["active_structure"]
        effective_lr = self.lr

        active_hidden, pooled_hidden, route_l2, route_cosine = self._compute_route_views(x, active)
        output = self._dual_path_output(pooled_hidden)

        if y.shape[0] == 1:
            target = np.zeros(self.out_size)
            target[int(y[0])] = 1.0
        else:
            target = y

        error = target - output
        loss = self.pool.learn_active(
            active_structure=active,
            x=np.atleast_2d(x),
            output_error=np.atleast_2d(error),
            lr=effective_lr,
            anchor_lambda=self.anchor_lambda,
        )

        self.W_out += effective_lr * np.outer(pooled_hidden, error)
        self.W_out = np.clip(self.W_out, -2.0, 2.0)
        self.pool.commit_pending_reinforcements()

        self.step += 1
        if not self._pool_frozen and self.step % self.evolve_every == 0:
            self.pool.evolve()

        health_info = {}
        if hasattr(self.pool, "get_health_modulation_info"):
            health_info = self.pool.get_health_modulation_info()

        self._history.append(
            {
                "step": self.step,
                "loss": float(loss),
                "event": observation["event"],
                "surprise": float(observation["surprise"]),
                "n_structures": int(observation["n_structures"]),
                "health_score": (
                    self._current_health_status.get("health_score", 1.0)
                    if self._current_health_status else 1.0
                ),
                "route_l2": route_l2,
                "route_cosine": route_cosine,
                **health_info,
            }
        )
        return float(loss)

    def _update_topo_health(self, topo_system) -> None:
        """Query TopoMem system for current health status."""
        try:
            health_diag = topo_system.get_health_dashboard()
            current = health_diag.get("current", {})
            trend = health_diag.get("trend", {})

            self.set_topo_health_status(
                health_score=current.get("health_score", 1.0),
                prune_aggressiveness=current.get("prune_aggressiveness", 0.0),
                consolidate_threshold=current.get("consolidate_threshold", 0.3),
                cluster_filter_enabled=current.get("cluster_filter_enabled", False),
                should_early_intervene=health_diag.get("prediction", {}).get("action_required", False),
                trend_direction=trend.get("direction", "green"),
            )
        except Exception:
            pass

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = x.flatten()
        hidden = self.pool.forward(np.atleast_2d(x)).flatten()
        output = self._dual_path_output(hidden)
        output -= output.max()
        exp_out = np.exp(output)
        return exp_out / (exp_out.sum() + 1e-8)

    def predict(self, x: np.ndarray) -> int:
        return int(np.argmax(self.predict_proba(x)))

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        correct = sum(1 for idx in range(len(X)) if self.predict(X[idx]) == int(y[idx]))
        return correct / len(X)

    def get_stats(self) -> Dict:
        stats = self.pool.get_stats()
        stats["total_steps"] = self.step
        if self._history:
            recent = self._history[-20:]
            stats["recent_avg_loss"] = float(np.mean([row["loss"] for row in recent]))
            stats["recent_avg_surprise"] = float(np.mean([row["surprise"] for row in recent]))
            stats["recent_route_l2"] = float(np.mean([row["route_l2"] for row in recent]))
            stats["recent_route_cosine"] = float(np.mean([row["route_cosine"] for row in recent]))
            stats["train_route_mode"] = "pooled_hidden"

        if self._current_health_status:
            stats["topo_health_score"] = self._current_health_status.get("health_score", 1.0)

        if hasattr(self.pool, "get_health_modulation_info"):
            stats["health_modulation"] = self.pool.get_health_modulation_info()

        return stats

    def get_event_counts(self) -> Dict:
        counts = {
            "reinforce": 0, "branch": 0, "create": 0,
            "reinforce_suppressed": 0, "boundary_stabilize": 0,
        }
        for row in self._history:
            event = row.get("event", "")
            if event in counts:
                counts[event] += 1
        return counts
