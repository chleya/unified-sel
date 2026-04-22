from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import numpy as np


# Status: EXPERIMENTAL SIDECAR — not promoted as primary governance signal.
# P0 result: predictive residual is detectable (12.8x domain shift separation)
# but not superior to BatchHealthMonitor (27.2x) and has no temporal advantage.
# Do not connect to routing or CEE. Use for ablation / fused health experiments only.


@dataclass(frozen=True)
class PredictiveHealthSignals:
    residual_mean: float
    residual_z_score: float
    residual_trend: Literal["stable", "rising", "spiking"]
    status: Literal["warmup", "healthy", "gradual_drift", "domain_shift", "anomaly"]


@dataclass
class ControlContext:
    routing_decision: Optional[str] = None
    solver_type: Optional[str] = None
    monitor_signal: Optional[float] = None
    verifier_result: Optional[bool] = None

    def to_vector(self, dim: int = 8) -> np.ndarray:
        parts: List[float] = []
        decision_map = {
            None: 0.0,
            "accepted": 1.0,
            "verified": 2.0,
            "escalated": 3.0,
        }
        parts.append(decision_map.get(self.routing_decision, 0.0) / 3.0)
        parts.append(1.0 if self.solver_type is not None else 0.0)
        parts.append(
            self.monitor_signal if self.monitor_signal is not None else 0.5
        )
        parts.append(
            1.0
            if self.verifier_result is True
            else (0.0 if self.verifier_result is False else 0.5)
        )
        while len(parts) < dim:
            parts.append(0.0)
        return np.array(parts[:dim], dtype=np.float32)


class PredictiveHealthMonitor:
    name = "predictive_health"

    def __init__(
        self,
        latent_dim: int = 384,
        control_dim: int = 8,
        window_size: int = 10,
        warmup_windows: int = 2,
        learning_rate: float = 0.05,
    ):
        self._latent_dim = latent_dim
        self._control_dim = control_dim
        self._window_size = window_size
        self._warmup_windows = warmup_windows
        self._lr = learning_rate

        self._embeddings: List[np.ndarray] = []
        self._contexts: List[np.ndarray] = []
        self._window_centroids: List[np.ndarray] = []
        self._window_contexts: List[np.ndarray] = []
        self._residuals: List[float] = []
        self._residual_mean: float = 0.0
        self._residual_var: float = 1.0
        self._step_count: int = 0

        self._W: Optional[np.ndarray] = None
        self._W_ctrl: Optional[np.ndarray] = None
        self._bias: Optional[np.ndarray] = None

    def _init_weights(self):
        self._W = np.eye(self._latent_dim, dtype=np.float32) * 0.9
        self._W_ctrl = np.zeros(
            (self._latent_dim, self._control_dim), dtype=np.float32
        )
        self._bias = np.zeros(self._latent_dim, dtype=np.float32)

    def _predict_centroid(
        self, prev_centroid: np.ndarray, prev_ctx: np.ndarray
    ) -> np.ndarray:
        if self._W is None:
            return prev_centroid.copy()
        return self._W @ prev_centroid + self._W_ctrl @ prev_ctx + self._bias

    def _update_predictor(
        self,
        prev_centroid: np.ndarray,
        prev_ctx: np.ndarray,
        actual_centroid: np.ndarray,
    ):
        if self._W is None:
            self._init_weights()

        pred = self._predict_centroid(prev_centroid, prev_ctx)
        error = pred - actual_centroid
        self._W -= self._lr * np.outer(error, prev_centroid)
        self._W_ctrl -= self._lr * np.outer(error, prev_ctx)
        self._bias -= self._lr * error

    def observe(
        self,
        z: np.ndarray,
        control_context: Optional[ControlContext] = None,
    ) -> PredictiveHealthSignals:
        ctx = (
            control_context.to_vector(self._control_dim)
            if control_context
            else np.zeros(self._control_dim, dtype=np.float32)
        )

        self._embeddings.append(z.copy())
        self._contexts.append(ctx.copy())
        self._step_count += 1

        if self._step_count < self._window_size:
            return PredictiveHealthSignals(
                residual_mean=0.0,
                residual_z_score=0.0,
                residual_trend="stable",
                status="warmup",
            )

        if self._step_count % self._window_size != 0:
            return PredictiveHealthSignals(
                residual_mean=self._residual_mean if self._residuals else 0.0,
                residual_z_score=self._residuals[-1]
                if self._residuals
                else 0.0,
                residual_trend=self._compute_trend(),
                status=self._current_status(),
            )

        window_embs = np.stack(self._embeddings[-self._window_size:])
        window_ctx = np.mean(self._contexts[-self._window_size:], axis=0)
        centroid = np.mean(window_embs, axis=0)

        residual = 0.0
        if len(self._window_centroids) > 0:
            prev_centroid = self._window_centroids[-1]
            prev_ctx = self._window_contexts[-1]
            pred_centroid = self._predict_centroid(prev_centroid, prev_ctx)
            residual = float(
                1.0
                - np.dot(
                    pred_centroid / (np.linalg.norm(pred_centroid) + 1e-10),
                    centroid / (np.linalg.norm(centroid) + 1e-10),
                )
            )
            residual = max(0.0, residual)

        self._window_centroids.append(centroid.copy())
        self._window_contexts.append(window_ctx.copy())

        if len(self._window_centroids) >= 2 and len(self._window_centroids) > self._warmup_windows:
            prev_c = self._window_centroids[-2]
            prev_ctx_w = self._window_contexts[-2]
            self._update_predictor(prev_c, prev_ctx_w, centroid)

        if len(self._window_centroids) <= self._warmup_windows:
            self._residuals.append(residual)
            return PredictiveHealthSignals(
                residual_mean=0.0,
                residual_z_score=0.0,
                residual_trend="stable",
                status="warmup",
            )

        self._residuals.append(residual)

        residual_mean = float(np.mean(self._residuals))
        n = len(self._residuals)
        residual_var = float(np.var(self._residuals, ddof=1)) if n >= 2 else 1.0
        self._residual_mean = residual_mean
        self._residual_var = max(residual_var, 1e-10)

        z_score = (residual - residual_mean) / (np.sqrt(self._residual_var) + 1e-10)
        trend = self._compute_trend()
        status = self._compute_status(residual, z_score, trend)

        return PredictiveHealthSignals(
            residual_mean=residual_mean,
            residual_z_score=float(z_score),
            residual_trend=trend,
            status=status,
        )

    def _current_status(
        self,
    ) -> Literal["warmup", "healthy", "gradual_drift", "domain_shift", "anomaly"]:
        if not self._residuals or len(self._window_centroids) <= self._warmup_windows:
            return "warmup"
        return self._compute_status(
            self._residuals[-1] if self._residuals else 0.0,
            0.0,
            self._compute_trend(),
        )

    def _compute_trend(self) -> Literal["stable", "rising", "spiking"]:
        if len(self._residuals) < 3:
            return "stable"
        recent_3 = self._residuals[-3:]
        if all(recent_3[i] > recent_3[i - 1] for i in range(1, len(recent_3))):
            mean_r = self._residual_mean if self._residual_mean > 0 else 1.0
            if recent_3[-1] > 3 * mean_r:
                return "spiking"
            return "rising"
        return "stable"

    def _compute_status(
        self,
        residual: float,
        z_score: float,
        trend: str,
    ) -> Literal["healthy", "gradual_drift", "domain_shift", "anomaly"]:
        if trend == "spiking":
            return "anomaly"
        if z_score > 3.0:
            return "domain_shift"
        if z_score > 1.5 or trend == "rising":
            return "gradual_drift"
        return "healthy"

    def health_report(self) -> Dict[str, Any]:
        if len(self._window_centroids) <= self._warmup_windows:
            return {
                "status": "warmup",
                "n_observed": self._step_count,
                "n_windows": len(self._window_centroids),
                "warmup_windows_remaining": max(
                    0, self._warmup_windows - len(self._window_centroids) + 1
                ),
            }

        return {
            "status": self._compute_status(
                self._residuals[-1] if self._residuals else 0.0,
                0.0,
                self._compute_trend(),
            ),
            "n_observed": self._step_count,
            "n_windows": len(self._window_centroids),
            "residual_mean": self._residual_mean,
            "residual_var": self._residual_var,
            "residual_trend": self._compute_trend(),
        }
