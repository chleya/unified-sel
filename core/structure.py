"""
core/structure.py — Archived DFA structure unit

This module is kept for backward compatibility (smoke tests).
It implements a small Direct Feedback Alignment (DFA) network unit
with local readout and anchor-based regularization.

Design sources:
- SDAS/src/structure_pool.py
- SEL-Lab/core/sel_core.py
- FCRS/src/fcrs/types.py

NOTE: The "surprise", "tension", and "utility" terminology below is
legacy nomenclature from the original SEL-Lab project. These are
simply input-novelty, loss-plateau, and activation-frequency signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class Structure:
    """
    单个结构单元。

    每个 Structure 维护自己的权重矩阵（一个小型 DFA 网络），
    以及两个核心信号：
      - tension: 学习饱和度（高 → 该 clone 出新结构）
      - surprise: 对当前输入的陌生程度（高 → 该创建新结构）
    """

    id: int
    weights: np.ndarray         # 权重矩阵 [in_size, out_size]
    feedback: np.ndarray        # DFA 反馈矩阵 [out_size, out_size]
    label: str = ""             # 可读标签（如"任务A"、"避障"）

    # Lifecycle signals (legacy terminology from SEL-Lab)
    tension: float = 0.3        # Loss-plateau signal [0,1]; high → trigger clone
    utility: float = 1.0        # Activation-frequency signal [0,1]; low → prune
    age: int = 0                # Training steps survived

    # Input-novelty history (legacy: "surprise_history")
    surprise_history: List[float] = field(default_factory=list)

    # Loss history (used to compute tension)
    loss_history: List[float] = field(default_factory=list)

    # 动量（DFA 学习用）
    velocity: Optional[np.ndarray] = field(default=None, repr=False)
    local_readout: Optional[np.ndarray] = field(default=None, repr=False)
    local_readout_velocity: Optional[np.ndarray] = field(default=None, repr=False)
    local_readout_episode_steps_remaining: int = 0

    anchor: Optional[np.ndarray] = field(default=None, repr=False)
    anchor_fisher: Optional[np.ndarray] = field(default=None, repr=False)
    anchor_set: bool = False
    frozen: bool = False

    def __post_init__(self):
        if self.velocity is None:
            self.velocity = np.zeros_like(self.weights)
        if self.local_readout is None:
            self.local_readout = np.zeros((self.weights.shape[1], self.feedback.shape[0]), dtype=float)
        if self.local_readout_velocity is None:
            self.local_readout_velocity = np.zeros_like(self.local_readout)

    # ------------------------------------------------------------------
    # 核心信号计算
    # ------------------------------------------------------------------

    def current_surprise(self, observation: np.ndarray, label: int = None, out_size: int = 2) -> float:
        """
        Compute input-novelty score for the current observation-label pair.

        Components:
        1. Input novelty: cosine distance between observation and weight prototype
        2. Prediction error: if label provided, compute prediction mismatch

        Returns value in [0, 1]; higher = more novel.
        """
        obs = observation.flatten()

        # 1. Input novelty
        prototype = np.mean(self.weights, axis=1)
        norm_obs = np.linalg.norm(obs)
        norm_proto = np.linalg.norm(prototype)
        if norm_obs < 1e-8 or norm_proto < 1e-8:
            input_novelty = 1.0
        else:
            cosine_sim = np.dot(obs, prototype) / (norm_obs * norm_proto)
            input_novelty = float(np.clip(1.0 - cosine_sim, 0.0, 1.0))

        # 2. If no label provided, return input novelty only
        if label is None:
            return input_novelty

        # 3. Prediction error
        hidden = self.forward(np.atleast_2d(obs))
        output = hidden.flatten()[:out_size]

        # Softmax
        output_shifted = output - np.max(output)
        exp_output = np.exp(output_shifted)
        probs = exp_output / (np.sum(exp_output) + 1e-8)

        # Predicted label
        predicted_label = int(np.argmax(probs))

        # If mispredicted, increase novelty
        if predicted_label != label:
            prediction_novelty = 1.0
        else:
            # Even if correct, low confidence contributes some novelty
            prediction_novelty = 1.0 - probs[label]

        # Combine both novelty signals (weighted average)
        return 0.3 * input_novelty + 0.7 * prediction_novelty

    def update_tension(self, loss: float, window: int = 8) -> None:
        """
        Update loss-plateau signal (legacy: "tension").

        Plateau = degree of loss stagnation. If loss does not decrease over time,
        plateau signal rises.
        Source: SEL-Lab core/sel_core.py SELModule._update_tension
        """
        self.loss_history.append(loss)
        if len(self.loss_history) > window:
            self.loss_history.pop(0)

        if len(self.loss_history) < 2:
            return

        improvements = [
            self.loss_history[i - 1] - self.loss_history[i]
            for i in range(1, len(self.loss_history))
        ]
        avg_improvement = float(np.mean(improvements))
        min_improvement = 1e-4

        # Plateau score: less improvement → higher plateau signal
        plateau = max(0.0, min_improvement - avg_improvement) / max(min_improvement, 1e-8)
        plateau = float(np.clip(plateau, 0.0, 1.0))

        # Residual: current loss magnitude
        residual = float(np.tanh(np.mean(self.loss_history)))

        self.tension = float(0.6 * plateau + 0.4 * residual)

    def decay_utility(self, rate: float = 0.002) -> None:
        """Slowly decay activation-frequency signal (legacy: "utility").
        Inactive structures gradually lose influence."""
        self.utility = max(0.0, self.utility - rate)

    def reinforce(self, amount: float = 0.05) -> None:
        """Increase activation-frequency signal when activated."""
        self.utility = min(1.0, self.utility + amount)

    def set_anchor(self) -> None:
        """Freeze current weights as anchor. Source: SEL-Lab phase3_model.py:1354-1367"""
        self.anchor = self.weights.copy()
        self.anchor_set = True

    def estimate_anchor_fisher(self, X: np.ndarray, y: np.ndarray, out_size: int) -> None:
        if not self.anchor_set:
            return
        fisher = np.zeros_like(self.weights)
        w_cols = self.weights.shape[1]
        for i in range(len(X)):
            x_i = X[i].flatten()
            hidden = self.forward(np.atleast_2d(x_i))
            output = hidden.flatten()[:w_cols]
            y_i = np.atleast_1d(y[i]).flatten()
            if y_i.shape[0] == 1:
                target = np.zeros(w_cols)
                idx = int(y_i[0]) if int(y_i[0]) < w_cols else 0
                target[idx] = 1.0
            else:
                target = np.zeros(w_cols)
                for j in range(min(len(y_i), w_cols)):
                    target[j] = float(y_i[j])
            error = target - output
            grad = np.outer(x_i, error)
            fisher += grad ** 2
        fisher /= max(len(X), 1)
        self.anchor_fisher = np.clip(fisher, 0.0, 5.0)

    def anchor_penalty(self, anchor_lambda: float = 1.0) -> np.ndarray:
        """计算锚点正则化梯度。距离越远拉力越大。来源：SEL-Lab phase3_model.py:1354-1367"""
        if not self.anchor_set or self.anchor is None:
            return np.zeros_like(self.weights)
        if self.anchor_fisher is not None:
            return anchor_lambda * self.anchor_fisher * (self.anchor - self.weights)
        return anchor_lambda * (self.anchor - self.weights)

    # ------------------------------------------------------------------
    # DFA 前向学习
    # ------------------------------------------------------------------

    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播：x @ weights，带 tanh 激活。"""
        x = np.atleast_2d(x)
        return np.tanh(x @ self.weights)

    def learn(
        self,
        x: np.ndarray,
        output_error: np.ndarray,
        lr: float = 0.05,
        momentum: float = 0.9,
        anchor_lambda: float = 0.0,
    ) -> float:
        """
        DFA (Direct Feedback Alignment) learning step. No backpropagation needed.

        Source: SEL-Lab core/sel_core.py SELModule.forward_learning
        """
        x = np.atleast_2d(x)
        output_error = np.atleast_2d(output_error)

        fb_error = (self.feedback.T @ output_error.T).T
        grad = x.T @ fb_error

        if anchor_lambda > 0:
            grad += self.anchor_penalty(anchor_lambda)

        loss = float(np.mean(output_error ** 2))

        if self.frozen:
            self.age += 1
            return loss

        self.velocity = momentum * self.velocity + lr * grad
        self.weights += self.velocity
        self.weights = np.clip(self.weights, -2.0, 2.0)

        self.age += 1
        self.update_tension(loss)
        return loss

    def readout(self, hidden: np.ndarray) -> np.ndarray:
        hidden = np.atleast_2d(hidden)
        return hidden @ self.local_readout

    def learn_readout(
        self,
        hidden: np.ndarray,
        output_error: np.ndarray,
        lr: float = 0.05,
        momentum: float = 0.9,
    ) -> None:
        hidden = np.atleast_2d(hidden)
        output_error = np.atleast_2d(output_error)
        grad = hidden.T @ output_error
        self.local_readout_velocity = momentum * self.local_readout_velocity + lr * grad
        self.local_readout += self.local_readout_velocity
        self.local_readout = np.clip(self.local_readout, -2.0, 2.0)

    # ------------------------------------------------------------------
    # 结构演化操作
    # ------------------------------------------------------------------

    def clone(self, new_id: int, perturbation: float = 0.05, rng=None) -> "Structure":
        """
        Clone self with small perturbation. Used for loss-plateau-triggered splitting.
        Source: SEL-Lab SELModule.clone
        """
        if rng is None:
            rng = np.random.default_rng()
        new_weights = self.weights.copy() + rng.normal(0.0, perturbation, size=self.weights.shape)
        new_feedback = self.feedback.copy() + rng.normal(0.0, perturbation, size=self.feedback.shape)
        return Structure(
            id=new_id,
            weights=new_weights,
            feedback=new_feedback,
            label=f"clone_of_{self.id}",
            tension=self.tension * 0.7,
            utility=0.5,
            age=0,
            local_readout=self.local_readout.copy(),
            local_readout_episode_steps_remaining=0,
        )

    def __repr__(self) -> str:
        return (
            f"Structure(id={self.id}, label='{self.label}', "
            f"tension={self.tension:.3f}, utility={self.utility:.3f}, age={self.age})"
        )


def make_structure(
    structure_id: int,
    in_size: int,
    out_size: int,
    label: str = "",
    rng=None,
) -> Structure:
    """工厂函数：创建一个随机初始化的 Structure。"""
    if rng is None:
        rng = np.random.default_rng()
    weights = rng.normal(0.0, np.sqrt(2.0 / in_size), size=(in_size, out_size))
    feedback = rng.normal(0.0, 0.1, size=(out_size, out_size))
    return Structure(
        id=structure_id,
        weights=weights,
        feedback=feedback,
        label=label or f"struct_{structure_id}",
        utility=0.5,
    )
