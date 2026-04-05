"""
core/structure.py — 核心数据结构

Structure 是系统的基本单元，对应人脑中的一个功能性神经回路。

设计来源：
- SDAS/src/structure_pool.py: utility, surprise_history, action_values 概念
- SEL-Lab/core/sel_core.py: tension 计算，DFA 权重结构
- FCRS/src/fcrs/types.py: 严格的类型验证
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

    # 进化信号
    tension: float = 0.3        # 学习饱和度 [0,1]，高则需要 clone
    utility: float = 1.0        # 效用值 [0,1]，低则被淘汰
    age: int = 0                # 生存步数

    # 惊讶度历史（SDAS 的核心创新）
    surprise_history: List[float] = field(default_factory=list)

    # 损失历史（用于计算 tension）
    loss_history: List[float] = field(default_factory=list)

    # 动量（DFA 学习用）
    velocity: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self):
        if self.velocity is None:
            self.velocity = np.zeros_like(self.weights)

    # ------------------------------------------------------------------
    # 核心信号计算
    # ------------------------------------------------------------------

    def current_surprise(self, observation: np.ndarray) -> float:
        """
        计算当前输入对本结构的陌生程度（惊讶度）。

        用余弦距离衡量：observation 和 weights 的第一主方向有多不同。
        返回值 [0, 1]，越高越陌生。
        """
        obs = observation.flatten()
        # 用权重的列均值作为结构的"原型"向量
        prototype = np.mean(self.weights, axis=1)
        norm_obs = np.linalg.norm(obs)
        norm_proto = np.linalg.norm(prototype)
        if norm_obs < 1e-8 or norm_proto < 1e-8:
            return 1.0
        cosine_sim = np.dot(obs, prototype) / (norm_obs * norm_proto)
        # 相似度 → 陌生度
        return float(np.clip(1.0 - cosine_sim, 0.0, 1.0))

    def update_tension(self, loss: float, window: int = 8) -> None:
        """
        更新张力值。

        张力 = 损失高原程度。如果损失长期不下降，张力升高。
        来源：SEL-Lab core/sel_core.py SELModule._update_tension
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

        # 高原分数：改进越少，张力越高
        plateau = max(0.0, min_improvement - avg_improvement) / max(min_improvement, 1e-8)
        plateau = float(np.clip(plateau, 0.0, 1.0))

        # 残差：当前损失本身
        residual = float(np.tanh(np.mean(self.loss_history)))

        self.tension = float(0.6 * plateau + 0.4 * residual)

    def decay_utility(self, rate: float = 0.002) -> None:
        """每步缓慢衰减效用。不被激活的结构逐渐失去影响力。"""
        self.utility = max(0.0, self.utility - rate)

    def reinforce(self, amount: float = 0.05) -> None:
        """被激活时增加效用。"""
        self.utility = min(1.0, self.utility + amount)

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
    ) -> float:
        """
        DFA（直接反馈对齐）学习步骤。不需要反向传播。

        来源：SEL-Lab core/sel_core.py SELModule.forward_learning
        """
        x = np.atleast_2d(x)
        output_error = np.atleast_2d(output_error)

        # DFA：用固定随机反馈矩阵投影误差到隐层
        fb_error = (self.feedback.T @ output_error.T).T
        grad = x.T @ fb_error

        # 带动量的梯度更新
        self.velocity = momentum * self.velocity + lr * grad
        self.weights += self.velocity
        self.weights = np.clip(self.weights, -2.0, 2.0)

        loss = float(np.mean(output_error ** 2))
        self.age += 1
        self.update_tension(loss)
        return loss

    # ------------------------------------------------------------------
    # 结构演化操作
    # ------------------------------------------------------------------

    def clone(self, new_id: int, perturbation: float = 0.05, rng=None) -> "Structure":
        """
        克隆自身，加上小扰动。用于 tension 触发的结构分裂。
        来源：SEL-Lab SELModule.clone
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
