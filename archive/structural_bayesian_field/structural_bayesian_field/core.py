"""
Structural Bayesian Field - Core Implementation

核心思想：
- 不是贝叶斯公式，而是动态信念结构
- structure distribution + observation + feedback -> evolved structure field

三个可测变量：
1. belief weight：结构/adapter 当前可信度
2. update operator：反馈如何改变权重
3. action policy：权重分布如何决定 solve / retry / branch / escalate

连接：
- Unified-SEL 的 utility
- TopoMem 的 effectiveness_score
- Double Helix 的能力边界调度
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from collections import defaultdict
import numpy as np
from enum import Enum


class Action(Enum):
    """系统可以采取的行动"""
    ACCEPT = "accept"           # ABOVE：直接接受，不需要反馈
    VERIFY = "verify"           # NEAR：验证并反馈
    ESCALATE = "escalate"       # BELOW：升级到更强的模型
    BRANCH = "branch"           # 分支：创建新结构
    REINFORCE = "reinforce"     # 强化：强化现有结构


@dataclass
class Belief:
    """单个结构的信念"""
    structure_id: str
    belief_weight: float = 0.5  # [0, 1]，当前可信度
    utility: float = 0.5         # 来自 Unified-SEL
    effectiveness_score: float = 0.5  # 来自 TopoMem
    surprise: float = 0.0        # 来自 Unified-SEL
    tension: float = 0.0         # 来自 Unified-SEL
    age: int = 0
    usage_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def combined_weight(self) -> float:
        """
        综合权重：结合 utility 和 effectiveness_score
        
        这是结构贝叶斯场的核心：
        - 不是简单的概率，而是多个信号的融合
        """
        # 可以调整权重，这是一个超参数
        w_utility = 0.5
        w_effectiveness = 0.5
        return w_utility * self.utility + w_effectiveness * self.effectiveness_score


@dataclass
class Observation:
    """一次观测（输入 + 反馈）"""
    observation_id: str
    input_data: Any
    feedback_signal: Optional[float] = None  # [0, 1]，反馈成功程度
    verifier_signals: Optional[Dict[str, Any]] = None  # 来自 Double Helix 的 first-pass 信号
    boundary_label: Optional[str] = None  # above/near/below


@dataclass
class StructureField:
    """
    结构贝叶斯场
    
    核心概念：
    - S_t = { (structure_1, w_1), (structure_2, w_2), ... }
    - S_{t+1} = F(S_t, I_t)
    """
    field_id: str
    beliefs: Dict[str, Belief] = field(default_factory=dict)
    step_count: int = 0
    config: Dict[str, Any] = field(default_factory=dict)

    # 默认配置
    DEFAULT_CONFIG = {
        "effectiveness_decay": 0.95,  # 来自 TopoMem
        "utility_decay": 0.99,        # 来自 Unified-SEL
        "surprise_threshold": 0.7,     # 来自 Unified-SEL
        "tension_threshold": 0.1,      # 来自 Unified-SEL
        "weight_decay": 0.98,          # 信念权重衰减
        "min_weight": 0.01,            # 最小权重
    }

    def __post_init__(self):
        # 合并默认配置
        for key, value in self.DEFAULT_CONFIG.items():
            if key not in self.config:
                self.config[key] = value

    def add_belief(self, belief: Belief) -> None:
        """添加一个新的信念结构"""
        self.beliefs[belief.structure_id] = belief

    def remove_belief(self, structure_id: str) -> None:
        """移除一个信念结构"""
        if structure_id in self.beliefs:
            del self.beliefs[structure_id]

    def get_belief(self, structure_id: str) -> Optional[Belief]:
        """获取一个信念结构"""
        return self.beliefs.get(structure_id)

    def update_belief_weights(self, observation: Observation) -> None:
        """
        根据观测更新所有信念权重
        
        这是结构贝叶斯场的核心更新算子：
        - 不是简单的乘法更新
        - 而是结合多个信号的动力系统演化
        """
        self.step_count += 1

        for belief in self.beliefs.values():
            # 年龄增加
            belief.age += 1

            # 基础衰减
            belief.belief_weight *= self.config["weight_decay"]
            belief.utility *= self.config["utility_decay"]
            belief.effectiveness_score *= self.config["effectiveness_decay"]

            # 如果有反馈信号，更新
            if observation.feedback_signal is not None:
                # 类似 TopoMem 的 evolve_adapter
                decay = self.config["effectiveness_decay"]
                belief.effectiveness_score = (
                    belief.effectiveness_score * decay +
                    observation.feedback_signal * (1 - decay)
                )

                # 类似 Unified-SEL 的 reinforce
                belief.utility = (
                    belief.utility * 0.9 +
                    observation.feedback_signal * 0.1
                )

                # 更新信念权重
                belief.belief_weight = belief.combined_weight()

            # 确保权重在合理范围内
            belief.belief_weight = np.clip(belief.belief_weight, 0.0, 1.0)
            belief.utility = np.clip(belief.utility, 0.0, 1.0)
            belief.effectiveness_score = np.clip(belief.effectiveness_score, 0.0, 1.0)

        # 移除权重过低的信念
        self._prune_low_weight_beliefs()

    def _prune_low_weight_beliefs(self) -> None:
        """移除权重过低的信念"""
        to_remove = [
            sid for sid, belief in self.beliefs.items()
            if belief.belief_weight < self.config["min_weight"]
        ]
        for sid in to_remove:
            self.remove_belief(sid)

    def decide_action(self, observation: Observation) -> Action:
        """
        根据信念分布决定采取什么行动
        
        这是连接到能力边界调度的关键：
        - 信念分布 → action policy
        
        连接 Double Helix 的发现：
        - ABOVE: accept
        - NEAR: verify
        - BELOW: escalate
        """
        # 如果有 verifier_signals，优先使用（来自 Double Helix）
        if observation.verifier_signals:
            return self._decide_from_verifier_signals(observation.verifier_signals)

        # 如果有 boundary_label，直接使用
        if observation.boundary_label:
            if observation.boundary_label == "above":
                return Action.ACCEPT
            elif observation.boundary_label == "near":
                return Action.VERIFY
            elif observation.boundary_label == "below":
                return Action.ESCALATE

        # 否则，根据信念分布决定
        return self._decide_from_belief_distribution()

    def _decide_from_verifier_signals(self, verifier_signals: Dict[str, Any]) -> Action:
        """
        根据 Double Helix 的 verifier 信号决定行动
        
        这是连接能力边界调度的关键！
        """
        # 使用 Phase G 发现的最强信号
        first_error_type = verifier_signals.get("first_error_type", "pass")

        if first_error_type == "pass":
            # ABOVE：直接接受
            return Action.ACCEPT

        # 对于 other 类型，计算 patch_size_to_message_len_ratio
        patch_size = verifier_signals.get("first_patch_size", 0)
        message_len = verifier_signals.get("first_error_message_len", 1)
        ratio = patch_size / max(message_len, 1)

        # 使用 Phase G 发现的阈值
        # NEAR: ratio 较高，BELOW: ratio 较低
        # 这里需要根据实际数据调整阈值
        if ratio > 3.0:
            # NEAR：验证并反馈
            return Action.VERIFY
        else:
            # BELOW：升级
            return Action.ESCALATE

    def _decide_from_belief_distribution(self) -> Action:
        """根据信念分布决定行动"""
        if not self.beliefs:
            return Action.BRANCH

        # 找出最高权重的信念
        best_belief = max(self.beliefs.values(), key=lambda b: b.belief_weight)

        if best_belief.belief_weight > 0.8:
            # 高置信度：接受或强化
            if best_belief.surprise < self.config["surprise_threshold"]:
                return Action.ACCEPT
            else:
                return Action.REINFORCE
        elif best_belief.belief_weight > 0.3:
            # 中等置信度：验证
            return Action.VERIFY
        else:
            # 低置信度：分支或升级
            if best_belief.tension > self.config["tension_threshold"]:
                return Action.BRANCH
            else:
                return Action.ESCALATE

    def get_field_stats(self) -> Dict[str, Any]:
        """获取场的统计信息"""
        if not self.beliefs:
            return {
                "n_beliefs": 0,
                "step_count": self.step_count,
            }

        weights = [b.belief_weight for b in self.beliefs.values()]
        utilities = [b.utility for b in self.beliefs.values()]
        effectiveness = [b.effectiveness_score for b in self.beliefs.values()]

        return {
            "n_beliefs": len(self.beliefs),
            "step_count": self.step_count,
            "avg_weight": float(np.mean(weights)),
            "std_weight": float(np.std(weights)),
            "max_weight": float(np.max(weights)),
            "min_weight": float(np.min(weights)),
            "avg_utility": float(np.mean(utilities)),
            "avg_effectiveness": float(np.mean(effectiveness)),
            "best_belief_id": max(self.beliefs.keys(), key=lambda k: self.beliefs[k].belief_weight),
        }
