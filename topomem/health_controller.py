"""
topomem/health_controller.py — 拓扑健康ECU

统一控制器：将 H1/H2 健康度信号收敛为一个综合的健康状态，
并输出行为参数决定 retrieval、pruning、consolidation 等行为。

设计原则：
- 单一信号源：所有健康相关决策都经过这里
- 可预测性：输入信号 -> 输出行为的映射清晰可调
- 可扩展性：以后加 H3 或其他指标只需扩展 HealthStatus
- 趋势感知：不仅看当前值，还看趋势和预测

OBD 故障码体系：
- C001: H1 快速衰退（slope < -0.02, steps_until < 10）
- C002: H2 快速衰退（同上）
- C003: 综合健康分低于 consolidation 阈值
- C004: 趋势 ORANGE 但未触发阈值（提前干预）
- C005: 连续不稳定波动（variance 突增）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any
import copy

import numpy as np


# ------------------------------------------------------------------
# 故障码定义
# ------------------------------------------------------------------

class FaultCode(Enum):
    """OBD 风格故障码。"""
    C001_H1_RAPID_DECAY = "C001"  # H1 快速衰退
    C002_H2_RAPID_DECAY = "C002"  # H2 快速衰退
    C003_HEALTH_THRESHOLD = "C003"  # 综合健康分触发 consolidation
    C004_TREND_ALERT = "C004"  # 趋势预警（提前干预）
    C005_UNSTABLE_FLUCTUATION = "C005"  # 不稳定波动


@dataclass
class FaultRecord:
    """单次故障记录。"""
    code: FaultCode
    step: int
    health_score: float
    details: Dict[str, Any]


# ------------------------------------------------------------------
# 数据结构
# ------------------------------------------------------------------

class TrendDirection(Enum):
    """健康趋势方向（类似仪表盘颜色）。"""
    GREEN = "green"   # 稳定或上升
    YELLOW = "yellow" # 轻微下降
    ORANGE = "orange" # 明显下降
    RED = "red"       # 严重下降，接近阈值


@dataclass
class HealthTrend:
    """健康趋势分析结果。"""
    direction: TrendDirection = TrendDirection.GREEN
    slope: float = 0.0          # 健康分变化率（归一化步距）
    steps_until_consolidation: Optional[int] = None  # 预测几步后触发，None=不会
    confidence: float = 1.0    # 趋势预测置信度 [0, 1]
    is_stable: bool = True     # 是否稳定（波动小）

    # 原始历史
    history_scores: List[float] = field(default_factory=list)
    history_steps: List[int] = field(default_factory=list)

    # 关联故障码（可选）
    active_faults: List[FaultCode] = field(default_factory=list)


@dataclass
class HealthStatus:
    """拓扑健康状态及其衍生的行为参数。

    所有数值范围 [0, 1]，1=最健康/最不激进。
    """
    # 综合健康分数
    health_score: float = 1.0

    # Retrieval 参数乘数
    retrieval_gamma_mult: float = 1.0  # persistence 权重乘数

    # Pruning 参数
    prune_aggressiveness: float = 0.0  # 0=不删, 1=最激进

    # Consolidation 参数
    consolidate_threshold: float = 0.3  # 触发 consolidation 的阈值

    # 簇过滤
    cluster_filter_enabled: bool = False  # 健康差时启用干扰过滤

    # 诊断信息
    h1_health: float = 1.0
    h2_health: float = 1.0
    betti_1_count: int = 0
    betti_2_count: int = 0

    # 趋势信息（由 ECU 计算后填充）
    trend: Optional[HealthTrend] = None

    def __post_init__(self):
        """确保数值在合法范围。"""
        self.health_score = max(0.0, min(1.0, self.health_score))
        self.retrieval_gamma_mult = max(0.0, min(1.0, self.retrieval_gamma_mult))
        self.prune_aggressiveness = max(0.0, min(1.0, self.prune_aggressiveness))
        self.consolidate_threshold = max(0.0, min(1.0, self.consolidate_threshold))


# ------------------------------------------------------------------
# 配置（附调参指南）
# ------------------------------------------------------------------

@dataclass
class HealthControllerConfig:
    """健康控制器的可配置参数。

    调参指南：
    - consolidate_trigger_threshold: 经验值 0.3。太低 → 经常告警不处理；太高 → 坏很久才修
    - trend_threshold_slope: -0.01 意味着每步健康分下降 1%。经验值 [-0.005, -0.02]
    - stable_variance_threshold: 波动多大算稳定。数据噪声大选大（如 0.01）
    - trend_window_size: 看多远的历史。太长 → 反应慢；太短 → 易抖动
    """
    # H1/H2 权重（可扩展 H3 后加 h3_weight）
    h1_weight: float = 0.5
    h2_weight: float = 0.5

    # H1 专项阈值（低于此值立即触发consolidation，不等综合分数）
    h1_action_threshold: float = 0.3  # 对应 TopoMemConfig.h1_health_action_threshold

    # 健康分数计算方式
    health_formula: str = "weighted_avg"  # "weighted_avg" | "min" | "geometric"

    # 触发阈值
    consolidate_trigger_threshold: float = 0.3  # 经验值：低于此触发 consolidation
    cluster_filter_trigger_threshold: float = 0.7  # 低于此启用簇过滤

    # Pruning 激进程度范围
    prune_aggressiveness_min: float = 0.0
    prune_aggressiveness_max: float = 0.5

    # Retrieval gamma 范围
    retrieval_gamma_min: float = 0.0
    retrieval_gamma_max: float = 1.0

    # 趋势预测参数
    trend_window_size: int = 10        # 窗口大小，太长→反应慢，太短→抖
    trend_threshold_slope: float = -0.01  # slope低于此→下降趋势（归一化）
    stable_variance_threshold: float = 0.005  # 方差低于此→稳定
    min_history_for_trend: int = 3     # 最少历史点数才计算趋势


# ------------------------------------------------------------------
# TopologyHealthController 实现
# ------------------------------------------------------------------

class TopologyHealthController:
    """拓扑健康ECU - 单一信号源。

    将 H1/H2 健康度信号收敛为一个综合的健康状态，
    并输出行为参数供 retrieval、pruning、consolidation 使用。
    """

    def __init__(self, config: Optional[HealthControllerConfig] = None):
        self.config = config or HealthControllerConfig()
        self._health_history: List[float] = []
        self._step_history: List[int] = []
        self._step_counter: int = 0
        # OBD 故障日志
        self._fault_log: List[FaultRecord] = []

    # ------------------------------------------------------------------
    # 持久化 API
    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        """获取可序列化的 ECU 状态（用于保存）。"""
        return {
            "health_history": list(self._health_history),
            "step_history": list(self._step_history),
            "step_counter": self._step_counter,
            "fault_log": [
                {"code": r.code.value, "step": r.step,
                 "health_score": r.health_score, "details": r.details}
                for r in self._fault_log[-50:]  # 只保留最近50条
            ],
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """从保存的状态恢复。"""
        self._health_history = list(state.get("health_history", []))
        self._step_history = list(state.get("step_history", []))
        self._step_counter = state.get("step_counter", 0)
        self._fault_log = [
            FaultRecord(
                code=FaultCode(r["code"]),
                step=r["step"],
                health_score=r["health_score"],
                details=r["details"],
            )
            for r in state.get("fault_log", [])
        ]

    # ------------------------------------------------------------------
    # 核心 API
    # ------------------------------------------------------------------

    def compute_health_status(
        self,
        h1_health: float,
        h2_health: float,
        betti_1_count: int,
        betti_2_count: int,
    ) -> HealthStatus:
        """根据 H1/H2 健康度计算综合健康状态。"""
        cfg = self.config

        # 综合健康分数
        if cfg.health_formula == "weighted_avg":
            total_weight = cfg.h1_weight + cfg.h2_weight
            health = (
                cfg.h1_weight * h1_health +
                cfg.h2_weight * h2_health
            ) / total_weight
        elif cfg.health_formula == "min":
            health = min(h1_health, h2_health)
        elif cfg.health_formula == "geometric":
            health = np.sqrt(h1_health * h2_health)
        else:
            health = (h1_health + h2_health) / 2

        # Retrieval gamma
        gamma_range = cfg.retrieval_gamma_max - cfg.retrieval_gamma_min
        retrieval_gamma_mult = cfg.retrieval_gamma_min + gamma_range * health

        # Prune 激进程度
        prune_range = cfg.prune_aggressiveness_max - cfg.prune_aggressiveness_min
        prune_aggressiveness = (
            cfg.prune_aggressiveness_min +
            prune_range * (1 - health)
        )

        # Consolidation 阈值
        consolidate_threshold = (
            cfg.consolidate_trigger_threshold +
            0.4 * (1 - health)
        )

        # 簇过滤
        cluster_filter_enabled = (
            health < cfg.cluster_filter_trigger_threshold
        )

        # 记录历史 + 趋势
        self._record_health(health)
        trend = self._compute_trend()

        # 检测关联故障码
        self._detect_faults(trend, h1_health, h2_health, health)

        return HealthStatus(
            health_score=health,
            retrieval_gamma_mult=retrieval_gamma_mult,
            prune_aggressiveness=prune_aggressiveness,
            consolidate_threshold=consolidate_threshold,
            cluster_filter_enabled=cluster_filter_enabled,
            h1_health=h1_health,
            h2_health=h2_health,
            betti_1_count=betti_1_count,
            betti_2_count=betti_2_count,
            trend=trend,
        )

    def _record_health(self, health_score: float) -> None:
        """记录健康分到历史。"""
        self._step_counter += 1
        self._health_history.append(health_score)
        self._step_history.append(self._step_counter)
        window = self.config.trend_window_size
        if len(self._health_history) > window:
            self._health_history = self._health_history[-window:]
            self._step_history = self._step_history[-window:]

    def _compute_trend(self) -> HealthTrend:
        """从历史计算健康趋势（线性回归 + 稳定性判断）。"""
        cfg = self.config
        history = self._health_history
        steps = self._step_history

        if len(history) < cfg.min_history_for_trend:
            return HealthTrend(history_scores=list(history), history_steps=list(steps))

        scores = np.array(history)
        steps_arr = np.array(steps)

        # 归一化步数
        if steps_arr[-1] > steps_arr[0]:
            steps_norm = (steps_arr - steps_arr[0]) / (steps_arr[-1] - steps_arr[0])
        else:
            steps_norm = np.zeros_like(steps_arr)

        # 线性回归斜率
        slope = float(np.polyfit(steps_norm, scores, 1)[0]) if len(steps_norm) > 1 else 0.0

        # 稳定性
        variance = float(np.var(scores))
        is_stable = variance < cfg.stable_variance_threshold

        # 趋势方向
        if slope >= 0:
            direction = TrendDirection.GREEN
        elif slope > cfg.trend_threshold_slope:
            direction = TrendDirection.YELLOW
        elif slope > 2 * cfg.trend_threshold_slope:
            direction = TrendDirection.ORANGE
        else:
            direction = TrendDirection.RED

        # 预测步数
        steps_until = None
        if slope < 0:
            current = scores[-1] if len(scores) > 0 else cfg.consolidate_trigger_threshold
            threshold = cfg.consolidate_trigger_threshold
            gap = current - threshold
            if gap > 0 and slope < 0:
                step_change = abs(slope) * (np.std(scores) if np.std(scores) > 1e-6 else 0.01)
                if step_change > 1e-6:
                    steps_until = int(gap / step_change)
                    confidence = min(1.0, len(history) / cfg.trend_window_size)
                    if not is_stable:
                        confidence *= 0.5
                else:
                    steps_until = None
                    confidence = 0.0
            else:
                steps_until = None
                confidence = 0.0
        else:
            confidence = 1.0

        return HealthTrend(
            direction=direction,
            slope=slope,
            steps_until_consolidation=steps_until,
            confidence=confidence if steps_until is not None else 1.0,
            is_stable=is_stable,
            history_scores=list(history),
            history_steps=list(steps),
        )

    def _detect_faults(
        self,
        trend: HealthTrend,
        h1_health: float,
        h2_health: float,
        health_score: float,
    ) -> None:
        """检测并记录故障码。"""
        cfg = self.config
        step = self._step_counter

        # C001/C002: 单指标快速衰退（slope < -0.02 且 steps_until < 10）
        if trend.steps_until_consolidation is not None and trend.steps_until_consolidation < 10:
            if trend.slope < 2 * cfg.trend_threshold_slope:  # RED 区
                if h1_health < h2_health - 0.1:
                    self._log_fault(FaultCode.C001_H1_RAPID_DECAY, step, health_score, {
                        "slope": trend.slope, "steps_until": trend.steps_until_consolidation
                    })
                elif h2_health < h1_health - 0.1:
                    self._log_fault(FaultCode.C002_H2_RAPID_DECAY, step, health_score, {
                        "slope": trend.slope, "steps_until": trend.steps_until_consolidation
                    })

        # C005: 突然不稳定波动
        if not trend.is_stable and len(trend.history_scores) >= 3:
            recent = trend.history_scores[-3:]
            if np.var(recent) > 2 * cfg.stable_variance_threshold:
                self._log_fault(FaultCode.C005_UNSTABLE_FLUCTUATION, step, health_score, {
                    "variance": float(np.var(recent))
                })

    def _log_fault(self, code: FaultCode, step: int, health_score: float,
                   details: Dict[str, Any]) -> None:
        """记录故障（去重：同 step 同 code 不重复）。"""
        if not any(r.code == code and r.step == step for r in self._fault_log[-10:]):
            self._fault_log.append(FaultRecord(code, step, health_score, details))
            # 最多保留100条
            if len(self._fault_log) > 100:
                self._fault_log = self._fault_log[-100:]

    # ------------------------------------------------------------------
    # 决策 API
    # ------------------------------------------------------------------

    def should_consolidate(self, health_status: HealthStatus) -> bool:
        """判断是否应该触发 consolidation。

        触发条件（OR）：
        - 综合健康分低于动态阈值
        - H1 健康度低于 H1 专项阈值（独立于综合分数）
        """
        by_score = health_status.health_score < health_status.consolidate_threshold
        by_h1 = health_status.h1_health < self.config.h1_action_threshold
        return by_score or by_h1

    def should_early_intervene(self, health_status: HealthStatus) -> bool:
        """判断是否应该提前干预（趋势恶化但还没触发阈值）。"""
        if health_status.trend is None:
            return False
        if health_status.trend.direction in (TrendDirection.ORANGE, TrendDirection.RED):
            # 记录预警故障码
            if not self.should_consolidate(health_status):
                self._log_fault(
                    FaultCode.C004_TREND_ALERT,
                    self._step_counter,
                    health_status.health_score,
                    {"direction": health_status.trend.direction.value,
                     "slope": health_status.trend.slope}
                )
            return not self.should_consolidate(health_status)
        return False

    def should_filter_clusters(self, health_status: HealthStatus) -> bool:
        """判断是否应该启用簇过滤。"""
        return health_status.cluster_filter_enabled

    def get_retrieval_gamma_multiplier(self, health_status: HealthStatus) -> float:
        """获取 retrieval persistence 权重乘数。"""
        return health_status.retrieval_gamma_mult

    def get_prune_aggressiveness(self, health_status: HealthStatus) -> float:
        """获取 pruning 激进程度 [0, 1]。"""
        return health_status.prune_aggressiveness

    def get_diagnostic_info(self, health_status: HealthStatus) -> dict:
        """获取诊断信息（用于日志和调试）。"""
        trend_info = {}
        if health_status.trend:
            t = health_status.trend
            trend_info = {
                "trend_direction": t.direction.value,
                "trend_slope": round(t.slope, 6),
                "is_stable": t.is_stable,
                "steps_until_consolidation": t.steps_until_consolidation,
                "trend_confidence": round(t.confidence, 3),
            }

        return {
            "health_score": round(health_status.health_score, 4),
            "h1_health": round(health_status.h1_health, 4),
            "h2_health": round(health_status.h2_health, 4),
            "betti_1": health_status.betti_1_count,
            "betti_2": health_status.betti_2_count,
            "should_consolidate": self.should_consolidate(health_status),
            "should_early_intervene": self.should_early_intervene(health_status),
            "should_filter_clusters": self.should_filter_clusters(health_status),
            "consolidate_threshold": round(health_status.consolidate_threshold, 4),
            "prune_aggressiveness": round(health_status.prune_aggressiveness, 4),
            "retrieval_gamma_mult": round(health_status.retrieval_gamma_mult, 4),
            **trend_info,
        }

    def get_fault_log(self, max_records: int = 50) -> List[Dict[str, Any]]:
        """获取故障日志（供诊断用）。"""
        log = self._fault_log[-max_records:]
        return [
            {"code": r.code.value, "step": r.step,
             "health_score": round(r.health_score, 4),
             "details": r.details}
            for r in log
        ]

    def reset_history(self) -> None:
        """重置健康历史（通常在 consolidation 后调用）。"""
        self._health_history.clear()
        self._step_history.clear()
