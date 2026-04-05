"""
topomem/self_awareness.py — 自我认知模块

维护拓扑指纹的历史序列，通过 Wasserstein 距离检测认知漂移，
通过校准流程维护一致性。

设计来源：
- SPEC_SELF_AWARENESS.md: 自我认知与一致性守护完整规格
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from topomem.config import SelfAwarenessConfig


if TYPE_CHECKING:
    from topomem.memory import MemoryGraph
    from topomem.topology import PersistenceDiagram, TopologyEngine
    from topomem.engine import ReasoningEngine


logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# 数据类型定义
# ------------------------------------------------------------------

@dataclass
class TimestampedFingerprint:
    timestamp: float
    step: int
    fingerprint: np.ndarray  # (100,)


@dataclass
class TimestampedDiagram:
    timestamp: float
    step: int
    diagram: "PersistenceDiagram"


@dataclass
class DriftReport:
    timestamp: float
    step: int
    short_drift: float        # 与上次的 Wasserstein 距离
    long_drift: float         # 与基线的 Wasserstein 距离
    trend: str                # "accelerating" | "decelerating" | "stable" | "oscillating"
    status: str               # "stable" | "evolving" | "drifting" | "restructured"
    n_clusters: int           # 当前簇数
    message: str              # 人可读的描述


@dataclass
class CalibrationReport:
    timestamp: float
    drift: DriftReport
    n_clusters: int
    cluster_sizes: Dict[int, int]
    cluster_balance: float     # 0=完全平衡, 越大越不平衡
    avg_persistence: float
    orphan_ratio: float        # 未分配簇的节点比例
    self_description_consistency: Optional[float]  # 与上次自述的相似度
    recommendations: List[str]  # 建议操作列表


# ------------------------------------------------------------------
# SelfAwareness 实现
# ------------------------------------------------------------------

class SelfAwareness:
    """系统的自我认知模块。

    维护拓扑指纹的历史序列，通过 Wasserstein 距离检测认知漂移，
    通过校准流程维护一致性。
    """

    def __init__(self, config: Optional[SelfAwarenessConfig] = None):
        from topomem.topology import TopologyEngine
        self.config = config or SelfAwarenessConfig()
        self._topo_engine = TopologyEngine()  # 单例，复用缓存
        self._fingerprint_history: List[TimestampedFingerprint] = []
        self._diagram_history: List[TimestampedDiagram] = []
        self._drift_reports: List[DriftReport] = []
        self._step_count: int = 0
        self._baseline_fingerprint: Optional[np.ndarray] = None
        self._baseline_diagram: Optional["PersistenceDiagram"] = None
        self._last_self_description: Optional[str] = None
        self._embedding_manager: Optional["EmbeddingManager"] = None

    def update_fingerprint(
        self,
        memory_graph: "MemoryGraph",
        topo_engine: "TopologyEngine",
    ) -> None:
        """记录新的拓扑指纹到历史。"""
        fingerprint = memory_graph.get_topological_summary(topo_engine)

        # 获取最近的 diagram
        diagram = None
        if hasattr(memory_graph, "_topology_cache") and memory_graph._topology_cache is not None:
            diagram = memory_graph._topology_cache.diagram

        now = time.time()
        ts_fp = TimestampedFingerprint(
            timestamp=now,
            step=self._step_count,
            fingerprint=fingerprint,
        )
        self._fingerprint_history.append(ts_fp)

        if diagram is not None:
            ts_diag = TimestampedDiagram(
                timestamp=now,
                step=self._step_count,
                diagram=diagram,
            )
            self._diagram_history.append(ts_diag)

        # 限制历史大小
        max_size = self.config.fingerprint_history_size
        if len(self._fingerprint_history) > max_size:
            self._fingerprint_history = self._fingerprint_history[-max_size:]
        if len(self._diagram_history) > max_size:
            self._diagram_history = self._diagram_history[-max_size:]

        self._step_count += 1

        # 设置基线
        if self._baseline_fingerprint is None:
            self._baseline_fingerprint = fingerprint.copy()
        if self._baseline_diagram is None and diagram is not None:
            self._baseline_diagram = [d.copy() for d in diagram]

    def detect_drift(self) -> DriftReport:
        """检测认知漂移。"""
        now = time.time()

        if len(self._diagram_history) < 2:
            return DriftReport(
                timestamp=now,
                step=self._step_count,
                short_drift=0.0,
                long_drift=0.0,
                trend="stable",
                status="stable",
                n_clusters=0,
                message="Not enough history for drift detection",
            )

        topo_engine = self._topo_engine

        # 短期漂移：最近 vs 上一次
        recent_diag = self._diagram_history[-1].diagram
        prev_diag = self._diagram_history[-2].diagram
        short_drift = topo_engine.wasserstein_distance(recent_diag, prev_diag, dim=0)

        # 长期漂移：最近 vs 基线
        if self._baseline_diagram is not None:
            long_drift = topo_engine.wasserstein_distance(
                recent_diag, self._baseline_diagram, dim=0
            )
        else:
            long_drift = short_drift

        # 趋势检测（最近 5 次的短期漂移序列）
        recent_drifts = self._compute_recent_short_drifts(n=5)
        trend = self._classify_trend(recent_drifts)

        # 状态判定
        threshold = self.config.drift_threshold
        status = self._classify_status(short_drift, long_drift, threshold)

        # 获取当前簇数
        n_clusters = self._get_current_cluster_count()

        message = self._generate_message(status, short_drift, long_drift, trend)

        report = DriftReport(
            timestamp=now,
            step=self._step_count,
            short_drift=float(short_drift),
            long_drift=float(long_drift),
            trend=trend,
            status=status,
            n_clusters=n_clusters,
            message=message,
        )
        self._drift_reports.append(report)
        return report

    def _compute_recent_short_drifts(self, n: int = 5) -> List[float]:
        """计算最近 n 次的短期漂移序列。"""
        if len(self._diagram_history) < 2:
            return [0.0]

        topo_engine = self._topo_engine
        drifts = []
        start = max(1, len(self._diagram_history) - n)
        for i in range(start, len(self._diagram_history)):
            d1 = self._diagram_history[i].diagram
            d0 = self._diagram_history[i - 1].diagram
            drifts.append(topo_engine.wasserstein_distance(d1, d0, dim=0))
        return drifts if drifts else [0.0]

    def _classify_trend(self, drifts: List[float]) -> str:
        """分类漂移趋势。"""
        if len(drifts) < 2:
            return "stable"

        threshold = self.config.drift_threshold
        if all(d < threshold for d in drifts):
            return "stable"

        # 检查单调递增/递减
        diffs = [drifts[i + 1] - drifts[i] for i in range(len(drifts) - 1)]
        if all(d > 0 for d in diffs):
            return "accelerating"
        if all(d < 0 for d in diffs):
            return "decelerating"

        # 检查振荡
        sign_changes = sum(1 for i in range(len(diffs) - 1) if diffs[i] * diffs[i + 1] < 0)
        if sign_changes > len(diffs) / 2:
            return "oscillating"

        return "stable"

    def _classify_status(self, short: float, long: float, threshold: float) -> str:
        """分类漂移状态。"""
        if short < threshold * 0.5 and long < threshold:
            return "stable"
        if short > threshold:
            return "drifting"
        if long > threshold * 2:
            return "restructured"
        return "evolving"

    def _get_current_cluster_count(self) -> int:
        """获取当前簇数。"""
        if not self._diagram_history:
            return 0
        # 从最近的 diagram 估算
        h0 = self._diagram_history[-1].diagram[0]
        if len(h0) == 0:
            return 0
        # H0 中 death=inf 的点数为 1（全局连通分支），其余为 merge 事件
        finite = h0[np.isfinite(h0[:, 1])]
        return len(finite) + 1 if len(finite) > 0 else 1

    def _generate_message(
        self, status: str, short: float, long: float, trend: str
    ) -> str:
        """生成漂移报告的人可读描述。"""
        messages = {
            "stable": f"System stable. Short drift={short:.4f}, long drift={long:.4f}.",
            "evolving": f"System evolving. Short drift={short:.4f}, trend={trend}.",
            "drifting": f"WARNING: System drifting! Short drift={short:.4f} exceeds threshold.",
            "restructured": f"System restructured. Long drift={long:.4f} is very high.",
        }
        return messages.get(status, f"Unknown status: {status}")

    def get_identity_vector(self) -> np.ndarray:
        """返回当前的身份向量 (2*K,)。"""
        if not self._diagram_history:
            return np.zeros(2 * self.config.top_k_features)

        diagram = self._diagram_history[-1].diagram
        K = self.config.top_k_features

        # 提取所有特征
        features = []
        for dim, dgm in enumerate(diagram):
            if len(dgm) == 0:
                continue
            for point in dgm:
                birth, death = float(point[0]), float(point[1])
                if not np.isfinite(death):
                    continue
                persistence = death - birth
                features.append((persistence, birth, death))

        if not features:
            return np.zeros(2 * K)

        # 按 persistence 降序排序
        features.sort(key=lambda x: x[0], reverse=True)

        # 取 top-K
        top_k = features[:K]
        values = []
        for _, birth, death in top_k:
            values.extend([birth, death])

        # 不足 2*K 补 0
        while len(values) < 2 * K:
            values.append(0.0)

        return np.array(values[:2 * K], dtype=np.float64)

    def calibrate(
        self,
        memory_graph: "MemoryGraph",
        topo_engine: "TopologyEngine",
        engine: Optional["ReasoningEngine"] = None,
    ) -> CalibrationReport:
        """完整的自我校准流程。"""
        # 1. 强制重计算拓扑
        memory_graph.update_topology(topo_engine)
        self.update_fingerprint(memory_graph, topo_engine)

        # 2. 漂移检测
        drift = self.detect_drift()

        # 3. 结构性分析
        cluster_ids = []
        persistence_scores = []
        orphan_count = 0
        total_nodes = 0

        for _, data in memory_graph._graph.nodes(data=True):
            node = data["node"]
            total_nodes += 1
            if node.cluster_id >= 0:
                cluster_ids.append(node.cluster_id)
                persistence_scores.append(node.persistence_score)
            else:
                orphan_count += 1

        cluster_sizes = Counter(cluster_ids) if cluster_ids else {}
        sizes_list = list(cluster_sizes.values()) if cluster_sizes else [0]

        if len(sizes_list) > 1:
            cluster_balance = float(np.std(sizes_list) / max(np.mean(sizes_list), 1e-8))
        else:
            cluster_balance = 0.0

        avg_persistence = float(np.mean(persistence_scores)) if persistence_scores else 0.0
        orphan_ratio = orphan_count / max(total_nodes, 1)

        # 4.（可选）自我描述一致性
        self_desc_consistency = None
        if engine and cluster_sizes:
            self_desc_consistency = self._check_self_description_consistency(
                memory_graph, engine, cluster_sizes
            )

        # 5. 建议
        recommendations = self._generate_recommendations(
            drift, cluster_balance, orphan_ratio, avg_persistence
        )

        report = CalibrationReport(
            timestamp=time.time(),
            drift=drift,
            n_clusters=len(cluster_sizes),
            cluster_sizes=dict(cluster_sizes),
            cluster_balance=cluster_balance,
            avg_persistence=avg_persistence,
            orphan_ratio=orphan_ratio,
            self_description_consistency=self_desc_consistency,
            recommendations=recommendations,
        )
        return report

    def _check_self_description_consistency(
        self,
        memory_graph: "MemoryGraph",
        engine: "ReasoningEngine",
        cluster_sizes: Counter,
    ) -> Optional[float]:
        """检查自我描述的一致性。"""
        try:
            # 从每个簇采样 1 条代表性记忆
            samples = []
            for cid in cluster_sizes:
                for _, data in memory_graph._graph.nodes(data=True):
                    node = data["node"]
                    if node.cluster_id == cid:
                        samples.append(node.content)
                        break

            if not samples:
                return None

            # 让 engine 生成自述
            context_text = "\n".join(f"- {s}" for s in samples[:5])
            prompt = (
                f"Based on the following knowledge samples, describe the system's capabilities:\n"
                f"{context_text}\n\nDescribe in 2-3 sentences."
            )
            new_description = engine.generate(prompt, max_tokens=128, temperature=0.1)

            if self._last_self_description is None:
                self._last_self_description = new_description
                return 1.0

            # 计算 embedding 相似度
            from topomem.embedding import EmbeddingManager
            if self._embedding_manager is None:
                self._embedding_manager = EmbeddingManager()
            emb_mgr = self._embedding_manager
            emb_old = emb_mgr.encode(self._last_self_description)
            emb_new = emb_mgr.encode(new_description)
            sim = emb_mgr.similarity(emb_old, emb_new)

            self._last_self_description = new_description
            return float(sim)
        except Exception as e:
            logger.warning(f"Self-description consistency check failed: {e}")
            return None

    def _generate_recommendations(
        self,
        drift: DriftReport,
        cluster_balance: float,
        orphan_ratio: float,
        avg_persistence: float,
    ) -> List[str]:
        """生成校准建议。"""
        recs = []
        if drift.status == "drifting":
            recs.append("System is drifting. Consider reviewing recent additions.")
        if drift.status == "restructured":
            recs.append("Major restructuring detected. Verify if this is an expected domain shift.")
        if cluster_balance > 1.0:
            recs.append("Cluster sizes are very imbalanced. Consider rebalancing.")
        if orphan_ratio > 0.3:
            recs.append(f"High orphan ratio ({orphan_ratio:.2f}). Run update_topology to assign clusters.")
        if avg_persistence < 0.1:
            recs.append("Low average persistence. Knowledge may be too fragmented.")
        if not recs:
            recs.append("System healthy. No action needed.")
        return recs

    def should_calibrate(self) -> bool:
        """判断是否需要校准。"""
        if self._step_count == 0:
            return True
        if self._step_count % self.config.calibration_interval == 0:
            return True
        # 检查最近一次漂移状态
        if self._drift_reports:
            last = self._drift_reports[-1]
            if last.status == "drifting":
                return True
        return False

    # ==================================================================
    # 序列化
    # ==================================================================

    def save(self, path: str) -> None:
        """保存指纹历史和配置。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "step_count": self._step_count,
            "fingerprint_history": [
                {
                    "timestamp": fp.timestamp,
                    "step": fp.step,
                    "fingerprint": fp.fingerprint.tolist(),
                }
                for fp in self._fingerprint_history
            ],
            "baseline_fingerprint": (
                self._baseline_fingerprint.tolist()
                if self._baseline_fingerprint is not None
                else None
            ),
            "last_self_description": self._last_self_description,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load(self, path: str) -> None:
        """加载指纹历史。"""
        path = Path(path)
        if not path.exists():
            return

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._step_count = data.get("step_count", 0)
        self._fingerprint_history = [
            TimestampedFingerprint(
                timestamp=fp["timestamp"],
                step=fp["step"],
                fingerprint=np.array(fp["fingerprint"]),
            )
            for fp in data.get("fingerprint_history", [])
        ]
        bf = data.get("baseline_fingerprint")
        self._baseline_fingerprint = np.array(bf) if bf is not None else None
        self._last_self_description = data.get("last_self_description")

    def __repr__(self) -> str:
        status = "no_history"
        if self._drift_reports:
            status = self._drift_reports[-1].status
        return (
            f"SelfAwareness(step={self._step_count}, "
            f"history_size={len(self._fingerprint_history)}, "
            f"status='{status}')"
        )
