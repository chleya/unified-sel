"""
topomem/topology.py — 拓扑分析引擎

基于 Persistent Homology（持久同调），从高维点云中提取稳定的拓扑特征。

设计来源：
- SPEC_TOPOLOGY.md: 拓扑分析引擎完整规格
- ripser: 主选计算库（快速，专注 VR 复形）
- gudhi: 备选计算库（功能更全）
- persim: 持久图比较（Wasserstein 距离）
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional

import numpy as np

from topomem.config import TopologyConfig


logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# 数据类型定义
# ------------------------------------------------------------------

# 持久图类型：List[np.ndarray]
# dgms[0]: H0 diagram, shape (n, 2) - 每个 (birth, death)
# dgms[1]: H1 diagram, shape (m, 2) - 每个 (birth, death)
PersistenceDiagram = List[np.ndarray]

# 拓扑指纹类型：固定长度向量，L2 归一化
TopologicalFingerprint = np.ndarray


@dataclass
class TopologicalFeature:
    """一个拓扑特征（一个持久的 hole）"""
    dimension: int          # 0=连通分支, 1=环
    birth: float            # 出现的 filtration 值
    death: float            # 消失的 filtration 值
    persistence: float      # = death - birth，稳定程度
    representative: Optional[List[int]] = None  # 构成该特征的点索引


@dataclass
class TopologyResult:
    """compute_persistence 的完整返回（富结果）"""
    diagram: PersistenceDiagram
    features: List[TopologicalFeature]
    fingerprint: TopologicalFingerprint
    n_clusters: int                    # H0 持久分支数（= 有意义的簇数）
    cluster_labels: Optional[np.ndarray] = None  # (N,) 聚类标签


# ------------------------------------------------------------------
# TopologyEngine 实现
# ------------------------------------------------------------------

class TopologyEngine:
    """拓扑数据分析引擎。

    基于 Persistent Homology，从高维点云中提取稳定的拓扑特征。

    主要库选择：
    - ripser（主选）：速度快，API 简洁，专注 VR 复形
    - gudhi（备选）：功能更全，支持更多复形类型
    - persim：持久图比较（Wasserstein 距离）
    """

    def __init__(self, config: Optional[TopologyConfig] = None):
        """
        参数：
            config: TopologyConfig 配置对象
                    如果为 None，使用默认配置
        """
        self.config = config or TopologyConfig()
        self.max_homology_dim = self.config.max_homology_dim
        self.persistence_threshold = self.config.persistence_threshold
        self.metric = self.config.metric
        # LRU cache for compute_persistence results
        self._persistence_cache: dict = {}
        self._persistence_cache_order: list = []
        self._MAX_PERSISTENCE_CACHE = 32

    # ------------------------------------------------------------------
    # 核心 API: compute_persistence
    # ------------------------------------------------------------------

    def compute_persistence(self, points: np.ndarray) -> PersistenceDiagram:
        """计算 Persistent Homology（带 LRU 缓存）。

        参数：
            points: shape (N, D)，N 个 D 维点

        返回：
            PersistenceDiagram = List[np.ndarray]
            - dgms[0]: H0 diagram, shape (n, 2)
            - dgms[1]: H1 diagram, shape (m, 2)
            每行 = (birth, death)

        异常：
            ValueError: 如果输入包含 NaN 或 Inf
        """
        points = np.asarray(points, dtype=np.float64)

        # 验证输入
        if not np.all(np.isfinite(points)):
            raise ValueError("points contains NaN or Inf")

        n_points = len(points)

        # 边界情况：点太少（ripser 至少需要 2 个点）
        if n_points < 2:
            return [np.empty((0, 2)), np.empty((0, 2))]

        # 警告：点太多
        if n_points > 500:
            logger.warning(
                f"Computing PH on {n_points} points (>500). "
                f"This may take >5s."
            )

        # ---- LRU 缓存查找 ----
        # points.tolist() 产生 nested lists，必须把每行也转成 tuple 才能 hash
        points_tuple = tuple(tuple(float(x) for x in row) for row in points.tolist())
        cache_key = (points_tuple, self.max_homology_dim)
        if cache_key in self._persistence_cache:
            logger.debug(f"PH cache hit for {n_points} points")
            # 移到最后（LRU）
            self._persistence_cache_order.remove(cache_key)
            self._persistence_cache_order.append(cache_key)
            return self._persistence_cache[cache_key]

        # ---- 缓存未命中，执行计算 ----
        try:
            result = self._compute_with_ripser(points)
        except Exception as e:
            logger.warning(f"ripser failed: {e}, trying gudhi fallback...")
            try:
                result = self._compute_with_gudhi(points)
            except Exception as e2:
                logger.error(f"gudhi fallback also failed: {e2}")
                raise RuntimeError(
                    f"Both ripser and gudhi failed: {e}, {e2}"
                ) from e2

        # ---- 存入缓存（deepcopy 避免外部修改） ----
        cached_result = [np.array(d, copy=True) for d in result]
        if len(self._persistence_cache) >= self._MAX_PERSISTENCE_CACHE:
            # LRU 淘汰最老的条目
            oldest = self._persistence_cache_order.pop(0)
            del self._persistence_cache[oldest]
        self._persistence_cache[cache_key] = cached_result
        self._persistence_cache_order.append(cache_key)
        return cached_result

    def _compute_with_ripser(self, points: np.ndarray) -> PersistenceDiagram:
        """使用 ripser 计算 PH。"""
        from ripser import ripser

        result = ripser(points, maxdim=self.max_homology_dim)
        return result["dgms"]

    def _compute_with_gudhi(self, points: np.ndarray) -> PersistenceDiagram:
        """使用 gudhi 备选实现。"""
        import gudhi as gd

        # 构建 Rips complex
        rips = gd.RipsComplex(points=points)
        simplex_tree = rips.create_simplex_tree(
            max_dimension=self.max_homology_dim
        )

        # 计算 persistence
        diag = simplex_tree.persistence()

        # 转换为 ripser 格式
        from gudhi.persistence import persistences_from_simplex_tree
        h0 = []
        h1 = []
        for feature in persistences_from_simplex_tree(simplex_tree):
            dim = feature[0]
            birth = feature[1]
            death = feature[2]
            if dim == 0:
                h0.append([birth, death])
            elif dim == 1:
                h1.append([birth, death])

        return [
            np.array(h0) if h0 else np.empty((0, 2)),
            np.array(h1) if h1 else np.empty((0, 2)),
        ]

    # ------------------------------------------------------------------
    # 特征提取: extract_persistent_features
    # ------------------------------------------------------------------

    def extract_persistent_features(
        self,
        diagram: PersistenceDiagram,
        threshold: Optional[float] = None,
    ) -> List[TopologicalFeature]:
        """从持久图中提取有意义的拓扑特征。

        过滤策略：
        1. 移除 death=inf 的点（H0 中的全局连通分支）
        2. 计算 persistence = death - birth
        3. 如果 threshold 为 None，自动使用中位数
        4. 保留 persistence > 阈值的特征
        5. 按 persistence 降序排序

        参数：
            diagram: compute_persistence 的输出
            threshold: 持久性过滤阈值，None 则自动选择

        返回：
            List[TopologicalFeature]，按 persistence 降序
        """
        features: List[TopologicalFeature] = []

        for dim, dgm in enumerate(diagram):
            if len(dgm) == 0:
                continue

            for point in dgm:
                birth, death = float(point[0]), float(point[1])

                # 跳过 death=inf 的点
                if not np.isfinite(death):
                    continue

                persistence = death - birth
                features.append(TopologicalFeature(
                    dimension=dim,
                    birth=birth,
                    death=death,
                    persistence=persistence,
                ))

        if not features:
            return []

        # 自动阈值
        if threshold is None:
            threshold = self._auto_threshold(features)

        # 过滤并排序
        filtered = [f for f in features if f.persistence > threshold]
        filtered.sort(key=lambda f: f.persistence, reverse=True)
        return filtered

    def _auto_threshold(
        self,
        features: List[TopologicalFeature],
    ) -> float:
        """自动计算持久性阈值。"""
        if not features:
            return 0.0

        persistences = [f.persistence for f in features]
        median_p = float(np.median(persistences))
        return max(median_p, 0.01)  # 防止全 0

    # ------------------------------------------------------------------
    # 持久图比较: wasserstein_distance
    # ------------------------------------------------------------------

    def wasserstein_distance(
        self,
        diag_a: PersistenceDiagram,
        diag_b: PersistenceDiagram,
        dim: int = 0,
    ) -> float:
        """计算两个持久图在指定维度上的 Wasserstein 距离。

        参数：
            diag_a, diag_b: 两个 PersistenceDiagram
            dim: 维度（0=H0, 1=H1）

        返回：
            float >= 0
            - 0 = 完全相同的拓扑结构
            - 越大 = 差异越大
        """
        dgm_a = diag_a[dim] if len(diag_a) > dim else np.empty((0, 2))
        dgm_b = diag_b[dim] if len(diag_b) > dim else np.empty((0, 2))

        # 过滤 death=inf
        dgm_a = dgm_a[np.isfinite(dgm_a[:, 1])] if len(dgm_a) > 0 else dgm_a
        dgm_b = dgm_b[np.isfinite(dgm_b[:, 1])] if len(dgm_b) > 0 else dgm_b

        # 边界情况
        if len(dgm_a) == 0 and len(dgm_b) == 0:
            return 0.0

        try:
            from persim import wasserstein
            return float(wasserstein(dgm_a, dgm_b))
        except Exception as e:
            logger.warning(f"wasserstein failed: {e}, returning inf")
            return float("inf")

    # ------------------------------------------------------------------
    # 拓扑指纹: topological_summary
    # ------------------------------------------------------------------

    def topological_summary(
        self,
        diagram: PersistenceDiagram,
        method: str = "betti_curve",
    ) -> TopologicalFingerprint:
        """将持久图转为固定长度向量（拓扑指纹）。

        方法：
        - "betti_curve"（推荐）：在 filtration 轴上采样，统计存活特征数
        - "top_k_features"：取 top-K 个最持久特征的 (birth, death)

        参数：
            diagram: PersistenceDiagram
            method: 方法名

        返回：
            TopologicalFingerprint, shape (100,) for betti_curve
            L2 归一化
        """
        if method == "betti_curve":
            return self._betti_curve(diagram)
        elif method == "top_k_features":
            return self._top_k_features(diagram)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _betti_curve(self, diagram: PersistenceDiagram) -> TopologicalFingerprint:
        """计算 Betti Curve 指纹。

        对 H0 和 H1 分别计算，各 K=50 维，拼接为 (100,) 向量。
        """
        K = 50

        def compute_curve_for_dim(dgm: np.ndarray) -> np.ndarray:
            if len(dgm) == 0:
                return np.zeros(K)

            # 过滤 death=inf
            dgm = dgm[np.isfinite(dgm[:, 1])]
            if len(dgm) == 0:
                return np.zeros(K)

            max_death = float(np.max(dgm[:, 1]))
            if max_death <= 0:
                return np.zeros(K)

            t_values = np.linspace(0, max_death, K)
            curve = np.zeros(K)

            for i, t in enumerate(t_values):
                # 统计 birth <= t < death 的特征数
                curve[i] = np.sum(
                    (dgm[:, 0] <= t) & (dgm[:, 1] > t)
                )

            return curve

        curve_h0 = compute_curve_for_dim(diagram[0])
        curve_h1 = compute_curve_for_dim(
            diagram[1] if len(diagram) > 1 else np.empty((0, 2))
        )

        fingerprint = np.concatenate([curve_h0, curve_h1])  # (100,)

        # L2 归一化
        norm = np.linalg.norm(fingerprint)
        if norm > 1e-10:
            fingerprint = fingerprint / norm

        return fingerprint.astype(np.float64)

    def _top_k_features(self, diagram: PersistenceDiagram) -> TopologicalFingerprint:
        """取 top-K 个最持久特征的 (birth, death) 拼接。"""
        K = self.config.max_homology_dim + 1  # 使用 config 中的 top_k 概念
        features = self.extract_persistent_features(diagram)

        if not features:
            return np.zeros(2 * K)

        # 取 top-K
        top_k = features[:K]
        values = []
        for f in top_k:
            values.extend([f.birth, f.death])

        # 不足 2*K 补 0
        while len(values) < 2 * K:
            values.append(0.0)

        fingerprint = np.array(values[:2 * K], dtype=np.float64)

        # L2 归一化
        norm = np.linalg.norm(fingerprint)
        if norm > 1e-10:
            fingerprint = fingerprint / norm

        return fingerprint

    # ------------------------------------------------------------------
    # 聚类标签: cluster_labels_from_h0
    # ------------------------------------------------------------------

    def cluster_labels_from_h0(
        self,
        diagram: PersistenceDiagram,
        points: np.ndarray,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """从 H0 持久分支推导聚类标签。

        原理：Vietoris-Rips H0 等价于 single-linkage 聚类。

        参数：
            diagram: compute_persistence 的输出
            points: 原始点云 (N, D)
            threshold: 切割阈值（None 则自动选择）

        返回：
            np.ndarray, shape (N,)，每个点的簇标签（从 0 开始）
        """
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform

        n_points = len(points)
        if n_points < 2:
            return np.array([0])

        # 计算距离矩阵
        from scipy.spatial.distance import pdist
        distances = pdist(points, metric=self.metric)

        # Single-linkage 层次聚类
        Z = linkage(distances, method="single")

        # 自动阈值选择
        if threshold is None:
            threshold = self._auto_cluster_threshold(diagram)

        # 切割树得到簇标签
        labels = fcluster(Z, t=threshold, criterion="distance")

        # 重新编号为从 0 开始
        unique_labels = np.unique(labels)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        return np.array([label_map[l] for l in labels])

    def _auto_cluster_threshold(self, diagram: PersistenceDiagram) -> float:
        """从 H0 persistence 的 gap 自动选择切割阈值。"""
        h0 = diagram[0]
        if len(h0) < 2:
            return 0.5  # 默认阈值

        # 过滤 death=inf
        h0 = h0[np.isfinite(h0[:, 1])]
        if len(h0) < 2:
            return 0.5

        # 取 death 值排序
        deaths = np.sort(h0[:, 1])

        # 计算相邻差值
        gaps = np.diff(deaths)

        # 最大 gap 处，返回 gap 中间位置（而非 gap 左侧）
        if len(gaps) > 0 and np.max(gaps) > 0:
            idx = np.argmax(gaps)
            return float((deaths[idx] + deaths[idx + 1]) / 2)

        return float(np.median(deaths))

    # ------------------------------------------------------------------
    # 完整结果: compute_full_result（便捷方法）
    # ------------------------------------------------------------------

    def compute_full_result(
        self,
        points: np.ndarray,
    ) -> TopologyResult:
        """一次性计算所有拓扑特征，返回富结果。

        参数：
            points: shape (N, D)

        返回：
            TopologyResult
        """
        diagram = self.compute_persistence(points)
        features = self.extract_persistent_features(diagram)
        fingerprint = self.topological_summary(diagram)
        cluster_labels = self.cluster_labels_from_h0(diagram, points)
        n_clusters = len(np.unique(cluster_labels))

        return TopologyResult(
            diagram=diagram,
            features=features,
            fingerprint=fingerprint,
            n_clusters=n_clusters,
            cluster_labels=cluster_labels,
        )

    def __repr__(self) -> str:
        return (
            f"TopologyEngine(max_dim={self.max_homology_dim}, "
            f"threshold={self.persistence_threshold}, "
            f"metric='{self.metric}')"
        )
