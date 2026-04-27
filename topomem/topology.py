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
        max_clusters: Optional[int] = None,
    ) -> np.ndarray:
        """从 H0 持久分支推导聚类标签。

        原理：Vietoris-Rips H0 等价于 single-linkage 聚类。

        参数：
            diagram: compute_persistence 的输出
            points: 原始点云 (N, D)
            threshold: 切割阈值（None 则自动选择）
            max_clusters: 目标最大簇数（用于控制微簇问题）

        返回：
            np.ndarray, shape (N,)，每个点的簇标签（从 0 开始）
        """
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import pdist

        n_points = len(points)
        if n_points < 2:
            return np.array([0])

        # 计算距离矩阵
        distances = pdist(points, metric=self.metric)

        # Single-linkage 层次聚类
        Z = linkage(distances, method="single")

        # 自动阈值选择
        if threshold is None:
            threshold = self._auto_cluster_threshold(diagram)

        # 切割树得到簇标签
        labels = fcluster(Z, t=threshold, criterion="distance")

        # 如果簇数过多，尝试用 max_clusters 限制
        if max_clusters is not None:
            n_current = len(np.unique(labels))
            if n_current > max_clusters:
                # 用 max_clusters 作为切割阈值（基于层次聚类的"深度"）
                # fcluster 支持 k 模式：用簇数切割
                labels = fcluster(Z, t=max_clusters, criterion="maxclust")

        # 重新编号为从 0 开始
        unique_labels = np.unique(labels)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        return np.array([label_map[l] for l in labels])

    def cluster_labels_from_dbscan(
        self,
        points: np.ndarray,
        eps: Optional[float] = None,
        min_samples: int = 3,
        auto_eps_percentile: float = 90,
        skip_umap: bool = False,
    ) -> np.ndarray:
        """DBSCAN 密度聚类（使用余弦距离）。

        相比 single-linkage H0，DBSCAN 的优势：
        1. 密度感知：只在高密度区域形成簇，噪声点标记为 -1
        2. 自动簇数：无需预设 k，由数据分布决定
        3. 对高维稀疏数据更鲁棒

        参数：
            points: shape (N, D)，归一化后的 embedding
            eps: 余弦距离阈值（None 则自动估计）
            min_samples: 形成簇的最少邻居数
            auto_eps_percentile: 自动估计 eps 时使用的分位数

        返回：
            np.ndarray, shape (N,)
            簇标签：0, 1, 2, ... （-1 = 噪声点）
        """
        from sklearn.cluster import DBSCAN
        from sklearn.neighbors import NearestNeighbors

        n_points = len(points)
        if n_points < 2:
            return np.array([0])

        # P0 修复：如果启用 UMAP，先降维到低维空间
        # 验证：384D ARI=0.000，UMAP(2D) ARI=0.945
        # 但如果已经是低维（<=10D），就不再降维
        # skip_umap=True 当调用者已经做过 UMAP 降维时使用
        points_for_clustering = points
        if not skip_umap and getattr(self.config, 'use_umap_before_clustering', False) and points.shape[1] > 10:
            points_for_clustering = self._umap_reduce(points)
            logger.debug(f"UMAP: {points.shape[1]}D → {points_for_clustering.shape[1]}D")

        # 自动估计 eps（使用降维后的点云）
        if eps is None:
            eps = self._estimate_dbscan_eps(points_for_clustering, min_samples)

        # 计算距离矩阵
        # P0 修复：UMAP 降维后使用 Euclidean 距离（DBSCAN 在欧氏空间更稳定）
        # 原因：UMAP 已经用 cosine 相似度构建了邻域结构，降维后的欧氏距离
        #       能更好地反映点之间的聚类关系
        is_umap_reduced = points_for_clustering.shape[1] <= 10
        if is_umap_reduced:
            # UMAP 降维后用 Euclidean 距离
            from scipy.spatial.distance import pdist, squareform
            dists = pdist(points_for_clustering, metric="euclidean")
            distances = squareform(dists)
        elif self.metric == "cosine":
            cos_sim = points_for_clustering @ points_for_clustering.T
            np.fill_diagonal(cos_sim, 1.0)
            distances = np.clip(1 - cos_sim, 0.0, 2.0)
        else:
            from scipy.spatial.distance import pdist, squareform
            dists = pdist(points_for_clustering, metric=self.metric)
            distances = squareform(dists)

        # DBSCAN 聚类
        clustering = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric="precomputed",
        )
        labels = clustering.fit_predict(distances)

        # 如果 DBSCAN 返回太多噪声，调整参数重试
        noise_ratio = np.sum(labels == -1) / len(labels)
        if noise_ratio > 0.5 and n_points > 10:
            logger.warning(
                f"DBSCAN: {noise_ratio:.0%} noise, retrying with eps={eps*0.7:.3f}"
            )
            labels = DBSCAN(
                eps=eps * 0.7,
                min_samples=max(2, min_samples - 1),
                metric="precomputed",
            ).fit_predict(distances)

        return labels

    def cluster_labels_hybrid(
        self,
        points: np.ndarray,
        diagram: Optional[PersistenceDiagram] = None,
        dbscan_eps: Optional[float] = None,
        dbscan_min_samples: int = 3,
    ) -> np.ndarray:
        """混合聚类：DBSCAN 预聚类 + TDA H0 细化。

        工作流程：
        1. 先用 DBSCAN 做密度聚类，得到粗粒度簇
        2. 对每个 DBSCAN 簇内，用 H0 持久性进一步细分
        3. 噪声点分配给最近的簇

        这解决了：
        - H0 在高维空间的单点簇问题
        - DBSCAN 无法识别拓扑结构的问题

        参数：
            points: shape (N, D) 点云
            diagram: 可选的持久图（如果已计算）
            dbscan_eps: DBSCAN 半径（None 则自动估计）
            dbscan_min_samples: DBSCAN 最小样本

        返回：
            np.ndarray, shape (N,)，簇标签
        """
        n_points = len(points)
        if n_points < 2:
            return np.array([0])

        # P0 修复：如果启用 UMAP，先降维到低维空间（如果还不是低维）
        points_for_clustering = points
        if getattr(self.config, 'use_umap_before_clustering', False) and points.shape[1] > 10:
            points_for_clustering = self._umap_reduce(points)
            logger.debug(f"UMAP hybrid: {points.shape[1]}D → {points_for_clustering.shape[1]}D")

        # Step 1: DBSCAN 预聚类（skip_umap=True 因为上面已经做过 UMAP）
        dbscan_labels = self.cluster_labels_from_dbscan(
            points_for_clustering, eps=dbscan_eps, min_samples=dbscan_min_samples, skip_umap=True
        )

        unique_dbscan_clusters = set(dbscan_labels)
        unique_dbscan_clusters.discard(-1)  # 移除噪声

        n_dbscan_clusters = len(unique_dbscan_clusters)

        # 如果 DBSCAN 只找到 0-1 个簇，fallback 到 H0
        if n_dbscan_clusters <= 1:
            if diagram is not None:
                return self.cluster_labels_from_h0(diagram, points)
            else:
                # 没有 diagram，返回 DBSCAN 结果
                labels = dbscan_labels.copy()
                noise_mask = labels == -1
                if noise_mask.any():
                    max_label = labels.max() if len(labels) > 0 else 0
                    noise_count = noise_mask.sum()
                    labels[noise_mask] = np.arange(max_label + 1, max_label + 1 + noise_count)
                return labels

        # Step 2: 对每个 DBSCAN 簇，计算内部 H0 持久性
        final_labels = np.full(n_points, -1, dtype=int)
        next_cluster_id = 0

        for cluster_id in unique_dbscan_clusters:
            cluster_mask = dbscan_labels == cluster_id
            cluster_points = points[cluster_mask]
            cluster_indices = np.where(cluster_mask)[0]

            if len(cluster_points) < 2:
                final_labels[cluster_indices[0]] = next_cluster_id
                next_cluster_id += 1
                continue

            # 在簇内计算 PH
            try:
                sub_diagram = self.compute_persistence(cluster_points)
                sub_labels = self.cluster_labels_from_h0(sub_diagram, cluster_points)

                # 将子簇标签映射到全局标签
                unique_sub_labels = np.unique(sub_labels)
                for sub_label in unique_sub_labels:
                    sub_mask = sub_labels == sub_label
                    final_labels[cluster_indices[sub_mask]] = next_cluster_id
                    next_cluster_id += 1
            except Exception as e:
                logger.warning(f"PH failed for cluster {cluster_id}: {e}")
                final_labels[cluster_indices] = next_cluster_id
                next_cluster_id += 1

        # Step 3: 处理噪声点 - 分配给最近的簇
        noise_mask = final_labels == -1
        if noise_mask.any():
            noise_indices = np.where(noise_mask)[0]
            noise_points = points[noise_mask]

            if next_cluster_id > 0:
                cluster_centers = np.array([
                    points[final_labels == cid].mean(axis=0)
                    for cid in range(next_cluster_id)
                ])

                from sklearn.neighbors import NearestNeighbors
                nn = NearestNeighbors(n_neighbors=1, metric="cosine")
                nn.fit(cluster_centers)
                _, nearest = nn.kneighbors(noise_points)
                final_labels[noise_indices] = nearest.flatten()
            else:
                final_labels[noise_indices] = 0

        return final_labels

    def _estimate_dbscan_eps(
        self,
        points: np.ndarray,
        min_samples: int,
    ) -> float:
        """自动估计 DBSCAN 的 eps 参数。

        方法：k-distance 图的 90 分位数

        注意：UMAP 降维后的低维点使用 Euclidean 距离，
        因为 UMAP 已经用 cosine 相似度构建了邻域结构。
        """
        from sklearn.neighbors import NearestNeighbors

        n_points = len(points)
        if n_points < min_samples:
            return 0.5  # 默认值

        # P0 修复：UMAP 降维后的低维点用 Euclidean 距离
        # 判断依据：维度 <= 10 认为是 UMAP 降维后的点
        use_metric = "euclidean" if points.shape[1] <= 10 else self.metric

        # 计算 k 近邻距离
        nn = NearestNeighbors(n_neighbors=min_samples, metric=use_metric)
        nn.fit(points)
        distances, _ = nn.kneighbors(points)

        # 取第 k 近邻距离
        k_distances = np.sort(distances[:, -1])

        # 用 90 分位数
        eps = float(np.percentile(k_distances, 90))

        # 确保 eps 合理（低维空间通常需要更小的 eps）
        if points.shape[1] <= 10:
            return max(min(eps, 2.0), 0.1)
        return max(min(eps, 1.5), 0.05)

    def _auto_cluster_threshold(self, diagram: PersistenceDiagram) -> float:
        """从 H0 persistence 的 gap 自动选择切割阈值。

        改进版本：对 cosine metric 做了特殊处理。
        cosine distance 在 [0, 2] 范围，0=相同，2=正交。
        """
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

        # 如果 gap 策略失败，用百分位数
        # 对于 cosine，0.5-0.7 通常是合理的簇阈值
        if self.metric == "cosine":
            # 40th percentile 作为阈值（产生较少、较大的簇）
            return float(np.percentile(deaths, 40))

        return float(np.median(deaths))

    def estimate_n_clusters(self, points: np.ndarray, target_ratio: float = 0.1) -> int:
        """基于数据分布自动估计最佳簇数。

        方法：使用 K-NN 距离的统计量估算

        Args:
            points: shape (N, D)
            target_ratio: 期望的簇大小比例（默认 10% 的点 = 一个簇）

        Returns:
            估计的簇数
        """
        from scipy.spatial.distance import pdist

        n = len(points)
        if n < 10:
            return 2

        # 计算成对距离的样本
        if n > 500:
            # 采样以加速
            rng = np.random.RandomState(42)
            indices = rng.choice(n, size=500, replace=False)
            sample = points[indices]
        else:
            sample = points

        distances = pdist(sample, metric=self.metric)

        # 使用距离分布的统计量估算
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)

        # 简单启发式：mean + 0.5*std 以上的距离对被认为是"簇间"
        # 统计这样的点对比例，推算簇数
        threshold = mean_dist + 0.3 * std_dist

        if self.metric == "cosine":
            # Cosine distance: 0=相同, 2=正交
            # 阈值 0.5-0.7 对于语义相似文本是合理的
            if threshold < 0.4:
                threshold = 0.5
            elif threshold > 1.0:
                threshold = 0.7

        # 估计簇数
        n_clusters = max(2, min(int(n * target_ratio), n // 2))
        return n_clusters

    # ------------------------------------------------------------------
    # UMAP 降维辅助（P0 修复：高维空间聚类失效问题）
    # ------------------------------------------------------------------

    def _umap_reduce(self, points: np.ndarray) -> np.ndarray:
        """使用 UMAP 将高维点云降到低维，用于聚类前处理。

        关键发现：384D 空间 DBSCAN ARI=0.000，UMAP(2D)+DBSCAN ARI=0.945
        UMAP 保留了足够的几何结构用于拓扑聚类，同时解决了维度灾难。

        参数：
            points: shape (N, D) 原始高维 embedding

        返回：
            shape (N, umap_n_components) 降维后的点云

        异常：
            RuntimeError: 如果 UMAP 不可用或降维失败
        """
        try:
            import umap
        except ImportError:
            raise RuntimeError(
                "UMAP is required for use_umap_before_clustering=True but is not installed. "
                "Install with: pip install umap-learn"
            )

        n = len(points)
        n_components = getattr(self.config, 'umap_n_components', 2)
        n_neighbors = getattr(self.config, 'umap_n_neighbors', 15)
        min_dist = getattr(self.config, 'umap_min_dist', 0.1)

        # 验证：n_components 必须小于原始维度
        if n_components >= points.shape[1]:
            logger.warning(
                f"UMAP: n_components={n_components} >= original dim={points.shape[1]}, "
                f"skipping reduction"
            )
            return points

        # UMAP 参数调整：小数据集需要更小的 n_neighbors
        actual_neighbors = min(n_neighbors, n - 1)
        if actual_neighbors < 2:
            # 数据太少无法降维
            logger.warning(f"UMAP: too few points ({n}), skipping reduction")
            return points

        try:
            reducer = umap.UMAP(
                n_neighbors=actual_neighbors,
                min_dist=min_dist,
                n_components=n_components,
                metric='cosine',
                random_state=42,
            )
            reduced = reducer.fit_transform(points)
            logger.debug(f"UMAP: {points.shape[1]}D → {reduced.shape[1]}D")
            return np.asarray(reduced, dtype=np.float64)
        except Exception as e:
            logger.error(f"UMAP reduction failed: {e}, clustering will use original high-D points")
            raise RuntimeError(f"UMAP reduction failed: {e}") from e

    # ------------------------------------------------------------------
    # 完整结果: compute_full_result（便捷方法）
    # ------------------------------------------------------------------

    def compute_full_result(
        self,
        points: np.ndarray,
        max_clusters: Optional[int] = None,
    ) -> TopologyResult:
        """一次性计算所有拓扑特征，返回富结果。

        参数：
            points: shape (N, D)
            max_clusters: 目标最大簇数（用于控制 H0 微簇问题）

        返回：
            TopologyResult
        """
        diagram = self.compute_persistence(points)
        features = self.extract_persistent_features(diagram)
        fingerprint = self.topological_summary(diagram)
        cluster_labels = self.cluster_labels_from_h0(diagram, points, max_clusters=max_clusters)
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
