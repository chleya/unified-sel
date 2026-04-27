"""
topomem/memory.py — 图结构记忆系统

带拓扑关系的结构化记忆存储。双层索引：
- ChromaDB：向量 ANN 检索
- NetworkX：拓扑关系图（簇归属、邻居查询）

设计来源：
- SPEC_MEMORY.md: 图结构记忆系统完整规格
- ChromaDB：向量存储和检索
- NetworkX：图结构管理
"""

from __future__ import annotations

import json
import logging
import math
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np

from topomem.config import MemoryConfig
from topomem.embedding import EmbeddingManager
from topomem.topology import TopologyEngine, TopologyResult
from topomem.health_controller import TopologyHealthController, HealthStatus


logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# 数据结构
# ------------------------------------------------------------------

@dataclass
class MemoryNode:
    """一个记忆节点 = 一条知识 + 其结构化标注。"""

    # 核心字段（创建时确定）
    id: str
    content: str
    embedding: np.ndarray
    created_at: float
    metadata: dict = field(default_factory=dict)

    # 访问统计（动态更新）
    last_accessed: float = 0.0
    access_count: int = 0

    # 拓扑标注（由 update_topology 更新）
    cluster_id: int = -1
    persistence_score: float = 0.0

    # 衍生
    importance_score: float = 0.0

    def to_dict(self) -> dict:
        """转为字典（embedding 转 list 以便 JSON 序列化）。"""
        d = asdict(self)
        d["embedding"] = self.embedding.tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryNode":
        """从字典恢复（embedding list 转 np.ndarray）。"""
        d = dict(d)
        d["embedding"] = np.array(d["embedding"], dtype=np.float32)
        return cls(**d)


def compute_importance(
    node: MemoryNode,
    current_time: float,
    max_access_count: int = 1,
    decay: float = 0.001,
) -> float:
    """综合重要性 = 拓扑持久性 × 访问频率 × 时间衰减。

    importance = (
        0.5 * persistence_score_normalized +
        0.3 * log(1 + access_count) / log(1 + max_access) +
        0.2 * exp(-decay * (current_time - last_accessed))
    )
    """
    persistence = max(0.0, min(1.0, node.persistence_score))
    access = math.log(1 + node.access_count) / max(math.log(1 + max_access_count), 1e-8)
    recency = math.exp(-decay * (current_time - node.last_accessed))
    return 0.5 * persistence + 0.3 * access + 0.2 * recency


# ------------------------------------------------------------------
# MemoryGraph 实现
# ------------------------------------------------------------------

class MemoryGraph:
    """带拓扑关系的图结构记忆系统。

    双层索引：
    - ChromaDB：负责向量 ANN 检索
    - NetworkX：负责拓扑关系图
    """

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        embedding_mgr: Optional[EmbeddingManager] = None,
    ):
        self.config = config or MemoryConfig()
        self._embedding_mgr = embedding_mgr

        # NetworkX 图
        self._graph = nx.Graph()

        # ChromaDB 连接
        import chromadb
        persist_dir = self.config.chroma_persist_dir
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self._chroma_client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._chroma_client.get_or_create_collection(
            name="topomem_memory",
            metadata={"hnsw:space": "cosine"},
        )

        # 拓扑更新计数器
        self._inserts_since_topo_update = 0
        self._topology_cache: Optional[TopologyResult] = None

        # 拓扑健康指标（P1-1: 用于动态调整检索权重）
        self._h1_health: float = 1.0
        self._h2_health: float = 1.0
        self._betti_1_count: int = 0
        self._betti_2_count: int = 0

        # 拓扑健康ECU - 统一控制器
        self._health_controller = TopologyHealthController()
        self._health_status: Optional[HealthStatus] = None

    # ==================================================================
    # 写入
    # ==================================================================

    def add_memory(
        self,
        content: str,
        embedding: np.ndarray,
        metadata: Optional[dict] = None,
        topo_engine: Optional[TopologyEngine] = None,
    ) -> MemoryNode:
        """添加一条记忆。

        参数：
            content: 文本内容
            embedding: 384 维特征向量
            metadata: 任意元数据
            topo_engine: 拓扑引擎（可选，传入后会在达到阈值时自动更新拓扑）

        返回：
            MemoryNode
        """
        if not content:
            logger.warning("Adding memory with empty content")

        node_id = str(uuid.uuid4())
        now = time.time()

        node = MemoryNode(
            id=node_id,
            content=content,
            embedding=embedding.astype(np.float32),
            created_at=now,
            metadata=metadata or {},
            last_accessed=now,
            access_count=0,
            cluster_id=-1,
            persistence_score=0.0,
        )

        # 添加到 NetworkX
        self._graph.add_node(node_id, node=node)

        # 添加到 ChromaDB
        chroma_metadata = metadata.copy() if metadata else {"_placeholder": True}
        self._collection.add(
            ids=[node_id],
            embeddings=[embedding.tolist()],
            documents=[content],
            metadatas=[chroma_metadata],
        )

        # 拓扑更新计数
        self._inserts_since_topo_update += 1
        self._topology_cache = None  # 使缓存失效

        if topo_engine and self._inserts_since_topo_update >= self.config.topo_recompute_interval:
            self.update_topology(topo_engine)
            self._inserts_since_topo_update = 0

        return node

    def add_memory_from_text(
        self,
        content: str,
        metadata: Optional[dict] = None,
        topo_engine: Optional[TopologyEngine] = None,
    ) -> MemoryNode:
        """便利方法：自动编码文本后添加。"""
        if self._embedding_mgr is None:
            raise RuntimeError(
                "EmbeddingManager not provided. "
                "Initialize MemoryGraph with embedding_mgr to use this method."
            )
        embedding = self._embedding_mgr.encode(content)
        return self.add_memory(content, embedding, metadata, topo_engine)

    def add_memory_batch(
        self,
        items: List[dict],
        topo_engine: Optional[TopologyEngine] = None,
    ) -> List[MemoryNode]:
        """批量添加多条记忆（一次 ChromaDB IO，而非 N 次）。

        参数：
            items: List[dict]，每个 dict 包含：
                - content (str): 文本内容
                - embedding (np.ndarray): 384 维特征向量
                - metadata (Optional[dict]): 元数据
            topo_engine: 拓扑引擎（达到阈值时自动更新拓扑）

        返回：
            List[MemoryNode]
        """
        if not items:
            return []

        now = time.time()
        node_ids = [str(uuid.uuid4()) for _ in items]
        nodes = []

        # ---- 1. 批量构建 MemoryNode + NetworkX ----
        for i, item in enumerate(items):
            node = MemoryNode(
                id=node_ids[i],
                content=item["content"],
                embedding=item["embedding"].astype(np.float32),
                created_at=now,
                metadata=item.get("metadata") or {},
                last_accessed=now,
                access_count=0,
                cluster_id=-1,
                persistence_score=0.0,
            )
            self._graph.add_node(node_ids[i], node=node)
            nodes.append(node)

        # ---- 2. 一次 ChromaDB add() ----
        chroma_metadatas = [
            (item.get("metadata") or {}).copy() or {"_placeholder": True}
            for item in items
        ]
        self._collection.add(
            ids=node_ids,
            embeddings=[item["embedding"].tolist() for item in items],
            documents=[item["content"] for item in items],
            metadatas=chroma_metadatas,
        )

        # ---- 3. 拓扑更新计数 ----
        self._inserts_since_topo_update += len(items)
        self._topology_cache = None

        if topo_engine and self._inserts_since_topo_update >= self.config.topo_recompute_interval:
            self.update_topology(topo_engine)
            self._inserts_since_topo_update = 0

        return nodes

    # ==================================================================
    # 检索
    # ==================================================================

    def retrieve(
        self,
        query_embedding: np.ndarray,
        strategy: str = "hybrid",
        k: int = 5,
    ) -> List[Tuple[MemoryNode, float]]:
        """根据策略检索相关记忆。

        参数：
            query_embedding: 查询向量 (384,)
            strategy: "vector" | "topological" | "hybrid"
            k: 返回节点数

        返回：
            List[Tuple[MemoryNode, float]]，按相关性降序。
            float 是相似度分数 (0~1, 越高越相关)。
        """
        n_nodes = self.node_count()
        if n_nodes == 0:
            return []

        k = min(k, n_nodes)

        if strategy == "vector":
            results = self._retrieve_vector(query_embedding, k)
        elif strategy == "topological":
            results = self._retrieve_topological(query_embedding, k)
        elif strategy == "hybrid":
            results = self._retrieve_hybrid(query_embedding, k)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # 更新访问统计（原地写回 graph，返回副本防止外部篡改内部状态）
        now = time.time()
        safe_results = []
        for node, score in results:
            # 原地更新 graph 内部节点
            graph_node = self._graph.nodes[node.id]["node"]
            graph_node.access_count += 1
            graph_node.last_accessed = now
            # 返回副本，外部修改不影响内部状态
            node_copy = replace(graph_node)
            safe_results.append((node_copy, score))

        return safe_results

    def _retrieve_vector(
        self,
        query_embedding: np.ndarray,
        k: int,
    ) -> List[Tuple[MemoryNode, float]]:
        """ChromaDB 向量检索。返回 (节点, 相似度分数)。"""
        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        scored_nodes = []
        if results["ids"] and results["ids"][0]:
            distances = results.get("distances", [[]])
            if distances and distances[0]:
                for i, node_id in enumerate(results["ids"][0]):
                    if self._graph.has_node(node_id):
                        dist = distances[0][i]
                        # L2距离 → 相似度分数: 1/(1+dist)，范围 (0,1]
                        score = 1.0 / (1.0 + dist)
                        node_data = self._graph.nodes[node_id]
                        scored_nodes.append((node_data["node"], score))

        return scored_nodes

    def _retrieve_topological(
        self,
        query_embedding: np.ndarray,
        k: int,
    ) -> List[Tuple[MemoryNode, float]]:
        """拓扑检索：选择最近簇，然后在簇内检索。返回 (节点, 相似度分数)。"""
        centers = self.get_cluster_centers()
        if not centers:
            return self._retrieve_vector(query_embedding, k)

        # 计算 query 与每个簇中心的余弦距离
        def cosine_sim(a, b):
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a < 1e-8 or norm_b < 1e-8:
                return 0.0
            return float(np.dot(a, b) / (norm_a * norm_b))

        similarities = {
            cid: cosine_sim(query_embedding, center)
            for cid, center in centers.items()
        }

        # 动态簇数量：碎片化时扩展覆盖，保证至少有 k 个候选
        n_clusters = len(centers)
        n_nodes = self.node_count()
        if n_nodes == 0 or n_clusters == 0:
            return self._retrieve_vector(query_embedding, k)
        
        avg_size = n_nodes / n_clusters
        # 关键启发式：碎片化时(n_clusters >= k)确保覆盖 ≥ k 节点
        if n_clusters <= 2:
            top_n = n_clusters
        elif n_clusters <= 5:
            top_n = min(3, n_clusters)
        else:
            top_n = max(2, min(n_clusters, int(np.ceil(k / avg_size))))

        sorted_clusters = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        top_cluster_ids = [cid for cid, _ in sorted_clusters[:top_n]]

        # 在选中的簇内检索
        candidates = []
        for cid in top_cluster_ids:
            cluster_nodes = self.retrieve_by_cluster(cid)
            for node in cluster_nodes:
                sim = cosine_sim(query_embedding, node.embedding)
                candidates.append((node, sim))

        # 按相似度排序
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:k]

    def _retrieve_hybrid(
        self,
        query_embedding: np.ndarray,
        k: int,
    ) -> List[Tuple[MemoryNode, float]]:
        """混合检索：vector + topological + H0 persistence 综合排序。

        方向4核心改动：引入 H0 persistence 作为 re-ranking 信号。

        综合分数 = alpha * vector_sim + beta * topo_sim + gamma * persistence
        其中 persistence 反映记忆的拓扑稳定性（高 = 更核心的记忆）
        """
        vector_results = self._retrieve_vector(query_embedding, k)
        topo_results = self._retrieve_topological(query_embedding, k)

        # 合并去重，追踪每个结果的来源
        seen: Dict[str, Dict[str, float]] = {}
        for node, score in vector_results:
            if node.id not in seen:
                seen[node.id] = {"vector": score, "topo": 0.0}
            elif score > seen[node.id]["vector"]:
                seen[node.id]["vector"] = score
        for node, score in topo_results:
            if node.id not in seen:
                seen[node.id] = {"vector": 0.0, "topo": score}
            elif score > seen[node.id]["topo"]:
                seen[node.id]["topo"] = score

        # 综合排序：引入 persistence 权重因子（方向4）
        alpha = self.config.retrieval_vector_weight
        beta = self.config.retrieval_topo_weight
        gamma = self.config.retrieval_persistence_weight

        # P1-1: 使用健康ECU统一控制器 - 获取健康状态
        health_status = self.get_health_status()

        # 从健康状态获取调整后的 gamma
        adjusted_gamma = gamma * health_status.retrieval_gamma_mult
        # 重新归一化权重
        total_weight = alpha + beta + adjusted_gamma
        alpha_norm = alpha / total_weight
        beta_norm = beta / total_weight
        gamma_norm = adjusted_gamma / total_weight

        # P0-2: 干扰抵抗 - 计算每个簇的质量分数
        cluster_quality = {}
        all_cluster_ids = self._get_all_cluster_ids()
        for cid in all_cluster_ids:
            cluster_nodes = self.retrieve_by_cluster(cid)
            if cluster_nodes:
                avg_persistence = np.mean([n.persistence_score for n in cluster_nodes])
                cluster_quality[cid] = avg_persistence

        # P0-2: 干扰过滤 - 健康差时启用
        cluster_filter_enabled = health_status.cluster_filter_enabled
        if cluster_filter_enabled and cluster_quality:
            quality_threshold = np.mean(list(cluster_quality.values())) * 0.5
        else:
            quality_threshold = 0.0

        def composite_score_from_seen(item: Tuple[str, Dict[str, float]]) -> float:
            node_id, scores = item
            vec_score = scores["vector"]
            topo_score = scores["topo"]
            # 获取 node 对象
            node = self._graph.nodes[node_id]["node"]
            persistence = max(0.0, min(1.0, node.persistence_score))

            # P0-2: 干扰簇惩罚 - 来自弱簇的节点被惩罚
            node_cluster_quality = cluster_quality.get(node.cluster_id, 0.0)
            if node_cluster_quality < quality_threshold and quality_threshold > 0:
                # 降低来自弱簇节点的分数
                penalty = 0.5
            else:
                penalty = 1.0

            return penalty * (alpha_norm * vec_score + beta_norm * topo_score + gamma_norm * persistence)

        # 排序
        sorted_items = sorted(seen.items(), key=composite_score_from_seen, reverse=True)

        # 返回 (node, 综合分数) 格式
        results = []
        for node_id, scores in sorted_items[:k]:
            node = self._graph.nodes[node_id]["node"]
            vec_score = scores["vector"]
            topo_score = scores["topo"]
            composite = composite_score_from_seen((node_id, scores))
            results.append((node, composite))

        return results

    def retrieve_by_cluster(self, cluster_id: int) -> List[MemoryNode]:
        """返回指定拓扑簇中的所有记忆。"""
        nodes = []
        for node_id, data in self._graph.nodes(data=True):
            node = data["node"]
            if node.cluster_id == cluster_id:
                nodes.append(node)
        return nodes

    def get_cluster_centers(self) -> Dict[int, np.ndarray]:
        """返回每个簇的中心 embedding。"""
        centers = {}
        for cid in self._get_all_cluster_ids():
            cluster_nodes = self.retrieve_by_cluster(cid)
            if cluster_nodes:
                embeddings = np.stack([n.embedding for n in cluster_nodes])
                centers[cid] = np.mean(embeddings, axis=0)
        return centers

    def _get_all_cluster_ids(self) -> List[int]:
        """获取所有已分配的簇 ID。"""
        cids = set()
        for _, data in self._graph.nodes(data=True):
            cid = data["node"].cluster_id
            if cid >= 0:
                cids.add(cid)
        return sorted(cids)

    def get_h1_health(self) -> float:
        """返回 H1 健康分数。"""
        return self._h1_health

    def get_h2_health(self) -> float:
        """返回 H2 健康分数。"""
        return self._h2_health

    def get_betti_counts(self) -> Tuple[int, int]:
        """返回 (betti_1_count, betti_2_count)。"""
        return (self._betti_1_count, self._betti_2_count)

    def get_health_status(self) -> HealthStatus:
        """返回当前健康状态（通过ECU统一控制器计算）。"""
        if self._health_status is None:
            # 如果还没计算过，返回默认健康状态
            self._health_status = self._health_controller.compute_health_status(
                h1_health=self._h1_health,
                h2_health=self._h2_health,
                betti_1_count=self._betti_1_count,
                betti_2_count=self._betti_2_count,
            )
        return self._health_status

    def get_diagnostic_info(self) -> dict:
        """返回诊断信息（用于日志和调试）。"""
        return self._health_controller.get_diagnostic_info(self.get_health_status())

    def get_fault_log(self, max_records: int = 50) -> list:
        """返回 OBD 故障日志。"""
        return self._health_controller.get_fault_log(max_records)

    # ==================================================================
    # 拓扑管理
    # ==================================================================

    def update_topology(self, topo_engine: TopologyEngine) -> TopologyResult:
        """重新计算所有节点的拓扑关系。

        参数：
            topo_engine: TopologyEngine 实例

        返回：
            TopologyResult
        """
        n_nodes = self.node_count()
        if n_nodes == 0:
            raise ValueError("Cannot update topology of empty graph")

        # 收集所有 embedding
        node_ids = list(self._graph.nodes())
        embeddings = np.stack([
            self._graph.nodes[nid]["node"].embedding
            for nid in node_ids
        ])

        # 计算拓扑
        diagram = topo_engine.compute_persistence(embeddings)
        features = topo_engine.extract_persistent_features(diagram)
        fingerprint = topo_engine.topological_summary(diagram)

        # 选择聚类方法
        # 默认使用混合聚类（DBSCAN 预聚类 + H0 细化）
        clustering_method = getattr(topo_engine.config, 'clustering_method', 'hybrid')
        
        if clustering_method == 'hybrid':
            # 混合聚类：解决 H0 单点簇问题
            cluster_labels = topo_engine.cluster_labels_hybrid(
                embeddings,
                diagram=diagram,
            )
        elif clustering_method == 'dbscan':
            # 纯 DBSCAN
            dbscan_eps = getattr(topo_engine.config, 'dbscan_eps', None)
            dbscan_min_samples = getattr(topo_engine.config, 'dbscan_min_samples', 3)
            cluster_labels = topo_engine.cluster_labels_from_dbscan(
                embeddings, eps=dbscan_eps, min_samples=dbscan_min_samples
            )
        else:
            # H0 single-linkage（原始方法）
            max_clusters = getattr(topo_engine.config, 'max_h0_clusters', None)
            cluster_labels = topo_engine.cluster_labels_from_h0(
                diagram, embeddings, max_clusters=max_clusters
            )

        # 构建拓扑结果
        n_clusters = len(set(cluster_labels) - {-1}) if -1 in cluster_labels else len(set(cluster_labels))
        topology_result = TopologyResult(
            diagram=diagram,
            features=features,
            fingerprint=fingerprint,
            n_clusters=n_clusters,
            cluster_labels=cluster_labels,
        )

        # 计算每个簇的 persistence score
        # 同一簇内的节点共享相同的 persistence（来自 H0 特征）
        h0_features = [f for f in features if f.dimension == 0]
        # 按簇 ID 映射 persistence
        cluster_persistence = {}
        if h0_features and cluster_labels is not None:
            # 简化：取所有 H0 特征的平均 persistence 作为基线
            avg_persistence = np.mean([f.persistence for f in h0_features])
            for cid in np.unique(cluster_labels):
                cluster_persistence[int(cid)] = float(avg_persistence)

        # 更新节点属性
        for i, nid in enumerate(node_ids):
            node = self._graph.nodes[nid]["node"]
            if cluster_labels is not None:
                node.cluster_id = int(cluster_labels[i])
                node.persistence_score = cluster_persistence.get(node.cluster_id, 0.0)

        # 同步 cluster_id 和 persistence_score 到 ChromaDB metadata
        if cluster_labels is not None:
            for i, nid in enumerate(node_ids):
                node = self._graph.nodes[nid]["node"]
                # 更新 ChromaDB metadata
                self._collection.update(
                    ids=[nid],
                    metadatas=[{
                        "cluster_id": node.cluster_id,
                        "persistence_score": node.persistence_score,
                    }]
                )

        # 更新 NetworkX 边：同一簇内两两建边
        self._graph.clear_edges()
        for cid in self._get_all_cluster_ids():
            cluster_node_ids = [
                nid for nid, data in self._graph.nodes(data=True)
                if data["node"].cluster_id == cid
            ]
            # 如果簇太大，只连 k-nearest 邻居
            if len(cluster_node_ids) > 50:
                self._add_knn_edges(cluster_node_ids, embeddings, node_ids, k=5)
            else:
                for i in range(len(cluster_node_ids)):
                    for j in range(i + 1, len(cluster_node_ids)):
                        self._graph.add_edge(cluster_node_ids[i], cluster_node_ids[j])

        # 缓存结果
        self._topology_cache = topology_result

        # 更新 H1/H2 健康指标（P1-1）
        h1_features = [f for f in features if f.dimension == 1]
        h2_features = [f for f in features if f.dimension == 2]
        self._betti_1_count = len(h1_features)
        self._betti_2_count = len(h2_features)
        if h1_features:
            self._h1_health = float(np.mean([f.persistence for f in h1_features]))
        if h2_features:
            self._h2_health = float(np.mean([f.persistence for f in h2_features]))

        # 计算健康状态（通过 ECU 统一控制器）
        self._health_status = self._health_controller.compute_health_status(
            h1_health=self._h1_health,
            h2_health=self._h2_health,
            betti_1_count=self._betti_1_count,
            betti_2_count=self._betti_2_count,
        )

        self._topology_cache = topology_result
        return topology_result

    def _add_knn_edges(
        self,
        cluster_node_ids: List[str],
        all_embeddings: np.ndarray,
        all_node_ids: List[str],
        k: int = 5,
    ):
        """为大簇添加 k-NN 边而非全连接。"""
        from sklearn.neighbors import NearestNeighbors

        # 提取该簇的 embedding 索引
        cluster_indices = [all_node_ids.index(nid) for nid in cluster_node_ids]
        cluster_embeddings = all_embeddings[cluster_indices]

        # 计算 k-NN
        nn = NearestNeighbors(n_neighbors=min(k + 1, len(cluster_embeddings)), metric="cosine")
        nn.fit(cluster_embeddings)
        distances, indices = nn.kneighbors(cluster_embeddings)

        # 添加边
        for i, nid in enumerate(cluster_node_ids):
            for j_idx in indices[i][1:]:  # 跳过自己
                neighbor_nid = cluster_node_ids[j_idx]
                self._graph.add_edge(nid, neighbor_nid)

    def get_topological_summary(
        self,
        topo_engine: TopologyEngine,
    ) -> np.ndarray:
        """返回当前记忆图的全局拓扑指纹。"""
        if self._topology_cache is not None:
            return self._topology_cache.fingerprint

        if self.node_count() == 0:
            return np.zeros(100)

        result = self.update_topology(topo_engine)
        return result.fingerprint

    # ==================================================================
    # 容量管理
    # ==================================================================

    def prune(self, max_nodes: Optional[int] = None) -> List[str]:
        """移除低重要性节点。

        参数：
            max_nodes: 最大节点数，None 则使用 config.max_nodes

        返回：
            被删除的 node_id 列表
        """
        limit = max_nodes if max_nodes is not None else self.config.max_nodes
        current = self.node_count()

        if current <= limit:
            return []

        n_to_remove = current - limit
        now = time.time()

        # 计算簇质量分数（P1-3: 用于 pruning 优先级）
        cluster_quality = {}
        all_cluster_ids = self._get_all_cluster_ids()
        for cid in all_cluster_ids:
            cluster_nodes = self.retrieve_by_cluster(cid)
            if cluster_nodes:
                avg_persistence = np.mean([n.persistence_score for n in cluster_nodes])
                cluster_quality[cid] = avg_persistence

        # P1-3: 使用健康ECU统一控制器
        health_status = self.get_health_status()

        # 计算重要性
        all_nodes = [data["node"] for _, data in self._graph.nodes(data=True)]
        max_access = max((n.access_count for n in all_nodes), default=1)

        for node in all_nodes:
            base_importance = compute_importance(
                node, now, max_access, decay=self.config.importance_decay
            )
            # P1-3: 簇质量因子 - 弱簇中的节点更容易被删除
            cq = cluster_quality.get(node.cluster_id, 0.0)
            avg_cq = np.mean(list(cluster_quality.values())) if cluster_quality else 0.0
            # 低于平均簇质量的节点受到惩罚
            if avg_cq > 0 and cq < avg_cq * 0.5:
                quality_penalty = 0.7  # 降低 30%
            else:
                quality_penalty = 1.0
            # P1-3: 全局健康惩罚 - 使用 ECU 输出的 aggressiveness
            prune_aggressive = health_status.prune_aggressiveness
            health_adjustment = 1.0 - 0.2 * prune_aggressive  # 范围 [0.8, 1.0]
            node.importance_score = base_importance * quality_penalty * health_adjustment

        # 保护规则
        protected_ids = set()
        # 1. 每个 cluster 至少保留 1 个
        for cid in self._get_all_cluster_ids():
            cluster_nodes = [
                n for n in all_nodes if n.cluster_id == cid
            ]
            if cluster_nodes:
                best = max(cluster_nodes, key=lambda n: n.importance_score)
                protected_ids.add(best.id)

        # 2. 最近创建的节点保护（ configurable，保护最近 N 步内创建的）
        recent_limit = self.config.prune_recent_protection
        sorted_by_time = sorted(all_nodes, key=lambda n: n.created_at, reverse=True)
        for node in sorted_by_time[:recent_limit]:
            protected_ids.add(node.id)

        # 选择要删除的节点
        candidates = [n for n in all_nodes if n.id not in protected_ids]
        candidates.sort(key=lambda n: n.importance_score)
        to_remove = candidates[:n_to_remove]

        removed_ids = []
        for node in to_remove:
            self._delete_node(node.id)
            removed_ids.append(node.id)

        return removed_ids

    def _delete_node(self, node_id: str) -> None:
        """内部方法：从 NetworkX 和 ChromaDB 中删除节点。"""
        # 从 NetworkX 删除
        if self._graph.has_node(node_id):
            self._graph.remove_node(node_id)

        # 从 ChromaDB 删除
        try:
            self._collection.delete(ids=[node_id])
        except Exception as e:
            logger.warning(f"Failed to delete from ChromaDB: {e}")

        self._topology_cache = None  # 缓存失效

    def node_count(self) -> int:
        """当前节点总数。"""
        return self._graph.number_of_nodes()

    # ==================================================================
    # 序列化
    # ==================================================================

    def save(self, path: str) -> None:
        """持久化到磁盘。"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # 准备可序列化的图数据
        # 将 MemoryNode 转换为字典
        serializable_graph = nx.Graph()
        for nid, data in self._graph.nodes(data=True):
            node = data["node"]
            node_dict = node.to_dict()  # 包含 embedding as list
            serializable_graph.add_node(nid, node=node_dict)

        # 复制边
        for u, v in self._graph.edges():
            serializable_graph.add_edge(u, v)

        graph_data = nx.node_link_data(serializable_graph)
        with open(path / "graph.json", "w", encoding="utf-8") as f:
            json.dump(graph_data, f)

        # 保存元数据
        meta = {
            "inserts_since_topo_update": self._inserts_since_topo_update,
            "node_count": self.node_count(),
        }
        with open(path / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

        logger.info(f"MemoryGraph saved to {path}")

    def load(self, path: str) -> None:
        """从磁盘加载。"""
        path = Path(path)

        # 加载 NetworkX 图
        with open(path / "graph.json", "r", encoding="utf-8") as f:
            graph_data = json.load(f)

        serializable_graph = nx.node_link_graph(graph_data)

        # 恢复 MemoryNode 对象
        self._graph = nx.Graph()
        for nid, data in serializable_graph.nodes(data=True):
            node_dict = data["node"]
            node = MemoryNode.from_dict(node_dict)
            self._graph.add_node(nid, node=node)

        # 复制边
        for u, v in serializable_graph.edges():
            self._graph.add_edge(u, v)

        # 加载元数据
        meta_path = path / "meta.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self._inserts_since_topo_update = meta.get("inserts_since_topo_update", 0)

        logger.info(f"MemoryGraph loaded from {path} ({self.node_count()} nodes)")

    # ==================================================================
    # 内部辅助
    # ==================================================================

    def _check_sync(self) -> bool:
        """检查 ChromaDB 和 NetworkX 是否同步。"""
        chroma_ids = set(self._collection.get()["ids"])
        networkx_ids = set(self._graph.nodes())
        return chroma_ids == networkx_ids

    def __repr__(self) -> str:
        n_clusters = len(self._get_all_cluster_ids())
        return (
            f"MemoryGraph(nodes={self.node_count()}, "
            f"clusters={n_clusters}, "
            f"pending_topo_update={self._inserts_since_topo_update})"
        )


def _json_default(obj):
    """JSON 序列化辅助函数。"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
