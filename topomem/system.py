"""
topomem/system.py — TopoMem Reasoner v0.1 完整系统

唯一的用户入口，协调所有内部模块。
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# 必须在导入任何 HF 库之前设置环境变量
HF_CACHE = r"F:\unified-sel\topomem\data\models\hf_cache"
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = HF_CACHE
if "TRANSFORMERS_CACHE" not in os.environ:
    os.environ["TRANSFORMERS_CACHE"] = HF_CACHE
if "SENTENCE_TRANSFORMERS_HOME" not in os.environ:
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = HF_CACHE


import psutil

from topomem.adapters import (
    AdapterPool,
    PromptAdapter,
    compute_surprise,
    compute_tension,
    decide_action,
    create_default_adapter,
)
from topomem.config import TopoMemConfig
from topomem.embedding import EmbeddingManager
from topomem.engine import ReasoningEngine, extract_knowledge
from topomem.guard import ConsistencyGuard
from topomem.memory import MemoryGraph, MemoryNode
from topomem.self_awareness import SelfAwareness
from topomem.topology import TopologyEngine


logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# 数据结构
# ------------------------------------------------------------------

@dataclass
class ProcessResult:
    """一次 process() 调用的完整输出。"""

    response_text: str
    retrieved_memories: List[dict]
    retrieval_strategy: str
    adapter_used: str
    surprise_score: float
    memory_accepted: bool
    memory_reject_reason: Optional[str]
    drift_status: Optional[str]
    calibrated: bool
    latency_ms: float
    step: int


@dataclass
class SystemMetrics:
    """系统运行时指标。用于监控和调优。"""
    # Operation counts
    ask_calls: int = 0
    retrieve_calls: int = 0
    add_memory_calls: int = 0
    add_memory_batch_calls: int = 0
    consolidation_calls: int = 0
    create_adapter_calls: int = 0
    create_adapter_skipped: int = 0  # similarity check
    update_topology_calls: int = 0

    # Latency (ms, accumulated)
    ask_latency_ms: float = 0.0
    retrieve_latency_ms: float = 0.0
    add_memory_latency_ms: float = 0.0
    consolidation_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0

    # TDA cache
    tda_cache_hits: int = 0
    tda_cache_misses: int = 0

    # Consolidation results
    orphans_detected: int = 0
    merge_candidates_found: int = 0

    # Current state snapshots
    current_node_count: int = 0
    current_cluster_count: int = 0
    current_adapter_count: int = 0

    # H1 geometric quality (P4)
    h1_health: float = 1.0
    h1_drift: float = 0.0
    # H2 domain-bridging (P4 extension)
    betti_2_count: int = 0
    h2_to_h1_ratio: float = 0.0
    h2_health: float = 1.0
    h2_drift: float = 0.0
    cavitation_rate: float = 0.0

    def to_dict(self) -> dict:
        """转换为可序列化的字典（包含计算字段）。"""
        total_tda = self.tda_cache_hits + self.tda_cache_misses
        tda_hit_rate = (
            self.tda_cache_hits / total_tda if total_tda > 0 else 0.0
        )
        return {
            # Counts
            "ask_calls": self.ask_calls,
            "retrieve_calls": self.retrieve_calls,
            "add_memory_calls": self.add_memory_calls,
            "add_memory_batch_calls": self.add_memory_batch_calls,
            "consolidation_calls": self.consolidation_calls,
            "create_adapter_calls": self.create_adapter_calls,
            "create_adapter_skipped": self.create_adapter_skipped,
            "update_topology_calls": self.update_topology_calls,
            # Latency (ms)
            "ask_latency_ms": round(self.ask_latency_ms, 2),
            "retrieve_latency_ms": round(self.retrieve_latency_ms, 2),
            "add_memory_latency_ms": round(self.add_memory_latency_ms, 2),
            "consolidation_latency_ms": round(self.consolidation_latency_ms, 2),
            "llm_latency_ms": round(self.llm_latency_ms, 2),
            # Averages
            "avg_ask_ms": round(self.ask_latency_ms / self.ask_calls, 1) if self.ask_calls > 0 else 0,
            "avg_retrieve_ms": round(self.retrieve_latency_ms / self.retrieve_calls, 1) if self.retrieve_calls > 0 else 0,
            # TDA
            "tda_cache_hits": self.tda_cache_hits,
            "tda_cache_misses": self.tda_cache_misses,
            "tda_hit_rate": round(tda_hit_rate, 3),
            # Consolidation
            "orphans_detected": self.orphans_detected,
            "merge_candidates_found": self.merge_candidates_found,
            # Current state
            "current_node_count": self.current_node_count,
            "current_cluster_count": self.current_cluster_count,
            "current_adapter_count": self.current_adapter_count,
            # H1 geometric quality (P4)
            "h1_health": round(self.h1_health, 3) if hasattr(self, 'h1_health') else 1.0,
            "h1_drift": round(self.h1_drift, 4) if hasattr(self, 'h1_drift') else 0.0,
            # H2 domain-bridging (P4 extension)
            "betti_2_count": self.betti_2_count,
            "h2_to_h1_ratio": round(self.h2_to_h1_ratio, 4),
            "h2_health": round(self.h2_health, 4),
            "h2_drift": round(self.h2_drift, 4),
            "cavitation_rate": round(self.cavitation_rate, 4),
        }


@dataclass
class SystemStatus:
    """系统状态快照。"""

    step: int
    memory_node_count: int
    memory_cluster_count: int
    adapter_count: int
    drift_status: str
    last_calibration_step: int
    ram_usage_mb: float
    fingerprint_shape: Optional[tuple]
    h1_health: float = 1.0          # H1 geometric health score (P4)
    h1_drift: float = 0.0           # H1 drift vs baseline (P4)
    # H2 domain-bridging (P4 extension)
    betti_2_count: int = 0
    h2_to_h1_ratio: float = 0.0
    h2_health: float = 1.0
    h2_drift: float = 0.0


# ------------------------------------------------------------------
# TopoMemSystem 实现
# ------------------------------------------------------------------

class TopoMemSystem:
    """TopoMem Reasoner v0.1 完整系统。

    这是唯一的用户入口。所有内部模块通过此类协调。
    """

    def __init__(self, config: Optional[TopoMemConfig] = None):
        """
        初始化顺序严格按依赖关系。
        """
        self.config = config or TopoMemConfig()
        self._step = 0
        self._last_calibration_step = 0
        self._process_log: List[ProcessResult] = []
        self._metrics = SystemMetrics()

        # 1. Embedding
        self.embedding = EmbeddingManager(self.config.embedding)

        # 2. Topology
        self.topology = TopologyEngine(self.config.topology)

        # 3. Memory
        self.memory = MemoryGraph(self.config.memory, self.embedding)

        # 4. Reasoning Engine
        self.engine = ReasoningEngine(self.config.engine)

        # 5. Self Awareness
        self.self_aware = SelfAwareness(self.config.awareness)

        # 6. Consistency Guard
        self.guard = ConsistencyGuard(self.config.awareness)

        # 7. Adapter Pool
        self.adapters = AdapterPool(self.config.adapter, self.embedding)

        logger.info("TopoMemSystem initialized")

    def process(self, input_text: str) -> ProcessResult:
        """完整的输入处理流程。

        1. 编码输入
        2. 选择 adapter
        3. 检索记忆
        4. 推理生成
        5. 知识提取与守护
        6. Adapter 反馈
        7. Surprise 驱动的 adapter 创建
        8. 自我认知更新

        返回 ProcessResult。
        """
        start_time = time.time()
        self._step += 1

        # ---- Step 1: 编码输入 ----
        query_embedding = self.embedding.encode(input_text)

        # ---- Step 2: 选择 adapter ----
        adapter, surprise = self.adapters.select_adapter(
            query_embedding, self.memory
        )

        # ---- Step 3: 检索记忆 ----
        strategy = "hybrid"
        retrieved = self.memory.retrieve(
            query_embedding,
            strategy=strategy,
            k=self.config.memory.similarity_top_k,
        )

        # ---- Step 4: 推理生成 ----
        system_prompt = adapter.apply(input_text)
        context_dicts = [
            {
                "content": m.content,
                "cluster_id": m.cluster_id,
                "access_count": m.access_count,
                "relevance_score": round(score, 3),
            }
            for m, score in retrieved
        ]

        try:
            response = self.engine.generate(
                prompt=input_text,
                context=context_dicts if context_dicts else None,
                system_prompt=system_prompt,
            )
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            response = f"[Error generating response: {e}]"

        # ---- Step 5: 知识提取与守护 ----
        new_knowledge = extract_knowledge(input_text, response)
        memory_accepted = False
        reject_reason = None

        if new_knowledge:
            knowledge_embedding = self.embedding.encode(new_knowledge)
            accepted, reason = self.guard.should_accept_memory(
                new_knowledge,
                knowledge_embedding,
                self.memory,
                self.self_aware,
                self.topology,
            )
            if accepted:
                self.memory.add_memory(
                    content=new_knowledge,
                    embedding=knowledge_embedding,
                    metadata={
                        "source": "process",
                        "step": self._step,
                    },
                    topo_engine=self.topology,
                )
                memory_accepted = True
            else:
                reject_reason = reason

        # ---- Step 6: Adapter 反馈 ----
        self.adapters.evolve_adapter(adapter.adapter_id, feedback=0.5)

        # ---- Step 7: Surprise 驱动的 adapter 创建 ----
        tension = compute_tension(self.self_aware)
        action = decide_action(
            surprise,
            tension,
            surprise_threshold=0.7,
            tension_threshold=self.config.awareness.drift_threshold,
        )

        if action in ("create_adapter", "consolidate_and_delay"):
            cluster_id = self._find_nearest_cluster(query_embedding)
            if cluster_id >= 0:
                cluster_memories = self.memory.retrieve_by_cluster(cluster_id)
                if len(cluster_memories) >= 5:
                    self.adapters.create_adapter(
                        cluster_id=cluster_id,
                        representative_memories=cluster_memories[:10],
                        engine=self.engine,
                    )
                    logger.info(
                        f"Created new adapter for cluster {cluster_id} "
                        f"(action={action})"
                    )

        # ---- Step 8: 自我认知更新 ----
        drift_status = None
        calibrated = False
        if self.self_aware.should_calibrate():
            try:
                report = self.self_aware.calibrate(
                    self.memory, self.topology, self.engine
                )
                drift_status = report.drift.status
                calibrated = True
                self._last_calibration_step = self._step
            except Exception as e:
                logger.warning(f"Calibration failed: {e}")
                drift_report = self.self_aware.detect_drift()
                drift_status = drift_report.status

        # ---- 构造结果 ----
        elapsed_ms = (time.time() - start_time) * 1000
        result = ProcessResult(
            response_text=response,
            retrieved_memories=context_dicts,
            retrieval_strategy=strategy,
            adapter_used=adapter.name
            if hasattr(adapter, "name")
            else "default",
            surprise_score=surprise,
            memory_accepted=memory_accepted,
            memory_reject_reason=reject_reason,
            drift_status=drift_status,
            calibrated=calibrated,
            latency_ms=elapsed_ms,
            step=self._step,
        )
        self._process_log.append(result)
        return result

    def add_knowledge(self, text: str, metadata: Optional[dict] = None) -> bool:
        """便捷方法：直接添加知识（不触发推理）。"""
        self._metrics.add_memory_calls += 1
        t0 = time.perf_counter()
        embedding = self.embedding.encode(text)
        accepted, reason = self.guard.should_accept_memory(
            text, embedding, self.memory, self.self_aware, self.topology
        )
        if accepted:
            self.memory.add_memory(
                content=text,
                embedding=embedding,
                metadata=metadata,
                topo_engine=self.topology,
            )
            self._metrics.add_memory_latency_ms += (time.perf_counter() - t0) * 1000
            return True
        logger.info(f"Knowledge rejected: {reason}")
        self._metrics.add_memory_latency_ms += (time.perf_counter() - t0) * 1000
        return False

    def add_knowledge_batch(self, texts: List[str], metadata_list: Optional[List[dict]] = None) -> int:
        """批量添加知识（一次 encode_batch + 一次 ChromaDB add）。

        注意：不经过 ConsistencyGuard（guard 用于单条拦截，批量场景默认信任）。

        参数：
            texts: 知识文本列表
            metadata_list: 元数据列表（可选，长度需与 texts 一致）

        返回：
            成功添加的数量
        """
        if not texts:
            return 0
        metas = metadata_list or [None] * len(texts)
        if len(metas) != len(texts):
            raise ValueError("texts and metadata_list must have the same length")

        self._metrics.add_memory_batch_calls += 1
        t0 = time.perf_counter()
        embeddings = self.embedding.encode_batch(texts)
        items = [
            {"content": text, "embedding": embeddings[i], "metadata": metas[i]}
            for i, text in enumerate(texts)
        ]
        nodes = self.memory.add_memory_batch(items, topo_engine=self.topology)
        self._metrics.add_memory_latency_ms += (time.perf_counter() - t0) * 1000
        return len(nodes)

    def ask(self, question: str, max_tokens: int = 256) -> str:
        """便捷方法：只提问，不添加知识。"""
        self._metrics.ask_calls += 1
        t0 = time.perf_counter()
        t_retrieve = time.perf_counter()
        query_embedding = self.embedding.encode(question)
        adapter, surprise = self.adapters.select_adapter(
            query_embedding, self.memory
        )
        retrieved = self.memory.retrieve(
            query_embedding,
            strategy="hybrid",
            k=self.config.memory.similarity_top_k,
        )
        self._metrics.retrieve_calls += 1
        self._metrics.retrieve_latency_ms += (time.perf_counter() - t_retrieve) * 1000

        system_prompt = adapter.apply(question)
        context_dicts = [
            {
                "content": m.content,
                "cluster_id": m.cluster_id,
                "access_count": m.access_count,
                "relevance_score": round(score, 3),
            }
            for m, score in retrieved
        ]

        t_llm = time.perf_counter()
        try:
            response = self.engine.generate(
                prompt=question,
                context=context_dicts if context_dicts else None,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
            )
        except Exception as e:
            logger.error(f"Ask failed: {e}")
            response = f"[Error: {e}]"
        self._metrics.llm_latency_ms += (time.perf_counter() - t_llm) * 1000
        self._metrics.ask_latency_ms += (time.perf_counter() - t0) * 1000
        return response

    def get_status(self) -> SystemStatus:
        """系统状态快照。"""
        cluster_ids = self.memory._get_all_cluster_ids()

        process = psutil.Process(os.getpid())
        ram_mb = process.memory_info().rss / (1024 * 1024)

        fp = None
        if self.self_aware._fingerprint_history:
            fp = self.self_aware._fingerprint_history[-1].fingerprint

        drift = self.self_aware.detect_drift()
        h1_m = self.self_aware.get_h1_metrics()
        h1_health = h1_m.h1_health_score if not h1_m.suppressed else 1.0
        h1_drift = self.self_aware.get_h1_drift()
        h2_m = self.self_aware.get_h2_metrics()

        return SystemStatus(
            step=self._step,
            memory_node_count=self.memory.node_count(),
            memory_cluster_count=len(cluster_ids),
            adapter_count=self.adapters.adapter_count,
            drift_status=drift.status,
            last_calibration_step=self._last_calibration_step,
            ram_usage_mb=ram_mb,
            fingerprint_shape=fp.shape if fp is not None else None,
            h1_health=h1_health,
            h1_drift=h1_drift,
            betti_2_count=h2_m.betti_2_count,
            h2_to_h1_ratio=h2_m.h2_to_h1_ratio,
            h2_health=h2_m.h2_health_score,
            h2_drift=h2_m.h2_drift_since_baseline,
        )

    def get_metrics(self) -> dict:
        """返回系统运行时指标。用于监控和调优。"""
        # 快照当前状态
        m = self._metrics
        m.current_node_count = self.memory.node_count()
        m.current_cluster_count = len(self.memory._get_all_cluster_ids())
        m.current_adapter_count = self.adapters.adapter_count
        # P4: H1 geometric quality snapshot
        h1_m = self.self_aware.get_h1_metrics()
        m.h1_health = h1_m.h1_health_score if not h1_m.suppressed else 1.0
        m.h1_drift = self.self_aware.get_h1_drift()
        # P4+: H2 domain-bridging snapshot
        h2_m = self.self_aware.get_h2_metrics()
        m.betti_2_count = h2_m.betti_2_count
        m.h2_to_h1_ratio = h2_m.h2_to_h1_ratio
        m.h2_health = h2_m.h2_health_score
        m.h2_drift = h2_m.h2_drift_since_baseline
        m.cavitation_rate = h2_m.cavitation_rate
        return m.to_dict()

    def save(self, path: str) -> None:
        """持久化整个系统状态。"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # 保存各模块
        self.memory.save(str(path / "memory"))
        self.adapters.save(str(path / "adapters.json"))
        self.self_aware.save(str(path / "self_awareness.json"))

        # 保存系统元数据
        meta = {
            "step": self._step,
            "last_calibration_step": self._last_calibration_step,
            "process_log": [
                {
                    "response_text": r.response_text,
                    "adapter_used": r.adapter_used,
                    "surprise_score": r.surprise_score,
                    "memory_accepted": r.memory_accepted,
                    "drift_status": r.drift_status,
                    "calibrated": r.calibrated,
                    "latency_ms": r.latency_ms,
                    "step": r.step,
                }
                for r in self._process_log[-100:]  # 只保存最近 100 条
            ],
        }
        with open(path / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

        logger.info(f"System saved to {path}")

    def load(self, path: str) -> None:
        """从磁盘恢复系统。"""
        path = Path(path)

        self.memory.load(str(path / "memory"))
        self.adapters.load(str(path / "adapters.json"))
        self.self_aware.load(str(path / "self_awareness.json"))

        meta_path = path / "meta.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self._step = meta.get("step", 0)
            self._last_calibration_step = meta.get("last_calibration_step", 0)

        logger.info(f"System loaded from {path}")

    def reset(self) -> None:
        """清空所有状态，回到初始状态。"""
        self._step = 0
        self._last_calibration_step = 0
        self._process_log.clear()

        # 重建核心模块
        self.memory = MemoryGraph(self.config.memory, self.embedding)
        self.engine = ReasoningEngine(self.config.engine)
        self.self_aware = SelfAwareness(self.config.awareness)
        self.guard = ConsistencyGuard(self.config.awareness)
        self.adapters = AdapterPool(self.config.adapter, self.embedding)

        logger.info("System reset")

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _find_nearest_cluster(self, query_embedding: np.ndarray) -> int:
        """找到 query 最近的簇 ID。"""
        centers = self.memory.get_cluster_centers()
        if not centers:
            return -1

        best_cid = -1
        best_sim = -1.0

        for cid, center in centers.items():
            norm_q = np.linalg.norm(query_embedding)
            norm_c = np.linalg.norm(center)
            if norm_q < 1e-8 or norm_c < 1e-8:
                continue
            sim = float(np.dot(query_embedding, center) / (norm_q * norm_c))
            if sim > best_sim:
                best_sim = sim
                best_cid = cid

        return best_cid

    def __repr__(self) -> str:
        return (
            f"TopoMemSystem(step={self._step}, "
            f"nodes={self.memory.node_count()}, "
            f"clusters={len(self.memory._get_all_cluster_ids())}, "
            f"adapters={self.adapters.adapter_count})"
        )

    def consolidation_pass(
        self,
        orphan_threshold: float = 0.05,
        merge_centroid_threshold: float = 0.92,
        update_topology: bool = False,
    ) -> dict:
        """记忆整合检查（借鉴 LLM Wiki Lint 模式）。"""
        self._metrics.consolidation_calls += 1
        t0 = time.perf_counter()

        node_count = self.memory.node_count()
        cluster_centers = self.memory.get_cluster_centers()
        cluster_ids = list(cluster_centers.keys())

        # ------------------------------------------------------------------
        # 1. 找孤立节点
        # ------------------------------------------------------------------
        orphans = []
        all_node_ids = []

        # 遍历所有节点，检查 cluster_id 和 persistence_score
        # ---- 1. 找孤立节点（ChromaDB 层直接过滤，不加载全部数据）----
        try:
            coll = self.memory._collection

            # 用两次 ChromaDB 查询找孤儿：
            # 1. $lt persistence_score — 直接数据库过滤
            # 2. cluster_id 为 None — ChromaDB 不支持 $eq:None，Python 过滤
            try:
                low_persist_data = coll.get(
                    where={"persistence_score": {"$lt": orphan_threshold}},
                    include=["metadatas"]
                )
                for i, mid in enumerate(low_persist_data["ids"]):
                    orphans.append(mid)
            except Exception:
                pass

            # cluster_id is None（ChromaDB 不支持 $eq:None，改用 Python 过滤）
            all_data = coll.get(include=["metadatas"])
            for i, mid in enumerate(all_data["ids"]):
                meta = all_data["metadatas"][i]
                if meta.get("cluster_id") is None and mid not in orphans:
                    orphans.append(mid)
        except Exception:
            pass

        # ------------------------------------------------------------------
        # 2. 找可合并的簇（centroid 相似度过高）
        # ------------------------------------------------------------------
        merge_candidates = []
        if len(cluster_ids) >= 2:
            cids = sorted(cluster_ids)
            # 向量化：一次矩阵运算替代双层循环
            centroids = np.array([cluster_centers[cid] for cid in cids])  # (c, d)
            norms = np.linalg.norm(centroids, axis=1, keepdims=True)       # (c, 1)
            # 避免除零
            norms = np.where(norms < 1e-8, 1.0, norms)
            normalized = centroids / norms                                     # (c, d)
            # Cosine similarity matrix
            sim_matrix = normalized @ normalized.T                            # (c, c)
            # 提取上三角（不含对角线）
            upper_tri_indices = np.triu_indices(len(cids), k=1)
            sims = sim_matrix[upper_tri_indices]  # 拉成一维
            mask = sims > merge_centroid_threshold
            valid_indices = np.where(mask)[0]
            merge_candidates = [
                (cids[upper_tri_indices[0][k]], cids[upper_tri_indices[1][k]], round(float(sims[k]), 4))
                for k in valid_indices
            ]

        # 按相似度降序排列
        merge_candidates.sort(key=lambda x: x[2], reverse=True)

        # ------------------------------------------------------------------
        # 3. 可选：拓扑更新（重新聚类）
        # ------------------------------------------------------------------
        topology_updated = False
        if update_topology:
            try:
                topo_result = self.memory.update_topology(self.topology)
                topology_updated = True
                logger.info(
                    f"Consolidation topology update: {topo_result.n_clusters} clusters, "
                    f"cluster_labels shape: {topo_result.cluster_labels.shape if topo_result.cluster_labels is not None else None}"
                )
            except ValueError as e:
                # 空图无法更新拓扑（还没有节点）
                logger.warning(f"Consolidation topology update skipped: {e}")

        report = {
            "orphans": orphans,
            "orphan_count": len(orphans),
            "merge_candidates": merge_candidates,
            "merge_count": len(merge_candidates),
            "cluster_count": len(cluster_ids),
            "node_count": node_count,
            "topology_updated": topology_updated,
        }

        self._metrics.orphans_detected += len(orphans)
        self._metrics.merge_candidates_found += len(merge_candidates)
        self._metrics.consolidation_latency_ms += (time.perf_counter() - t0) * 1000

        # TDA cache stats from topology engine
        if hasattr(self.topology, '_persistence_cache'):
            hits = len(self.topology._persistence_cache)
            self._metrics.tda_cache_hits += hits

        logger.info(
            f"Consolidation pass: {len(orphans)} orphans, "
            f"{len(merge_candidates)} merge candidates, "
            f"{len(cluster_ids)} clusters"
        )
        return report
