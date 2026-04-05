"""
topomem/guard.py — 一致性守护模块

在记忆写入前做预检，防止有害变更。

设计来源：
- SPEC_SELF_AWARENESS.md: 一致性守护完整规格
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

from topomem.config import SelfAwarenessConfig


if TYPE_CHECKING:
    from topomem.memory import MemoryGraph
    from topomem.self_awareness import SelfAwareness
    from topomem.topology import TopologyEngine


logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# 数据类型定义
# ------------------------------------------------------------------

@dataclass
class ConsolidationAction:
    action_type: str           # "merge" | "strengthen" | "remove" | "reassign"
    node_ids: List[str]
    reason: str


# ------------------------------------------------------------------
# ConsistencyGuard 实现
# ------------------------------------------------------------------

class ConsistencyGuard:
    """一致性守护。在记忆写入前做预检，防止有害变更。

    设计哲学：
    - 宁可误拒（false reject）也不误入（false accept）
    - 但不能太严格导致系统无法学习新知识
    - 所有拒绝都返回原因，方便调试和人工覆盖
    """

    def __init__(self, config: Optional[SelfAwarenessConfig] = None):
        self.config = config or SelfAwarenessConfig()

    def should_accept_memory(
        self,
        new_content: str,
        new_embedding: np.ndarray,
        memory_graph: "MemoryGraph",
        self_awareness: "SelfAwareness",
        topo_engine: Optional["TopologyEngine"] = None,
    ) -> Tuple[bool, str]:
        """判断是否接受新记忆。

        检查清单：
        1. 重复检测（similarity > 0.95 → 拒绝）
        2. 矛盾检测（similarity > 0.8 + 否定词 → 警告但不拒绝）
        3. 拓扑稳定性预估（漂移状态下要求 similarity > 0.3）
        4. 容量检查（满容量时建议 prune）

        返回：(accept, reason)
        """
        # 检查 1: 重复检测
        dup_check = self._check_duplicate(new_embedding, memory_graph)
        if not dup_check[0]:
            return dup_check

        # 检查 2: 矛盾检测
        contradiction_check = self._check_contradiction(new_content, new_embedding, memory_graph)
        if not contradiction_check[0]:
            # 矛盾是 warning，不拒绝
            pass  # 继续

        # 检查 3: 拓扑稳定性预估
        stability_check = self._check_stability(new_embedding, memory_graph, self_awareness)
        if not stability_check[0]:
            return stability_check

        # 检查 4: 容量检查
        if topo_engine is not None:
            capacity_check = self._check_capacity(memory_graph, topo_engine)
            if not capacity_check[0]:
                return capacity_check

        # 构建最终 reason
        reasons = ["accepted"]
        if contradiction_check[1]:
            reasons.append(f"warning: {contradiction_check[1]}")
        if not stability_check[0]:
            reasons.append(stability_check[1])

        return True, "; ".join(reasons)

    def _check_duplicate(
        self,
        new_embedding: np.ndarray,
        memory_graph: "MemoryGraph",
    ) -> Tuple[bool, str]:
        """检查是否与已有记忆重复。"""
        if memory_graph.node_count() == 0:
            return True, ""

        # 检索最相似的节点
        results = memory_graph.retrieve(new_embedding, strategy="vector", k=1)
        if not results:
            return True, ""

        best_node = results[0][0]
        norm_new = np.linalg.norm(new_embedding)
        norm_old = np.linalg.norm(best_node.embedding)
        if norm_new < 1e-8 or norm_old < 1e-8:
            return True, ""

        similarity = float(np.dot(new_embedding, best_node.embedding) / (norm_new * norm_old))

        if similarity > 0.95:
            return False, f"duplicate: similarity {similarity:.4f} with node {best_node.id}"

        return True, ""

    def _check_contradiction(
        self,
        new_content: str,
        new_embedding: np.ndarray,
        memory_graph: "MemoryGraph",
    ) -> Tuple[bool, str]:
        """检查潜在矛盾。

        不阻塞（accept=True），但返回 warning。
        """
        if memory_graph.node_count() == 0:
            return True, ""

        results = memory_graph.retrieve(new_embedding, strategy="vector", k=3)
        if not results:
            return True, ""

        # 检查否定词
        negation_words = ["not", "never", "false", "no", "不", "非", "没有", "否"]
        new_has_negation = any(w in new_content.lower() for w in negation_words)

        for node, _ in results:
            norm_new = np.linalg.norm(new_embedding)
            norm_old = np.linalg.norm(node.embedding)
            if norm_new < 1e-8 or norm_old < 1e-8:
                continue

            similarity = float(np.dot(new_embedding, node.embedding) / (norm_new * norm_old))

            if similarity > 0.8:
                old_has_negation = any(w in node.content.lower() for w in negation_words)
                if new_has_negation != old_has_negation:
                    return True, (
                        f"potential contradiction with node {node.id[:8]}... "
                        f"(similarity {similarity:.3f})"
                    )

        return True, ""

    def _check_stability(
        self,
        new_embedding: np.ndarray,
        memory_graph: "MemoryGraph",
        self_awareness: "SelfAwareness",
    ) -> Tuple[bool, str]:
        """检查漂移状态下的稳定性。"""
        if memory_graph.node_count() == 0:
            return True, ""

        # 获取最近漂移状态
        if not self_awareness._drift_reports:
            return True, ""

        last_drift = self_awareness._drift_reports[-1]
        if last_drift.status != "drifting":
            return True, ""

        # 漂移状态下，要求新记忆与已有知识有一定相关性
        results = memory_graph.retrieve(new_embedding, strategy="vector", k=1)
        if not results:
            return True, ""

        best_node = results[0][0]
        norm_new = np.linalg.norm(new_embedding)
        norm_old = np.linalg.norm(best_node.embedding)
        if norm_new < 1e-8 or norm_old < 1e-8:
            return True, ""

        similarity = float(np.dot(new_embedding, best_node.embedding) / (norm_new * norm_old))

        if similarity < 0.3:
            return False, (
                f"drifting state: new knowledge too unrelated "
                f"(similarity {similarity:.3f} < 0.3)"
            )

        return True, ""

    def _check_capacity(
        self,
        memory_graph: "MemoryGraph",
        topo_engine: "TopologyEngine",
    ) -> Tuple[bool, str]:
        """检查容量。"""
        max_nodes = memory_graph.config.max_nodes
        if memory_graph.node_count() >= max_nodes:
            return True, f"at capacity ({max_nodes} nodes), prune recommended"
        return True, ""

    # ==================================================================
    # 建议整理
    # ==================================================================

    def recommend_consolidation(
        self,
        memory_graph: "MemoryGraph",
        topo_engine: Optional["TopologyEngine"] = None,
    ) -> List[ConsolidationAction]:
        """建议的记忆整理操作。

        扫描策略：
        1. 合并候选：similarity > 0.9 的节点对
        2. 强化候选：persistence_score > 90th percentile
        3. 清理候选：importance_score < 10th percentile
        4. 孤儿处理：cluster_id == -1 的节点
        """
        actions: List[ConsolidationAction] = []

        all_nodes = []
        for nid, data in memory_graph._graph.nodes(data=True):
            all_nodes.append((nid, data["node"]))

        if len(all_nodes) < 2:
            return actions

        # 1. 合并候选
        merge_pairs = self._find_merge_candidates(all_nodes)
        for nid_a, nid_b, sim in merge_pairs:
            actions.append(ConsolidationAction(
                action_type="merge",
                node_ids=[nid_a, nid_b],
                reason=f"high similarity ({sim:.3f}), consider merging",
            ))

        # 2. 强化候选
        strengthen_ids = self._find_strengthen_candidates(all_nodes)
        if strengthen_ids:
            actions.append(ConsolidationAction(
                action_type="strengthen",
                node_ids=strengthen_ids,
                reason="high persistence, mark as core knowledge",
            ))

        # 3. 清理候选
        remove_ids = self._find_remove_candidates(all_nodes)
        if remove_ids:
            actions.append(ConsolidationAction(
                action_type="remove",
                node_ids=remove_ids,
                reason="low importance, consider deletion",
            ))

        # 4. 孤儿处理
        orphan_ids = [
            nid for nid, node in all_nodes if node.cluster_id == -1
        ]
        if orphan_ids:
            actions.append(ConsolidationAction(
                action_type="reassign",
                node_ids=orphan_ids,
                reason=f"unassigned cluster ({len(orphan_ids)} nodes)",
            ))

        return actions

    def _find_merge_candidates(self, all_nodes):
        """查找高相似度节点对（向量化实现，O(n²) → O(n)）。"""
        n = len(all_nodes)
        if n < 2:
            return []

        # 一次性堆叠所有 embedding → (n, d)
        embeddings = np.stack([node.embedding for _, node in all_nodes])

        # L2 归一化
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normalized = embeddings / norms

        # Cosine similarity matrix → (n, n)，上三角即所有节点对
        sim_matrix = normalized @ normalized.T

        # 提取上三角索引 (不含对角线)，similarity > 0.9
        threshold = 0.9
        pairs = []
        node_ids = [nid for nid, _ in all_nodes]
        for i in range(n):
            for j in range(i + 1, n):
                sim = float(sim_matrix[i, j])
                if sim > threshold:
                    pairs.append((node_ids[i], node_ids[j], sim))

        return pairs

    def _find_strengthen_candidates(self, all_nodes):
        """查找高持久性节点。"""
        scores = [node.persistence_score for _, node in all_nodes]
        if not scores:
            return []

        threshold = float(np.percentile(scores, 90))
        return [nid for nid, node in all_nodes if node.persistence_score >= threshold]

    def _find_remove_candidates(self, all_nodes):
        """查找低重要性节点。"""
        now = time.time()
        from topomem.memory import compute_importance

        scores = []
        max_access = max((node.access_count for _, node in all_nodes), default=1)

        for nid, node in all_nodes:
            imp = compute_importance(node, now, max_access)
            scores.append((nid, imp))

        if not scores:
            return []

        threshold = float(np.percentile([s for _, s in scores], 10))
        return [nid for nid, imp in scores if imp < threshold]

    def __repr__(self) -> str:
        return "ConsistencyGuard()"
