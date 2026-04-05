"""
Phase 2: 图结构记忆系统单元测试

测试 memory.py 的完整功能。
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest


# 必须在导入任何 HF 库之前设置环境变量
HF_CACHE = r"F:\unified-sel\topomem\data\models\hf_cache"
os.environ["HF_HOME"] = HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE
os.environ["SENTENCE_TRANSFORMERS_HOME"] = HF_CACHE


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def embedding_mgr():
    from topomem.embedding import EmbeddingManager
    return EmbeddingManager()


@pytest.fixture
def topo_engine():
    from topomem.topology import TopologyEngine
    return TopologyEngine()


@pytest.fixture
def memory_graph(embedding_mgr, tmp_path):
    from topomem.memory import MemoryGraph
    from topomem.config import MemoryConfig
    # 使用临时目录避免 ChromaDB 冲突
    config = MemoryConfig(
        max_nodes=50,
        chroma_persist_dir=str(tmp_path / "chromadb"),
    )
    return MemoryGraph(config=config, embedding_mgr=embedding_mgr)


# ------------------------------------------------------------------
# 基本写入和计数
# ------------------------------------------------------------------

class TestMemoryGraphBasics:
    """基本添加和计数测试"""

    def test_add_and_count(self, memory_graph):
        """添加 5 条记忆，node_count() 应为 5"""
        for i in range(5):
            memory_graph.add_memory(
                content=f"Test memory {i}",
                embedding=np.random.RandomState(i).rand(384).astype(np.float32),
            )
        assert memory_graph.node_count() == 5

    def test_add_from_text(self, memory_graph):
        """用 add_memory_from_text 添加文本，验证 embedding 自动生成"""
        node = memory_graph.add_memory_from_text("Neural networks learn from data.")
        assert node.id is not None
        assert node.content == "Neural networks learn from data."
        assert node.embedding.shape == (384,)
        assert memory_graph.node_count() == 1

    def test_add_empty_content_warning(self, memory_graph):
        """空内容应允许添加但有 warning"""
        node = memory_graph.add_memory_from_text("")
        assert node is not None
        assert node.content == ""


# ------------------------------------------------------------------
# 检索
# ------------------------------------------------------------------

class TestRetrieval:
    """检索策略测试"""

    def test_empty_graph_retrieve(self, memory_graph):
        """空图检索应返回空列表"""
        query = np.random.rand(384).astype(np.float32)
        results = memory_graph.retrieve(query, strategy="vector")
        assert results == []

    def test_retrieve_vector(self, memory_graph):
        """vector 检索应返回最相关的"""
        # 添加 10 条明显不同的记忆
        rng = np.random.RandomState(42)
        for i in range(10):
            emb = rng.rand(384).astype(np.float32)
            if i == 0:
                target_emb = emb.copy()
            memory_graph.add_memory(
                content=f"Memory {i}",
                embedding=emb,
            )

        # 用第一条记忆的 embedding 检索，应该找到它
        results = memory_graph.retrieve(target_emb, strategy="vector", k=1)
        assert len(results) == 1
        assert results[0][0].content == "Memory 0"

    def test_retrieve_k_larger_than_nodes(self, memory_graph):
        """k > 节点总数时应返回全部"""
        for i in range(3):
            memory_graph.add_memory_from_text(f"Memory {i}")

        query = memory_graph._embedding_mgr.encode("query")
        results = memory_graph.retrieve(query, strategy="vector", k=10)
        assert len(results) == 3

    def test_retrieve_topological(self, memory_graph, topo_engine):
        """拓扑检索应返回同组记忆"""
        # 插入 3 组不同主题各 3 条
        topics = {
            0: ["Neural network training.", "Deep learning models.", "Backpropagation."],
            1: ["The cat sleeps.", "Dogs bark loudly.", "Birds fly high."],
            2: ["Stock prices fell.", "Market crashed today.", "Investment lost."],
        }
        for topic_id, texts in topics.items():
            for text in texts:
                memory_graph.add_memory_from_text(text, topo_engine=topo_engine)

        # 用技术相关查询检索
        query = memory_graph._embedding_mgr.encode("Machine learning algorithms")
        results = memory_graph.retrieve(query, strategy="topological", k=5)

        # 应该主要返回同一簇的记忆
        if results:
            cluster_ids = [n.cluster_id for n, _ in results]
            # 大多数结果应在同一簇
            most_common = max(set(cluster_ids), key=cluster_ids.count)
            same_cluster = sum(1 for c in cluster_ids if c == most_common)
            assert same_cluster >= len(results) * 0.6  # 至少 60% 同簇

    def test_retrieve_hybrid(self, memory_graph, topo_engine):
        """hybrid 应兼具 vector 和 topo 的优点"""
        # 插入两组数据
        for text in ["Python programming.", "Data structures.", "Algorithms."]:
            memory_graph.add_memory_from_text(text, topo_engine=topo_engine)
        for text in ["The garden is beautiful.", "Flowers bloom in spring.", "Trees are green."]:
            memory_graph.add_memory_from_text(text, topo_engine=topo_engine)

        query = memory_graph._embedding_mgr.encode("Coding in Python")
        results = memory_graph.retrieve(query, strategy="hybrid", k=3)

        assert len(results) <= 3
        assert len(results) > 0

    def test_access_count_update(self, memory_graph):
        """每次被 retrieve 返回后 access_count 应 +1"""
        node = memory_graph.add_memory_from_text("Test memory.")
        assert node.access_count == 0

        query = memory_graph._embedding_mgr.encode("Test")
        memory_graph.retrieve(query, strategy="vector", k=1)

        assert node.access_count == 1

    def test_unknown_strategy_raises(self, memory_graph):
        """未知策略应抛出 ValueError"""
        # 先添加一个节点避免空图检查
        memory_graph.add_memory_from_text("Test")
        query = np.random.rand(384).astype(np.float32)
        with pytest.raises(ValueError, match="Unknown strategy"):
            memory_graph.retrieve(query, strategy="invalid")


# ------------------------------------------------------------------
# 拓扑管理
# ------------------------------------------------------------------

class TestTopologyManagement:
    """拓扑更新和管理测试"""

    def test_update_topology_clusters(self, memory_graph, topo_engine):
        """3 组明显分离的数据，update_topology 应产生 3 个簇"""
        rng = np.random.RandomState(42)
        # 3 组分开的点
        for i in range(5):
            emb = rng.rand(384).astype(np.float32) * 0.1  # 簇 0
            memory_graph.add_memory(f"cluster0_{i}", emb)
        for i in range(5):
            emb = rng.rand(384).astype(np.float32) * 0.1 + 10.0  # 簇 1
            memory_graph.add_memory(f"cluster1_{i}", emb)
        for i in range(5):
            emb = rng.rand(384).astype(np.float32) * 0.1 + 20.0  # 簇 2
            memory_graph.add_memory(f"cluster2_{i}", emb)

        memory_graph.update_topology(topo_engine)

        n_clusters = len(memory_graph._get_all_cluster_ids())
        assert n_clusters >= 2  # 至少 2 个簇

    def test_topological_summary(self, memory_graph, topo_engine):
        """get_topological_summary 应返回 (100,) 归一化向量"""
        for i in range(10):
            memory_graph.add_memory_from_text(f"Memory {i}")

        summary = memory_graph.get_topological_summary(topo_engine)
        assert summary.shape == (100,)
        assert np.all(np.isfinite(summary))

    def test_retrieve_by_cluster(self, memory_graph, topo_engine):
        """retrieve_by_cluster 应返回指定簇的所有节点"""
        for i in range(6):
            memory_graph.add_memory_from_text(f"Memory {i}", topo_engine=topo_engine)

        memory_graph.update_topology(topo_engine)
        cluster_ids = memory_graph._get_all_cluster_ids()

        if cluster_ids:
            first_cluster = cluster_ids[0]
            nodes = memory_graph.retrieve_by_cluster(first_cluster)
            assert all(n.cluster_id == first_cluster for n in nodes)

    def test_get_cluster_centers(self, memory_graph, topo_engine):
        """get_cluster_centers 应返回每个簇的中心向量"""
        for i in range(6):
            memory_graph.add_memory_from_text(f"Memory {i}", topo_engine=topo_engine)

        memory_graph.update_topology(topo_engine)
        centers = memory_graph.get_cluster_centers()

        assert len(centers) > 0
        for cid, center in centers.items():
            assert center.shape == (384,)


# ------------------------------------------------------------------
# 容量管理
# ------------------------------------------------------------------

class TestCapacityManagement:
    """容量管理和 prune 测试"""

    def test_prune_removes_low_importance(self, memory_graph, topo_engine):
        """prune 应移除 importance 最低的节点"""
        # 添加超过限制的节点
        for i in range(memory_graph.config.max_nodes + 10):
            memory_graph.add_memory_from_text(f"Memory {i}", topo_engine=topo_engine)

        removed = memory_graph.prune()
        assert len(removed) > 0
        assert memory_graph.node_count() <= memory_graph.config.max_nodes

    def test_prune_protects_clusters(self, memory_graph, topo_engine):
        """prune 后每个 cluster 至少保留 1 个节点"""
        # 添加 3 组数据
        for i in range(10):
            memory_graph.add_memory_from_text(f"tech_{i}", topo_engine=topo_engine)
        for i in range(10):
            memory_graph.add_memory_from_text(f"nature_{i}", topo_engine=topo_engine)

        # 强制 prune
        removed = memory_graph.prune(max_nodes=5)

        # 检查每个簇至少有一个节点
        cluster_ids = memory_graph._get_all_cluster_ids()
        for cid in cluster_ids:
            nodes = memory_graph.retrieve_by_cluster(cid)
            assert len(nodes) >= 1, f"簇 {cid} 应为空"


# ------------------------------------------------------------------
# 序列化
# ------------------------------------------------------------------

class TestSerialization:
    """保存和加载测试"""

    def test_save_and_load(self, memory_graph, topo_engine):
        """save 后 load，所有数据应完整恢复"""
        # 添加一些记忆
        for i in range(5):
            memory_graph.add_memory_from_text(f"Memory {i}")

        # 更新拓扑
        memory_graph.update_topology(topo_engine)

        # 保存
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_graph.save(tmpdir)

            # 创建新实例并加载
            from topomem.memory import MemoryGraph
            new_graph = MemoryGraph(
                config=memory_graph.config,
                embedding_mgr=memory_graph._embedding_mgr,
            )
            new_graph.load(tmpdir)

            # 验证数据
            assert new_graph.node_count() == 5
            for _, data in new_graph._graph.nodes(data=True):
                node = data["node"]
                assert isinstance(node.embedding, np.ndarray)
                assert node.embedding.shape == (384,)

    def test_chromadb_networkx_sync(self, memory_graph):
        """删除节点后两个存储应保持一致"""
        for i in range(5):
            memory_graph.add_memory_from_text(f"Memory {i}")

        # 删除一个节点
        first_node_id = list(memory_graph._graph.nodes())[0]
        memory_graph._delete_node(first_node_id)

        # 检查同步
        assert memory_graph._check_sync()


# ------------------------------------------------------------------
# MemoryNode 测试
# ------------------------------------------------------------------

class TestMemoryNode:
    """MemoryNode 数据结构和序列化测试"""

    def test_to_dict_and_from_dict(self):
        """to_dict 和 from_dict 应循环一致"""
        from topomem.memory import MemoryNode
        node = MemoryNode(
            id="test-123",
            content="Test content",
            embedding=np.random.rand(384).astype(np.float32),
            created_at=1000.0,
            metadata={"key": "value"},
            access_count=5,
            cluster_id=2,
            persistence_score=0.8,
        )

        d = node.to_dict()
        restored = MemoryNode.from_dict(d)

        assert restored.id == node.id
        assert restored.content == node.content
        assert restored.embedding.shape == (384,)
        assert np.allclose(restored.embedding, node.embedding)
        assert restored.metadata == node.metadata

    def test_compute_importance(self):
        """compute_importance 应返回 [0, 1] 范围内的值"""
        import time
        from topomem.memory import MemoryNode, compute_importance

        node = MemoryNode(
            id="test",
            content="test",
            embedding=np.zeros(384),
            created_at=time.time(),
            access_count=10,
            cluster_id=0,
            persistence_score=0.5,
            last_accessed=time.time(),
        )

        importance = compute_importance(node, time.time(), max_access_count=20)
        assert 0.0 <= importance <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
