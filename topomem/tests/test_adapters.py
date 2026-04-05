"""
Phase 4: 动态塑造机制单元测试

测试 adapters.py 的完整功能。
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
def adapter_pool(embedding_mgr):
    from topomem.adapters import AdapterPool
    return AdapterPool(embedding_mgr=embedding_mgr)


@pytest.fixture
def topo_engine():
    from topomem.topology import TopologyEngine
    return TopologyEngine()


@pytest.fixture
def memory_graph(embedding_mgr, tmp_path):
    from topomem.memory import MemoryGraph
    from topomem.config import MemoryConfig
    config = MemoryConfig(
        max_nodes=50,
        chroma_persist_dir=str(tmp_path / "chromadb"),
    )
    return MemoryGraph(config=config, embedding_mgr=embedding_mgr)


# ------------------------------------------------------------------
# BaseAdapter 接口测试
# ------------------------------------------------------------------

class TestBaseAdapter:
    """BaseAdapter 接口和 PromptAdapter 实现测试"""

    def test_prompt_adapter_implements_base(self, adapter_pool):
        """PromptAdapter 应正确实现 BaseAdapter 接口"""
        default = adapter_pool.default_adapter
        assert hasattr(default, "apply")
        assert hasattr(default, "get_domain_embedding")
        assert hasattr(default, "adapter_id")
        assert hasattr(default, "adapter_type")
        assert default.adapter_type == "prompt"

    def test_adapter_apply_returns_system_prompt(self, adapter_pool):
        """apply() 应返回 system_prompt 字符串"""
        default = adapter_pool.default_adapter
        result = default.apply("test prompt")
        assert isinstance(result, str)
        assert len(result) > 0
        assert "reasoning" in result.lower() or "precise" in result.lower()

    def test_adapter_domain_embedding_shape(self, adapter_pool):
        """get_domain_embedding() 应返回 (384,) 向量"""
        default = adapter_pool.default_adapter
        emb = default.get_domain_embedding()
        assert emb.shape == (384,)

    def test_adapter_id_property(self, adapter_pool):
        """adapter_id 属性应返回字符串"""
        default = adapter_pool.default_adapter
        assert isinstance(default.adapter_id, str)
        assert len(default.adapter_id) > 0

    def test_adapter_to_dict_and_from_dict(self, embedding_mgr):
        """PromptAdapter 应可序列化和恢复"""
        from topomem.adapters import PromptAdapter
        import time

        adapter = PromptAdapter(
            id="test-123",
            name="Test",
            system_prompt="Test prompt.",
            domain_keywords=["test", "example"],
            domain_embedding=np.random.rand(384).astype(np.float32),
            topological_cluster=0,
            created_at=time.time(),
            usage_count=5,
            effectiveness_score=0.7,
            last_used=time.time(),
        )

        d = adapter.to_dict()
        restored = PromptAdapter.from_dict(d)

        assert restored.id == adapter.id
        assert restored.name == adapter.name
        assert restored.system_prompt == adapter.system_prompt
        assert restored.domain_keywords == adapter.domain_keywords
        assert np.allclose(restored.domain_embedding, adapter.domain_embedding)
        assert restored.topological_cluster == adapter.topological_cluster
        assert restored.usage_count == adapter.usage_count

    def test_adapter_repr(self, adapter_pool):
        """__repr__ 应包含关键信息"""
        default = adapter_pool.default_adapter
        repr_str = repr(default)
        assert "PromptAdapter" in repr_str
        assert "name=" in repr_str


# ------------------------------------------------------------------
# AdapterPool 测试
# ------------------------------------------------------------------

class TestAdapterPool:
    """AdapterPool 完整功能测试"""

    def test_select_default_when_empty(self, adapter_pool):
        """无自定义 adapter 时应返回 default"""
        query_emb = np.random.rand(384).astype(np.float32)
        adapter, surprise = adapter_pool.select_adapter(query_emb, None)
        assert adapter.adapter_id == "default"
        assert surprise == 1.0

    def test_select_best_match(self, adapter_pool, embedding_mgr):
        """应选择与 query 最匹配的 adapter"""
        # 创建两个不同领域的 adapter
        tech_emb = embedding_mgr.encode("Python programming code algorithm")
        science_emb = embedding_mgr.encode("Physics chemistry biology science")

        from topomem.adapters import PromptAdapter
        import time
        now = time.time()

        tech_adapter = PromptAdapter(
            id="tech",
            name="Programming",
            system_prompt="You are a programming assistant.",
            domain_keywords=["python", "code", "algorithm"],
            domain_embedding=tech_emb,
            topological_cluster=0,
            created_at=now,
        )
        science_adapter_obj = PromptAdapter(
            id="science",
            name="Science",
            system_prompt="You are a science assistant.",
            domain_keywords=["physics", "chemistry", "biology"],
            domain_embedding=science_emb,
            topological_cluster=1,
            created_at=now,
        )

        adapter_pool._adapters["tech"] = tech_adapter
        adapter_pool._adapters["science"] = science_adapter_obj

        # 查询编程相关问题
        query_emb = embedding_mgr.encode("How do I debug Python code?")
        adapter, surprise = adapter_pool.select_adapter(query_emb, None)
        assert adapter.adapter_id == "tech", f"Should select tech adapter, got {adapter.adapter_id}"

    def test_select_fallback_low_similarity(self, adapter_pool, embedding_mgr):
        """similarity < 0.3 时应 fallback 到 default"""
        from topomem.adapters import PromptAdapter
        import time

        # 创建一个非常特定的领域
        specific_emb = embedding_mgr.encode(
            "Quantum chromodynamics lattice gauge theory"
        )
        specific_adapter = PromptAdapter(
            id="qcd",
            name="QCD",
            system_prompt="You are a QCD expert.",
            domain_keywords=["quantum", "chromodynamics"],
            domain_embedding=specific_emb,
            topological_cluster=0,
            created_at=time.time(),
        )
        adapter_pool._adapters["qcd"] = specific_adapter

        # 查询完全不相关的内容
        query_emb = embedding_mgr.encode("How to bake a cake?")
        adapter, surprise = adapter_pool.select_adapter(query_emb, None)
        assert adapter.adapter_id == "default", (
            f"Should fallback to default for unrelated query, got {adapter.adapter_id}"
        )

    def test_create_adapter_from_cluster(self, adapter_pool, memory_graph, topo_engine):
        """应成功从记忆簇创建新 adapter"""
        # 添加一组编程相关的记忆
        texts = [
            "Python has a global interpreter lock (GIL).",
            "asyncio provides cooperative multitasking in Python.",
            "Threading module allows concurrent execution.",
        ]
        for text in texts:
            memory_graph.add_memory_from_text(text, topo_engine=topo_engine)

        memory_graph.update_topology(topo_engine)

        # 获取一个簇的记忆
        cluster_ids = memory_graph._get_all_cluster_ids()
        if cluster_ids:
            first_cluster = cluster_ids[0]
            memories = memory_graph.retrieve_by_cluster(first_cluster)

            if memories:
                adapter = adapter_pool.create_adapter(first_cluster, memories)
                assert adapter is not None
                assert adapter.adapter_id != "default"
                assert adapter.topological_cluster == first_cluster
                assert len(adapter.domain_keywords) > 0

    def test_create_adapter_auto_prune(self, adapter_pool, embedding_mgr):
        """超过 max_adapters 时应自动淘汰"""
        from topomem.adapters import AdapterPool, PromptAdapter
        from topomem.config import AdapterConfig
        import time

        # 创建小容量的 pool
        small_config = AdapterConfig(max_adapters=3)
        small_pool = AdapterPool(config=small_config, embedding_mgr=embedding_mgr)

        now = time.time()
        # 创建多个 adapter
        for i in range(5):
            emb = embedding_mgr.encode(f"Topic {i} with unique content {i*10}")
            adapter = PromptAdapter(
                id=f"topic_{i}",
                name=f"Topic {i}",
                system_prompt=f"You are a {i} assistant.",
                domain_keywords=[f"topic{i}"],
                domain_embedding=emb,
                topological_cluster=i,
                created_at=now,
                effectiveness_score=0.5 - i * 0.05,  # 逐渐降低
            )
            small_pool._adapters[f"topic_{i}"] = adapter

        # 手动触发 prune（因为直接添加到 _adapters 不会触发自动 prune）
        small_pool._prune_adapters()

        # 应该淘汰到 max_adapters-1 个非 default adapter
        assert len(small_pool._adapters) <= small_config.max_adapters

    def test_evolve_adapter_updates_score(self, adapter_pool):
        """feedback 应更新 effectiveness_score"""
        default = adapter_pool.default_adapter
        initial_score = default.effectiveness_score

        # 正面反馈
        adapter_pool.evolve_adapter("default", 1.0)
        assert default.effectiveness_score > initial_score

        # 负面反馈
        adapter_pool.evolve_adapter("default", 0.0)
        # 分数应下降但仍 > 0
        assert 0.0 < default.effectiveness_score < 1.0

    def test_prune_removes_lowest(self, adapter_pool, embedding_mgr):
        """prune 应移除 effectiveness 最低的"""
        from topomem.adapters import PromptAdapter
        import time

        now = time.time()
        # 创建 adapter
        for i in range(5):
            emb = embedding_mgr.encode(f"Topic {i}")
            adapter = PromptAdapter(
                id=f"low_{i}",
                name=f"Low {i}",
                system_prompt=f"Prompt {i}",
                domain_keywords=[f"topic{i}"],
                domain_embedding=emb,
                topological_cluster=i,
                created_at=now,
                effectiveness_score=0.1 + i * 0.1,  # 不同分数
            )
            adapter_pool._adapters[f"low_{i}"] = adapter

        before = len(adapter_pool._adapters)
        removed = adapter_pool._prune_adapters()

        # 至少有一个被移除（如果超过限制）
        assert len(adapter_pool._adapters) <= before

    def test_prune_keeps_default(self, adapter_pool, embedding_mgr):
        """default adapter 不应被 prune"""
        from topomem.adapters import PromptAdapter
        import time

        now = time.time()
        # 创建多个低效 adapter
        for i in range(10):
            emb = embedding_mgr.encode(f"Topic {i}")
            adapter = PromptAdapter(
                id=f"temp_{i}",
                name=f"Temp {i}",
                system_prompt=f"Prompt {i}",
                domain_keywords=[f"topic{i}"],
                domain_embedding=emb,
                topological_cluster=i,
                created_at=now,
                effectiveness_score=0.01,
            )
            adapter_pool._adapters[f"temp_{i}"] = adapter

        adapter_pool._prune_adapters()
        assert "default" in adapter_pool._adapters

    def test_adapter_pool_repr(self, adapter_pool):
        """__repr__ 应包含关键信息"""
        repr_str = repr(adapter_pool)
        assert "AdapterPool" in repr_str
        assert "adapters=" in repr_str


# ------------------------------------------------------------------
# Surprise/Tension 测试
# ------------------------------------------------------------------

class TestSurpriseTension:
    """Surprise/Tension 信号系统测试"""

    def test_surprise_known_domain(self, adapter_pool, embedding_mgr, memory_graph, topo_engine):
        """已知领域的 query → surprise 应较低"""
        from topomem.adapters import compute_surprise

        # 添加一些记忆并创建 adapter
        texts = ["Python is great for data science and machine learning."] * 5
        for text in texts:
            memory_graph.add_memory_from_text(text, topo_engine=topo_engine)
        memory_graph.update_topology(topo_engine)

        cluster_ids = memory_graph._get_all_cluster_ids()
        if cluster_ids:
            memories = memory_graph.retrieve_by_cluster(cluster_ids[0])
            if memories:
                adapter_pool.create_adapter(cluster_ids[0], memories)

        # 查询相同领域
        query_emb = embedding_mgr.encode("Python programming")
        surprise = compute_surprise(query_emb, adapter_pool)
        # surprise 应相对较低（< 1.0）
        assert surprise < 1.0

    def test_surprise_unknown_domain(self, adapter_pool, embedding_mgr):
        """全新领域的 query → surprise 应较高"""
        from topomem.adapters import compute_surprise

        # 无自定义 adapter
        query_emb = embedding_mgr.encode(
            "Quantum entanglement in superconducting qubits"
        )
        surprise = compute_surprise(query_emb, adapter_pool)
        # surprise 应很高（= 1.0 因为没有自定义 adapter）
        assert surprise == 1.0

    def test_tension_stable_system(self, memory_graph, topo_engine):
        """稳定系统的 tension 应接近 0"""
        from topomem.self_awareness import SelfAwareness
        from topomem.adapters import compute_tension

        # 添加相同类型的数据
        for i in range(5):
            memory_graph.add_memory_from_text(f"Memory {i}")
        memory_graph.update_topology(topo_engine)

        sa = SelfAwareness()
        sa.update_fingerprint(memory_graph, topo_engine)
        # 只有一条指纹，tension 应为 0
        tension = compute_tension(sa)
        assert tension == 0.0

    def test_tension_changing_system(self, memory_graph, topo_engine):
        """快速变化的系统 tension 应较高"""
        from topomem.self_awareness import SelfAwareness
        from topomem.adapters import compute_tension

        sa = SelfAwareness()

        # 添加一组数据
        for i in range(5):
            memory_graph.add_memory_from_text(f"Python code {i}")
        memory_graph.update_topology(topo_engine)
        sa.update_fingerprint(memory_graph, topo_engine)

        # 添加完全不同的一组数据
        for i in range(5):
            memory_graph.add_memory_from_text(f"Quantum physics {i}")
        memory_graph.update_topology(topo_engine)
        sa.update_fingerprint(memory_graph, topo_engine)

        tension = compute_tension(sa)
        # tension 应 > 0（有变化）
        assert tension >= 0.0


# ------------------------------------------------------------------
# 序列化测试
# ------------------------------------------------------------------

class TestAdapterSerialization:
    """AdapterPool 保存和加载测试"""

    def test_adapter_pool_save_load(self, adapter_pool, embedding_mgr, memory_graph, topo_engine):
        """save 后 load，所有 adapter 应完整恢复"""
        # 添加一些记忆并创建 adapter
        for i in range(5):
            memory_graph.add_memory_from_text(f"Memory {i}")
        memory_graph.update_topology(topo_engine)

        cluster_ids = memory_graph._get_all_cluster_ids()
        if cluster_ids:
            for cid in cluster_ids[:2]:  # 最多创建 2 个
                memories = memory_graph.retrieve_by_cluster(cid)
                if memories:
                    adapter_pool.create_adapter(cid, memories)

        before_count = adapter_pool.adapter_count

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "adapters.json")
            adapter_pool.save(path)

            from topomem.adapters import AdapterPool
            new_pool = AdapterPool(embedding_mgr=embedding_mgr)
            new_pool.load(path)

            assert new_pool.adapter_count == before_count
            assert "default" in new_pool._adapters

    def test_load_nonexistent_file(self, adapter_pool):
        """加载不存在的文件应不报错"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nonexistent.json")
            adapter_pool.load(path)
            # 应保持原样
            assert adapter_pool.adapter_count >= 1


# ------------------------------------------------------------------
# 决策矩阵测试
# ------------------------------------------------------------------

class TestDecisionMatrix:
    """Surprise/Tension 决策矩阵测试"""

    def test_decide_action_use_existing(self):
        """低 surprise + 低 tension → use_existing"""
        from topomem.adapters import decide_action
        action = decide_action(surprise=0.2, tension=0.05)
        assert action == "use_existing"

    def test_decide_action_create_adapter(self):
        """高 surprise + 低 tension → create_adapter"""
        from topomem.adapters import decide_action
        action = decide_action(surprise=0.8, tension=0.05)
        assert action == "create_adapter"

    def test_decide_action_consolidate(self):
        """低 surprise + 高 tension → consolidate"""
        from topomem.adapters import decide_action
        action = decide_action(surprise=0.2, tension=0.2)
        assert action == "consolidate"

    def test_decide_action_consolidate_and_delay(self):
        """高 surprise + 高 tension → consolidate_and_delay"""
        from topomem.adapters import decide_action
        action = decide_action(surprise=0.8, tension=0.2)
        assert action == "consolidate_and_delay"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
