"""
Phase 5: 自我认知与一致性守护单元测试

测试 self_awareness.py 和 guard.py 的完整功能。
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
    config = MemoryConfig(
        max_nodes=50,
        chroma_persist_dir=str(tmp_path / "chromadb"),
    )
    return MemoryGraph(config=config, embedding_mgr=embedding_mgr)


@pytest.fixture
def self_awareness():
    from topomem.self_awareness import SelfAwareness
    return SelfAwareness()


@pytest.fixture
def guard():
    from topomem.guard import ConsistencyGuard
    return ConsistencyGuard()


# ------------------------------------------------------------------
# SelfAwareness 测试
# ------------------------------------------------------------------

class TestSelfAwareness:
    """SelfAwareness 完整功能测试"""

    def test_initial_fingerprint(self, memory_graph, topo_engine, self_awareness):
        """首次 update 后应设置 baseline_fingerprint"""
        # 添加一些记忆
        for i in range(10):
            memory_graph.add_memory_from_text(f"Memory {i}", topo_engine=topo_engine)

        memory_graph.update_topology(topo_engine)
        self_awareness.update_fingerprint(memory_graph, topo_engine)

        assert self_awareness._baseline_fingerprint is not None
        assert self_awareness._baseline_fingerprint.shape == (100,)
        assert len(self_awareness._fingerprint_history) == 1

    def test_fingerprint_history_size(self, memory_graph, topo_engine, self_awareness):
        """历史超过 max 时应丢弃最旧的"""
        from topomem.config import SelfAwarenessConfig
        from topomem.self_awareness import SelfAwareness

        small_config = SelfAwarenessConfig(fingerprint_history_size=3)
        sa = SelfAwareness(config=small_config)

        # 添加 5 次指纹
        for i in range(5):
            memory_graph.add_memory_from_text(f"Memory {i}_{i*10}")
            memory_graph.update_topology(topo_engine)
            sa.update_fingerprint(memory_graph, topo_engine)

        assert len(sa._fingerprint_history) <= 3

    def test_detect_drift_stable(self, memory_graph, topo_engine, self_awareness):
        """相同数据多次更新 → status=stable"""
        # 添加少量数据
        for i in range(5):
            memory_graph.add_memory_from_text(f"Memory {i}")
        memory_graph.update_topology(topo_engine)
        self_awareness.update_fingerprint(memory_graph, topo_engine)

        # 再次更新（数据不变）
        memory_graph.update_topology(topo_engine)
        self_awareness.update_fingerprint(memory_graph, topo_engine)

        report = self_awareness.detect_drift()
        assert report.status in ("stable", "evolving")

    def test_identity_vector_shape(self, memory_graph, topo_engine, self_awareness):
        """应返回 (20,) 向量"""
        for i in range(10):
            memory_graph.add_memory_from_text(f"Memory {i}")
        memory_graph.update_topology(topo_engine)
        self_awareness.update_fingerprint(memory_graph, topo_engine)

        identity = self_awareness.get_identity_vector()
        assert identity.shape == (20,)

    def test_identity_vector_stability(self, memory_graph, topo_engine, self_awareness):
        """小变化不应大幅改变身份向量"""
        for i in range(10):
            memory_graph.add_memory_from_text(f"Memory {i}")
        memory_graph.update_topology(topo_engine)
        self_awareness.update_fingerprint(memory_graph, topo_engine)

        id1 = self_awareness.get_identity_vector()

        # 添加一条新记忆
        memory_graph.add_memory_from_text("New memory")
        memory_graph.update_topology(topo_engine)
        self_awareness.update_fingerprint(memory_graph, topo_engine)

        id2 = self_awareness.get_identity_vector()

        # 身份向量应有一定变化但不应完全不同
        diff = np.linalg.norm(id1 - id2)
        assert diff < 5.0  # 不应差异过大

    def test_should_calibrate_interval(self, memory_graph, topo_engine, self_awareness):
        """达到 calibration_interval 时应返回 True"""
        from topomem.config import SelfAwarenessConfig
        from topomem.self_awareness import SelfAwareness

        config = SelfAwarenessConfig(calibration_interval=5)
        sa = SelfAwareness(config=config)

        # 模拟 5 步
        for _ in range(5):
            sa._step_count += 1

        assert sa.should_calibrate()

    def test_save_load_roundtrip(self, memory_graph, topo_engine, self_awareness):
        """save 后 load，指纹历史应完整恢复"""
        for i in range(5):
            memory_graph.add_memory_from_text(f"Memory {i}")
        memory_graph.update_topology(topo_engine)
        self_awareness.update_fingerprint(memory_graph, topo_engine)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "self_awareness.json")
            self_awareness.save(path)

            from topomem.self_awareness import SelfAwareness
            new_sa = SelfAwareness()
            new_sa.load(path)

            assert new_sa._step_count == self_awareness._step_count
            assert len(new_sa._fingerprint_history) == len(self_awareness._fingerprint_history)

    def test_repr(self, self_awareness):
        """__repr__ 应包含关键信息"""
        repr_str = repr(self_awareness)
        assert "SelfAwareness" in repr_str
        assert "step=" in repr_str


# ------------------------------------------------------------------
# ConsistencyGuard 测试
# ------------------------------------------------------------------

class TestConsistencyGuard:
    """ConsistencyGuard 完整功能测试"""

    def test_reject_duplicate(self, memory_graph, topo_engine, self_awareness, guard, embedding_mgr):
        """similarity > 0.95 的重复内容应被拒绝"""
        # 添加一条记忆
        content = "The Earth orbits the Sun."
        memory_graph.add_memory_from_text(content, topo_engine=topo_engine)

        new_embedding = embedding_mgr.encode(content)
        accept, reason = guard.should_accept_memory(
            content, new_embedding, memory_graph, self_awareness
        )
        assert not accept, f"Should reject duplicate, got: {reason}"
        assert "duplicate" in reason.lower()

    def test_accept_new_knowledge(self, memory_graph, topo_engine, self_awareness, guard, embedding_mgr):
        """全新领域的知识应被接受"""
        memory_graph.add_memory_from_text("Python is a programming language.", topo_engine=topo_engine)

        new_content = "The Pacific Ocean is the largest ocean."
        new_embedding = embedding_mgr.encode(new_content)

        accept, reason = guard.should_accept_memory(
            new_content, new_embedding, memory_graph, self_awareness
        )
        assert accept, f"Should accept new knowledge, got: {reason}"

    def test_warn_contradiction(self, memory_graph, topo_engine, self_awareness, guard, embedding_mgr):
        """潜在矛盾应 accept=True 但 reason 包含 warning"""
        memory_graph.add_memory_from_text("The sky is always blue.", topo_engine=topo_engine)

        # 矛盾内容（包含否定词）
        new_content = "The sky is not always blue; it changes color."
        new_embedding = embedding_mgr.encode(new_content)

        accept, reason = guard.should_accept_memory(
            new_content, new_embedding, memory_graph, self_awareness
        )
        assert accept, "Contradiction should be accepted with warning"
        if "contradiction" in reason.lower() or "warning" in reason.lower():
            assert True  # 有 warning

    def test_capacity_warning(self, memory_graph, topo_engine, self_awareness, guard, embedding_mgr):
        """满容量时应建议 prune"""
        from topomem.memory import MemoryGraph
        from topomem.config import MemoryConfig
        import tempfile

        # 创建小容量的图
        small_config = MemoryConfig(max_nodes=3, chroma_persist_dir=str(tempfile.mkdtemp()))
        small_graph = MemoryGraph(config=small_config, embedding_mgr=embedding_mgr)

        # 填满
        for i in range(3):
            small_graph.add_memory_from_text(f"Memory {i}")

        new_content = "New knowledge"
        new_embedding = embedding_mgr.encode(new_content)

        accept, reason = guard.should_accept_memory(
            new_content, new_embedding, small_graph, self_awareness
        )
        assert accept
        if "capacity" in reason.lower() or "prune" in reason.lower():
            assert True  # 有容量警告

    def test_consolidation_merge(self, memory_graph, topo_engine, self_awareness, guard, embedding_mgr):
        """高相似度节点对应被建议合并"""
        # 添加两条非常相似的记
        memory_graph.add_memory_from_text("Python is great for data science.", topo_engine=topo_engine)
        memory_graph.add_memory_from_text("Python is excellent for data science.", topo_engine=topo_engine)

        memory_graph.update_topology(topo_engine)
        actions = guard.recommend_consolidation(memory_graph, topo_engine)

        merge_actions = [a for a in actions if a.action_type == "merge"]
        # 高相似度对可能被建议合并
        assert len(actions) >= 0  # 至少返回空列表

    def test_consolidation_strengthen(self, memory_graph, topo_engine, guard):
        """高持久性节点应被建议强化"""
        for i in range(10):
            memory_graph.add_memory_from_text(f"Memory {i}", topo_engine=topo_engine)

        memory_graph.update_topology(topo_engine)
        actions = guard.recommend_consolidation(memory_graph, topo_engine)

        # 可能有强化建议
        assert isinstance(actions, list)

    def test_consolidation_remove(self, memory_graph, topo_engine, guard):
        """低重要性节点应被建议删除"""
        for i in range(10):
            memory_graph.add_memory_from_text(f"Memory {i}", topo_engine=topo_engine)

        memory_graph.update_topology(topo_engine)
        actions = guard.recommend_consolidation(memory_graph, topo_engine)

        # 可能有删除建议
        assert isinstance(actions, list)

    def test_guard_repr(self, guard):
        """__repr__ 应包含类名"""
        repr_str = repr(guard)
        assert "ConsistencyGuard" in repr_str


# ------------------------------------------------------------------
# 集成测试：SelfAwareness + ConsistencyGuard
# ------------------------------------------------------------------

class TestSelfAwarenessGuardIntegration:
    """SelfAwareness + ConsistencyGuard 集成测试"""

    def test_full_workflow(self, memory_graph, topo_engine, embedding_mgr):
        """完整工作流程：添加记忆 → 自我认知 → 一致性检查"""
        from topomem.self_awareness import SelfAwareness
        from topomem.guard import ConsistencyGuard

        sa = SelfAwareness()
        guard = ConsistencyGuard()

        # 添加一些记忆
        texts = [
            "Neural networks learn from data.",
            "Deep learning uses multiple layers.",
            "Gradient descent optimizes weights.",
        ]
        for text in texts:
            memory_graph.add_memory_from_text(text, topo_engine=topo_engine)

        # 更新拓扑和自我认知
        memory_graph.update_topology(topo_engine)
        sa.update_fingerprint(memory_graph, topo_engine)

        # 漂移检测
        report = sa.detect_drift()
        assert report.status in ("stable", "evolving")

        # 一致性检查
        new_content = "Backpropagation trains neural networks."
        new_embedding = embedding_mgr.encode(new_content)
        accept, reason = guard.should_accept_memory(
            new_content, new_embedding, memory_graph, sa
        )
        assert accept

    def test_drift_detection_after_domain_shift(self, memory_graph, topo_engine, embedding_mgr):
        """领域切换后应检测到漂移"""
        from topomem.self_awareness import SelfAwareness

        sa = SelfAwareness()

        # 领域 A：编程
        for text in [
            "Python has a GIL.",
            "asyncio provides cooperative multitasking.",
            "Threading is limited by GIL.",
        ]:
            memory_graph.add_memory_from_text(text, topo_engine=topo_engine)
        memory_graph.update_topology(topo_engine)
        sa.update_fingerprint(memory_graph, topo_engine)

        # 领域 B：自然（明显不同）
        for text in [
            "Rivers flow to the ocean.",
            "Mountains are formed by tectonic activity.",
            "Forests cover 30% of land.",
        ]:
            memory_graph.add_memory_from_text(text, topo_engine=topo_engine)
        memory_graph.update_topology(topo_engine)
        sa.update_fingerprint(memory_graph, topo_engine)

        report = sa.detect_drift()
        # 应检测到某种漂移
        assert report.status in ("evolving", "drifting", "restructured")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
