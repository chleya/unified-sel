"""
Phase 1: Embedding + TDA 引擎单元测试

测试 embedding.py 和 topology.py 的完整功能。
"""

from __future__ import annotations

import os
import sys
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


# ------------------------------------------------------------------
# EmbeddingManager 测试
# ------------------------------------------------------------------

class TestEmbeddingManager:
    """EmbeddingManager 完整功能测试"""

    @pytest.fixture
    def manager(self):
        from topomem.embedding import EmbeddingManager
        return EmbeddingManager()

    def test_encode_single_text(self, manager):
        """编码单段文本"""
        text = "This is a test sentence."
        embedding = manager.encode(text)
        assert embedding.shape == (384,)
        assert np.all(np.isfinite(embedding))

    def test_encode_empty_text(self, manager):
        """编码空文本应返回零向量"""
        embedding = manager.encode("")
        assert embedding.shape == (384,)
        assert np.allclose(embedding, 0.0)

    def test_encode_batch(self, manager):
        """批量编码"""
        texts = ["Hello world.", "Testing batch encoding.", "Third sentence."]
        embeddings = manager.encode_batch(texts)
        assert embeddings.shape == (3, 384)
        assert np.all(np.isfinite(embeddings))

    def test_encode_batch_empty(self, manager):
        """空列表应返回空数组"""
        embeddings = manager.encode_batch([])
        assert embeddings.shape == (0, 384)

    def test_similarity_same(self, manager):
        """相同向量的相似度应为 1"""
        text = "Test sentence."
        emb = manager.encode(text)
        sim = manager.similarity(emb, emb)
        assert abs(sim - 1.0) < 1e-5

    def test_similarity_different(self, manager):
        """不同句子的相似度应 < 1"""
        emb1 = manager.encode("The cat sits on the mat.")
        emb2 = manager.encode("Stock prices crashed today.")
        sim = manager.similarity(emb1, emb2)
        assert sim < 0.8  # 语义不同的句子应该相似度较低

    def test_similarity_semantic(self, manager):
        """语义相似的句子应该有更高相似度"""
        emb1 = manager.encode("The cat is sleeping.")
        emb2 = manager.encode("A kitten rests peacefully.")
        emb3 = manager.encode("The stock market crashed.")
        sim_similar = manager.similarity(emb1, emb2)
        sim_different = manager.similarity(emb1, emb3)
        assert sim_similar > sim_different

    def test_similarity_matrix(self, manager):
        """相似度矩阵计算"""
        texts = ["Hello.", "World.", "Test."]
        embeddings = manager.encode_batch(texts)
        sim_matrix = manager.similarity_matrix(embeddings)
        assert sim_matrix.shape == (3, 3)
        # 对角线应该接近 1
        assert np.allclose(np.diag(sim_matrix), 1.0, atol=1e-5)
        # 矩阵应该对称
        assert np.allclose(sim_matrix, sim_matrix.T, atol=1e-6)

    def test_dimension_property(self, manager):
        """dimension 属性应返回 384"""
        assert manager.dimension == 384

    def test_repr(self, manager):
        """__repr__ 应包含关键信息"""
        repr_str = repr(manager)
        assert "EmbeddingManager" in repr_str
        assert "dim=384" in repr_str

    def test_unload(self, manager):
        """unload 后模型应为 None"""
        _ = manager.model  # 触发加载
        manager.unload()
        assert manager._model is None


# ------------------------------------------------------------------
# TopologyEngine 测试
# ------------------------------------------------------------------

class TestTopologyEngine:
    """TopologyEngine 完整功能测试"""

    @pytest.fixture
    def engine(self):
        from topomem.topology import TopologyEngine
        return TopologyEngine()

    # --- compute_persistence 测试 ---

    def test_empty_points(self, engine):
        """0 个或 1 个点 → 返回空 diagram"""
        points = np.array([[1.0, 2.0]])
        diagram = engine.compute_persistence(points)
        assert len(diagram) == 2
        assert len(diagram[0]) == 0
        assert len(diagram[1]) == 0

    def test_two_points(self, engine):
        """2 个点 → H0 有一个 merge 事件"""
        points = np.array([[0.0, 0.0], [10.0, 0.0]])
        diagram = engine.compute_persistence(points)
        assert len(diagram) == 2
        # H0 应至少有一个点
        assert len(diagram[0]) >= 1

    def test_three_clusters(self, engine):
        """3 组明显分开的点 → H0 应有 3 个高 persistence 特征"""
        rng = np.random.RandomState(42)
        cluster_0 = rng.rand(10, 2) * 0.1
        cluster_1 = rng.rand(10, 2) * 0.1 + np.array([10.0, 0.0])
        cluster_2 = rng.rand(10, 2) * 0.1 + np.array([0.0, 10.0])
        points = np.vstack([cluster_0, cluster_1, cluster_2])

        diagram = engine.compute_persistence(points)
        features = engine.extract_persistent_features(diagram)

        # 应该有高 persistence 的 H0 特征
        h0_features = [f for f in features if f.dimension == 0]
        assert len(h0_features) >= 2  # 至少 2 个 merge 事件（3 个簇）

    def test_circle_h1(self, engine):
        """圆上的点 → H1 应有一个高 persistence 环"""
        n_points = 30
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        points = np.column_stack([np.cos(angles), np.sin(angles)])

        diagram = engine.compute_persistence(points)
        features = engine.extract_persistent_features(diagram)

        # 应该有 H1 特征（环）
        h1_features = [f for f in features if f.dimension == 1]
        assert len(h1_features) >= 1, "圆应该产生一个 H1 环特征"

    def test_noise_robustness(self, engine):
        """同一数据加不同程度噪声 → 高 persistence 特征应稳定"""
        rng = np.random.RandomState(42)
        base_points = rng.rand(20, 2) * 10

        diagram_1 = engine.compute_persistence(base_points)
        noisy_points = base_points + rng.normal(0, 0.1, base_points.shape)
        diagram_2 = engine.compute_persistence(noisy_points)

        # 高 persistence 特征应该相似
        features_1 = engine.extract_persistent_features(diagram_1)
        features_2 = engine.extract_persistent_features(diagram_2)

        # 特征数量不应相差太多
        assert abs(len(features_1) - len(features_2)) <= 3

    def test_nan_raises_error(self, engine):
        """NaN 输入应抛出 ValueError"""
        points = np.array([[1.0, float("nan")], [2.0, 3.0]])
        with pytest.raises(ValueError, match="NaN or Inf"):
            engine.compute_persistence(points)

    def test_inf_raises_error(self, engine):
        """Inf 输入应抛出 ValueError"""
        points = np.array([[1.0, float("inf")], [2.0, 3.0]])
        with pytest.raises(ValueError, match="NaN or Inf"):
            engine.compute_persistence(points)

    # --- extract_persistent_features 测试 ---

    def test_empty_diagram(self, engine):
        """空 diagram 应返回空列表"""
        diagram = [np.empty((0, 2)), np.empty((0, 2))]
        features = engine.extract_persistent_features(diagram)
        assert features == []

    def test_filters_infinite_death(self, engine):
        """应过滤掉 death=inf 的点"""
        diagram = [
            np.array([[0.0, np.inf], [0.5, 1.0], [0.3, 0.8]]),
            np.empty((0, 2)),
        ]
        features = engine.extract_persistent_features(diagram)
        # 不应该有 death=inf 的特征
        for f in features:
            assert np.isfinite(f.death)

    def test_sorted_by_persistence(self, engine):
        """特征应按 persistence 降序排序"""
        diagram = [
            np.array([[0.1, 0.9], [0.2, 0.5], [0.3, 1.5]]),
            np.empty((0, 2)),
        ]
        features = engine.extract_persistent_features(diagram)
        persistences = [f.persistence for f in features]
        assert persistences == sorted(persistences, reverse=True)

    # --- wasserstein_distance 测试 ---

    def test_wasserstein_self_zero(self, engine):
        """同一 diagram 的 Wasserstein 距离应为 0"""
        diagram = engine.compute_persistence(np.random.RandomState(42).rand(20, 2))
        dist = engine.wasserstein_distance(diagram, diagram, dim=0)
        assert dist == 0.0

    def test_wasserstein_different(self, engine):
        """不同 diagram 的距离应 > 0"""
        rng = np.random.RandomState(42)
        points_a = rng.rand(20, 2)
        points_b = rng.rand(20, 2) + 10.0  # 完全不同的点集

        diag_a = engine.compute_persistence(points_a)
        diag_b = engine.compute_persistence(points_b)

        dist = engine.wasserstein_distance(diag_a, diag_b, dim=0)
        assert dist > 0.0

    def test_wasserstein_both_empty(self, engine):
        """两个空 diagram 的距离应为 0"""
        empty = [np.empty((0, 2)), np.empty((0, 2))]
        dist = engine.wasserstein_distance(empty, empty, dim=0)
        assert dist == 0.0

    # --- topological_summary 测试 ---

    def test_fingerprint_shape(self, engine):
        """指纹维度应为 (100,)"""
        points = np.random.RandomState(42).rand(30, 2)
        diagram = engine.compute_persistence(points)
        fingerprint = engine.topological_summary(diagram)
        assert fingerprint.shape == (100,)

    def test_fingerprint_deterministic(self, engine):
        """同一数据两次计算的指纹应完全相同"""
        points = np.random.RandomState(42).rand(30, 2)
        diagram = engine.compute_persistence(points)
        fp1 = engine.topological_summary(diagram)
        fp2 = engine.topological_summary(diagram)
        assert np.allclose(fp1, fp2)

    def test_fingerprint_l2_normalized(self, engine):
        """指纹应 L2 归一化"""
        points = np.random.RandomState(42).rand(50, 2)
        diagram = engine.compute_persistence(points)
        fingerprint = engine.topological_summary(diagram)
        norm = np.linalg.norm(fingerprint)
        assert abs(norm - 1.0) < 1e-5 or norm < 1e-10  # 允许全零向量

    def test_fingerprint_empty_diagram(self, engine):
        """空 diagram 的指纹应为零向量"""
        diagram = [np.empty((0, 2)), np.empty((0, 2))]
        fingerprint = engine.topological_summary(diagram)
        assert fingerprint.shape == (100,)
        assert np.allclose(fingerprint, 0.0)

    def test_top_k_features_method(self, engine):
        """top_k_features 方法应返回正确形状"""
        points = np.random.RandomState(42).rand(30, 2)
        diagram = engine.compute_persistence(points)
        fingerprint = engine.topological_summary(diagram, method="top_k_features")
        assert len(fingerprint.shape) == 1
        assert len(fingerprint) > 0

    # --- cluster_labels_from_h0 测试 ---

    def test_cluster_labels_count(self, engine):
        """3 组分开的点 → 应返回 3 种标签"""
        rng = np.random.RandomState(42)
        cluster_0 = rng.rand(10, 2) * 0.1
        cluster_1 = rng.rand(10, 2) * 0.1 + np.array([10.0, 0.0])
        cluster_2 = rng.rand(10, 2) * 0.1 + np.array([0.0, 10.0])
        points = np.vstack([cluster_0, cluster_1, cluster_2])

        diagram = engine.compute_persistence(points)
        labels = engine.cluster_labels_from_h0(diagram, points)

        assert len(labels) == 30
        assert len(np.unique(labels)) >= 2  # 至少 2 个簇

    def test_cluster_labels_single_point(self, engine):
        """单个点应返回 [0]"""
        points = np.array([[1.0, 2.0]])
        diagram = engine.compute_persistence(points)
        labels = engine.cluster_labels_from_h0(diagram, points)
        assert labels[0] == 0

    # --- compute_full_result 测试 ---

    def test_full_result(self, engine):
        """完整结果应包含所有字段"""
        from topomem.topology import TopologyResult
        points = np.random.RandomState(42).rand(30, 2)
        result = engine.compute_full_result(points)

        assert isinstance(result, TopologyResult)
        assert len(result.diagram) == 2
        assert isinstance(result.features, list)
        assert result.fingerprint.shape == (100,)
        assert result.n_clusters >= 1
        assert result.cluster_labels is not None
        assert len(result.cluster_labels) == 30

    def test_repr(self, engine):
        """__repr__ 应包含关键信息"""
        repr_str = repr(engine)
        assert "TopologyEngine" in repr_str
        assert "max_dim=" in repr_str


# ------------------------------------------------------------------
# 集成测试：Embedding + Topology
# ------------------------------------------------------------------

class TestEmbeddingTopologyIntegration:
    """Embedding + Topology 集成测试"""

    @pytest.fixture
    def manager(self):
        from topomem.embedding import EmbeddingManager
        return EmbeddingManager()

    @pytest.fixture
    def engine(self):
        from topomem.topology import TopologyEngine
        return TopologyEngine()

    def test_text_clusters(self, manager, engine):
        """不同主题的文本应形成不同的拓扑簇"""
        # 两组不同主题的文本
        tech_texts = [
            "The neural network uses backpropagation.",
            "Deep learning models require large datasets.",
            "Transformer architecture improves NLP performance.",
        ]
        nature_texts = [
            "The cat sleeps on the warm mat.",
            "Birds fly south for the winter.",
            "Trees lose leaves in autumn.",
        ]

        # 编码
        all_texts = tech_texts + nature_texts
        embeddings = manager.encode_batch(all_texts)

        # 拓扑分析
        diagram = engine.compute_persistence(embeddings)
        labels = engine.cluster_labels_from_h0(diagram, embeddings)

        # 应该至少有 2 个簇
        n_clusters = len(np.unique(labels))
        assert n_clusters >= 2, f"应至少有 2 个簇，实际 {n_clusters}"

    def test_fingerprint_from_text(self, manager, engine):
        """从文本计算拓扑指纹"""
        texts = [
            "Machine learning is a subset of AI.",
            "Python is popular for data science.",
            "The weather is sunny today.",
            "Cats are independent pets.",
        ]
        embeddings = manager.encode_batch(texts)
        diagram = engine.compute_persistence(embeddings)
        fingerprint = engine.topological_summary(diagram)

        assert fingerprint.shape == (100,)
        assert np.all(np.isfinite(fingerprint))

    def test_wasserstein_topic_shift(self, manager, engine):
        """主题切换应产生较大的 Wasserstein 距离"""
        # 主题 A：技术
        tech_texts = [
            "Neural networks learn from data.",
            "Gradient descent optimizes weights.",
            "Convolution layers extract features.",
        ]
        # 主题 B：自然
        nature_texts = [
            "Rivers flow to the ocean.",
            "Mountains are formed by tectonic activity.",
            "Forests provide oxygen.",
        ]

        emb_tech = manager.encode_batch(tech_texts)
        emb_nature = manager.encode_batch(nature_texts)

        diag_tech = engine.compute_persistence(emb_tech)
        diag_nature = engine.compute_persistence(emb_nature)

        dist = engine.wasserstein_distance(diag_tech, diag_nature, dim=0)
        assert dist > 0.0, "主题切换应产生非零距离"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
