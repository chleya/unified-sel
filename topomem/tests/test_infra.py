"""
Phase 0 基础设施冒烟测试

验证所有底层工具可用，不依赖完整模型下载。

重要：所有 HF 缓存必须存储在 F 盘，避免占用 C 盘空间。
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
TOPOMEM_ROOT = PROJECT_ROOT / "topomem"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestCoreLibraries:
    """核心计算库可用性测试"""

    def test_numpy(self):
        """numpy 可用"""
        arr = np.array([1.0, 2.0, 3.0])
        assert arr.mean() == 2.0

    def test_scipy(self):
        """scipy 可用"""
        from scipy.spatial.distance import cosine
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        dist = cosine(a, b)
        assert dist > 0.9

    def test_scikit_learn(self):
        """scikit-learn 可用"""
        from sklearn.metrics.pairwise import cosine_similarity
        a = np.array([[1.0, 0.0, 0.0]])
        b = np.array([[0.0, 1.0, 0.0]])
        sim = cosine_similarity(a, b)
        assert sim[0, 0] == 0.0

    def test_networkx(self):
        """networkx 可用"""
        import networkx as nx
        G = nx.Graph()
        G.add_edge("A", "B")
        assert G.has_edge("A", "B")
        assert nx.number_of_nodes(G) == 2


class TestTDALibraries:
    """TDA 库可用性测试"""

    def test_gudhi(self):
        """gudhi 可用"""
        import gudhi as gd
        # 创建简单的 Rips complex
        points = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        rips = gd.RipsComplex(points=points)
        tree = rips.create_simplex_tree(max_dimension=1)
        assert tree.num_vertices() == 4

    def test_ripser(self):
        """ripser 可用"""
        from ripser import ripser
        # 简单的点云
        points = np.random.RandomState(42).rand(20, 2)
        result = ripser(points, maxdim=1)
        assert "dgms" in result
        assert len(result["dgms"]) >= 1  # 至少 H0

    def test_persim(self):
        """persim 可用"""
        from ripser import ripser
        from persim import bottleneck
        # 计算两个简单持久图之间的 bottleneck 距离
        points = np.random.RandomState(42).rand(10, 2)
        result1 = ripser(points, maxdim=0)
        result2 = ripser(points + 0.1, maxdim=0)
        dist = bottleneck(result1["dgms"][0], result2["dgms"][0])
        assert dist >= 0.0


class TestEmbeddingModel:
    """Embedding 模型可用性测试"""

    def test_sentence_transformers_import(self):
        """sentence-transformers 可导入"""
        import sentence_transformers
        assert hasattr(sentence_transformers, "SentenceTransformer")

    def test_sentence_transformers_encode(self):
        """sentence-transformers 可编码文本"""
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        sentences = ["This is a test sentence.", "Another sentence here."]
        embeddings = model.encode(sentences)
        assert embeddings.shape == (2, 384)
        # 相似句子应该更接近
        sim_sentences = ["The cat sits.", "A cat is sitting."]
        diff_sentences = ["The stock market crashed.", "I like pizza."]
        sim_emb = model.encode(sim_sentences)
        diff_emb = model.encode(diff_sentences)
        sim_sim = float(np.dot(sim_emb[0], sim_emb[1]) / (np.linalg.norm(sim_emb[0]) * np.linalg.norm(sim_emb[1])))
        diff_sim = float(np.dot(diff_emb[0], diff_emb[1]) / (np.linalg.norm(diff_emb[0]) * np.linalg.norm(diff_emb[1])))
        assert sim_sim > diff_sim, "相似句子应该具有更高余弦相似度"


class TestChromaDB:
    """ChromaDB 可用性测试"""

    def test_chromadb_import(self):
        """chromadb 可导入"""
        import chromadb
        assert hasattr(chromadb, "Client")

    def test_chromadb_crud(self):
        """ChromaDB CRUD 操作"""
        import chromadb
        client = chromadb.Client()
        collection = client.create_collection("test_collection")

        # Create
        collection.add(
            documents=["Hello world", "Testing ChromaDB"],
            ids=["doc1", "doc2"]
        )

        # Read
        results = collection.get()
        assert len(results["ids"]) == 2

        # Query
        query_results = collection.query(
            query_texts=["Hello"],
            n_results=1
        )
        assert len(query_results["ids"][0]) == 1
        assert query_results["ids"][0][0] == "doc1"

        # Delete
        collection.delete(ids=["doc2"])
        results = collection.get()
        assert len(results["ids"]) == 1


class TestTransformersFallback:
    """Transformers fallback 可用性测试"""

    def test_transformers_import(self):
        """transformers 可导入"""
        import transformers
        assert hasattr(transformers, "AutoTokenizer")
        assert hasattr(transformers, "AutoModelForCausalLM")

    def test_tokenizer_available(self):
        """tokenizer 可用"""
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        tokens = tokenizer("Hello, world!")
        assert "input_ids" in tokens
        assert len(tokens["input_ids"]) > 0


class TestPSUtil:
    """psutil 可用性测试"""

    def test_psutil_import(self):
        """psutil 可导入"""
        import psutil
        assert hasattr(psutil, "virtual_memory")
        assert hasattr(psutil, "cpu_percent")

    def test_system_info(self):
        """可获取系统信息"""
        import psutil
        mem = psutil.virtual_memory()
        assert mem.total > 0
        assert mem.available > 0


class TestTopoMemConfig:
    """TopoMem 配置可用性测试"""

    def test_config_import(self):
        """TopoMemConfig 可导入"""
        from topomem.config import TopoMemConfig
        config = TopoMemConfig()
        assert config.embedding.dimension == 384
        assert config.memory.max_nodes == 500

    def test_paths_exist(self):
        """配置中的路径应该存在"""
        from topomem.config import DATA_DIR, MODELS_DIR, CORPUS_DIR, RESULTS_DIR
        assert DATA_DIR.exists()
        # MODELS_DIR 和 CORPUS_DIR 可能还不存在，但父目录应该存在
        assert DATA_DIR.parent.exists()
        assert RESULTS_DIR.exists()


class TestEnvironmentVariables:
    """环境变量配置测试"""

    def test_hf_home_set(self):
        """HF_HOME 应该被设置到 F 盘"""
        hf_home = os.environ.get("HF_HOME", "")
        assert "F:\\" in hf_home, f"HF_HOME 应该指向 F 盘，当前为: {hf_home}"
        assert "C:\\" not in hf_home, f"HF_HOME 不应该指向 C 盘: {hf_home}"

    def test_transformers_cache_set(self):
        """TRANSFORMERS_CACHE 应该被设置到 F 盘"""
        cache = os.environ.get("TRANSFORMERS_CACHE", "")
        assert "F:\\" in cache, f"TRANSFORMERS_CACHE 应该指向 F 盘，当前为: {cache}"

    def test_sentence_transformers_home_set(self):
        """SENTENCE_TRANSFORMERS_HOME 应该被设置到 F 盘"""
        home = os.environ.get("SENTENCE_TRANSFORMERS_HOME", "")
        assert "F:\\" in home, f"SENTENCE_TRANSFORMERS_HOME 应该指向 F 盘，当前为: {home}"

    def test_hf_hub_cache_directory(self):
        """huggingface_hub 应该使用 F 盘缓存目录"""
        from huggingface_hub import constants
        cache_dir = constants.HF_HUB_CACHE
        assert "F:\\" in cache_dir, f"HF_HUB_CACHE 应该指向 F 盘，当前为: {cache_dir}"
        assert "C:\\" not in cache_dir, f"HF_HUB_CACHE 不应该指向 C 盘: {cache_dir}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
