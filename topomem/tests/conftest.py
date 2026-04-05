"""
conftest.py — pytest 全局 fixtures

使用 session-scoped fixtures 避免每个测试重新加载模型，
将完整测试套件时间从 ~16 分钟降低到 ~3-5 分钟。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# 确保项目根目录在 sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 设置 HuggingFace 缓存路径（必须在导入 sentence_transformers 之前）
HF_CACHE = str(PROJECT_ROOT / "topomem" / "data" / "models" / "hf_cache")
os.environ["HF_HOME"] = HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE
os.environ["SENTENCE_TRANSFORMERS_HOME"] = HF_CACHE


@pytest.fixture(scope="session")
def embedding_mgr():
    """Session-scoped: 所有测试共享同一个 EmbeddingManager 实例。

    EmbeddingManager 内部已经有类级别模型缓存，
    这个 fixture 确保模型只加载一次。
    """
    from topomem.embedding import EmbeddingManager

    # 触发模型加载（将在全局缓存中）
    mgr = EmbeddingManager()
    _ = mgr.model  # 懒加载
    return mgr


@pytest.fixture(scope="session")
def topo_engine():
    """Session-scoped: TopologyEngine 是轻量级无状态工具，共享实例安全。"""
    from topomem.topology import TopologyEngine

    return TopologyEngine()


@pytest.fixture(scope="session")
def embedding_mgr_for_memory(embedding_mgr):
    """兼容 test_memory.py 的 embedding_mgr fixture 签名。"""
    return embedding_mgr


@pytest.fixture(scope="session")
def topo_engine_for_memory(topo_engine):
    """兼容 test_memory.py 的 topo_engine fixture 签名。"""
    return topo_engine
