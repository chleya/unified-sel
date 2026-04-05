"""
topomem/embedding.py — Embedding 特征层

将文本输入转换为固定维度的特征向量，作为后续拓扑分析的基础。

设计来源：
- SPEC_TOPOLOGY.md: Embedding Manager 设计
- SPEC_ENGINE.md: EmbeddingManager 规范
- sentence-transformers: all-MiniLM-L6-v2 模型
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np


# 必须在导入 sentence_transformers 之前设置
HF_CACHE = r"F:\unified-sel\topomem\data\models\hf_cache"
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = HF_CACHE
if "SENTENCE_TRANSFORMERS_HOME" not in os.environ:
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = HF_CACHE

from sentence_transformers import SentenceTransformer
from topomem.config import EmbeddingConfig


class EmbeddingManager:
    """管理文本到特征向量的转换。

    使用 sentence-transformers 的预训练 embedding 模型，
    将文本编码为固定维度的稠密向量。

    当前默认模型：all-MiniLM-L6-v2（384 维，快速，CPU 友好）
    """

    # 类级别模型缓存 — 所有实例共享同一个模型，避免重复加载
    _global_model: Optional[SentenceTransformer] = None
    _global_config: Optional[EmbeddingConfig] = None

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        参数：
            config: EmbeddingConfig 配置对象
                    如果为 None，使用默认配置
        """
        self.config = config or EmbeddingConfig()
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        """懒加载模型 + 全局缓存。同一模型的多个实例共享底层 SentenceTransformer。"""
        if self._model is None:
            if EmbeddingManager._global_model is None:
                # 全局缓存为空 → 首次加载
                EmbeddingManager._global_model = SentenceTransformer(
                    self.config.model_name,
                    device=self.config.device,
                )
                EmbeddingManager._global_config = self.config
            elif (
                EmbeddingManager._global_config is not None
                and EmbeddingManager._global_config.model_name != self.config.model_name
            ):
                # 配置不同 → 创建独立模型
                self._model = SentenceTransformer(
                    self.config.model_name,
                    device=self.config.device,
                )
                return self._model
            # 复用全局缓存
            self._model = EmbeddingManager._global_model
        return self._model

    @property
    def dimension(self) -> int:
        """返回 embedding 维度。"""
        return self.config.dimension

    def encode(self, text: str) -> np.ndarray:
        """编码单段文本为特征向量。

        参数：
            text: 输入文本

        返回：
            np.ndarray, shape (D,) 其中 D=384（默认）
        """
        if not text or not text.strip():
            return np.zeros(self.dimension, dtype=np.float32)

        embedding = self.model.encode(
            [text.strip()],
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embedding[0]

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """批量编码多段文本。

        参数：
            texts: 文本列表

        返回：
            np.ndarray, shape (N, D) 其中 N=len(texts)
        """
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        cleaned = [t.strip() if t else "" for t in texts]
        embeddings = self.model.encode(
            cleaned,
            show_progress_bar=False,
            normalize_embeddings=True,
            batch_size=self.config.batch_size,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """计算两个特征向量的余弦相似度。

        参数：
            a: 向量 A, shape (D,)
            b: 向量 B, shape (D,)

        返回：
            float, 范围 [-1, 1]，1 表示完全相同
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """计算一组特征向量两两之间的余弦相似度矩阵。

        参数：
            embeddings: shape (N, D)

        返回：
            np.ndarray, shape (N, N)，对称矩阵
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normalized = embeddings / norms
        return normalized @ normalized.T

    def unload(self) -> None:
        """释放模型，回收内存。"""
        if self._model is not None:
            del self._model
            self._model = None

    def __repr__(self) -> str:
        status = "loaded" if self._model is not None else "lazy"
        return (
            f"EmbeddingManager(model='{self.config.model_name}', "
            f"dim={self.dimension}, device='{self.config.device}', "
            f"status='{status}')"
        )
