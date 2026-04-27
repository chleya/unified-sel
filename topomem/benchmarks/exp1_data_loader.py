"""
topomem/benchmarks/exp1_data_loader.py

实验1数据加载器：20 Newsgroups 数据集

目标：
- 加载有语义标签的多主题数据集
- 每个文档 = embedding + 主题标签
- 用于验证 H0/H1 是否编码语义信息
"""

import json
import os
import sys
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np

PROJECT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT))

HF_CACHE = PROJECT / "topomem" / "data" / "models" / "hf_cache"
os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE)
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(HF_CACHE)

import warnings
warnings.filterwarnings("ignore")


# ------------------------------------------------------------------
# 20 Newsgroups 加载
# ------------------------------------------------------------------

def load_20newsgroups(
    subset: str = "train",
    categories: Optional[List[str]] = None,
    remove: Optional[List[str]] = None,
) -> List[Dict]:
    """加载 20 Newsgroups 数据集。

    Args:
        subset: "train" 或 "test"
        categories: 要加载的子集（如 ["rec.autos", "sci.physics"]）
                   None = 加载全部 20 个类别
        remove: 要移除的内容（如 ["headers", "footers", "quotes"]）

    Returns:
        List[Dict], 每项包含:
        - text: str, 文档内容
        - label: int, 类别索引 (0-19)
        - label_name: str, 类别名称
        - topic: str, 简化后的主题标签
    """
    try:
        from sklearn.datasets import fetch_20newsgroups
    except ImportError:
        raise ImportError("需要 sklearn: pip install scikit-learn")

    # 默认移除元信息，只保留正文
    if remove is None:
        remove = ("headers", "footers", "quotes")

    print(f"Loading 20 Newsgroups ({subset})...")
    data = fetch_20newsgroups(
        subset=subset,
        categories=categories,
        remove=remove,
        shuffle=True,
        random_state=42,
        return_X_y=False,
    )

    # 类别信息
    target_names = data.target_names
    topic_map = _create_topic_map(target_names)

    items = []
    for i, (text, label) in enumerate(zip(data.data, data.target)):
        if len(text.strip()) < 50:
            continue  # 跳过太短的文档

        items.append({
            "text": text[:2000],  # 截断到 2000 字符
            "label": int(label),
            "label_name": target_names[label],
            "topic": topic_map[label],
            "doc_id": f"doc_{i}",
        })

    print(f"  Loaded {len(items)} documents from {len(target_names)} categories")
    return items


def _create_topic_map(target_names: List[str]) -> Dict[int, str]:
    """将 20 个细粒度类别映射到 6 个粗粒度主题。

    粗粒度主题便于计算纯度和熵。
    """
    topic_mapping = {
        # alt (1)
        "alt.atheism": "religion",
        # comp (5)
        "comp.graphics": "computing",
        "comp.os.ms-windows.misc": "computing",
        "comp.sys.ibm.pc.hardware": "computing",
        "comp.sys.mac.hardware": "computing",
        "comp.windows.x": "computing",
        # misc (1)
        "misc.forsale": "commerce",
        # rec (4)
        "rec.autos": "recreation",
        "rec.motorcycles": "recreation",
        "rec.sport.baseball": "recreation",
        "rec.sport.hockey": "recreation",
        # sci (4)
        "sci.crypt": "science",
        "sci.electronics": "science",
        "sci.med": "science",
        "sci.space": "science",
        # soc (1)
        "talk.politics.mideast": "politics",
        # talk (3)
        "talk.politics.guns": "politics",
        "talk.politics.misc": "politics",
        "talk.religion.misc": "religion",
    }

    # 填充未映射的类别（兜底）
    for name in target_names:
        if name not in topic_mapping:
            topic_mapping[name] = name.split(".")[0]

    return {i: topic_mapping.get(name, name.split(".")[0]) for i, name in enumerate(target_names)}


# ------------------------------------------------------------------
# Embedding 编码
# ------------------------------------------------------------------

def encode_corpus(
    items: List[Dict],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    cache_path: Optional[str] = None,
) -> Tuple[List[Dict], np.ndarray]:
    """对语料库中所有文档进行 embedding 编码。

    Args:
        items: load_20newsgroups() 的输出
        model_name: sentence-transformers 模型名
        batch_size: 批处理大小
        cache_path: 缓存文件路径（可选）

    Returns:
        (items_with_embedding, embeddings_matrix)
        items_with_embedding: 每项新增 embedding 字段（384维 numpy array）
        embeddings_matrix: (N, 384) numpy array
    """
    from topomem.embedding import EmbeddingManager, EmbeddingConfig

    # 检查缓存
    if cache_path and os.path.exists(cache_path):
        print(f"  Loading embeddings from cache: {cache_path}")
        cached = np.load(cache_path)
        embeddings = cached["embeddings"]
        for i, item in enumerate(items):
            item["embedding"] = embeddings[i]
        return items, embeddings

    print(f"  Encoding {len(items)} documents with {model_name}...")

    texts = [item["text"] for item in items]

    emb_config = EmbeddingConfig(model_name=model_name)
    emb_manager = EmbeddingManager(config=emb_config)

    embeddings = emb_manager.encode_batch(texts)
    emb_manager.unload()

    # 写回 items
    for i, item in enumerate(items):
        item["embedding"] = embeddings[i]

    # 缓存
    if cache_path:
        cache_dir = os.path.dirname(cache_path)
        os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(cache_path, embeddings=embeddings)
        print(f"  Cached embeddings to: {cache_path}")

    return items, embeddings


# ------------------------------------------------------------------
# 持久化
# ------------------------------------------------------------------

def get_cache_path(dataset_name: str = "20newsgroups") -> str:
    """获取缓存文件路径。"""
    cache_dir = PROJECT / "topomem" / "data" / "benchmarks" / "exp1"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir / f"{dataset_name}_embeddings.npz")


def load_or_fetch_corpus(
    subset: str = "train",
    categories: Optional[List[str]] = None,
    use_cache: bool = True,
) -> Tuple[List[Dict], np.ndarray]:
    """加载或获取语料库的便捷入口。

    自动处理缓存逻辑。
    """
    cache_path = get_cache_path() if use_cache else None

    items = load_20newsgroups(subset=subset, categories=categories)
    items, embeddings = encode_corpus(items, cache_path=cache_path)

    return items, embeddings


# ------------------------------------------------------------------
# 统计分析辅助
# ------------------------------------------------------------------

def compute_label_distribution(items: List[Dict]) -> Dict[str, int]:
    """计算每个主题的文档数量。"""
    from collections import Counter
    topics = [item["topic"] for item in items]
    return dict(Counter(topics))


def get_label_array(items: List[Dict]) -> np.ndarray:
    """获取所有文档的标签数组（用于 sklearn.metrics）。"""
    return np.array([item["label"] for item in items])


def get_topic_array(items: List[Dict]) -> np.ndarray:
    """获取所有文档的粗粒度主题数组。"""
    topic_to_id = {
        "computing": 0,
        "recreation": 1,
        "science": 2,
        "politics": 3,
        "religion": 4,
        "commerce": 5,
    }
    return np.array([topic_to_id.get(item["topic"], -1) for item in items])


def get_embeddings_matrix(items: List[Dict]) -> np.ndarray:
    """从 items 中提取 embeddings 矩阵。"""
    return np.array([item["embedding"] for item in items])


# ------------------------------------------------------------------
# 主入口
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("20 Newsgroups Data Loader - Experiment 1")
    print("=" * 60)

    # 加载训练集
    items, embeddings = load_or_fetch_corpus(subset="train")

    # 统计分析
    print("\n[Dataset Summary]")
    print(f"  Total documents: {len(items)}")
    print(f"  Embedding dim: {embeddings.shape[1]}")

    label_dist = compute_label_distribution(items)
    print(f"  Topics: {len(label_dist)}")
    for topic, count in sorted(label_dist.items(), key=lambda x: -x[1]):
        print(f"    {topic}: {count}")

    # 保存完整数据
    out_path = Path(__file__).parent / "results" / "exp1_corpus.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # JSON 序列化（embedding 转 list）
    serializable = []
    for item in items:
        s = dict(item)
        s["embedding"] = item["embedding"].tolist()
        serializable.append(s)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "n_documents": len(items),
            "n_topics": len(label_dist),
            "topic_distribution": label_dist,
            "items": serializable,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nCorpus saved to: {out_path}")
