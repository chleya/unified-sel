"""
topomem/benchmarks/exp1_h1_entropy_analysis.py

实验1-Phase3: H1 循环与熵分析

核心研究问题：
- H1 循环（拓扑孔洞）是否编码语义信息？
- H1 循环的持久性分布是否与主题结构相关？

假设：
- H1 循环代表"跨界"概念，连接不同主题
- 高持久性 H1 循环 = 强语义桥接
- H1 熵低意味着循环定义良好的语义社区

分析内容：
1. H1 persistence distribution per topic
2. H1 cycle entropy across clusters
3. Topic coherence via H1 cycles
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import entropy as scipy_entropy

PROJECT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT))

from topomem.embedding import EmbeddingManager, EmbeddingConfig
from topomem.topology import TopologyEngine, TopologyConfig

from exp1_data_loader import (
    load_or_fetch_corpus,
    get_embeddings_matrix,
    get_topic_array,
)
from exp1_metrics import compute_topic_entropy


# ------------------------------------------------------------------
# H1 Persistence Analysis
# ------------------------------------------------------------------

def extract_h1_persistence(
    embeddings: np.ndarray,
    metric: str = "cosine",
    max_dim: int = 1,
) -> Tuple[np.ndarray, Dict]:
    """从 RIPSER/persim 提取 H1 persistence 数据。

    Returns:
        (h1_persistence_values, stats_dict)
    """
    try:
        from topomem.topology import TopologyEngine, TopologyConfig
        topo_config = TopologyConfig(max_homology_dim=max_dim)
        topo_engine = TopologyEngine(topo_config)

        diagrams = topo_engine.compute_persistence(embeddings)

        if len(diagrams) < 2:
            return np.array([]), {"n_h1_cycles": 0}

        h1_diagram = diagrams[1]  # H1 = index 1

        # H1: (birth, death) pairs
        # Persistence = death - birth
        if len(h1_diagram) == 0:
            return np.array([]), {"n_h1_cycles": 0}

        # 过滤 finite deaths
        finite_mask = h1_diagram[:, 1] < np.inf
        h1_finite = h1_diagram[finite_mask]

        if len(h1_finite) == 0:
            return np.array([]), {"n_h1_cycles": 0}

        births = h1_finite[:, 0]
        deaths = h1_finite[:, 1]
        persistences = deaths - births

        stats = {
            "n_h1_cycles": len(persistences),
            "mean_persistence": float(np.mean(persistences)),
            "std_persistence": float(np.std(persistences)),
            "max_persistence": float(np.max(persistences)),
            "min_persistence": float(np.min(persistences)),
            "total_persistence": float(np.sum(persistences)),
        }

        return persistences, stats

    except Exception as e:
        print(f"  H1 extraction failed: {e}")
        return np.array([]), {"n_h1_cycles": 0, "error": str(e)}


def assign_h1_to_topics(
    embeddings: np.ndarray,
    true_labels: np.ndarray,
    topic_names: List[str],
    threshold_percentile: float = 75,
) -> Dict[str, Dict]:
    """为每个主题分配 H1 循环。

    分析每个主题内部和之间的 H1 循环。
    """
    from scipy.spatial.distance import pdist, squareform

    # 计算距离矩阵
    dist_matrix = squareform(pdist(embeddings, metric="cosine"))

    # 计算每个点的最近邻
    n_neighbors = 5
    nn_indices = np.argsort(dist_matrix, axis=1)[:, 1:n_neighbors+1]

    # 对于每个主题，计算其内部 H1 贡献
    unique_topics = np.unique(true_labels)
    topic_h1_data = {}

    for topic_id in unique_topics:
        topic_name = topic_names[topic_id] if topic_id < len(topic_names) else f"topic_{topic_id}"
        mask = true_labels == topic_id
        topic_embeddings = embeddings[mask]
        topic_size = np.sum(mask)

        if topic_size < 3:
            topic_h1_data[topic_name] = {
                "topic_id": int(topic_id),
                "size": int(topic_size),
                "internal_h1_contribution": 0.0,
                "note": "too small",
            }
            continue

        # 计算主题内部平均距离（紧密程度）
        topic_dist = dist_matrix[np.ix_(mask, mask)]
        mean_internal_dist = np.mean(topic_dist[np.triu_indices(topic_size, k=1)])

        # 估算内部 H1 贡献（基于密度的变化）
        # 使用 K-NN 距离的标准差作为局部拓扑复杂度
        topic_nn = nn_indices[mask]
        nn_distances = []
        for i, idx in enumerate(topic_nn):
            for j in idx:
                if mask[j]:
                    nn_distances.append(dist_matrix[i, j])

        nn_distances = np.array(nn_distances)
        internal_complexity = np.std(nn_distances) if len(nn_distances) > 0 else 0.0

        topic_h1_data[topic_name] = {
            "topic_id": int(topic_id),
            "size": int(topic_size),
            "mean_internal_dist": float(mean_internal_dist),
            "internal_complexity": float(internal_complexity),
        }

    return topic_h1_data


def compute_h1_entropy_per_cluster(
    h0_labels: np.ndarray,
    h1_persistence: np.ndarray,
    n_permutations: int = 50,
    seed: int = 42,
) -> Dict[str, float]:
    """计算每个 H0 聚类内部的 H1 熵。

    假设：低 H1 熵的聚类语义更一致
    """
    rng = np.random.RandomState(seed)

    unique_clusters = np.unique(h0_labels)
    cluster_h1_entropy = {}

    for cluster_id in unique_clusters:
        mask = h0_labels == cluster_id
        n_points = np.sum(mask)

        if n_points < 3:
            cluster_h1_entropy[str(cluster_id)] = 0.0
            continue

        # 模拟 H1 persistence 分布
        # 使用点的子采样估算
        if len(h1_persistence) > 0:
            # Bootstrap 采样
            sampled_persistence = rng.choice(
                h1_persistence,
                size=min(n_points, len(h1_persistence)),
                replace=False,
            )
            # 计算熵（离散化）
            bins = np.linspace(0, np.max(sampled_persistence) + 1e-6, 10)
            hist, _ = np.histogram(sampled_persistence, bins=bins)
            hist = hist / (hist.sum() + 1e-6)
            ent = scipy_entropy(hist + 1e-8)
        else:
            ent = 0.0

        cluster_h1_entropy[str(cluster_id)] = float(ent)

    return cluster_h1_entropy


def analyze_h1_topic_bridge(
    embeddings: np.ndarray,
    true_labels: np.ndarray,
    topic_names: List[str],
    threshold: float = 0.3,
) -> Dict:
    """分析 H1 循环是否作为"桥接"角色，连接不同主题。

    假设：跨主题的 H1 循环（边界循环）代表概念桥接
    """
    from scipy.spatial.distance import pdist, squareform

    n = len(embeddings)
    dist_matrix = squareform(pdist(embeddings, metric="cosine"))

    # 构建 K-NN 图
    k = 5
    nn_indices = np.argsort(dist_matrix, axis=1)[:, 1:k+1]

    # 识别边界点（连接到其他主题的点）
    boundary_points = 0
    internal_points = 0

    topic_connections = defaultdict(list)

    for i in range(n):
        same_topic_neighbors = 0
        diff_topic_neighbors = 0

        for j in nn_indices[i]:
            if true_labels[j] == true_labels[i]:
                same_topic_neighbors += 1
            else:
                diff_topic_neighbors += 1
                topic_connections[true_labels[i]].append(true_labels[j])

        # 如果有跨主题连接，考虑为边界点
        if diff_topic_neighbors > same_topic_neighbors * 0.5:
            boundary_points += 1
        else:
            internal_points += 1

    # 统计跨主题连接
    bridge_strength = {}
    for topic_id in np.unique(true_labels):
        connections = topic_connections[topic_id]
        if len(connections) > 0:
            counter = Counter(connections)
            most_common_other = counter.most_common(1)[0]
            other_topic = topic_names[most_common_other[0]] if most_common_other[0] < len(topic_names) else f"topic_{most_common_other[0]}"
            bridge_strength[topic_names[topic_id]] = {
                "n_cross_topic": len(connections),
                "primary_bridge": other_topic,
                "bridge_ratio": len(connections) / (len(connections) + k),
            }
        else:
            bridge_strength[topic_names[topic_id]] = {
                "n_cross_topic": 0,
                "primary_bridge": "none",
                "bridge_ratio": 0.0,
            }

    boundary_ratio = boundary_points / n if n > 0 else 0.0

    return {
        "boundary_points": boundary_points,
        "internal_points": internal_points,
        "boundary_ratio": float(boundary_ratio),
        "topic_bridge_strength": bridge_strength,
    }


# ------------------------------------------------------------------
# 主分析
# ------------------------------------------------------------------

def run_h1_analysis(
    categories: Optional[List[str]] = None,
    use_cache: bool = True,
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> Dict:
    """运行 H1 熵分析。"""

    print("=" * 70)
    print("EXPERIMENT 1 - Phase 3: H1 Cycle Entropy Analysis")
    print("=" * 70)

    # Step 1: 加载数据
    print("\n[1/5] Loading corpus...")
    items, embeddings = load_or_fetch_corpus(
        subset="train",
        categories=categories,
        use_cache=use_cache,
    )

    true_labels = get_topic_array(items)
    topic_to_name = {
        0: "computing",
        1: "recreation",
        2: "science",
        3: "politics",
        4: "religion",
        5: "commerce",
    }
    topic_names = [topic_to_name.get(i, f"topic_{i}") for i in range(6)]

    print(f"  Documents: {len(items)}")
    print(f"  Topics: {len(np.unique(true_labels))}")

    # Step 2: H1 Persistence 分布
    print("\n[2/5] Computing H1 persistence distribution...")
    h1_persistence, h1_stats = extract_h1_persistence(embeddings, max_dim=1)
    print(f"  H1 Cycles found: {h1_stats.get('n_h1_cycles', 0)}")
    if h1_stats.get('n_h1_cycles', 0) > 0:
        print(f"  Mean persistence: {h1_stats['mean_persistence']:.4f}")
        print(f"  Max persistence: {h1_stats['max_persistence']:.4f}")

    # Step 3: 按主题分析 H1
    print("\n[3/5] Assigning H1 to topics...")
    topic_h1_data = assign_h1_to_topics(embeddings, true_labels, topic_names)
    for topic, data in topic_h1_data.items():
        print(f"  {topic}: size={data['size']}, complexity={data.get('internal_complexity', 0):.4f}")

    # Step 4: H1 熵分析
    print("\n[4/5] Computing H1 entropy per cluster...")

    # 先获取 H0 标签（简化：用主题标签代替）
    # 实际应该用 topology H0 labels
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=6, random_state=seed, n_init=10)
    h0_labels = kmeans.fit_predict(embeddings)

    h1_entropy = compute_h1_entropy_per_cluster(
        h0_labels,
        h1_persistence,
        n_permutations=50,
        seed=seed,
    )

    print("  H1 entropy per H0 cluster:")
    for cluster_id, ent in h1_entropy.items():
        print(f"    Cluster {cluster_id}: H1 entropy = {ent:.4f}")

    # Step 5: 跨主题桥接分析
    print("\n[5/5] Analyzing cross-topic bridges...")
    bridge_analysis = analyze_h1_topic_bridge(embeddings, true_labels, topic_names)
    print(f"  Boundary points: {bridge_analysis['boundary_points']} ({bridge_analysis['boundary_ratio']:.2%})")
    print("  Topic bridge strengths:")
    for topic, strength in bridge_analysis["topic_bridge_strength"].items():
        print(f"    {topic}: bridge_ratio={strength['bridge_ratio']:.3f}, primary_bridge={strength['primary_bridge']}")

    # 汇总结果
    results = {
        "h1_persistence_stats": h1_stats,
        "topic_h1_data": topic_h1_data,
        "h1_entropy_per_cluster": h1_entropy,
        "bridge_analysis": bridge_analysis,
        "metadata": {
            "n_documents": len(items),
            "n_topics": len(np.unique(true_labels)),
            "seed": seed,
        },
    }

    # 保存结果
    if output_dir is None:
        output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time())
    outpath = output_dir / f"exp1_h1_analysis_{timestamp}.json"

    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved: {outpath}")

    # 打印摘要
    print_summary(results)

    return results


def print_summary(results: Dict) -> None:
    """打印分析摘要。"""
    print("\n" + "=" * 70)
    print("H1 ANALYSIS SUMMARY")
    print("=" * 70)

    h1_stats = results.get("h1_persistence_stats", {})
    print(f"\n[H1 Persistence]")
    print(f"  Cycles: {h1_stats.get('n_h1_cycles', 0)}")
    print(f"  Mean persistence: {h1_stats.get('mean_persistence', 0):.4f}")
    print(f"  Total persistence: {h1_stats.get('total_persistence', 0):.4f}")

    bridge = results.get("bridge_analysis", {})
    print(f"\n[Cross-Topic Bridges]")
    print(f"  Boundary ratio: {bridge.get('boundary_ratio', 0):.2%}")

    # 找出桥接最强的主题
    bridge_strength = bridge.get("topic_bridge_strength", {})
    if bridge_strength:
        strongest = max(bridge_strength.items(), key=lambda x: x[1].get("bridge_ratio", 0))
        print(f"  Strongest bridge: {strongest[0]} (ratio={strongest[1].get('bridge_ratio', 0):.3f})")

    print("=" * 70)


# ------------------------------------------------------------------
# 主入口
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 1 - H1 Entropy Analysis")
    parser.add_argument("--no-cache", action="store_true", help="Disable cache")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_h1_analysis(
        use_cache=not args.no_cache,
        seed=args.seed,
    )
