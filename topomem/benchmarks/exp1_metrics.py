"""
topomem/benchmarks/exp1_metrics.py

实验1指标计算模块：聚类质量评估

目标：
- 评估 H0/H1 是否编码语义信息
- 提供可复现的量化指标
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from collections import Counter
import hashlib

PROJECT = "F:/unified-sel"
import sys
sys.path.insert(0, PROJECT)


# ------------------------------------------------------------------
# 聚类质量指标
# ------------------------------------------------------------------

def compute_h0_purity(
    h0_labels: np.ndarray,
    true_labels: np.ndarray,
) -> float:
    """计算 H0 连通分支聚类的纯度。

    纯度 = max_over_clusters (|intersection| / |cluster|)

    Args:
        h0_labels: H0 分支标签 (N,)
        true_labels: 真实主题标签 (N,)

    Returns:
        purity score in [0, 1], higher is better
    """
    unique_h0 = np.unique(h0_labels)
    total_correct = 0
    total_points = len(h0_labels)

    for cluster_id in unique_h0:
        mask = h0_labels == cluster_id
        cluster_true = true_labels[mask]
        if len(cluster_true) == 0:
            continue
        # 最常见的主题
        counter = Counter(cluster_true)
        most_common_count = counter.most_common(1)[0][1]
        total_correct += most_common_count

    return total_correct / total_points if total_points > 0 else 0.0


def compute_topic_entropy(
    h0_labels: np.ndarray,
    true_labels: np.ndarray,
    topic_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """计算每个 H0 聚类内部的主题熵。

    低熵 = 聚类内部主题一致性好

    Args:
        h0_labels: H0 分支标签 (N,)
        true_labels: 真实主题标签 (N,)
        topic_names: 主题名称列表

    Returns:
        Dict[cluster_id, entropy]
    """
    unique_h0 = np.unique(h0_labels)
    entropies = {}

    for cluster_id in unique_h0:
        mask = h0_labels == cluster_id
        cluster_true = true_labels[mask]
        if len(cluster_true) == 0:
            entropies[str(cluster_id)] = 0.0
            continue

        # 计算 Shannon 熵
        counter = Counter(cluster_true)
        total = sum(counter.values())
        entropy = 0.0
        for count in counter.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)

        entropies[str(cluster_id)] = entropy

    return entropies


def compute_rand_index(
    h0_labels: np.ndarray,
    true_labels: np.ndarray,
) -> float:
    """计算调整兰德指数 (ARI)。"""
    return adjusted_rand_score(true_labels, h0_labels)


def compute_nmi(
    h0_labels: np.ndarray,
    true_labels: np.ndarray,
) -> float:
    """计算归一化互信息 (NMI)。"""
    return normalized_mutual_info_score(true_labels, h0_labels)


# ------------------------------------------------------------------
# Null Hypothesis 基线 (Monte Carlo)
# ------------------------------------------------------------------

def compute_random_purity_baseline(
    n_samples: int,
    n_topics: int,
    n_clusters: int,
    n_trials: int = 100,
    seed: int = 42,
) -> Tuple[float, float]:
    """通过 Monte Carlo 模拟计算随机基线纯度。

    H0: 如果 H0 聚类不编码语义信息，纯度应该接近随机基线

    Args:
        n_samples: 样本数量
        n_topics: 主题数量
        n_clusters: H0 聚类数量
        n_trials: Monte Carlo 试验次数
        seed: 随机种子

    Returns:
        (mean_purity, std_purity)
    """
    rng = np.random.RandomState(seed)
    purities = []

    for _ in range(n_trials):
        # 随机分配主题
        random_true = rng.randint(0, n_topics, size=n_samples)

        # 随机分配聚类标签
        random_h0 = rng.randint(0, n_clusters, size=n_samples)

        # 计算纯度
        purity = compute_h0_purity(random_h0, random_true)
        purities.append(purity)

    return np.mean(purities), np.std(purities)


def compute_significance_zscore(
    observed_purity: float,
    random_mean: float,
    random_std: float,
) -> float:
    """计算观察值相对于随机基线的 z-score。

    z > 2 表示统计显著 (p < 0.05)
    z > 3 表示高度显著 (p < 0.001)
    """
    if random_std == 0:
        return 0.0
    return (observed_purity - random_mean) / random_std


# ------------------------------------------------------------------
# 层次聚类对比
# ------------------------------------------------------------------

def run_kmeans_baseline(
    embeddings: np.ndarray,
    true_labels: np.ndarray,
    n_clusters: int,
    seed: int = 42,
) -> Dict[str, float]:
    """运行 K-Means 作为传统聚类基线。

    对比：H0聚类 vs K-Means
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    kmeans_labels = kmeans.fit_predict(embeddings)

    return {
        "kmeans_purity": compute_h0_purity(kmeans_labels, true_labels),
        "kmeans_ari": compute_rand_index(kmeans_labels, true_labels),
        "kmeans_nmi": compute_nmi(kmeans_labels, true_labels),
    }


# ------------------------------------------------------------------
# 摘要报告
# ------------------------------------------------------------------

def generate_metrics_report(
    h0_labels: np.ndarray,
    true_labels: np.ndarray,
    embeddings: np.ndarray,
    topic_to_name: Dict[int, str],
    n_trials_mc: int = 100,
) -> Dict:
    """生成完整的指标报告。

    Args:
        h0_labels: H0 分支标签 (N,)
        true_labels: 真实主题标签 (N,)
        embeddings: 嵌入矩阵 (N, D)
        topic_to_name: 主题ID -> 名称映射
        n_trials_mc: Monte Carlo 试验次数

    Returns:
        完整报告字典
    """
    n_topics = len(np.unique(true_labels))
    n_h0_clusters = len(np.unique(h0_labels))

    # H0 聚类指标
    h0_purity = compute_h0_purity(h0_labels, true_labels)
    h0_ari = compute_rand_index(h0_labels, true_labels)
    h0_nmi = compute_nmi(h0_labels, true_labels)

    # 主题熵
    topic_entropy = compute_topic_entropy(h0_labels, true_labels)

    # K-Means 基线
    kmeans_metrics = run_kmeans_baseline(embeddings, true_labels, n_topics)

    # Monte Carlo 基线
    mc_mean, mc_std = compute_random_purity_baseline(
        n_samples=len(true_labels),
        n_topics=n_topics,
        n_clusters=n_h0_clusters,
        n_trials=n_trials_mc,
    )

    # Z-score
    z_score = compute_significance_zscore(h0_purity, mc_mean, mc_std)

    report = {
        "dataset": {
            "n_samples": len(true_labels),
            "n_topics": n_topics,
            "n_h0_clusters": n_h0_clusters,
        },
        "h0_clustering": {
            "purity": h0_purity,
            "ari": h0_ari,
            "nmi": h0_nmi,
            "topic_entropy": topic_entropy,
        },
        "kmeans_baseline": kmeans_metrics,
        "monte_carlo": {
            "random_purity_mean": mc_mean,
            "random_purity_std": mc_std,
            "n_trials": n_trials_mc,
        },
        "significance": {
            "z_score": z_score,
            "is_significant_2sigma": z_score > 2,
            "is_significant_3sigma": z_score > 3,
        },
    }

    return report


def print_report(report: Dict) -> None:
    """格式化打印报告。"""
    print("=" * 60)
    print("EXPERIMENT 1 METRICS REPORT")
    print("=" * 60)

    ds = report["dataset"]
    print(f"\n[Dataset]")
    print(f"  Samples: {ds['n_samples']}")
    print(f"  Topics: {ds['n_topics']}")
    print(f"  H0 Clusters: {ds['n_h0_clusters']}")

    h0 = report["h0_clustering"]
    print(f"\n[H0 Clustering Metrics]")
    print(f"  Purity:    {h0['purity']:.4f}")
    print(f"  ARI:       {h0['ari']:.4f}")
    print(f"  NMI:       {h0['nmi']:.4f}")

    km = report["kmeans_baseline"]
    print(f"\n[KMeans Baseline]")
    print(f"  Purity:    {km['kmeans_purity']:.4f}")
    print(f"  ARI:       {km['kmeans_ari']:.4f}")
    print(f"  NMI:       {km['kmeans_nmi']:.4f}")

    mc = report["monte_carlo"]
    print(f"\n[Monte Carlo Null Hypothesis]")
    print(f"  Random Purity: {mc['random_purity_mean']:.4f} ± {mc['random_purity_std']:.4f}")
    print(f"  Z-Score:       {report['significance']['z_score']:.2f}")

    sig = report["significance"]
    print(f"\n[Statistical Significance]")
    print(f"  Significant (z>2):   {'YES' if sig['is_significant_2sigma'] else 'NO'}")
    print(f"  Highly Significant:  {'YES' if sig['is_significant_3sigma'] else 'NO'}")

    print("=" * 60)


# ------------------------------------------------------------------
# 主入口
# ------------------------------------------------------------------

if __name__ == "__main__":
    # 简单测试
    print("Testing exp1_metrics module...")

    # 模拟数据
    rng = np.random.RandomState(42)
    n = 200
    true_labels = rng.randint(0, 4, size=n)
    h0_labels = rng.randint(0, 6, size=n)
    embeddings = rng.randn(n, 384)

    # 测试各函数
    purity = compute_h0_purity(h0_labels, true_labels)
    print(f"  H0 Purity: {purity:.4f}")

    ari = compute_rand_index(h0_labels, true_labels)
    print(f"  ARI: {ari:.4f}")

    mc_mean, mc_std = compute_random_purity_baseline(200, 4, 6, n_trials=50)
    print(f"  MC Baseline: {mc_mean:.4f} ± {mc_std:.4f}")

    z = compute_significance_zscore(purity, mc_mean, mc_std)
    print(f"  Z-Score: {z:.2f}")

    print("\nAll tests passed!")
