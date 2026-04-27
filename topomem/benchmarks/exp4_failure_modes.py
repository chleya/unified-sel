"""
topomem/benchmarks/exp4_failure_modes.py

实验4：失败模式分析

核心问题：
- TopoMem 在什么情况下会失败？
- H0/H1 结构何时失效？
- 边界条件和退化场景是什么？

实验设计：
1. 干扰敏感性：添加噪声/混淆文档时的鲁棒性
2. 尺度敏感性：不同数据规模下的表现
3. 主题重叠：当主题边界模糊时的表现
4. 稀疏性：数据稀疏时的 TDA 失效分析
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.datasets import make_blobs

PROJECT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT))

from topomem.embedding import EmbeddingManager, EmbeddingConfig
from topomem.topology import TopologyEngine, TopologyConfig

sys.path.insert(0, str(Path(__file__).parent))
from exp1_data_loader import load_or_fetch_corpus, get_topic_array
from exp1_metrics import compute_h0_purity, compute_rand_index, compute_nmi


# ------------------------------------------------------------------
# 干扰敏感性测试
# ------------------------------------------------------------------

def test_noise_sensitivity(
    embeddings: np.ndarray,
    true_labels: np.ndarray,
    noise_levels: List[float] = [0.0, 0.05, 0.1, 0.2, 0.5],
    n_clusters: int = 6,
    seed: int = 42,
) -> Dict:
    """测试噪声敏感性。

    向 embedding 添加高斯噪声，观察性能退化。
    """
    print("\n  [Test 1] Noise Sensitivity...")

    results = []

    for noise in noise_levels:
        rng = np.random.RandomState(seed)
        if noise > 0:
            # 添加噪声
            emb_std = np.std(embeddings)
            noise_matrix = rng.randn(*embeddings.shape) * emb_std * noise
            noisy_emb = embeddings + noise_matrix
        else:
            noisy_emb = embeddings

        # K-Means 聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        labels = kmeans.fit_predict(noisy_emb)

        ari = compute_rand_index(labels, true_labels)
        purity = compute_h0_purity(labels, true_labels)

        results.append({
            "noise_level": noise,
            "ari": float(ari),
            "purity": float(purity),
        })

        print(f"    Noise={noise:.2f}: ARI={ari:.4f}, Purity={purity:.4f}")

    # 计算退化率
    baseline_ari = results[0]["ari"]
    for r in results[1:]:
        r["ari_degradation"] = r["ari"] - baseline_ari

    return {
        "test": "noise_sensitivity",
        "baseline_ari": float(baseline_ari),
        "results": results,
    }


# ------------------------------------------------------------------
# 尺度敏感性测试
# ------------------------------------------------------------------

def test_scale_sensitivity(
    embeddings: np.ndarray,
    true_labels: np.ndarray,
    sample_sizes: List[int] = [500, 1000, 2000, 5000],
    n_clusters: int = 6,
    seed: int = 42,
) -> Dict:
    """测试数据规模敏感性。

    使用不同大小的子样本，观察性能随规模的变化。
    """
    print("\n  [Test 2] Scale Sensitivity...")

    results = []
    baseline_n_clusters = len(np.unique(true_labels))

    for size in sample_sizes:
        if size > len(embeddings):
            continue

        rng = np.random.RandomState(seed)
        indices = rng.choice(len(embeddings), size=size, replace=False)
        sample_emb = embeddings[indices]
        sample_labels = true_labels[indices]

        # K-Means
        n_c = min(n_clusters, len(np.unique(sample_labels)))
        kmeans = KMeans(n_clusters=n_c, random_state=seed, n_init=10)
        labels = kmeans.fit_predict(sample_emb)

        ari = compute_rand_index(labels, sample_labels)

        # 计算 H0 统计
        topo_config = TopologyConfig(max_homology_dim=1)
        topo_engine = TopologyEngine(topo_config)
        try:
            diagrams = topo_engine.compute_persistence(sample_emb)
            n_h0 = len(diagrams[0]) if len(diagrams) > 0 else 0
            n_h1 = len(diagrams[1]) if len(diagrams) > 1 else 0
        except:
            n_h0, n_h1 = 0, 0

        results.append({
            "sample_size": size,
            "ari": float(ari),
            "n_h0_features": n_h0,
            "n_h1_features": n_h1,
        })

        print(f"    Size={size}: ARI={ari:.4f}, H0={n_h0}, H1={n_h1}")

    return {
        "test": "scale_sensitivity",
        "results": results,
    }


# ------------------------------------------------------------------
# 主题重叠测试
# ------------------------------------------------------------------

def test_topic_overlap(
    embeddings: np.ndarray,
    true_labels: np.ndarray,
    topic_names: Dict[int, str],
    overlap_ratios: List[float] = [0.0, 0.1, 0.2, 0.3, 0.5],
    n_clusters: int = 6,
    seed: int = 42,
) -> Dict:
    """测试主题重叠时的表现。

    通过混合不同主题的 embedding 来模拟主题重叠。
    """
    print("\n  [Test 3] Topic Overlap...")

    results = []
    rng = np.random.RandomState(seed)

    for overlap in overlap_ratios:
        # 创建"混合"标签：通过打乱部分标签模拟重叠
        mixed_labels = true_labels.copy()
        n_mix = int(len(mixed_labels) * overlap)
        mix_indices = rng.choice(len(mixed_labels), size=n_mix, replace=False)

        # 打乱这些位置的标签
        shuffled = true_labels[mix_indices]
        rng.shuffle(shuffled)
        mixed_labels[mix_indices] = shuffled

        # K-Means on original embeddings
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        # 对比：聚类结果 vs 原始标签 vs 混合标签
        ari_original = compute_rand_index(labels, true_labels)
        ari_mixed = compute_rand_index(labels, mixed_labels)

        results.append({
            "overlap_ratio": overlap,
            "ari_vs_original": float(ari_original),
            "ari_vs_mixed": float(ari_mixed),
            "degradation": float(ari_original - ari_mixed),
        })

        print(f"    Overlap={overlap:.1f}: ARI(original)={ari_original:.4f}, ARI(mixed)={ari_mixed:.4f}")

    return {
        "test": "topic_overlap",
        "results": results,
    }


# ------------------------------------------------------------------
# 稀疏性测试
# ------------------------------------------------------------------

def test_sparsity(
    sample_size: int = 500,
    n_clusters: int = 6,
    sparsity_levels: List[float] = [0.0, 0.5, 0.7, 0.9, 0.95],
    seed: int = 42,
) -> Dict:
    """测试稀疏数据下的 TDA 失效。

    使用 make_blobs 生成不同稀疏度的数据。
    """
    print("\n  [Test 4] Sparsity (Simulated)...")

    results = []
    rng = np.random.RandomState(seed)

    for sparsity in sparsity_levels:
        # 生成标准 blobs
        X, y = make_blobs(
            n_samples=sample_size,
            centers=n_clusters,
            n_features=384,
            cluster_std=1.0,
            random_state=seed,
        )

        # 随机置零模拟稀疏性
        if sparsity > 0:
            mask = rng.rand(*X.shape) < sparsity
            X_sparse = X.copy()
            X_sparse[mask] = 0
        else:
            X_sparse = X

        # K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        labels = kmeans.fit_predict(X_sparse)
        ari = adjusted_rand_score(y, labels)

        # TDA
        topo_config = TopologyConfig(max_homology_dim=1)
        topo_engine = TopologyEngine(topo_config)
        try:
            diagrams = topo_engine.compute_persistence(X_sparse)
            n_h0 = len(diagrams[0]) if len(diagrams) > 0 else 0
            n_h1 = len(diagrams[1]) if len(diagrams) > 1 else 0
            topo_success = True
        except Exception as e:
            n_h0, n_h1 = 0, 0
            topo_success = False

        results.append({
            "sparsity": sparsity,
            "ari": float(ari),
            "n_h0": n_h0,
            "n_h1": n_h1,
            "topo_success": topo_success,
        })

        print(f"    Sparsity={sparsity:.2f}: ARI={ari:.4f}, H0={n_h0}, H1={n_h1}, TDA={'OK' if topo_success else 'FAIL'}")

    return {
        "test": "sparsity",
        "sample_size": sample_size,
        "results": results,
    }


# ------------------------------------------------------------------
# 边界情况测试
# ------------------------------------------------------------------

def test_edge_cases(
    seed: int = 42,
) -> Dict:
    """测试边界情况。

    包括：极小数据、极高维、重复点等。
    """
    print("\n  [Test 5] Edge Cases...")

    results = []
    rng = np.random.RandomState(seed)

    # Test 1: 极小数据 (n < 5)
    print("    Testing n=5 (minimum)...")
    try:
        X_mini, y_mini = make_blobs(n_samples=5, centers=2, n_features=384, random_state=seed)
        topo_config = TopologyConfig(max_homology_dim=1)
        topo_engine = TopologyEngine(topo_config)
        diagrams = topo_engine.compute_persistence(X_mini)
        result = {"case": "n=5", "topo_success": True, "n_h0": len(diagrams[0]) if len(diagrams) > 0 else 0}
    except Exception as e:
        result = {"case": "n=5", "topo_success": False, "error": str(e)}
    results.append(result)
    print(f"      Result: {'OK' if result['topo_success'] else 'FAIL'}")

    # Test 2: 重复点
    print("    Testing duplicate points...")
    try:
        X_dup = np.tile(rng.randn(5, 384), (2, 1))  # 5个点各重复2次
        y_dup = np.array([0] * 5 + [1] * 5)
        topo_engine = TopologyEngine(TopologyConfig(max_homology_dim=1))
        diagrams = topo_engine.compute_persistence(X_dup)
        result = {"case": "duplicates", "topo_success": True, "n_h0": len(diagrams[0]) if len(diagrams) > 0 else 0}
    except Exception as e:
        result = {"case": "duplicates", "topo_success": False, "error": str(e)}
    results.append(result)
    print(f"      Result: {'OK' if result['topo_success'] else 'FAIL'}")

    # Test 3: 极高维 (D >> N)
    print("    Testing D=10000, N=50...")
    try:
        X_high, y_high = make_blobs(n_samples=50, centers=3, n_features=10000, random_state=seed)
        topo_engine = TopologyEngine(TopologyConfig(max_homology_dim=1))
        diagrams = topo_engine.compute_persistence(X_high)
        result = {"case": "high_dim", "topo_success": True, "n_h0": len(diagrams[0]) if len(diagrams) > 0 else 0}
    except Exception as e:
        result = {"case": "high_dim", "topo_success": False, "error": str(e)}
    results.append(result)
    print(f"      Result: {'OK' if result['topo_success'] else 'FAIL'}")

    return {
        "test": "edge_cases",
        "results": results,
    }


# ------------------------------------------------------------------
# 主实验
# ------------------------------------------------------------------

def run_failure_mode_analysis(
    categories: Optional[List[str]] = None,
    use_cache: bool = True,
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> Dict:
    """运行失败模式分析。"""

    print("=" * 70)
    print("EXPERIMENT 4: Failure Mode Analysis")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Step 1: 加载数据
    print("\n[1/5] Loading corpus...")
    start_total = time.time()

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

    print(f"  Corpus: {len(items)} documents")
    print(f"  Topics: {len(np.unique(true_labels))}")

    # Step 2: 运行各项测试
    print("\n[2/5] Running failure mode tests...")

    results = {}

    # Test 1: Noise Sensitivity
    results["noise_sensitivity"] = test_noise_sensitivity(
        embeddings, true_labels,
        noise_levels=[0.0, 0.05, 0.1, 0.2, 0.5],
        seed=seed
    )

    # Test 2: Scale Sensitivity
    results["scale_sensitivity"] = test_scale_sensitivity(
        embeddings, true_labels,
        sample_sizes=[500, 1000, 2000, 5000],
        seed=seed
    )

    # Test 3: Topic Overlap
    results["topic_overlap"] = test_topic_overlap(
        embeddings, true_labels, topic_to_name,
        overlap_ratios=[0.0, 0.1, 0.2, 0.3, 0.5],
        seed=seed
    )

    # Test 4: Sparsity
    results["sparsity"] = test_sparsity(
        sample_size=500,
        sparsity_levels=[0.0, 0.5, 0.7, 0.9, 0.95],
        seed=seed
    )

    # Test 5: Edge Cases
    results["edge_cases"] = test_edge_cases(seed=seed)

    # Step 3: 保存结果
    elapsed_total = time.time() - start_total

    print(f"\n[4/4] Saving results... (total time: {elapsed_total:.1f}s)")

    if output_dir is None:
        output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time())
    outpath = output_dir / f"exp4_failure_modes_{timestamp}.json"

    output_data = {
        "experiment": "Failure Mode Analysis",
        "timestamp": timestamp,
        "datetime": datetime.now().isoformat(),
        "config": {
            "n_documents": len(items),
            "n_topics": len(np.unique(true_labels)),
            "seed": seed,
        },
        "results": results,
        "total_elapsed_sec": elapsed_total,
    }

    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved: {outpath}")

    # 打印摘要
    print_summary(results)

    return output_data


def print_summary(results: Dict) -> None:
    """打印结果摘要。"""
    print("\n" + "=" * 70)
    print("FAILURE MODE SUMMARY")
    print("=" * 70)

    # Noise Sensitivity
    print("\n[1. Noise Sensitivity]")
    ns = results["noise_sensitivity"]["results"]
    print(f"  Baseline ARI (no noise): {ns[0]['ari']:.4f}")
    for r in ns[1:]:
        deg = r.get("ari_degradation", 0)
        print(f"  Noise={r['noise_level']:.2f}: ARI={r['ari']:.4f} (degradation: {deg:+.4f})")

    # Scale Sensitivity
    print("\n[2. Scale Sensitivity]")
    ss = results["scale_sensitivity"]["results"]
    for r in ss:
        print(f"  Size={r['sample_size']}: ARI={r['ari']:.4f}, H1 cycles={r['n_h1_features']}")

    # Topic Overlap
    print("\n[3. Topic Overlap]")
    to_ = results["topic_overlap"]["results"]
    print(f"  Baseline (no overlap): ARI={to_[0]['ari_vs_original']:.4f}")
    for r in to_[1:]:
        print(f"  Overlap={r['overlap_ratio']:.1f}: degradation={r['degradation']:+.4f}")

    # Sparsity
    print("\n[4. Sparsity]")
    sp = results["sparsity"]["results"]
    for r in sp:
        status = "OK" if r["topo_success"] else "FAIL"
        print(f"  Sparsity={r['sparsity']:.2f}: ARI={r['ari']:.4f}, TDA={status}")

    # Edge Cases
    print("\n[5. Edge Cases]")
    ec = results["edge_cases"]["results"]
    for r in ec:
        status = "OK" if r.get("topo_success", False) else "FAIL"
        print(f"  {r['case']}: TDA={status}")

    # Critical Failures
    print("\n[Critical Failures]")
    failures = []
    for test_name, test_results in results.items():
        if test_name == "edge_cases":
            for r in test_results.get("results", []):
                if not r.get("topo_success", True):
                    failures.append(r["case"])
        elif test_name == "sparsity":
            for r in test_results.get("results", []):
                if not r.get("topo_success", True):
                    failures.append(f"sparsity={r['sparsity']}")

    if failures:
        print(f"  Detected: {', '.join(failures)}")
    else:
        print("  None detected - system is robust to standard conditions")

    print("=" * 70)


# ------------------------------------------------------------------
# 主入口
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 4: Failure Mode Analysis")
    parser.add_argument("--no-cache", action="store_true", help="Disable cache")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_failure_mode_analysis(
        use_cache=not args.no_cache,
        seed=args.seed,
    )
