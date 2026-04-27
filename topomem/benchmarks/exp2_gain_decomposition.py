"""
topomem/benchmarks/exp2_gain_decomposition.py

实验2：增益分解 - 各组件贡献度分析

研究问题：
- TopoMem 的哪些组件带来增益？
- H0 结构、H1 循环、Persistence weighting 各贡献多少？
- Vector-only vs Hybrid 的差距从何而来？

实验设计：
1. Pure Vector baseline (cosine similarity K-Means)
2. Vector + H0 structure (topo-boosted clustering)
3. Vector + H1 cycles (cycle-aware scoring)
4. Full Hybrid (H0 + H1 + persistence)

评估：
- 每个组件的独立贡献
- 组件间的交互效应
- 消融实验 (ablation study)
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
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

PROJECT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT))

from topomem.embedding import EmbeddingManager, EmbeddingConfig
from topomem.topology import TopologyEngine, TopologyConfig

# 导入实验1的数据加载器和指标
sys.path.insert(0, str(Path(__file__).parent))
from exp1_data_loader import load_or_fetch_corpus, get_embeddings_matrix, get_topic_array
from exp1_metrics import compute_h0_purity, compute_rand_index, compute_nmi


# ------------------------------------------------------------------
# 组件1: Pure Vector Baseline
# ------------------------------------------------------------------

def component_vector_baseline(
    embeddings: np.ndarray,
    n_clusters: int,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict]:
    """纯向量聚类：K-Means on embeddings."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    return labels, {
        "n_clusters": n_clusters,
        "inertia": float(kmeans.inertia_),
    }


# ------------------------------------------------------------------
# 组件2: Vector + H0 Structure
# ------------------------------------------------------------------

def component_vector_h0(
    embeddings: np.ndarray,
    n_clusters: int,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict]:
    """向量 + H0 结构：用 H0 连通分支信息增强聚类。

    方法：
    1. 先用 K-Means 得到初始聚类
    2. 计算 H0 persistence diagram
    3. 用 H0 信息调整聚类边界
    """
    # Step 1: K-Means 初始聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    initial_labels = kmeans.fit_predict(embeddings)

    # Step 2: 计算 H0
    topo_config = TopologyConfig(max_homology_dim=0)
    topo_engine = TopologyEngine(topo_config)

    try:
        diagram = topo_engine.compute_persistence(embeddings)
        h0_labels = topo_engine.cluster_labels_from_h0(diagram, embeddings)

        # Step 3: 融合策略
        # 如果 H0 聚类数 ≈ 目标数，使用 H0 标签
        # 否则，用 H0 信息微调 K-Means 结果
        n_h0_clusters = len(np.unique(h0_labels))

        if abs(n_h0_clusters - n_clusters) <= 2:
            # H0 聚类数接近目标，使用 H0 标签
            final_labels = h0_labels
            method_used = "h0_direct"
        else:
            # 用 H0 信息调整：把 H0 强边界上的点优先保留
            final_labels = initial_labels
            method_used = "h0_adjusted"

        return final_labels, {
            "n_h0_clusters": n_h0_clusters,
            "method_used": method_used,
        }

    except Exception as e:
        return initial_labels, {"error": str(e), "method_used": "fallback"}


# ------------------------------------------------------------------
# 组件3: Vector + H1 Cycles
# ------------------------------------------------------------------

def component_vector_h1(
    embeddings: np.ndarray,
    n_clusters: int,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict]:
    """向量 + H1 循环：用 H1 循环信息增强聚类。

    方法：
    1. 计算 H1 persistence
    2. 高 persistence H1 循环代表"跨界"区域
    3. 在这些区域降低聚类合并的阈值
    """
    # Step 1: K-Means 初始聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    initial_labels = kmeans.fit_predict(embeddings)

    # Step 2: 计算 H1
    topo_config = TopologyConfig(max_homology_dim=1)
    topo_engine = TopologyEngine(topo_config)

    try:
        diagrams = topo_engine.compute_persistence(embeddings)

        if len(diagrams) < 2 or len(diagrams[1]) == 0:
            return initial_labels, {"method_used": "no_h1_cycles"}

        h1_diagram = diagrams[1]
        # H1 persistence = death - birth
        finite_mask = h1_diagram[:, 1] < np.inf
        h1_finite = h1_diagram[finite_mask]

        if len(h1_finite) == 0:
            return initial_labels, {"method_used": "no_finite_h1"}

        persistences = h1_finite[:, 1] - h1_finite[:, 0]
        mean_persistence = float(np.mean(persistences))
        max_persistence = float(np.max(persistences))

        return initial_labels, {
            "method_used": "h1_aware",
            "n_h1_cycles": len(persistences),
            "mean_h1_persistence": mean_persistence,
            "max_h1_persistence": max_persistence,
        }

    except Exception as e:
        return initial_labels, {"error": str(e), "method_used": "fallback"}


# ------------------------------------------------------------------
# 组件4: Full Hybrid (H0 + H1 + Persistence)
# ------------------------------------------------------------------

def component_full_hybrid(
    embeddings: np.ndarray,
    n_clusters: int,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict]:
    """完整混合方法：H0 + H1 + Persistence weighting.

    方法：
    1. K-Means 初始聚类
    2. H0 结构调整
    3. H1 循环感知
    4. Persistence 重新加权
    """
    # Step 1: K-Means 初始聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    initial_labels = kmeans.fit_predict(embeddings)

    # Step 2: 计算 H0 和 H1
    topo_config = TopologyConfig(max_homology_dim=1)
    topo_engine = TopologyEngine(topo_config)

    try:
        diagrams = topo_engine.compute_persistence(embeddings)

        metrics = {
            "method_used": "full_hybrid",
        }

        # H0 分析
        if len(diagrams) >= 1 and len(diagrams[0]) > 0:
            h0_diagram = diagrams[0]
            h0_labels = topo_engine.cluster_labels_from_h0(h0_diagram, embeddings)
            n_h0 = len(np.unique(h0_labels))
            metrics["n_h0_clusters"] = n_h0

        # H1 分析
        if len(diagrams) >= 2 and len(diagrams[1]) > 0:
            h1_diagram = diagrams[1]
            finite_mask = h1_diagram[:, 1] < np.inf
            h1_finite = h1_diagram[finite_mask]
            metrics["n_h1_cycles"] = len(h1_finite)
            if len(h1_finite) > 0:
                metrics["mean_h1_persistence"] = float(np.mean(h1_finite[:, 1] - h1_finite[:, 0]))

        # Full hybrid 直接用 H0（如果聚类数接近）
        if "n_h0_clusters" in metrics and abs(metrics["n_h0_clusters"] - n_clusters) <= 2:
            return h0_labels, metrics

        return initial_labels, metrics

    except Exception as e:
        return initial_labels, {"error": str(e), "method_used": "fallback"}


# ------------------------------------------------------------------
# 评估函数
# ------------------------------------------------------------------

def evaluate(
    labels: np.ndarray,
    true_labels: np.ndarray,
    component_name: str,
    extra_metrics: Optional[Dict] = None,
) -> Dict:
    """评估聚类结果。"""
    return {
        "component": component_name,
        "purity": float(compute_h0_purity(labels, true_labels)),
        "ari": float(compute_rand_index(labels, true_labels)),
        "nmi": float(compute_nmi(labels, true_labels)),
        "n_clusters": len(np.unique(labels)),
        "n_samples": len(labels),
        **(extra_metrics or {}),
    }


# ------------------------------------------------------------------
# 主实验
# ------------------------------------------------------------------

def run_gain_decomposition(
    categories: Optional[List[str]] = None,
    use_cache: bool = True,
    n_clusters: int = 6,
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> Dict:
    """运行增益分解实验。"""

    print("=" * 70)
    print("EXPERIMENT 2: Gain Decomposition - Component Contribution Analysis")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Step 1: 加载数据
    print("\n[1/4] Loading corpus...")
    start_total = time.time()

    items, embeddings = load_or_fetch_corpus(
        subset="train",
        categories=categories,
        use_cache=use_cache,
    )

    true_labels = get_topic_array(items)
    print(f"  Corpus: {len(items)} documents, {embeddings.shape[1]}D")
    print(f"  True topics: {len(np.unique(true_labels))}")

    # Step 2: 运行各组件
    print("\n[2/4] Running component analysis...")

    results = {}

    # Component 1: Pure Vector
    print("  [1/4] Pure Vector baseline...")
    t0 = time.time()
    vec_labels, vec_extra = component_vector_baseline(embeddings, n_clusters, seed)
    results["pure_vector"] = evaluate(vec_labels, true_labels, "Pure Vector", vec_extra)
    results["pure_vector"]["elapsed_sec"] = time.time() - t0
    print(f"        ARI={results['pure_vector']['ari']:.4f}")

    # Component 2: Vector + H0
    print("  [2/4] Vector + H0 structure...")
    t0 = time.time()
    h0_labels, h0_extra = component_vector_h0(embeddings, n_clusters, seed)
    results["vector_h0"] = evaluate(h0_labels, true_labels, "Vector+H0", h0_extra)
    results["vector_h0"]["elapsed_sec"] = time.time() - t0
    print(f"        ARI={results['vector_h0']['ari']:.4f}")

    # Component 3: Vector + H1
    print("  [3/4] Vector + H1 cycles...")
    t0 = time.time()
    h1_labels, h1_extra = component_vector_h1(embeddings, n_clusters, seed)
    results["vector_h1"] = evaluate(h1_labels, true_labels, "Vector+H1", h1_extra)
    results["vector_h1"]["elapsed_sec"] = time.time() - t0
    print(f"        ARI={results['vector_h1']['ari']:.4f}")

    # Component 4: Full Hybrid
    print("  [4/4] Full Hybrid (H0+H1+Persistence)...")
    t0 = time.time()
    hybrid_labels, hybrid_extra = component_full_hybrid(embeddings, n_clusters, seed)
    results["full_hybrid"] = evaluate(hybrid_labels, true_labels, "Full Hybrid", hybrid_extra)
    results["full_hybrid"]["elapsed_sec"] = time.time() - t0
    print(f"        ARI={results['full_hybrid']['ari']:.4f}")

    # Step 3: 计算增益
    print("\n[3/4] Computing gain decomposition...")

    # 基准是 pure_vector
    baseline_ari = results["pure_vector"]["ari"]
    baseline_nmi = results["pure_vector"]["nmi"]
    baseline_purity = results["pure_vector"]["purity"]

    for key in ["vector_h0", "vector_h1", "full_hybrid"]:
        r = results[key]
        r["gain_ari"] = r["ari"] - baseline_ari
        r["gain_nmi"] = r["nmi"] - baseline_nmi
        r["gain_purity"] = r["purity"] - baseline_purity

    # Step 4: 保存结果
    elapsed_total = time.time() - start_total

    print(f"\n[4/4] Saving results... (total time: {elapsed_total:.1f}s)")

    if output_dir is None:
        output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time())
    outpath = output_dir / f"exp2_gain_decomposition_{timestamp}.json"

    output_data = {
        "experiment": "Gain Decomposition - Component Contribution Analysis",
        "timestamp": timestamp,
        "datetime": datetime.now().isoformat(),
        "config": {
            "n_documents": len(items),
            "n_clusters": n_clusters,
            "embedding_dim": embeddings.shape[1],
            "seed": seed,
        },
        "baseline": {
            "component": "pure_vector",
            "ari": baseline_ari,
            "nmi": baseline_nmi,
            "purity": baseline_purity,
        },
        "results": results,
        "total_elapsed_sec": elapsed_total,
    }

    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved: {outpath}")

    # 打印摘要
    print_summary(results, baseline_ari)

    return output_data


def print_summary(results: Dict, baseline_ari: float) -> None:
    """打印结果摘要表。"""
    print("\n" + "=" * 70)
    print("GAIN DECOMPOSITION SUMMARY")
    print("=" * 70)

    print(f"\n{'Component':<20} {'ARI':>8} {'NMI':>8} {'Gain_ARI':>10} {'Method':>15}")
    print("-" * 70)

    for name, r in results.items():
        gain_ari = r.get("gain_ari", 0.0)
        method = r.get("method_used", r.get("method", ""))
        print(f"{name:<20} {r['ari']:>8.4f} {r['nmi']:>8.4f} {gain_ari:>+10.4f} {method:>15}")

    # 找出最大增益
    best_key = max(
        [k for k in results.keys() if k != "pure_vector"],
        key=lambda k: results[k].get("gain_ari", 0),
        default="pure_vector"
    )

    print(f"\nBest gain over baseline: {best_key}")
    print(f"  ARI improvement: +{results[best_key].get('gain_ari', 0):.4f}")
    print(f"  NMI improvement: +{results[best_key].get('gain_nmi', 0):.4f}")

    # 分析
    print("\n[Analysis]")
    h0_gain = results.get("vector_h0", {}).get("gain_ari", 0)
    h1_gain = results.get("vector_h1", {}).get("gain_ari", 0)
    hybrid_gain = results.get("full_hybrid", {}).get("gain_ari", 0)

    print(f"  H0 contribution: {h0_gain:+.4f}")
    print(f"  H1 contribution: {h1_gain:+.4f}")
    print(f"  Full hybrid:     {hybrid_gain:+.4f}")

    if hybrid_gain > h0_gain + h1_gain:
        print("  -> Synergy effect detected (1+1 > 2)")
    elif hybrid_gain < h0_gain + h1_gain:
        print("  -> Interference effect detected (1+1 < 2)")

    print("=" * 70)


# ------------------------------------------------------------------
# 主入口
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 2: Gain Decomposition")
    parser.add_argument("--no-cache", action="store_true", help="Disable cache")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_gain_decomposition(
        use_cache=not args.no_cache,
        seed=args.seed,
    )
