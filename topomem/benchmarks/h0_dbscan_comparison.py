"""
benchmarks/h0_dbscan_comparison.py

P0 实验：H0 聚类修复验证

目标：对比 DBSCAN 预聚类 vs single-linkage H0 的聚类质量
指标：ARI（Adjusted Rand Index）vs 20 Newsgroups 真实标签

预期：DBSCAN 应该产生有意义的簇（ARI > 0.1），而 single-linkage 产生微簇（ARI ≈ 0）
"""

import sys, io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT_DIR))
sys.path.insert(0, str(_SCRIPT_DIR))

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from topomem.embedding import EmbeddingManager
from topomem.topology import TopologyEngine
from topomem.config import TopologyConfig

from exp1_data_loader import load_20newsgroups


def normalize(vecs):
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
    return vecs / norms


def evaluate_clustering(labels, true_labels, method_name):
    """评估聚类质量。"""
    # 过滤噪声点（-1）
    mask = labels != -1
    if mask.sum() < len(labels) * 0.5:
        print(f"  {method_name}: Too many noise points ({mask.sum()}/{len(labels)}), skipping ARI")
        return None

    # 只在有效点上计算
    valid_labels = labels[mask]
    valid_true = np.array(true_labels)[mask]

    ari = adjusted_rand_score(valid_true, valid_labels)
    nmi = normalized_mutual_info_score(valid_true, valid_labels)
    n_clusters = len(set(valid_labels))
    n_noise = (labels == -1).sum()

    print(f"  {method_name}:")
    print(f"    ARI = {ari:.4f}")
    print(f"    NMI = {nmi:.4f}")
    print(f"    clusters = {n_clusters} (+ {n_noise} noise)")
    print(f"    noise% = {n_noise/len(labels)*100:.1f}%")
    return {"ari": ari, "nmi": nmi, "n_clusters": n_clusters, "n_noise": n_noise}


def main():
    print("="*70)
    print("P0: H0 Clustering Fix Validation")
    print("="*70)

    # 加载数据
    print("\nLoading 20 Newsgroups (subsample for speed)...")
    all_items = load_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))

    # 每类取 20 篇，加速实验
    from collections import defaultdict
    by_domain = defaultdict(list)
    for item in all_items:
        by_domain[item["label_name"]].append(item)

    # 取 5 个不相关的类
    selected_domains = [
        "comp.sys.ibm.pc.hardware",
        "rec.sport.baseball",
        "sci.crypt",
        "rec.autos",
        "talk.politics.mideast",
    ]
    items = []
    true_labels = []
    for domain in selected_domains:
        for item in by_domain[domain][:20]:
            items.append(item)
            true_labels.append(domain)

    print(f"  Selected {len(items)} docs from {len(selected_domains)} domains")

    # 编码
    print("\nEncoding...")
    emb_mgr = EmbeddingManager()
    texts = [item["text"] for item in items]
    embeddings = normalize(np.array(emb_mgr.encode_batch(texts)))
    print(f"  Embeddings shape: {embeddings.shape}")

    # 编码真实标签为整数
    label_to_int = {d: i for i, d in enumerate(selected_domains)}
    true_label_ids = np.array([label_to_int[d] for d in true_labels])

    # ========== 方法 1: Single-linkage H0 ==========
    print("\n" + "="*50)
    print("Method 1: Single-linkage H0 (OLD)")
    print("="*50)
    topo_single = TopologyEngine(TopologyConfig(
        max_homology_dim=1,
        metric="cosine",
    ))
    diagram_single = topo_single.compute_persistence(embeddings)
    labels_h0 = topo_single.cluster_labels_from_h0(diagram_single, embeddings)
    evaluate_clustering(labels_h0, true_label_ids, "Single-linkage H0")

    # ========== 方法 2: DBSCAN ==========
    print("\n" + "="*50)
    print("Method 2: DBSCAN (NEW)")
    print("="*50)

    # 尝试不同 eps
    for eps in [0.3, 0.5, 0.7]:
        print(f"\n  --- eps={eps} ---")
        labels_dbscan = topo_single.cluster_labels_from_dbscan(
            embeddings, eps=eps, min_samples=3
        )
        evaluate_clustering(labels_dbscan, true_label_ids, f"DBSCAN eps={eps}")

    # ========== 方法 3: DBSCAN + H0 refinement ==========
    print("\n" + "="*50)
    print("Method 3: DBSCAN + H0 sub-clustering (HYBRID)")
    print("="*50)

    labels_dbscan = topo_single.cluster_labels_from_dbscan(embeddings, eps=0.5, min_samples=3)

    # 对每个 DBSCAN 簇内部做 H0 refinement
    hybrid_labels = np.zeros(len(embeddings), dtype=int) - 1
    next_label = 0
    unique_dbscan = set(labels_dbscan) - {-1}

    for dbscan_cid in unique_dbscan:
        mask = labels_dbscan == dbscan_cid
        cluster_embeddings = embeddings[mask]

        if len(cluster_embeddings) < 3:
            # 太小的簇不再细分
            hybrid_labels[mask] = next_label
            next_label += 1
        else:
            # 在簇内做 H0
            try:
                sub_diagram = topo_single.compute_persistence(cluster_embeddings)
                sub_labels = topo_single.cluster_labels_from_h0(
                    sub_diagram, cluster_embeddings, max_clusters=3
                )
                # 重新映射子簇标签
                for new_lbl in np.unique(sub_labels):
                    sub_mask = mask & (sub_labels == new_lbl)
                    hybrid_labels[sub_mask] = next_label
                    next_label += 1
            except Exception:
                hybrid_labels[mask] = next_label
                next_label += 1

    evaluate_clustering(hybrid_labels, true_label_ids, "DBSCAN+H0 hybrid")

    # ========== 分析：为什么 DBSCAN 的 ARI 不会很高 ==========
    print("\n" + "="*50)
    print("Analysis: Cluster size distribution")
    print("="*50)

    for name, labels in [
        ("Single-linkage H0", labels_h0),
        ("DBSCAN eps=0.5", labels_dbscan),
        ("DBSCAN+H0 hybrid", hybrid_labels),
    ]:
        unique, counts = np.unique(labels, return_counts=True)
        print(f"\n  {name}:")
        print(f"    Cluster sizes: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")
        if len(unique) <= 10:
            for lbl, cnt in zip(unique, counts):
                print(f"      Cluster {lbl}: {cnt} points")


if __name__ == "__main__":
    main()
