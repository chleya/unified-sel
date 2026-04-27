"""
实验：聚类方法有效性对比

比较三种聚类方法：
1. H0 single-linkage（原始方法）
2. DBSCAN 密度聚类
3. Hybrid 混合聚类（新方法）

目标：证明混合聚类解决了 H0 单点簇问题，
同时保留了拓扑结构的优势。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from topomem.embedding import EmbeddingManager
from topomem.topology import TopologyEngine
from topomem.memory import MemoryGraph
from topomem.config import TopologyConfig, MemoryConfig


def generate_synthetic_data(n_domains: int = 3, n_per_domain: int = 20) -> tuple:
    """生成合成测试数据。"""
    from sklearn.datasets import make_blobs
    
    # 生成 3 个簇的嵌入
    embeddings, labels = make_blobs(
        n_samples=n_domains * n_per_domain,
        n_features=50,  # 降维表示
        centers=n_domains,
        random_state=42,
    )
    
    # 映射到 384 维
    rng = np.random.default_rng(42)
    projection = rng.normal(0, 1, (50, 384))
    full_embeddings = embeddings @ projection
    
    # 归一化
    norms = np.linalg.norm(full_embeddings, axis=1, keepdims=True)
    full_embeddings = full_embeddings / (norms + 1e-8)
    
    # 生成文本
    domain_names = ["programming", "physics", "geography", "biology", "history"][:n_domains]
    texts = [f"Sample from {domain_names[l]} domain #{i}" for i, l in enumerate(labels)]
    true_labels = [domain_names[l] for l in labels]
    
    # 保存到全局供后续使用
    global _synthetic_embeddings
    _synthetic_embeddings = full_embeddings
    
    return texts, true_labels


_synthetic_embeddings = None


def load_test_corpus():
    corpus_dir = PROJECT_ROOT / "topomem" / "data" / "test_corpus"
    all_texts = []
    all_labels = []
    
    for domain_file in corpus_dir.glob("*.json"):
        with open(domain_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        domain = domain_file.stem
        
        # 支持两种格式：列表或字典
        if isinstance(data, list):
            # 列表格式：每个元素是一个文本
            for item in data:
                if isinstance(item, dict):
                    text = item.get("text", item.get("content", ""))
                else:
                    text = str(item)
                if text.strip():
                    all_texts.append(text)
                    all_labels.append(domain)
        elif isinstance(data, dict):
            # 字典格式：可能有 "knowledge" 或其他键
            items = data.get("knowledge", data.get("items", []))
            for item in items:
                if isinstance(item, dict):
                    text = item.get("text", item.get("content", ""))
                    if text.strip():
                        all_texts.append(text)
                        all_labels.append(domain)
    
    return all_texts, all_labels


def evaluate_clustering(method_name: str, labels: np.ndarray, true_labels: list) -> dict:
    """评估聚类结果。"""
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score
    
    n_clusters = len(set(labels) - {-1})
    n_noise = np.sum(labels == -1)
    
    # NMI（忽略噪声点）
    non_noise_mask = labels >= 0
    if non_noise_mask.sum() > 0:
        nmi = normalized_mutual_info_score(
            [true_labels[i] for i in range(len(true_labels)) if non_noise_mask[i]],
            labels[non_noise_mask]
        )
    else:
        nmi = 0.0
    
    return {
        "method": method_name,
        "n_clusters": n_clusters,
        "n_noise": int(n_noise),
        "noise_ratio": float(n_noise / len(labels)),
        "nmi": float(nmi),
    }


def experiment_clustering_comparison():
    """主实验：对比三种聚类方法。"""
    print("=" * 60)
    print("TopoMem 聚类方法有效性实验")
    print("=" * 60)
    
    # 1. 加载数据 - 如果语料为空，使用合成数据
    print("\n[1/5] 加载数据...")
    texts, true_labels = load_test_corpus()
    
    if len(texts) < 10:
        print(f"  语料库仅 {len(texts)} 条，使用合成数据测试...")
        texts, true_labels = generate_synthetic_data(n_domains=3, n_per_domain=20)
    
    print(f"  共 {len(texts)} 条知识，{len(set(true_labels))} 个领域")
    print(f"  领域分布: {dict(zip(*np.unique(true_labels, return_counts=True)))}")
    
    # 2. 编码
    print("\n[2/5] 编码文本...")
    embedding_mgr = EmbeddingManager()
    embeddings = embedding_mgr.encode_batch(texts)
    print(f"  嵌入维度: {embeddings.shape}")
    
    # 3. 测试三种聚类方法
    print("\n[3/5] 测试聚类方法...")
    topo_engine = TopologyEngine()
    
    # 计算 PH 一次
    diagram = topo_engine.compute_persistence(embeddings)
    
    results = []
    
    # 方法 1: H0 single-linkage
    print("\n  测试 H0 single-linkage...")
    t0 = time.time()
    labels_h0 = topo_engine.cluster_labels_from_h0(diagram, embeddings)
    time_h0 = time.time() - t0
    result_h0 = evaluate_clustering("h0", labels_h0, true_labels)
    result_h0["time_ms"] = time_h0 * 1000
    results.append(result_h0)
    print(f"    簇数: {result_h0['n_clusters']}, NMI: {result_h0['nmi']:.3f}, 时间: {time_h0*1000:.1f}ms")
    
    # 方法 2: DBSCAN
    print("\n 测试 DBSCAN...")
    t0 = time.time()
    labels_dbscan = topo_engine.cluster_labels_from_dbscan(embeddings)
    time_dbscan = time.time() - t0
    result_dbscan = evaluate_clustering("dbscan", labels_dbscan, true_labels)
    result_dbscan["time_ms"] = time_dbscan * 1000
    results.append(result_dbscan)
    print(f"    簇数: {result_dbscan['n_clusters']}, NMI: {result_dbscan['nmi']:.3f}, 时间: {time_dbscan*1000:.1f}ms")
    
    # 方法 3: Hybrid
    print("\n 测试 Hybrid...")
    t0 = time.time()
    labels_hybrid = topo_engine.cluster_labels_hybrid(embeddings, diagram=diagram)
    time_hybrid = time.time() - t0
    result_hybrid = evaluate_clustering("hybrid", labels_hybrid, true_labels)
    result_hybrid["time_ms"] = time_hybrid * 1000
    results.append(result_hybrid)
    print(f"    簇数: {result_hybrid['n_clusters']}, NMI: {result_hybrid['nmi']:.3f}, 时间: {time_hybrid*1000:.1f}ms")
    
    # 4. 测试 MemoryGraph 使用不同聚类方法
    print("\n[4/5] 测试 MemoryGraph 聚类...")
    
    # 使用 Hybrid 的 MemoryGraph
    memory_hybrid = MemoryGraph()
    memory_hybrid._embedding_mgr = embedding_mgr
    
    for text in texts:
        memory_hybrid.add_memory_from_text(text, topo_engine=topo_engine)
    
    n_clusters_hybrid = len(memory_hybrid._get_all_cluster_ids())
    print(f"  MemoryGraph (Hybrid): {memory_hybrid.node_count()} 节点, {n_clusters_hybrid} 簇")
    
    # 5. 总结
    print("\n[5/5] 实验结果:")
    print("-" * 60)
    print(f"{'方法':<15} {'簇数':<6} {'噪声':<6} {'NMI':<8} {'时间/ms':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['method']:<15} {r['n_clusters']:<6} {r['n_noise']:<6} {r['nmi']:<8.3f} {r['time_ms']:<10.1f}")
    print("-" * 60)
    
    # 找出最佳方法
    best = max(results, key=lambda x: x['nmi'])
    print(f"\n✅ 最佳方法: {best['method']} (NMI={best['nmi']:.3f})")
    
    # 保存结果
    output = {
        "experiment": "clustering_comparison",
        "n_texts": len(texts),
        "n_domains": len(set(true_labels)),
        "results": results,
        "best_method": best['method'],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    results_dir = PROJECT_ROOT / "topomem" / "results" / "clustering_comparison"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"{time.strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_path}")
    return output


if __name__ == "__main__":
    experiment_clustering_comparison()
