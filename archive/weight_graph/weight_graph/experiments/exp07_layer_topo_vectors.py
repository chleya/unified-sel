"""
实验 7：逐层拓扑向量提取

对 Qwen2.5-0.5B 的每一层提取 ~10 维拓扑特征向量。
这是后续回归分析（Task 8）的前置条件。

预计耗时：24 层 × ~1 分钟 ≈ 30 分钟（单层图约 5K 节点，Louvain 很快）
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict
import pickle
import numpy as np

from weight_graph.config import GraphBuildConfig, AnalysisConfig
from weight_graph.graph_builder import GraphBuilder, WeightGraph
from weight_graph.analyzers import basic_stats, degree_distribution, detect_communities, compute_pagerank
from weight_graph.utils import ensure_dir, save_results


def extract_layer_topo_vector(
    layer_matrices: List,
    graph_config: GraphBuildConfig,
    analysis_config: AnalysisConfig,
) -> Dict:
    """
    对单层的所有 MLP 组件构建子图，计算拓扑向量。

    返回 ~10 维特征：
    {
        "modularity": float,
        "num_communities": int,
        "density": float,
        "avg_in_degree": float,
        "max_in_degree": int,
        "max_out_degree": int,
        "degree_std": float,
        "pagerank_entropy": float,
        "pagerank_gini": float,
        "reciprocity": float,
    }
    """
    builder = GraphBuilder(graph_config)

    # 合并该层所有组件（gate/up/down）为一个图
    merged = WeightGraph()
    for matrix in layer_matrices:
        sub = builder.build_single_layer(matrix)
        _merge_into(merged, sub)

    # 计算各项指标
    stats = basic_stats(merged)
    dd = degree_distribution(merged)
    comm = detect_communities(merged, analysis_config)
    pr = compute_pagerank(merged, analysis_config)

    # PageRank entropy
    pr_values = np.array(list(pr["scores"].values()))
    if pr_values.sum() > 0:
        pr_values = pr_values / pr_values.sum()
        pr_entropy = float(-np.sum(pr_values[pr_values > 0] * np.log2(pr_values[pr_values > 0])))
    else:
        pr_entropy = 0.0

    # PageRank Gini coefficient
    pr_sorted = np.sort(pr_values)
    n = len(pr_sorted)
    if n > 0 and pr_sorted.sum() > 0:
        index = np.arange(1, n + 1)
        gini = float((2 * np.sum(index * pr_sorted) / (n * np.sum(pr_sorted))) - (n + 1) / n)
    else:
        gini = 0.0

    return {
        "modularity": comm.modularity,
        "num_communities": comm.num_communities,
        "density": stats["density"],
        "avg_in_degree": stats["avg_in_degree"],
        "max_in_degree": stats["max_in_degree"],
        "max_out_degree": stats["max_out_degree"],
        "degree_std": float(np.std(dd["in_degrees"])) if len(dd["in_degrees"]) > 0 else 0.0,
        "pagerank_entropy": pr_entropy,
        "pagerank_gini": gini,
        "reciprocity": stats.get("reciprocity", 0.0),
    }


def _merge_into(dst: WeightGraph, src: WeightGraph) -> None:
    """将 src 图合并到 dst 图中。"""
    for node_name, attrs in src.nodes.items():
        if node_name not in dst.nodes:
            dst.add_node(node_name, **attrs)
    for src_node, dst_node, edge_attrs in src.edges:
        if src_node not in dst.node_to_idx:
            dst.add_node(src_node, **src.nodes.get(src_node, {}))
        if dst_node not in dst.node_to_idx:
            dst.add_node(dst_node, **src.nodes.get(dst_node, {}))
        dst.add_edge(src_node, dst_node, **edge_attrs)


def run():
    output_dir = ensure_dir(Path("results/weight_graph/exp07"))
    print(f"[exp07] Output: {output_dir}")

    graph_config = GraphBuildConfig(
        sparsify_method="topk",
        topk=32,
        add_residual=False,  # 单层分析不加残差
    )
    analysis_config = AnalysisConfig(
        community_method="louvain",
        community_resolution=1.0,
    )

    # 加载缓存的矩阵
    cache_path = Path("weight_graph/cache_matrices.pkl")
    if cache_path.exists():
        print("[exp07] Loading cached matrices...")
        with open(cache_path, "rb") as f:
            all_matrices = pickle.load(f)
        print(f"[exp07] Loaded {len(all_matrices)} matrices")
    else:
        print("[exp07] ERROR: cache_matrices.pkl not found. Run exp01 first.")
        return

    # 按层分组
    from collections import defaultdict
    by_layer = defaultdict(list)
    for m in all_matrices:
        by_layer[m.layer_index].append(m)

    num_layers = max(by_layer.keys()) + 1 if by_layer else 0
    print(f"[exp07] Model has {num_layers} layers")

    # 对每层计算拓扑向量
    topo_matrix = {}  # layer_index → topo_vector dict
    for layer_idx in range(num_layers):
        print(f"  Layer {layer_idx}/{num_layers-1}...", end=" ", flush=True)
        matrices = by_layer[layer_idx]
        topo = extract_layer_topo_vector(matrices, graph_config, analysis_config)
        topo_matrix[layer_idx] = topo
        print(f"mod={topo['modularity']:.3f}, pr_ent={topo['pagerank_entropy']:.1f}")

    # 保存结果
    save_results(topo_matrix, output_dir / "layer_topo_vectors.json")
    print(f"[exp07] Saved to {output_dir / 'layer_topo_vectors.json'}")

    # 转为 numpy 矩阵方便后续回归
    feature_names = list(topo_matrix[0].keys())
    X = np.array([[topo_matrix[l][f] for f in feature_names] for l in range(num_layers)])
    np.save(output_dir / "topo_matrix.npy", X)
    save_results(
        {"feature_names": feature_names, "shape": list(X.shape)},
        output_dir / "topo_matrix_meta.json"
    )
    print(f"[exp07] Topo matrix shape: {X.shape}")
    print(f"[exp07] Done.")


if __name__ == "__main__":
    run()