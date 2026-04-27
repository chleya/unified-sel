"""
weight_graph/analyzers.py — 图分析算法

所有分析器接收 WeightGraph，返回分析结果字典。
设计为无状态函数，方便组合和测试。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from weight_graph.config import AnalysisConfig
from weight_graph.graph_builder import WeightGraph


# ============================================================
# 1. 基础拓扑指标
# ============================================================

def basic_stats(graph: WeightGraph) -> Dict:
    """计算基础图统计量。"""
    from collections import defaultdict

    if graph.num_nodes == 0:
        return {
            "num_nodes": 0,
            "num_edges": 0,
            "density": 0.0,
            "avg_in_degree": 0.0,
            "avg_out_degree": 0.0,
            "max_in_degree": 0,
            "max_out_degree": 0,
            "reciprocity": 0.0,
        }

    in_deg = defaultdict(int)
    out_deg = defaultdict(int)
    edges_set = set()
    for src, dst, _ in graph.edges:
        out_deg[src] += 1
        in_deg[dst] += 1
        edges_set.add((src, dst))

    # 双向边比例
    bidirectional = sum(1 for (a, b) in edges_set if (b, a) in edges_set)
    reciprocity = bidirectional / max(len(edges_set), 1)

    all_in = list(in_deg.values())
    all_out = list(out_deg.values())

    return {
        "num_nodes": graph.num_nodes,
        "num_edges": graph.num_edges,
        "density": graph.num_edges / max(1, graph.num_nodes * (graph.num_nodes - 1)),
        "avg_in_degree": sum(all_in) / max(1, len(all_in)),
        "avg_out_degree": sum(all_out) / max(1, len(all_out)),
        "max_in_degree": max(all_in) if all_in else 0,
        "max_out_degree": max(all_out) if all_out else 0,
        "reciprocity": reciprocity,
    }


def degree_distribution(graph: WeightGraph) -> Dict:
    """计算入度和出度分布。"""
    from collections import defaultdict

    if graph.num_nodes == 0:
        return {
            "in_degrees": np.array([]),
            "out_degrees": np.array([]),
            "in_degree_hist": None,
            "out_degree_hist": None,
            "powerlaw_fit": None,
        }

    in_deg = defaultdict(int)
    out_deg = defaultdict(int)
    for src, dst, _ in graph.edges:
        out_deg[src] += 1
        in_deg[dst] += 1

    node_names = list(graph.nodes.keys())
    in_arr = np.array([in_deg[n] for n in node_names], dtype=int)
    out_arr = np.array([out_deg[n] for n in node_names], dtype=int)

    return {
        "in_degrees": in_arr,
        "out_degrees": out_arr,
        "in_degree_hist": np.histogram(in_arr, bins=50),
        "out_degree_hist": np.histogram(out_arr, bins=50),
        "powerlaw_fit": None,  # 可选：pip install powerlaw 后实现
    }


# ============================================================
# 2. 社区检测
# ============================================================

@dataclass
class CommunityResult:
    """社区检测结果。"""
    # 每个节点的社区标签
    partition: Dict[str, int]
    # 社区数量
    num_communities: int
    # modularity 分数 [0, 1]，> 0.3 认为有显著社区结构
    modularity: float
    # 每个社区的节点数量
    community_sizes: Dict[int, int]
    # 每个社区的内部边密度
    community_densities: Optional[Dict[int, float]] = None


@dataclass
class CycleResult:
    """环路检测结果。"""
    strongly_connected_components: List[List[str]]
    num_significant_scc: int
    max_scc_size: int
    sample_cycles: List[List[str]]
    cycle_length_distribution: Dict[int, int]


def detect_communities(graph: WeightGraph, config: AnalysisConfig) -> CommunityResult:
    """
    社区检测。

    自动选择算法：
    - python-louvain：用于中等图（<100K 节点）
    - NetworkX greedy_modularity：用于超大图（>=100K 节点）或 Louvain 失败时
    """
    import networkx as nx

    if graph.num_nodes == 0:
        return CommunityResult(
            partition={},
            num_communities=0,
            modularity=0.0,
            community_sizes={},
        )

    # For very large graphs, greedy_modularity is more practical
    # Louvain can be slow (>10min) on graphs with >200K nodes
    use_greedy = graph.num_nodes >= 100_000

    if not use_greedy:
        try:
            import community as community_louvain

            G_ud = graph.to_networkx().to_undirected()
            partition = community_louvain.best_partition(
                G_ud,
                weight="weight",
                resolution=config.community_resolution,
            )
            modularity = community_louvain.modularity(
                partition, G_ud, weight="weight"
            )
            # Build community structures from partition
            comm_map: Dict[int, List[str]] = {}
            for node, comm_id in partition.items():
                comm_map.setdefault(comm_id, []).append(node)
            community_sizes = {cid: len(nodes) for cid, nodes in comm_map.items()}
            num_communities = len(community_sizes)
            # Re-index communities to 0..num_communities-1 for consistency
            reindex = {old: new for new, old in enumerate(comm_map.keys())}
            partition = {node: reindex[cid] for node, cid in partition.items()}
            community_sizes = {reindex[cid]: len(nodes) for cid, nodes in comm_map.items()}
        except Exception:
            use_greedy = True

    if use_greedy:
        # Use NetworkX greedy_modularity (faster for very large graphs)
        from networkx.algorithms.community import greedy_modularity_communities

        G = graph.to_networkx().to_undirected()
        communities = greedy_modularity_communities(
            G, weight="weight", resolution=config.community_resolution
        )
        partition = {}
        community_sizes = {}
        for idx, comm in enumerate(communities):
            community_sizes[idx] = len(comm)
            for node in comm:
                partition[node] = idx
        num_communities = len(communities)
        modularity = nx.algorithms.community.modularity(
            G, communities, weight="weight", resolution=config.community_resolution
        )

    return CommunityResult(
        partition=partition,
        num_communities=num_communities,
        modularity=modularity,
        community_sizes=community_sizes,
    )


def community_profile(
    graph: WeightGraph,
    community_result: CommunityResult,
) -> Dict:
    """分析每个社区的特征。"""
    if not community_result.partition:
        return {}

    communities: Dict[int, List[str]] = {}
    for node, comm_id in community_result.partition.items():
        communities.setdefault(comm_id, []).append(node)

    profiles = {}
    for comm_id, nodes in communities.items():
        layers = set()
        components = set()
        internal_weights = []
        external_weights = []
        node_set = set(nodes)

        for src, dst, attrs in graph.edges:
            w = abs(attrs.get("weight", 1.0))
            if src in node_set and dst in node_set:
                internal_weights.append(w)
            else:
                external_weights.append(w)

        for node in nodes:
            attrs = graph.nodes.get(node, {})
            layers.add(attrs.get("layer", "?"))
            components.add(attrs.get("component", "?"))

        profiles[comm_id] = {
            "size": len(nodes),
            "layers": sorted(layers),
            "components": sorted(components),
            "avg_internal_weight": float(np.mean(internal_weights)) if internal_weights else 0.0,
            "avg_external_weight": float(np.mean(external_weights)) if external_weights else 0.0,
        }

    return profiles


# ============================================================
# 3. PageRank 分析
# ============================================================

def compute_pagerank(
    graph: WeightGraph,
    config: AnalysisConfig,
) -> Dict:
    """在有向图上计算 PageRank。"""
    import networkx as nx

    if graph.num_nodes == 0:
        return {
            "scores": {},
            "top_n": [],
            "layer_distribution": {},
            "component_distribution": {},
        }

    G = graph.to_networkx()
    scores = nx.pagerank(G, alpha=config.pagerank_alpha, weight="weight")
    top_n = sorted(scores.items(), key=lambda x: x[1], reverse=True)[: config.pagerank_top_n]

    layer_dist: Dict[int, float] = {}
    comp_dist: Dict[str, float] = {}
    for node, score in scores.items():
        attrs = graph.nodes.get(node, {})
        layer_dist[attrs.get("layer", "?")] = layer_dist.get(attrs.get("layer", "?"), 0.0) + score
        comp_dist[attrs.get("component", "?")] = comp_dist.get(attrs.get("component", "?"), 0.0) + score

    return {
        "scores": scores,
        "top_n": top_n,
        "layer_distribution": layer_dist,
        "component_distribution": comp_dist,
    }


# ============================================================
# 4. 环路检测
# ============================================================

def detect_cycles(graph: WeightGraph, config: AnalysisConfig) -> CycleResult:
    """环路和强连通分量检测。"""
    import networkx as nx
    from collections import Counter

    if graph.num_nodes == 0:
        return CycleResult(
            strongly_connected_components=[],
            num_significant_scc=0,
            max_scc_size=0,
            sample_cycles=[],
            cycle_length_distribution={},
        )

    G = graph.to_networkx()
    sccs = list(nx.strongly_connected_components(G))
    sig_sccs = [c for c in sccs if len(c) >= config.scc_min_size]
    max_scc_size = max((len(c) for c in sccs), default=0)

    sample_cycles = []
    cycle_lengths: Counter = Counter()
    for scc in sig_sccs:
        subgraph = G.subgraph(scc)
        try:
            for cycle in nx.simple_cycles(subgraph):
                if len(cycle) <= config.max_cycle_length:
                    sample_cycles.append(cycle)
                    cycle_lengths[len(cycle)] += 1
                    if len(sample_cycles) >= 100:
                        break
        except Exception:
            pass
        if len(sample_cycles) >= 100:
            break

    return CycleResult(
        strongly_connected_components=[list(c) for c in sig_sccs],
        num_significant_scc=len(sig_sccs),
        max_scc_size=max_scc_size,
        sample_cycles=sample_cycles[:100],
        cycle_length_distribution=dict(cycle_lengths),
    )


def characterize_cycles(graph: WeightGraph, cycles: List[List[str]]) -> Dict:
    """分析环路的特征。"""
    from collections import Counter

    if not cycles:
        return {
            "total_cycles": 0,
            "single_layer_cycles": 0,
            "cross_layer_cycles": 0,
            "mlp_only_cycles": 0,
            "mixed_cycles": 0,
            "cycles_via_residual": 0,
            "avg_cycle_length": 0.0,
            "avg_cycle_strength": 0.0,
        }

    single_layer = cross_layer = mlp_only = mixed = via_residual = 0
    total_len = total_strength = 0
    edge_weights = {(src, dst): abs(a.get("weight", 1.0)) for src, dst, a in graph.edges}

    for cycle in cycles:
        layers = set()
        components = set()
        for node in cycle:
            attrs = graph.nodes.get(node, {})
            layers.add(attrs.get("layer", "?"))
            comp = attrs.get("component", "?")
            if comp:
                components.add(comp.split("_")[0])
            if "R_" in node:
                via_residual += 1

        if len(layers) == 1:
            single_layer += 1
        else:
            cross_layer += 1

        if components == {"mlp"}:
            mlp_only += 1
        else:
            mixed += 1

        total_len += len(cycle)
        total_strength += sum(
            edge_weights.get((cycle[i], cycle[(i + 1) % len(cycle)]), 0.0)
            for i in range(len(cycle))
        )

    n = len(cycles)
    return {
        "total_cycles": n,
        "single_layer_cycles": single_layer,
        "cross_layer_cycles": cross_layer,
        "mlp_only_cycles": mlp_only,
        "mixed_cycles": mixed,
        "cycles_via_residual": via_residual,
        "avg_cycle_length": total_len / n,
        "avg_cycle_strength": total_strength / n,
    }


# ============================================================
# 5. 跨模型对比
# ============================================================

def compare_models(
    graph_a: WeightGraph,
    graph_b: WeightGraph,
    config: AnalysisConfig,
    label_a: str = "trained",
    label_b: str = "random",
) -> Dict:
    """对比两个模型的图拓扑指标。"""
    from scipy.stats import entropy

    def _stats(g: WeightGraph) -> Dict:
        if g.num_nodes == 0:
            return {k: 0.0 for k in (
                "num_nodes", "num_edges", "density", "modularity",
                "num_communities", "max_scc_size", "degree_entropy")}
        comm = detect_communities(g, config)
        cycles = detect_cycles(g, config)
        degrees = [sum(1 for _, dst, _ in g.edges if dst == n) for n in g.nodes]
        pk = np.array(degrees, dtype=float)
        pk = pk / pk.sum() if pk.sum() > 0 else pk
        dent = float(entropy(pk)) if len(pk) > 0 else 0.0
        return {
            "num_nodes": g.num_nodes,
            "num_edges": g.num_edges,
            "density": g.num_edges / max(1, g.num_nodes * (g.num_nodes - 1)),
            "modularity": comm.modularity,
            "num_communities": comm.num_communities,
            "max_scc_size": cycles["max_scc_size"],
            "degree_entropy": dent,
        }

    sa = _stats(graph_a)
    sb = _stats(graph_b)
    metrics = {}
    for key in sa:
        metrics[f"{label_a}_{key}"] = sa[key]
        metrics[f"{label_b}_{key}"] = sb[key]
        metrics[f"{key}_diff"] = sa[key] - sb[key]

    return {
        f"{label_a}_stats": sa,
        f"{label_b}_stats": sb,
        "metrics": metrics,
        "label_a": label_a,
        "label_b": label_b,
    }
