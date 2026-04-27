"""
weight_graph/viz.py — 可视化

所有可视化函数接收分析结果，输出到文件或 matplotlib axes。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from weight_graph.graph_builder import WeightGraph
from weight_graph.analyzers import CommunityResult, CycleResult


def plot_degree_distribution(
    in_degrees: np.ndarray,
    out_degrees: np.ndarray,
    title: str = "Degree Distribution",
    save_path: Optional[Path] = None,
    log_scale: bool = True,
) -> None:
    """绘制入度/出度分布直方图。"""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title)

    ax1.hist(in_degrees, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    ax1.set_title("In-Degree Distribution")
    ax1.set_xlabel("In-Degree")
    ax1.set_ylabel("Count")
    if log_scale:
        ax1.set_yscale("log")

    ax2.hist(out_degrees, bins=50, color="darkorange", edgecolor="white", alpha=0.8)
    ax2.set_title("Out-Degree Distribution")
    ax2.set_xlabel("Out-Degree")
    ax2.set_ylabel("Count")
    if log_scale:
        ax2.set_yscale("log")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
    plt.close()


def plot_community_structure(
    graph: WeightGraph,
    community_result: CommunityResult,
    title: str = "Community Structure",
    save_path: Optional[Path] = None,
    max_nodes: int = 2000,
) -> None:
    """
    可视化社区结构。

    对于大图：显示社区规模分布而非完整图。
    对于小图：用 networkx spring_layout + matplotlib 散点图渲染。
    """
    import matplotlib.pyplot as plt

    if not community_result.community_sizes:
        return

    sizes = list(community_result.community_sizes.values())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title)

    axes[0].hist(sizes, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
    axes[0].set_title("Community Size Distribution")
    axes[0].set_xlabel("Community Size")
    axes[0].set_ylabel("Count")
    axes[0].set_yscale("log")

    sorted_communities = sorted(community_result.community_sizes.items(), key=lambda x: x[1], reverse=True)
    top_k = sorted_communities[:20]
    if top_k:
        comm_ids = [f"Comm {c[0]}" for c in top_k]
        comm_sizes = [c[1] for c in top_k]
        axes[1].barh(comm_ids[::-1], comm_sizes[::-1], color="darkorange", edgecolor="white")
        axes[1].set_title("Top 20 Communities by Size")
        axes[1].set_xlabel("Size")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
    plt.close()


def plot_modularity_comparison(
    labels: List[str],
    modularities: List[float],
    title: str = "Trained vs Random Modularity",
    save_path: Optional[Path] = None,
    errs: Optional[List[float]] = None,
) -> None:
    """
    柱状图对比不同模型的 modularity。

    用于 exp03（trained vs random）和 exp04（cross-scale）。
    如果有多个种子，传入 errs 画 error bar。
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.6), 5))
    fig.suptitle(title)

    colors = ["steelblue"] + ["lightgray"] * (len(labels) - 1)
    if errs:
        ax.bar(labels, modularities, yerr=errs, color=colors, edgecolor="white", capsize=4)
    else:
        ax.bar(labels, modularities, color=colors, edgecolor="white")

    ax.set_ylabel("Modularity")
    ax.set_ylim(0, max(modularities) * 1.2 if modularities else 1.0)
    ax.axhline(y=0.3, color="red", linestyle="--", alpha=0.5, label="Significance threshold (0.3)")
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
    plt.close()


def plot_pagerank_distribution(
    scores: Dict[str, float],
    top_n: int = 30,
    title: str = "PageRank Top Nodes",
    save_path: Optional[Path] = None,
) -> None:
    """
    PageRank 分数分布。

    上方：水平条形图显示 top-N 节点。
    下方：PageRank 分数直方图。
    """
    import matplotlib.pyplot as plt

    if not scores:
        return

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_nodes = sorted_scores[:top_n]
    all_values = list(scores.values())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(title)

    node_names = [n for n, _ in top_nodes]
    node_scores = [s for _, s in top_nodes]
    ax1.barh(node_names[::-1], node_scores[::-1], color="steelblue", edgecolor="white")
    ax1.set_title(f"Top {top_n} Nodes by PageRank")
    ax1.set_xlabel("PageRank Score")

    ax2.hist(all_values, bins=50, color="darkorange", edgecolor="white", alpha=0.8)
    ax2.set_title("PageRank Score Distribution")
    ax2.set_xlabel("PageRank Score")
    ax2.set_ylabel("Count")
    ax2.set_yscale("log")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
    plt.close()


def plot_cycle_stats(
    cycle_result: CycleResult,
    title: str = "Cycle Analysis",
    save_path: Optional[Path] = None,
) -> None:
    """
    环路分析可视化。

    左图：SCC 大小分布直方图。
    右图：环路长度分布。
    """
    import matplotlib.pyplot as plt

    sccs = cycle_result.strongly_connected_components
    if not sccs:
        return

    scc_sizes = [len(c) for c in sccs]
    cycle_lengths = cycle_result.cycle_length_distribution

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title)

    ax1.hist(scc_sizes, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
    ax1.set_title("SCC Size Distribution")
    ax1.set_xlabel("SCC Size")
    ax1.set_ylabel("Count")
    ax1.set_yscale("log")

    if cycle_lengths:
        lengths = sorted(cycle_lengths.keys())
        counts = [cycle_lengths[l] for l in lengths]
        ax2.bar(lengths, counts, color="darkorange", edgecolor="white")
        ax2.set_title("Cycle Length Distribution")
        ax2.set_xlabel("Cycle Length")
        ax2.set_ylabel("Count")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
    plt.close()


def export_graph(
    graph: WeightGraph,
    path: Path,
    fmt: str = "gexf",
) -> None:
    """
    导出图到文件，供外部工具（Gephi、graph-tool）使用。

    支持格式：
    - "gexf": XML 格式，Gephi 可直接打开
    - "graphml": 通用图格式
    - "edgelist": 简单文本（src dst weight）
    """
    import networkx as nx

    G = graph.to_networkx()
    if fmt == "gexf":
        nx.write_gexf(G, str(path))
    elif fmt == "graphml":
        nx.write_graphml(G, str(path))
    elif fmt == "edgelist":
        with open(path, "w") as f:
            for src, dst, attrs in graph.edges:
                w = attrs.get("weight", 1.0)
                f.write(f"{src} {dst} {w}\n")
    else:
        raise ValueError(f"Unknown export format: {fmt}. Use 'gexf', 'graphml', or 'edgelist'.")
