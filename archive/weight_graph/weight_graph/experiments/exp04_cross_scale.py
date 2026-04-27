"""
实验 4：不同规模模型的图拓扑对比（H3 验证）

核心问题：
  模型规模（0.5B → 1.5B → 3B）增大时，图的拓扑属性如何变化？
  特别是：环路数量/复杂度是否和推理能力正相关？

方法：
  对 Qwen2.5-0.5B、1.5B、3B 分别构建权重图，对比：
  - modularity（模块化程度是否随规模变化？）
  - SCC 数量和规模（环路复杂度是否随规模增加？）
  - PageRank entropy（关键节点分布是否更均匀？）
  - degree distribution 的幂律指数

预期耗时：1-2 小时（3B 模型的图构建较慢）

注意：
  - 3B 模型权重约 6GB，确保足够内存
  - 如果内存不够，只分析 MLP 层（跳过 attention）
  - 或只分析前 N 层做采样
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from weight_graph.config import ExperimentConfig, ExtractionConfig, GraphBuildConfig, AnalysisConfig
from weight_graph.extractor import WeightExtractor
from weight_graph.graph_builder import GraphBuilder
from weight_graph.analyzers import (
    basic_stats,
    degree_distribution,
    detect_communities,
    compute_pagerank,
    detect_cycles,
)
from weight_graph.utils import ensure_dir, save_results


MODELS = [
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",
]


def run(models: List[str] | None = None) -> None:
    """运行跨规模对比实验。"""
    if models is None:
        models = MODELS

    output_dir = ensure_dir(Path("results/weight_graph/exp04"))
    print(f"[exp04] Output: {output_dir}")

    all_results: Dict[str, Dict] = {}

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"[exp04] Analyzing {model_name}")
        print(f"{'='*60}")

        config = ExperimentConfig(
            extraction=ExtractionConfig(
                model_name=model_name,
                layer_types=["mlp"],  # 先只做 MLP，省内存
                layer_indices=None,
            ),
            graph=GraphBuildConfig(
                sparsify_method="topk",
                topk=32,
                add_residual=True,
            ),
            analysis=AnalysisConfig(
                community_method="louvain",
                scc_min_size=3,
            ),
        )

        extractor = WeightExtractor(config.extraction)
        matrices = extractor.extract()

        builder = GraphBuilder(config.graph)
        graph = builder.build_full_model(matrices)

        stats = basic_stats(graph)
        communities = detect_communities(graph, config.analysis)
        pr = compute_pagerank(graph, config.analysis)
        cycles = detect_cycles(graph, config.analysis)
        dd = degree_distribution(graph)

        result = {
            "model": model_name,
            "num_nodes": stats["num_nodes"],
            "num_edges": stats["num_edges"],
            "density": stats["density"],
            "modularity": communities.modularity,
            "num_communities": communities.num_communities,
            "max_scc_size": cycles.max_scc_size,
            "num_significant_scc": cycles.num_significant_scc,
            "pagerank_entropy": _entropy(list(pr["scores"].values())),
            "powerlaw_alpha": dd.get("powerlaw_fit", {}).get("alpha"),
        }
        all_results[model_name] = result

        print(f"  Nodes: {result['num_nodes']}")
        print(f"  Edges: {result['num_edges']}")
        print(f"  Modularity: {result['modularity']:.4f}")
        print(f"  Communities: {result['num_communities']}")
        print(f"  Max SCC: {result['max_scc_size']}")
        print(f"  PR Entropy: {result['pagerank_entropy']:.4f}")

    # 汇总对比
    print(f"\n{'='*60}")
    print(f"[exp04] Cross-Scale Summary")
    print(f"{'='*60}")
    header = f"{'Model':<25} {'Nodes':>8} {'Edges':>10} {'Mod':>6} {'Comm':>5} {'MaxSCC':>7} {'PREnt':>7}"
    print(header)
    for model_name, r in all_results.items():
        short_name = model_name.split("/")[-1]
        print(
            f"{short_name:<25} {r['num_nodes']:>8} {r['num_edges']:>10} "
            f"{r['modularity']:>6.3f} {r['num_communities']:>5} "
            f"{r['max_scc_size']:>7} {r['pagerank_entropy']:>7.3f}"
        )

    save_results(all_results, output_dir / "cross_scale_results.json")
    print(f"\n[exp04] Done. Results saved to {output_dir}")


def _entropy(values: list) -> float:
    """计算 PageRank 分布的 Shannon 熵。"""
    import numpy as np
    v = np.array(values, dtype=float)
    v = v / v.sum()
    v = v[v > 0]
    return float(-np.sum(v * np.log2(v)))


if __name__ == "__main__":
    run()
