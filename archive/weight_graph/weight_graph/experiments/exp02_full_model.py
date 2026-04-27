"""
实验 2：全模型图构建 + 社区检测 + PageRank（Phase 2-3）

目标：
  构建完整模型的有向图（含残差连接），
  分析社区结构和关键节点。

前置条件：
  exp01 通过（度分布非均匀）

预期耗时：10-30 分钟（取决于模型大小和稀疏化程度）

步骤：
  1. 加载 Qwen2.5-0.5B 全部层的权重
  2. 构建全模型图（含残差连接）
  3. 社区检测 + modularity
  4. PageRank + 关键节点识别
  5. 环路检测（SCC + 短环采样）
  6. 保存所有结果 + 可视化

预期输出：
  - exp02_full_stats.json
  - exp02_communities.json
  - exp02_pagerank.json
  - exp02_cycles.json
  - 若干 .png 可视化

关键判断点：
  - modularity > 0.3? → 有显著社区结构
  - SCC 中有 size > 10 的? → 存在非平凡环路
  - PageRank 分布是否有明显峰值?
"""

from __future__ import annotations

from pathlib import Path

from weight_graph.config import ExperimentConfig, ExtractionConfig, GraphBuildConfig, AnalysisConfig
from weight_graph.extractor import WeightExtractor
from weight_graph.graph_builder import GraphBuilder
from weight_graph.analyzers import (
    basic_stats,
    detect_communities,
    community_profile,
    compute_pagerank,
    detect_cycles,
    characterize_cycles,
)
from weight_graph.viz import (
    plot_community_structure,
    plot_pagerank_distribution,
    plot_cycle_stats,
)
from weight_graph.utils import ensure_dir, save_results


def run(config: ExperimentConfig | None = None) -> None:
    """运行实验 2。"""
    if config is None:
        config = ExperimentConfig(
            extraction=ExtractionConfig(
                model_name="Qwen/Qwen2.5-0.5B",
                layer_types=["mlp", "attention"],
                layer_indices=None,  # 全部层
            ),
            graph=GraphBuildConfig(
                sparsify_method="topk",
                topk=32,
                add_residual=True,
                collapse_attention=True,
            ),
            analysis=AnalysisConfig(
                community_method="louvain",
                community_resolution=1.0,
                pagerank_top_n=50,
                scc_min_size=3,
                max_cycle_length=8,
            ),
            output_dir="results/weight_graph/exp02",
        )

    output_dir = ensure_dir(Path(config.output_dir))
    print(f"[exp02] Output: {output_dir}")

    # Step 1: 提取全部权重
    print("[exp02] Extracting all weights...")
    extractor = WeightExtractor(config.extraction)
    matrices = extractor.extract()
    print(f"[exp02] Got {len(matrices)} weight matrices across {len(set(m.layer_index for m in matrices))} layers")

    # Step 2: 构建全模型图
    print("[exp02] Building full model graph...")
    builder = GraphBuilder(config.graph)
    graph = builder.build_full_model(matrices)
    stats = basic_stats(graph)
    print(f"[exp02] Graph: {stats['num_nodes']} nodes, {stats['num_edges']} edges")
    save_results(stats, output_dir / "full_stats.json")

    # Step 3: 社区检测
    print("[exp02] Detecting communities...")
    communities = detect_communities(graph, config.analysis)
    print(f"[exp02] Communities: {communities.num_communities}, Modularity: {communities.modularity:.4f}")
    if communities.modularity > 0.3:
        print("[exp02] *** SIGNIFICANT community structure detected! ***")
    else:
        print("[exp02] Weak community structure (modularity < 0.3)")

    profile = community_profile(graph, communities)
    save_results(
        {
            "num_communities": communities.num_communities,
            "modularity": communities.modularity,
            "community_sizes": communities.community_sizes,
            "profile": profile,
        },
        output_dir / "communities.json",
    )
    plot_community_structure(graph, communities, save_path=output_dir / "community_viz.png")

    # Step 4: PageRank
    print("[exp02] Computing PageRank...")
    pr = compute_pagerank(graph, config.analysis)
    print(f"[exp02] Top 10 PageRank nodes:")
    for name, score in pr["top_n"][:10]:
        print(f"  {name}: {score:.6f}")
    save_results(pr, output_dir / "pagerank.json")
    plot_pagerank_distribution(pr["scores"], save_path=output_dir / "pagerank_dist.png")

    # Step 5: 环路检测
    print("[exp02] Detecting cycles...")
    cycles = detect_cycles(graph, config.analysis)
    print(f"[exp02] Significant SCCs (size >= {config.analysis.scc_min_size}): {cycles.num_significant_scc}")
    print(f"[exp02] Max SCC size: {cycles.max_scc_size}")
    if cycles.sample_cycles:
        print(f"[exp02] Sample cycle (first): {cycles.sample_cycles[0]}")
        char = characterize_cycles(graph, cycles.sample_cycles)
        save_results(
            {
                "num_significant_scc": cycles.num_significant_scc,
                "max_scc_size": cycles.max_scc_size,
                "num_sample_cycles": len(cycles.sample_cycles),
                "characterization": char,
            },
            output_dir / "cycles.json",
        )
    else:
        print("[exp02] No significant cycles found (graph is DAG-like)")
        save_results(
            {"num_significant_scc": 0, "max_scc_size": cycles.max_scc_size},
            output_dir / "cycles.json",
        )

    plot_cycle_stats(cycles, save_path=output_dir / "cycle_stats.png")
    print(f"\n[exp02] Done. Results saved to {output_dir}")


if __name__ == "__main__":
    run()
