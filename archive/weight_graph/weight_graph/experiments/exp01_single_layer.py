"""
实验 1：单层 MLP 图分析（Phase 1 入门实验）

目标：
  验证"权重矩阵 → 图"的 pipeline 可以跑通，
  观察单层 MLP 的基础拓扑特征。

预期耗时：几分钟（只处理一层）

步骤：
  1. 加载 Qwen2.5-0.5B 的第 0 层 MLP 权重（gate/up/down）
  2. 分别转为二部有向图
  3. 计算基础指标：节点数、边数、度分布
  4. 画度分布直方图
  5. 保存结果到 results/weight_graph/exp01/

预期输出：
  - exp01_stats.json: 基础统计
  - exp01_degree_dist.png: 度分布图
  - 控制台打印核心数字

判断标准（Phase 1 止损点）：
  - 如果度分布是纯均匀分布 → 图结构没有信息量，考虑止损
  - 如果度分布呈现幂律/长尾 → 有结构，继续 Phase 2
"""

from __future__ import annotations

from pathlib import Path

from weight_graph.config import ExperimentConfig, ExtractionConfig, GraphBuildConfig
from weight_graph.extractor import WeightExtractor
from weight_graph.graph_builder import GraphBuilder
from weight_graph.analyzers import basic_stats, degree_distribution
from weight_graph.viz import plot_degree_distribution
from weight_graph.utils import ensure_dir, save_results


def run(config: ExperimentConfig | None = None) -> None:
    """运行实验 1。"""
    if config is None:
        config = ExperimentConfig(
            extraction=ExtractionConfig(
                model_name="Qwen/Qwen2.5-0.5B",
                layer_types=["mlp"],
                layer_indices=[0],
            ),
            graph=GraphBuildConfig(
                sparsify_method="percentile",
                percentile=95.0,
                add_residual=False,  # 单层不需要残差
            ),
            output_dir="results/weight_graph/exp01",
        )

    output_dir = ensure_dir(Path(config.output_dir))
    print(f"[exp01] Output: {output_dir}")

    # Step 1: 提取权重
    print("[exp01] Extracting weights...")
    extractor = WeightExtractor(config.extraction)
    matrices = extractor.extract_single_layer(layer_index=0)
    print(f"[exp01] Got {len(matrices)} weight matrices")
    for m in matrices:
        print(f"  - {m.name}: {m.weight.shape} ({m.component})")

    # Step 2: 对每个矩阵构建图
    builder = GraphBuilder(config.graph)
    for matrix in matrices:
        print(f"\n[exp01] Building graph for {matrix.component}...")
        graph = builder.build_single_layer(matrix)

        # Step 3: 基础统计
        stats = basic_stats(graph)
        print(f"  Nodes: {stats['num_nodes']}")
        print(f"  Edges: {stats['num_edges']}")
        print(f"  Density: {stats['density']:.6f}")
        print(f"  Avg in-degree: {stats['avg_in_degree']:.1f}")
        print(f"  Avg out-degree: {stats['avg_out_degree']:.1f}")

        # Step 4: 度分布
        dd = degree_distribution(graph)

        # Step 5: 可视化
        plot_degree_distribution(
            dd["in_degrees"],
            dd["out_degrees"],
            title=f"Degree Distribution - Layer 0 {matrix.component}",
            save_path=output_dir / f"degree_dist_{matrix.component}.png",
        )

        # 保存
        save_results(
            {"component": matrix.component, "shape": list(matrix.weight.shape), **stats},
            output_dir / f"stats_{matrix.component}.json",
        )

    print(f"\n[exp01] Done. Results saved to {output_dir}")


if __name__ == "__main__":
    run()
