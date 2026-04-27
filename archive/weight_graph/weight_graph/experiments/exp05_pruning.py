"""
实验 5：基于图的剪枝实验（H2 + H4 验证）

核心问题：
  基于图拓扑的剪枝（PageRank / 社区）是否优于传统方法（magnitude）？

方法：
  1. 构建权重图，计算每个神经元的 PageRank 分数
  2. 三种剪枝策略，各剪掉 X% 的参数：
     a. Magnitude pruning: 按权重绝对值，删最小的
     b. PageRank pruning: 按 PageRank，删最低的
     c. Community pruning: 删除整个低 PageRank 社区
     d. Random pruning: 随机删除（baseline）
  3. 每种策略在多个剪枝率（10%, 20%, 30%, 50%）下评估
  4. 评估指标：perplexity on WikiText-2（或其他 benchmark）

前置条件：
  - exp02 完成（有 PageRank 和社区数据）
  - 需要能跑模型推理（至少 CPU）

预期耗时：2-4 小时（多个剪枝率 × 多种策略）

注意：
  - 剪枝只修改权重（置零），不改变架构
  - 实际剪枝操作：找到对应的 (layer, component, neuron_idx) → 将该行/列权重置零
  - 需要从图节点名反向映射回权重矩阵坐标
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from weight_graph.config import ExperimentConfig, ExtractionConfig, GraphBuildConfig, AnalysisConfig
from weight_graph.extractor import WeightExtractor
from weight_graph.graph_builder import GraphBuilder
from weight_graph.analyzers import compute_pagerank, detect_communities
from weight_graph.utils import ensure_dir, save_results


PRUNING_RATIOS = [0.10, 0.20, 0.30, 0.50]


def run(config: ExperimentConfig | None = None) -> None:
    """运行剪枝对比实验。"""
    if config is None:
        config = ExperimentConfig(
            extraction=ExtractionConfig(
                model_name="Qwen/Qwen2.5-0.5B",
                layer_types=["mlp"],
                layer_indices=None,
            ),
            graph=GraphBuildConfig(
                sparsify_method="topk",
                topk=32,
                add_residual=True,
            ),
            analysis=AnalysisConfig(
                community_method="louvain",
            ),
            output_dir="results/weight_graph/exp05",
        )

    output_dir = ensure_dir(Path(config.output_dir))
    print(f"[exp05] Output: {output_dir}")

    # Step 1: 构建图 + 分析
    print("[exp05] Building graph and computing metrics...")
    extractor = WeightExtractor(config.extraction)
    matrices = extractor.extract()
    builder = GraphBuilder(config.graph)
    graph = builder.build_full_model(matrices)
    pr = compute_pagerank(graph, config.analysis)
    communities = detect_communities(graph, config.analysis)

    # Step 2: 生成剪枝 mask
    # 实现者注意：这里需要将图节点映射回权重矩阵坐标
    # 节点名格式："L{layer}_mlp_{component}_{type}_{idx}"
    # → 解析出 (layer, component, idx)
    # → 在对应权重矩阵中将该行/列置零

    results: Dict[str, List[Dict]] = {
        "magnitude": [],
        "pagerank": [],
        "community": [],
        "random": [],
    }

    for ratio in PRUNING_RATIOS:
        print(f"\n[exp05] Pruning ratio: {ratio*100:.0f}%")

        for strategy in ["magnitude", "pagerank", "community", "random"]:
            print(f"  Strategy: {strategy}...", end=" ")

            # TODO: 实现每种策略
            # pruned_model = apply_pruning(model, strategy, ratio, pr, communities)
            # ppl = evaluate_perplexity(pruned_model, "wikitext-2")
            # results[strategy].append({"ratio": ratio, "perplexity": ppl})
            # print(f"PPL = {ppl:.2f}")
            raise NotImplementedError(f"需要实现 {strategy} pruning")

    # Step 3: 结果对比
    print(f"\n[exp05] === Results ===")
    print(f"{'Ratio':<8} {'Magnitude':>12} {'PageRank':>12} {'Community':>12} {'Random':>12}")
    for i, ratio in enumerate(PRUNING_RATIOS):
        row = f"{ratio*100:.0f}%{'':<5}"
        for strategy in ["magnitude", "pagerank", "community", "random"]:
            if i < len(results[strategy]):
                ppl = results[strategy][i]["perplexity"]
                row += f"{ppl:>12.2f}"
            else:
                row += f"{'N/A':>12}"
        print(row)

    save_results(results, output_dir / "pruning_results.json")
    print(f"\n[exp05] Done. Results saved to {output_dir}")


def apply_pruning(model, strategy: str, ratio: float, pr: Dict, communities) -> object:
    """
    对模型应用剪枝。

    参数：
        model: HuggingFace 模型
        strategy: "magnitude" / "pagerank" / "community" / "random"
        ratio: 剪枝比例 (0-1)
        pr: PageRank 结果（from compute_pagerank）
        communities: 社区检测结果

    返回剪枝后的模型（原模型的 copy，权重被置零）。

    实现提示：
    - magnitude: 收集所有权重的绝对值 → sort → 取 bottom ratio% → 置零
    - pagerank: 按 PageRank 排序 → 取 bottom ratio% 的节点 → 对应权重行/列置零
    - community: 按社区总 PageRank 排序 → 删除最弱社区直到达到 ratio%
    - random: 随机选 ratio% 的参数置零
    """
    raise NotImplementedError


def evaluate_perplexity(model, dataset: str = "wikitext-2") -> float:
    """
    计算模型在数据集上的 perplexity。

    实现提示：
    - 用 datasets 库加载 wikitext-2-raw-v1
    - 滑动窗口计算 loss
    - PPL = exp(avg_loss)
    - 参考 HuggingFace 的 perplexity 计算示例
    """
    raise NotImplementedError


if __name__ == "__main__":
    run()
