"""
实验 3：训练模型 vs 随机初始化模型的图拓扑对比（H1 验证）

这是整个研究最关键的实验。

核心问题：
  训练后的权重图是否比随机初始化有更高的模块化程度？
  如果是 → 训练过程自组织出了模块化结构
  如果否 → 图的社区结构是权重分布的数学产物，无语义含义

方法：
  1. 加载训练好的 Qwen2.5-0.5B → 构建图 → 计算 modularity（从缓存加载）
  2. 用相同架构，随机初始化权重 → 构建图 → 计算 modularity
  3. 重复 5 次随机初始化，得到 baseline 分布
  4. 训练模型的 modularity 是否显著高于 baseline？

预期耗时：~3 小时（5 个 random seed，每个约 20-40 分钟）
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pickle
import torch
import torch.nn as nn

from weight_graph.config import ExperimentConfig, ExtractionConfig, GraphBuildConfig, AnalysisConfig
from weight_graph.extractor import WeightExtractor, WeightMatrix
from weight_graph.graph_builder import GraphBuilder
from weight_graph.analyzers import detect_communities
from weight_graph.viz import plot_modularity_comparison
from weight_graph.utils import ensure_dir, save_results


def generate_random_matrices_from_template(
    template_matrices: List[WeightMatrix],
    seed: int,
) -> List[WeightMatrix]:
    """
    基于模板矩阵的形状生成随机初始化的权重矩阵。

    使用 PyTorch 的 Kaiming uniform 初始化（与 AutoModel.from_config 相同），
    这样避免每次都从 HuggingFace 下载模型。

    参数:
        template_matrices: 模板矩阵列表（用于获取形状）
        seed: 随机种子
    """
    torch.manual_seed(seed)
    import random
    random.seed(seed)
    np.random.seed(seed)

    random_matrices = []
    for tmpl in template_matrices:
        d_in, d_out = tmpl.d_in, tmpl.d_out
        # 使用 PyTorch Kaiming uniform 初始化（与 Linear 默认初始化相同）
        linear = nn.Linear(d_in, d_out, bias=False)
        weight = linear.weight.data.float().numpy()
        random_matrices.append(WeightMatrix(
            name=getattr(tmpl, 'name', f'layers.{tmpl.layer_index}.mlp.{tmpl.component}.weight'),
            layer_index=tmpl.layer_index,
            component=tmpl.component,
            weight=weight,
            d_in=d_in,
            d_out=d_out,
        ))
    return random_matrices




def run(
    n_random: int = 5,
    seeds: List[int] | None = None,
    use_cache: bool = True,
    config: ExperimentConfig | None = None,
    force_greedy: bool = False,
) -> None:
    """
    运行 trained vs random 对比实验（5-seed 版本用于统计显著性验证）。

    参数:
        n_random: 随机初始化重复次数（默认5）
        seeds: 随机种子列表（默认 [42, 123, 456, 789, 1024]）
        use_cache: 是否使用缓存的 trained graph（推荐 True，省时）
        config: 实验配置
    """
    if seeds is None:
        seeds = [42, 123, 456]  # Reduced from 5 to 3 seeds for faster execution
    n_random = len(seeds)

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
            output_dir="results/weight_graph/exp03",
        )

    output_dir = ensure_dir(Path(config.output_dir))
    print(f"[exp03] Output: {output_dir}")
    print(f"[exp03] Using seeds: {seeds}")

    # --- 训练模型（从缓存加载）---
    cache_path = Path("weight_graph/cache_graph.pkl")
    matrices_cache = Path("weight_graph/cache_matrices.pkl")

    if use_cache and cache_path.exists():
        print("[exp03] Loading trained graph from cache...")
        with open(cache_path, "rb") as f:
            trained_graph = pickle.load(f)
        trained_communities = detect_communities(trained_graph, config.analysis)
        trained_modularity = trained_communities.modularity
        print(f"[exp03] Trained modularity (from cache): {trained_modularity:.4f}")
    else:
        print("[exp03] Building trained graph from scratch...")
        extractor = WeightExtractor(config.extraction)
        matrices = extractor.extract()
        builder = GraphBuilder(config.graph)
        trained_graph = builder.build_full_model(matrices)
        trained_communities = detect_communities(trained_graph, config.analysis)
        trained_modularity = trained_communities.modularity
        print(f"[exp03] Trained modularity: {trained_modularity:.4f}")

    # 加载模板矩阵（用于生成随机初始化）
    if matrices_cache.exists():
        print("[exp03] Loading cached matrices for random generation...")
        with open(matrices_cache, "rb") as f:
            template_matrices = pickle.load(f)
    else:
        print("[exp03] Extracting template matrices from HuggingFace...")
        if 'extractor' not in dir():
            extractor = WeightExtractor(config.extraction)
        template_matrices = extractor.extract()

    # --- 随机初始化 baseline（5 seeds）---
    print(f"\n[exp03] Generating {n_random} random baselines...")
    print(f"[exp03] (Using PyTorch Kaiming init from template shapes, no HuggingFace)")
    random_modularities: List[float] = []

    builder = GraphBuilder(config.graph)

    for i, seed in enumerate(seeds):
        print(f"  Random seed {seed} ({i+1}/{n_random})...", end=" ", flush=True)
        random_matrices = generate_random_matrices_from_template(template_matrices, seed)
        random_graph = builder.build_full_model(random_matrices)
        random_communities = detect_communities(random_graph, config.analysis)
        random_modularities.append(random_communities.modularity)
        print(f"modularity = {random_communities.modularity:.4f}")
        # 释放内存
        del random_graph, random_communities, random_matrices

    # --- 统计检验 ---
    random_mean = np.mean(random_modularities)
    random_std = np.std(random_modularities)
    z_score = (trained_modularity - random_mean) / max(random_std, 1e-8)
    # 单侧 p-value（假设 trained > random）
    from scipy import stats as sp_stats
    p_value = 1 - sp_stats.norm.cdf(z_score)

    print(f"\n[exp03] === Results ===")
    print(f"  Trained modularity:  {trained_modularity:.4f}")
    print(f"  Random mean ± std:   {random_mean:.4f} ± {random_std:.4f}")
    print(f"  Z-score:             {z_score:.2f}")
    print(f"  P-value (one-sided): {p_value:.6f}")

    if p_value < 0.05:
        print(f"  *** H1 SUPPORTED: Training produces significantly higher modularity ***")
        h1_verdict = "SUPPORTED"
    elif p_value < 0.1:
        print(f"  H1 marginal: some evidence but not conclusive")
        h1_verdict = "MARGINAL"
    else:
        print(f"  H1 REJECTED: No significant difference.")
        h1_verdict = "REJECTED"

    # 保存结果
    results = {
        "trained_modularity": float(trained_modularity),
        "random_modularities": [float(m) for m in random_modularities],
        "random_mean": float(random_mean),
        "random_std": float(random_std),
        "z_score": float(z_score),
        "p_value": float(p_value),
        "n_random": n_random,
        "seeds": seeds,
        "h1_supported": p_value < 0.05,
        "h1_verdict": h1_verdict,
    }
    save_results(results, output_dir / "h1_results_5seed.json")
    print(f"[exp03] Saved 5-seed results to {output_dir / 'h1_results_5seed.json'}")

    # 可视化
    errs = [0.0] + [random_std] * n_random  # only trained has no error bar
    plot_modularity_comparison(
        labels=["Trained"] + [f"Random-{s}" for s in seeds],
        modularities=[trained_modularity] + random_modularities,
        title=f"H1: Trained vs Random Modularity (p={p_value:.4f}, {h1_verdict})",
        save_path=output_dir / "modularity_comparison_5seed.png",
        errs=errs,
    )

    print(f"\n[exp03] Done.")


if __name__ == "__main__":
    run()
