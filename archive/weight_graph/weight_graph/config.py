"""
weight_graph/config.py — 配置定义

所有可调参数集中管理。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ExtractionConfig:
    """权重提取配置。"""

    model_name: str = "Qwen/Qwen2.5-0.5B"
    # 要提取的层类型：mlp, attention, all
    layer_types: List[str] = field(default_factory=lambda: ["mlp"])
    # 指定层索引（None = 所有层）
    layer_indices: Optional[List[int]] = None
    # 设备：cpu / cuda / auto
    device: str = "cpu"
    # 是否只加载权重不加载完整模型（省内存）
    weights_only: bool = True
    # 随机种子（用于 extract_random_init）
    seed: Optional[int] = None


@dataclass
class GraphBuildConfig:
    """图构建配置。"""

    # --- 稀疏化 ---
    # 方法：percentile / topk / sigma
    sparsify_method: str = "percentile"
    # percentile 方法：保留 top X% 的边（按绝对值）
    percentile: float = 95.0
    # topk 方法：每个输出神经元保留 top-k 条入边
    topk: int = 32
    # sigma 方法：保留 mean + n_sigma * std 以上的边
    n_sigma: float = 2.0

    # --- 多层连接 ---
    # 是否添加残差连接（跨层同维度连边）
    add_residual: bool = True
    # 残差连接的边权（相对于权重边的比例）
    residual_weight: float = 1.0

    # --- Attention 处理 ---
    # 是否将 attention head 折叠为超节点
    collapse_attention: bool = True


@dataclass
class AnalysisConfig:
    """图分析配置。"""

    # 社区检测
    community_method: str = "louvain"  # louvain / leiden / spectral
    community_resolution: float = 1.0  # Louvain resolution 参数

    # PageRank
    pagerank_alpha: float = 0.85  # damping factor
    pagerank_top_n: int = 50  # 报告前 N 个节点

    # 环路检测
    max_cycle_length: int = 10  # 最大环路搜索长度
    # 强连通分量（SCC）最小规模
    scc_min_size: int = 3

    # 度分布拟合
    degree_fit: str = "powerlaw"  # powerlaw / lognormal


@dataclass
class ExperimentConfig:
    """实验总配置。"""

    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    graph: GraphBuildConfig = field(default_factory=GraphBuildConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    # 输出目录（相对于项目根目录）
    output_dir: str = "results/weight_graph"

    # 随机种子
    seed: int = 42
