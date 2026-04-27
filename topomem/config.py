"""
Global configuration for TopoMem Reasoner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOPOMEM_ROOT = Path(__file__).resolve().parent
DATA_DIR = TOPOMEM_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
CORPUS_DIR = DATA_DIR / "test_corpus"
RESULTS_DIR = TOPOMEM_ROOT / "results"


@dataclass
class EmbeddingConfig:
    model_name: str = "all-MiniLM-L6-v2"
    dimension: int = 384
    batch_size: int = 32
    device: str = "cpu"


@dataclass
class TopologyConfig:
    max_homology_dim: int = 2          # compute H0, H1, and H2 (H2 adds < 12ms overhead, validated P0+P1)
    persistence_threshold: Optional[float] = None  # None = use median
    filtration_steps: int = 30
    metric: str = "euclidean"
    max_h0_clusters: Optional[int] = None  # 限制 H0 聚类数，None = 自动
    
    # 聚类方法选择（P0 修复：解决 H0 单点簇问题）
    clustering_method: str = "hybrid"    # "hybrid" | "dbscan" | "h0"
                                         # hybrid = DBSCAN 预聚类 + H0 细化（推荐）
                                         # dbscan = 纯密度聚类
                                         # h0 = 原始 single-linkage
    
    # DBSCAN 预聚类参数
    dbscan_eps: Optional[float] = None   # None = 自动估计（k-distance 90 分位数）
    dbscan_min_samples: int = 3          # 形成簇的最少点数
    auto_eps_percentile: float = 90      # 自动估计 eps 时使用的分位数

    # UMAP 降维参数（P0 修复：解决高维空间聚类失效问题）
    # 验证结果：384D DBSCAN ARI=0.000，UMAP(2D)+DBSCAN ARI=0.945
    use_umap_before_clustering: bool = True  # 在聚类前先用 UMAP 降到低维
    umap_n_components: int = 2           # UMAP 降维目标维度（2D 效果最好，ARI=0.945）
    umap_n_neighbors: int = 15           # UMAP n_neighbors 参数
    umap_min_dist: float = 0.1           # UMAP min_dist 参数


@dataclass
class MemoryConfig:
    max_nodes: int = 500
    chroma_persist_dir: str = str(DATA_DIR / "chromadb")
    similarity_top_k: int = 5
    topo_recompute_interval: int = 20  # recompute topology every N inserts
    prune_threshold: float = 0.1       # persistence below this → pruneable
    prune_recent_protection: int = 10  # 保护最近 N 步内创建的节点
    # Retrieval weighting (方向4: H0 persistence 作为检索信号)
    retrieval_vector_weight: float = 0.6   # alpha: vector similarity weight
    retrieval_topo_weight: float = 0.3      # beta: topological score weight
    retrieval_persistence_weight: float = 0.1  # gamma: H0 persistence weight
    # 重要性时间衰减（原来硬编码 0.001）
    importance_decay: float = 0.001


@dataclass
class EngineConfig:
    model_path: str = str(MODELS_DIR / "qwen2.5-0.5b-instruct-q4_k_m.gguf")
    n_ctx: int = 2048
    n_threads: int = 4
    max_tokens: int = 256
    temperature: float = 0.7
    # Fallback: if llama-cpp-python not available, use transformers
    fallback_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    use_fallback: bool = False


@dataclass
class AdapterConfig:
    """Prompt Adapter Pool config.
    
    Interface is designed to be swappable with LoRA adapters in the future.
    Key abstraction: an Adapter modifies model behavior for a domain/cluster.
    Current impl: prompt-based. Future: LoRA weight-based.
    """
    max_adapters: int = 10
    min_usage_for_keep: int = 3
    effectiveness_decay: float = 0.95
    # Future LoRA switch point
    adapter_backend: str = "prompt"  # "prompt" or "lora" (future)


@dataclass
class SelfAwarenessConfig:
    fingerprint_history_size: int = 100
    drift_threshold: float = 0.1       # Wasserstein distance threshold
    calibration_interval: int = 50     # calibrate every N steps
    top_k_features: int = 10           # top-K persistent features for fingerprint


@dataclass
class TopoMemConfig:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    awareness: SelfAwarenessConfig = field(default_factory=SelfAwarenessConfig)
    # Consolidation settings (direction 2)
    consolidation_merge_threshold: float = 0.80  # lowered from 0.92 to enable real merge candidates
    # H1 health action threshold (direction 1)
    h1_health_action_threshold: float = 0.3     # trigger consolidation when h1_health falls below this
    seed: int = 42
