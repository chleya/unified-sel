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


@dataclass
class MemoryConfig:
    max_nodes: int = 500
    chroma_persist_dir: str = str(DATA_DIR / "chromadb")
    similarity_top_k: int = 5
    topo_recompute_interval: int = 20  # recompute topology every N inserts
    prune_threshold: float = 0.1       # persistence below this → pruneable
    prune_recent_protection: int = 10  # 保护最近 N 步内创建的节点


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
    seed: int = 42
