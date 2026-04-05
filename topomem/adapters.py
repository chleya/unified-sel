"""
topomem/adapters.py — 动态塑造机制

Prompt Adapter Pool 管理：根据 query 的拓扑位置自动选择最匹配的 adapter，
并随使用经验进化。

设计来源：
- SPEC_ADAPTERS.md: 动态塑造机制完整规格
- 从 unified-sel 的 surprise/tension 机制迁移
"""

from __future__ import annotations

import json
import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from topomem.config import AdapterConfig, SelfAwarenessConfig


if TYPE_CHECKING:
    from topomem.memory import MemoryGraph, MemoryNode
    from topomem.embedding import EmbeddingManager
    from topomem.engine import ReasoningEngine
    from topomem.self_awareness import SelfAwareness


logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# System Prompt 模板
# ------------------------------------------------------------------

ADAPTER_PROMPT_TEMPLATE = """You are a specialized assistant for {domain_name}.

Your expertise includes: {domain_keywords}.

When answering questions in this domain:
- Prioritize accuracy and domain-specific terminology
- Reference relevant context from your knowledge base
- Be precise and technical when appropriate
- If unsure, clearly state the limitation

{custom_instructions}"""

DEFAULT_CUSTOM_INSTRUCTIONS = """- Provide clear, structured answers
- Use examples when helpful
- Acknowledge uncertainty if present"""


# ------------------------------------------------------------------
# 抽象接口（为 LoRA 预留）
# ------------------------------------------------------------------

class BaseAdapter(ABC):
    """Adapter 抽象基类。

    所有 adapter（无论 Prompt 还是 LoRA）必须实现这些方法。
    """

    @abstractmethod
    def apply(self, prompt: str) -> str:
        """将 adapter 的效果应用到推理过程中。

        Prompt 实现：返回定制化 system prompt
        LoRA 实现：加载权重，返回原始 prompt
        """

    @abstractmethod
    def get_domain_embedding(self) -> np.ndarray:
        """返回此 adapter 所代表的领域的 embedding。"""

    @property
    @abstractmethod
    def adapter_id(self) -> str: ...

    @property
    @abstractmethod
    def adapter_type(self) -> str: ...  # "prompt" or "lora"


# ------------------------------------------------------------------
# PromptAdapter 实现
# ------------------------------------------------------------------

@dataclass
class PromptAdapter(BaseAdapter):
    """基于 system prompt 的行为适配器（MVP 实现）。"""

    id: str
    name: str
    system_prompt: str
    domain_keywords: List[str]
    domain_embedding: np.ndarray
    topological_cluster: int

    # 生命周期统计
    created_at: float
    usage_count: int = 0
    effectiveness_score: float = 0.5
    last_used: float = 0.0

    # 自定义指令
    custom_instructions: str = DEFAULT_CUSTOM_INSTRUCTIONS

    def apply(self, prompt: str = "") -> str:
        """注入定制化 system prompt。"""
        self.usage_count += 1
        self.last_used = time.time()
        return self.system_prompt

    def get_domain_embedding(self) -> np.ndarray:
        return self.domain_embedding

    @property
    def adapter_id(self) -> str:
        return self.id

    @property
    def adapter_type(self) -> str:
        return "prompt"

    def to_dict(self) -> dict:
        """序列化为字典。"""
        return {
            "id": self.id,
            "name": self.name,
            "system_prompt": self.system_prompt,
            "domain_keywords": self.domain_keywords,
            "domain_embedding": self.domain_embedding.tolist(),
            "topological_cluster": self.topological_cluster,
            "created_at": self.created_at,
            "usage_count": self.usage_count,
            "effectiveness_score": self.effectiveness_score,
            "last_used": self.last_used,
            "custom_instructions": self.custom_instructions,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PromptAdapter":
        """从字典恢复。"""
        d = dict(d)
        d["domain_embedding"] = np.array(d["domain_embedding"], dtype=np.float32)
        return cls(**d)

    def __repr__(self) -> str:
        return (
            f"PromptAdapter(id='{self.id[:8]}...', name='{self.name}', "
            f"cluster={self.topological_cluster}, "
            f"usage={self.usage_count}, "
            f"effectiveness={self.effectiveness_score:.3f})"
        )


# ------------------------------------------------------------------
# 默认 Adapter
# ------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = """You are a precise and consistent reasoning assistant.
You answer questions based on the provided context.
If the context doesn't contain relevant information, say so clearly.
Be concise and factual."""


def create_default_adapter(embedding_mgr: Optional["EmbeddingManager"] = None) -> PromptAdapter:
    """创建默认通用 adapter。"""
    now = time.time()
    emb = np.zeros(384, dtype=np.float32)
    if embedding_mgr is not None:
        emb = embedding_mgr.encode("general purpose reasoning and knowledge")

    return PromptAdapter(
        id="default",
        name="General",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        domain_keywords=["general", "reasoning", "knowledge"],
        domain_embedding=emb,
        topological_cluster=-1,
        created_at=now,
        effectiveness_score=0.5,
        last_used=now,
    )


# ------------------------------------------------------------------
# AdapterPool 实现
# ------------------------------------------------------------------

class AdapterPool:
    """管理 Adapter 的创建、选择、进化和淘汰。"""

    def __init__(
        self,
        config: Optional[AdapterConfig] = None,
        embedding_mgr: Optional["EmbeddingManager"] = None,
    ):
        self.config = config or AdapterConfig()
        self._embedding_mgr = embedding_mgr
        self._adapters: Dict[str, BaseAdapter] = {}
        self._default_adapter = create_default_adapter(embedding_mgr)
        self._adapters["default"] = self._default_adapter

    def select_adapter(
        self,
        query_embedding: np.ndarray,
        memory_graph: "MemoryGraph",
    ) -> Tuple[BaseAdapter, float]:
        """根据 query 的拓扑位置选择最合适的 adapter。

        返回：
            (best_adapter, surprise_score)
            surprise_score = 1.0 - best_similarity
        """
        non_default = {
            k: v for k, v in self._adapters.items() if k != "default"
        }

        if not non_default:
            return self._default_adapter, 1.0

        best_adapter = self._default_adapter
        best_sim = 0.0

        for aid, adapter in non_default.items():
            domain_emb = adapter.get_domain_embedding()
            sim = self._cosine_similarity(query_embedding, domain_emb)
            if sim > best_sim:
                best_sim = sim
                best_adapter = adapter

        # 置信度门槛
        if best_sim < 0.3:
            return self._default_adapter, 1.0 - best_sim

        return best_adapter, 1.0 - best_sim

    def create_adapter(
        self,
        cluster_id: int,
        representative_memories: List["MemoryNode"],
        engine: Optional["ReasoningEngine"] = None,
    ) -> BaseAdapter:
        """从拓扑簇中自动生成新 adapter。"""
        import uuid

        # 1. 领域 embedding
        embeddings = np.stack([m.embedding for m in representative_memories])
        domain_embedding = np.mean(embeddings, axis=0).astype(np.float32)

        # ========== 质量验证：相似 adapter 检查 ==========
        SIMILARITY_THRESHOLD = 0.85
        def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a < 1e-8 or norm_b < 1e-8:
                return 0.0
            return float(np.dot(a, b) / (norm_a * norm_b))

        for existing in self._adapters.values():
            if existing.domain_embedding is not None:
                sim = _cosine_sim(domain_embedding, existing.domain_embedding)
                if sim > SIMILARITY_THRESHOLD:
                    logger.info(
                        f"Skipping adapter: similar adapter "
                        f"'{existing.name}' exists (sim={sim:.3f})"
                    )
                    return existing

        # 2. 领域关键词提取
        if engine and self._embedding_mgr:
            keywords = self._extract_keywords_with_llm(
                representative_memories, engine
            )
        else:
            keywords = self._extract_keywords_simple(representative_memories)

        # 3. 领域名称
        domain_name = keywords[0].title() if keywords else f"Cluster {cluster_id}"

        # 4. System Prompt 生成
        system_prompt = ADAPTER_PROMPT_TEMPLATE.format(
            domain_name=domain_name,
            domain_keywords=", ".join(keywords),
            custom_instructions=DEFAULT_CUSTOM_INSTRUCTIONS,
        )

        # 5. 创建 PromptAdapter
        adapter_id = str(uuid.uuid4())
        now = time.time()
        adapter = PromptAdapter(
            id=adapter_id,
            name=domain_name,
            system_prompt=system_prompt,
            domain_keywords=keywords,
            domain_embedding=domain_embedding,
            topological_cluster=cluster_id,
            created_at=now,
            effectiveness_score=0.5,
            last_used=now,
        )

        self._adapters[adapter_id] = adapter

        # 6. 淘汰低效 adapter
        if len(self._adapters) > self.config.max_adapters:
            self._prune_adapters()

        logger.info(f"Created adapter '{domain_name}' for cluster {cluster_id}")
        return adapter

    def _extract_keywords_with_llm(
        self,
        memories: List["MemoryNode"],
        engine: "ReasoningEngine",
    ) -> List[str]:
        """用 LLM 提取关键词。"""
        texts = "\n".join(f"- {m.content}" for m in memories[:10])
        prompt = (
            f"Extract 5-8 domain keywords from these texts. "
            f"Return only comma-separated keywords:\n{texts}"
        )
        response = engine.generate(prompt, max_tokens=64, temperature=0.1)
        keywords = [k.strip().lower() for k in response.split(",") if k.strip()]
        return keywords[:8] if keywords else ["general"]

    def _extract_keywords_simple(
        self,
        memories: List["MemoryNode"],
    ) -> List[str]:
        """简单词频统计提取关键词。"""
        from collections import Counter

        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "dare", "ought", "used", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into",
            "through", "during", "before", "after", "above", "below",
            "between", "under", "again", "further", "then", "once",
            "and", "but", "or", "nor", "not", "so", "yet", "both",
            "either", "neither", "each", "every", "all", "any", "few",
            "more", "most", "other", "some", "such", "no", "only",
            "own", "same", "than", "too", "very", "just", "because",
            "if", "when", "where", "why", "how", "what", "which",
            "who", "whom", "this", "that", "these", "those", "it",
            "its", "i", "me", "my", "myself", "we", "our", "ours",
            "you", "your", "yours", "he", "him", "his", "she", "her",
            "they", "them", "their", "theirs", "about",
        }

        word_counts = Counter()
        for m in memories:
            words = m.content.lower().split()
            for w in words:
                w = w.strip(".,;:!?\"'()[]{}")
                if w not in stop_words and len(w) > 2:
                    word_counts[w] += 1

        return [w for w, _ in word_counts.most_common(8)]

    def evolve_adapter(
        self,
        adapter_id: str,
        feedback: float,
    ) -> None:
        """根据反馈调整 adapter 的效果评分。

        feedback ∈ [0, 1]:
        - 1.0 = 非常好
        - 0.5 = 中性
        - 0.0 = 很差
        """
        if adapter_id not in self._adapters:
            logger.warning(f"Adapter {adapter_id} not found")
            return

        adapter = self._adapters[adapter_id]
        decay = self.config.effectiveness_decay
        adapter.effectiveness_score = (
            adapter.effectiveness_score * decay + feedback * (1 - decay)
        )

    def _prune_adapters(self) -> List[str]:
        """淘汰低效 adapter。"""
        non_default = {
            k: v for k, v in self._adapters.items() if k != "default"
        }

        if len(non_default) <= self.config.max_adapters - 1:
            return []

        # 计算综合得分
        scores = []
        for aid, adapter in non_default.items():
            usage_factor = math.log(1 + adapter.usage_count)
            score = adapter.effectiveness_score * max(usage_factor, 0.1)
            scores.append((aid, score))

        scores.sort(key=lambda x: x[1])

        # 淘汰最低分的
        n_to_remove = len(non_default) - (self.config.max_adapters - 1)
        removed = []
        for aid, _ in scores[:n_to_remove]:
            del self._adapters[aid]
            removed.append(aid)

        logger.info(f"Pruned {len(removed)} adapters: {removed}")
        return removed

    def get_all_adapters(self) -> List[BaseAdapter]:
        """返回所有 adapter。"""
        return list(self._adapters.values())

    def get_adapter_by_id(self, adapter_id: str) -> Optional[BaseAdapter]:
        """按 ID 获取 adapter。"""
        return self._adapters.get(adapter_id)

    @property
    def default_adapter(self) -> BaseAdapter:
        """获取默认 adapter。"""
        return self._default_adapter

    @property
    def adapter_count(self) -> int:
        """adapter 总数（包括 default）。"""
        return len(self._adapters)

    # ==================================================================
    # 序列化
    # ==================================================================

    def save(self, path: str) -> None:
        """保存 adapter pool。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "adapters": {
                aid: adapter.to_dict()
                for aid, adapter in self._adapters.items()
                if aid != "default"
            },
            "default_adapter": self._default_adapter.to_dict(),
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, default=_json_default)

    def load(self, path: str) -> None:
        """加载 adapter pool。"""
        path = Path(path)
        if not path.exists():
            return

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._adapters.clear()

        # 加载 default
        default_data = data.get("default_adapter")
        if default_data:
            self._default_adapter = PromptAdapter.from_dict(default_data)
            self._adapters["default"] = self._default_adapter
        else:
            self._default_adapter = create_default_adapter(self._embedding_mgr)
            self._adapters["default"] = self._default_adapter

        # 加载自定义
        for aid, adapter_data in data.get("adapters", {}).items():
            adapter = PromptAdapter.from_dict(adapter_data)
            self._adapters[aid] = adapter

    def __repr__(self) -> str:
        return (
            f"AdapterPool(adapters={self.adapter_count}, "
            f"max={self.config.max_adapters})"
        )

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度。"""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


# ------------------------------------------------------------------
# Surprise / Tension 信号系统
# ------------------------------------------------------------------

def compute_surprise(
    query_embedding: np.ndarray,
    adapter_pool: AdapterPool,
) -> float:
    """query 的"意外度"。

    surprise = 1.0 - max_similarity_to_any_adapter

    用途：
    - surprise > 0.7 → 触发新 adapter 创建
    """
    _, surprise = adapter_pool.select_adapter(query_embedding, None)
    return surprise


def compute_tension(
    self_awareness: "SelfAwareness",
    window: int = 5,
) -> float:
    """系统的"张力" = 最近 N 次拓扑指纹变化的平均速率。

    tension = mean(wasserstein_distance(fp[i], fp[i-1]) for i in last N)
    """
    if len(self_awareness._diagram_history) < 2:
        return 0.0

    topo_engine = self_awareness._topo_engine

    history = self_awareness._diagram_history[-window:]
    drifts = []
    for i in range(1, len(history)):
        d1 = history[i].diagram
        d0 = history[i - 1].diagram
        drift = topo_engine.wasserstein_distance(d1, d0, dim=0)
        drifts.append(drift)

    return float(np.mean(drifts)) if drifts else 0.0


def decide_action(
    surprise: float,
    tension: float,
    surprise_threshold: float = 0.7,
    tension_threshold: float = 0.1,
) -> str:
    """根据 surprise 和 tension 决定行动。

    返回：
    - "use_existing": 使用现有 adapter
    - "create_adapter": 创建新 adapter
    - "consolidate": 触发记忆整理
    - "consolidate_and_delay": 整理 + 暂缓 adapter 创建
    """
    if tension < tension_threshold:
        if surprise < surprise_threshold:
            return "use_existing"
        else:
            return "create_adapter"
    else:
        if surprise < surprise_threshold:
            return "consolidate"
        else:
            return "consolidate_and_delay"


def _json_default(obj):
    """JSON 序列化辅助函数。"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
