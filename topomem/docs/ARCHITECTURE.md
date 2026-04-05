# TopoMem Reasoner v0.1 — 系统架构总纲

> **文档性质**：权威架构参考。所有模块实现必须符合本文定义的层次、接口和数据流。  
> **读者**：执行 agent、未来的人类开发者、架构审查者。

---

## 1. 设计哲学

**核心命题**：把所有知识压缩进海量参数并常驻显存是低效的。人类智能不靠海量符号记忆，而靠多模态特征记忆 + 组合联想。

**系统策略**：
- 用一个**小而强的注意力/推理引擎**（0.5B 量化模型）做路由和推理
- 用一个**外部拓扑结构化记忆系统**存储和组织知识
- 用**动态塑造机制**让系统行为适应不同知识域
- 用**自我认知层**防止记忆漂移，维护长期一致性

**MVP 验证目标**：在固定的小模型上，拓扑结构化记忆是否比纯向量检索提供更好的长期一致性和知识组织？

---

## 2. 三层架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                      TopoMem Reasoner v0.1                        │
│                                                                     │
│  ┌─────────────────┐  ┌─────────────────────┐  ┌─────────────────┐ │
│  │    Layer 1       │  │      Layer 2         │  │    Layer 3       │ │
│  │  REASONING       │  │  TOPOLOGICAL MEMORY   │  │  DYNAMIC SHAPING │ │
│  │  ENGINE          │  │  SYSTEM               │  │  & SELF-AWARENESS│ │
│  │                  │  │                       │  │                  │ │
│  │  ┌────────────┐  │  │  ┌───────────────┐   │  │  ┌────────────┐ │ │
│  │  │ LLM Engine │  │  │  │ TopologyEngine│   │  │  │ Adapter    │ │ │
│  │  │ (Qwen2.5   │  │  │  │ (PH + Wass.) │   │  │  │ Pool       │ │ │
│  │  │  -0.5B Q4) │  │  │  └───────────────┘   │  │  │ (Prompt /  │ │ │
│  │  └────────────┘  │  │  ┌───────────────┐   │  │  │  LoRA*)    │ │ │
│  │  ┌────────────┐  │  │  │ MemoryGraph   │   │  │  └────────────┘ │ │
│  │  │ Embedding  │  │  │  │ (NetworkX +   │   │  │  ┌────────────┐ │ │
│  │  │ Manager    │  │  │  │  ChromaDB)    │   │  │  │ Self-      │ │ │
│  │  │ (MiniLM)   │  │  │  └───────────────┘   │  │  │ Awareness  │ │ │
│  │  └────────────┘  │  │                       │  │  │ (Topo      │ │ │
│  │                  │  │                       │  │  │ Fingerprint│ │ │
│  │                  │  │                       │  │  │ + Guard)   │ │ │
│  └─────────────────┘  └─────────────────────┘  └─────────────────┘ │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                    ORCHESTRATION LAYER                          ││
│  │                   TopoMemSystem (system.py)                    ││
│  │            统一入口：process() → ProcessResult                  ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘

* Prompt adapter 为 MVP 实现；LoRA adapter 为未来升级路径
```

### 2.1 Layer 1: 推理引擎层（Reasoning Engine）

**职责**：接收输入、编码特征、生成输出文本。

| 模块 | 文件 | 职责 |
|------|------|------|
| `EmbeddingManager` | `embedding.py` | 文本 → 384 维向量；相似度计算 |
| `ReasoningEngine` | `engine.py` | 接收 prompt + 记忆上下文 → 生成文本回答 |

**关键约束**：
- 推理引擎**不持有知识**，它只是一个"思考器"
- 所有知识来自 Layer 2 的记忆系统
- 引擎层对外暴露统一的 `generate(prompt, context)` 接口

### 2.2 Layer 2: 拓扑记忆系统（Topological Memory）

**职责**：知识的存储、组织、检索和维护。**这是系统的核心差异化层。**

| 模块 | 文件 | 职责 |
|------|------|------|
| `TopologyEngine` | `topology.py` | 对 embedding 点云计算 Persistent Homology；提取持久特征；计算拓扑指纹和 Wasserstein 距离 |
| `MemoryGraph` | `memory.py` | 管理记忆节点的图结构（NetworkX）；支持三种检索策略；容量管理和 prune |

**关键设计**：
- 每个记忆节点 = 一段文本 + 其 embedding + 拓扑标注（所属簇、持久性分数）
- 节点之间的**边**由拓扑分析决定（同一持久 H0 连通分支内的节点互连）
- 提供三种检索策略：`vector`（纯余弦相似）、`topological`（拓扑簇引导）、`hybrid`

### 2.3 Layer 3: 动态塑造与自我认知（Dynamic Shaping & Self-Awareness）

**职责**：让系统行为适应不同领域；监控长期一致性。

| 模块 | 文件 | 职责 |
|------|------|------|
| `AdapterPool` | `adapters.py` | 管理 Prompt Adapter 的创建/选择/进化/淘汰 |
| `SelfAwareness` | `self_awareness.py` | 维护全局拓扑指纹（核心自我认知）；漂移检测 |
| `ConsistencyGuard` | `guard.py` | 拦截可能破坏一致性的记忆写入 |

**关键设计**：
- **全局拓扑摘要**是自我认知的核心指纹——它是整个记忆图的 Persistent Diagram 的向量化表示
- Adapter 选择由 query 的拓扑位置（属于哪个簇）驱动
- ConsistencyGuard 在**每次记忆写入前**做预检

---

## 3. 核心数据流

### 3.1 主处理流水线（process 调用）

```
用户输入 (text)
    │
    ▼
┌─ EmbeddingManager.encode(text) ──────────────────────────────┐
│  输出：query_embedding (384-dim np.ndarray)                    │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌─ AdapterPool.select_adapter(query_embedding, memory_graph) ──┐
│  1. 计算 query 与各拓扑簇中心的距离                             │
│  2. 计算 surprise 信号（= 与最近簇中心的距离）                   │
│  3. 选择最匹配的 PromptAdapter                                 │
│  输出：selected_adapter, surprise_score                        │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌─ MemoryGraph.retrieve(query_embedding, strategy) ────────────┐
│  根据 strategy（vector/topological/hybrid）检索相关记忆          │
│  输出：List[MemoryNode]（top-k 相关记忆）                       │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌─ ReasoningEngine.generate(prompt, context, adapter) ─────────┐
│  1. 用 adapter.system_prompt 构造系统提示                       │
│  2. 将检索到的记忆格式化为上下文                                 │
│  3. 调用 LLM 生成回答                                          │
│  输出：response_text                                           │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌─ ConsistencyGuard.check(new_knowledge, memory_graph) ────────┐
│  1. 从 response 中提取可存储的新知识                             │
│  2. 预检：是否与现有高持久性记忆矛盾？                           │
│  3. 预检：是否会导致拓扑指纹剧变？                               │
│  输出：(accept: bool, reason: str)                              │
└──────────────────────────────────────────────────────────────┘
    │
    ▼ (if accept)
┌─ MemoryGraph.add_memory(content, embedding, metadata) ───────┐
│  1. 创建 MemoryNode                                            │
│  2. 写入 ChromaDB 向量索引                                      │
│  3. 如果达到 topo_recompute_interval → 触发拓扑重计算            │
└──────────────────────────────────────────────────────────────┘
    │
    ▼ (定期，每 calibration_interval 步)
┌─ SelfAwareness.update_and_check() ───────────────────────────┐
│  1. 计算当前全局拓扑指纹                                        │
│  2. 与历史指纹做 Wasserstein 距离                               │
│  3. 判断漂移状态：stable / drifting / restructuring             │
│  4. 如果 drifting → 触发校准流程                                │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌─ 返回 ProcessResult ─────────────────────────────────────────┐
│  response_text, retrieved_memories, adapter_used,              │
│  surprise_score, drift_status, memory_accepted                 │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 拓扑重计算流程（后台/定期）

```
MemoryGraph.update_topology() 被触发
    │
    ▼
收集所有 N 个记忆节点的 embedding → (N, 384) 矩阵
    │
    ▼
TopologyEngine.compute_persistence(points, max_dim=1)
    │
    ├─ 构建 Vietoris-Rips 复形
    ├─ 计算 H0（连通分支）和 H1（环）
    │
    ▼
TopologyEngine.extract_persistent_features(diagram)
    │
    ├─ 按 persistence = death - birth 排序
    ├─ 过滤：persistence > median → 信号；其余 → 噪声
    │
    ▼
更新 MemoryGraph 中的拓扑关系
    │
    ├─ H0 持久分支 → 定义拓扑簇（同分支内节点互连）
    ├─ 每个节点标注 persistence_score 和 cluster_id
    ├─ H1 环 → 标记冗余路径（可选，用于高级分析）
    │
    ▼
TopologyEngine.topological_summary(diagram) → fingerprint
    │
    ▼
SelfAwareness.update_fingerprint(fingerprint)
```

---

## 4. 模块依赖关系

```
embedding.py ◄── 无外部模块依赖（基础层）
     │
     ▼
topology.py ◄── 依赖 embedding.py（间接，通过共享 ndarray 格式）
     │
     ▼
memory.py ◄── 依赖 topology.py, embedding.py
     │
     ▼
engine.py ◄── 依赖 embedding.py（用于 prompt 上下文构造）
     │
     ▼
self_awareness.py ◄── 依赖 topology.py, memory.py
     │
     ▼
guard.py ◄── 依赖 memory.py, self_awareness.py, embedding.py
     │
     ▼
adapters.py ◄── 依赖 memory.py, embedding.py
     │
     ▼
system.py ◄── 依赖以上所有模块（集成层）
```

**实现顺序**（严格遵循依赖关系）：
```
embedding.py → topology.py → memory.py → engine.py → 
self_awareness.py → guard.py → adapters.py → system.py
```

---

## 5. 跨模块接口契约

### 5.1 核心数据类型

```python
# 所有模块共享的数据类型定义在 config.py 或各自模块的顶部

# 向量表示
Embedding = np.ndarray  # shape: (384,), dtype: float32

# 持久图
PersistenceDiagram = List[np.ndarray]  
# dgms[0]: H0 diagram, shape (n, 2), columns = (birth, death)
# dgms[1]: H1 diagram, shape (m, 2)

# 拓扑指纹
TopologicalFingerprint = np.ndarray  # 固定长度向量，由 PersistenceLandscape 或 BettiCurve 生成

# 拓扑特征
@dataclass
class TopologicalFeature:
    dimension: int          # 0=连通分支, 1=环
    birth: float
    death: float
    persistence: float      # = death - birth
    representative: Optional[List[int]]  # 构成该特征的节点索引
```

### 5.2 关键接口签名

```python
# embedding.py
class EmbeddingManager:
    def encode(self, text: str) -> Embedding: ...
    def encode_batch(self, texts: List[str]) -> np.ndarray: ...  # (N, 384)
    def similarity(self, a: Embedding, b: Embedding) -> float: ...

# topology.py
class TopologyEngine:
    def compute_persistence(self, points: np.ndarray) -> PersistenceDiagram: ...
    def extract_persistent_features(self, diagram: PersistenceDiagram) -> List[TopologicalFeature]: ...
    def wasserstein_distance(self, diag_a: PersistenceDiagram, diag_b: PersistenceDiagram) -> float: ...
    def topological_summary(self, diagram: PersistenceDiagram) -> TopologicalFingerprint: ...

# memory.py
class MemoryGraph:
    def add_memory(self, content: str, embedding: Embedding, metadata: dict) -> MemoryNode: ...
    def retrieve(self, query_embedding: Embedding, strategy: str, k: int) -> List[MemoryNode]: ...
    def update_topology(self, topo_engine: TopologyEngine) -> None: ...
    def get_topological_summary(self, topo_engine: TopologyEngine) -> TopologicalFingerprint: ...
    def prune(self, max_nodes: int) -> List[str]: ...  # 返回被删除的 node_id

# engine.py
class ReasoningEngine:
    def generate(self, prompt: str, context: List[MemoryNode], adapter: Optional[PromptAdapter]) -> str: ...

# self_awareness.py
class SelfAwareness:
    def update_fingerprint(self, memory_graph: MemoryGraph, topo_engine: TopologyEngine) -> None: ...
    def detect_drift(self) -> DriftReport: ...
    def get_identity_vector(self) -> TopologicalFingerprint: ...

# guard.py
class ConsistencyGuard:
    def should_accept_memory(self, new_embedding: Embedding, new_content: str, 
                              memory_graph: MemoryGraph, self_awareness: SelfAwareness) -> Tuple[bool, str]: ...

# adapters.py (接口设计兼容未来 LoRA)
class BaseAdapter(ABC):
    """抽象基类，Prompt 和 LoRA adapter 共用"""
    @abstractmethod
    def apply(self, engine: ReasoningEngine, prompt: str) -> str: ...
    @abstractmethod
    def get_domain_embedding(self) -> Embedding: ...

class PromptAdapter(BaseAdapter): ...     # MVP 实现
# class LoRAAdapter(BaseAdapter): ...     # 未来实现

class AdapterPool:
    def select_adapter(self, query_embedding: Embedding, memory_graph: MemoryGraph) -> Tuple[BaseAdapter, float]: ...
    def create_adapter(self, cluster_id: int, representative_memories: List[MemoryNode]) -> BaseAdapter: ...
    def evolve_adapter(self, adapter_id: str, feedback: float) -> None: ...

# system.py
class TopoMemSystem:
    def process(self, input_text: str) -> ProcessResult: ...
    def get_status(self) -> SystemStatus: ...
```

---

## 6. 硬件约束与性能预算

| 资源 | 预算 | 分配 |
|------|------|------|
| RAM 总计 | < 4 GB | 模型 ~500MB + embedding ~200MB + ChromaDB ~200MB + NetworkX ~100MB + 工作内存 ~1GB |
| 磁盘 | F: 盘，< 10 GB | 模型文件 + ChromaDB 持久化 + 记忆快照 |
| 单次推理延迟 | < 60s（目标 < 30s） | embedding ~0.1s + 检索 ~0.5s + LLM 生成 ~20-50s |
| 记忆容量 | ≤ 500 节点 | 受 PH 计算复杂度限制（N^2 距离矩阵） |
| 拓扑重计算 | 每 20 次插入 | 500 节点时约 2-5 秒 |

---

## 7. 文件结构

```
F:\unified-sel\
├── core/                          # 原 unified-sel 代码（保留不动）
├── topomem/
│   ├── __init__.py                # 包入口，版本号
│   ├── config.py                  # 全局配置 dataclass
│   ├── embedding.py               # Layer 1: EmbeddingManager
│   ├── topology.py                # Layer 2: TopologyEngine
│   ├── memory.py                  # Layer 2: MemoryGraph + MemoryNode
│   ├── engine.py                  # Layer 1: ReasoningEngine
│   ├── self_awareness.py          # Layer 3: SelfAwareness
│   ├── guard.py                   # Layer 3: ConsistencyGuard
│   ├── adapters.py                # Layer 3: AdapterPool + BaseAdapter
│   ├── system.py                  # 集成层: TopoMemSystem
│   ├── docs/
│   │   ├── ARCHITECTURE.md        # 本文件
│   │   ├── SETUP_GUIDE.md         # 环境搭建指南
│   │   ├── SPEC_TOPOLOGY.md       # 拓扑引擎规格
│   │   ├── SPEC_MEMORY.md         # 记忆系统规格
│   │   ├── SPEC_ENGINE.md         # 推理引擎规格
│   │   ├── SPEC_SELF_AWARENESS.md # 自我认知规格
│   │   ├── SPEC_ADAPTERS.md       # 动态塑造规格
│   │   └── SPEC_INTEGRATION.md    # 集成与测试规格
│   ├── tests/
│   │   ├── test_infra.py          # 基础设施冒烟测试
│   │   ├── test_embedding.py      # embedding 单元测试
│   │   ├── test_topology.py       # 拓扑引擎单元测试
│   │   ├── test_memory.py         # 记忆系统单元测试
│   │   ├── test_engine.py         # 推理引擎单元测试
│   │   ├── test_self.py           # 自我认知单元测试
│   │   ├── test_adapters.py       # adapter 单元测试
│   │   └── test_integration.py    # 端到端集成测试
│   ├── benchmarks/
│   │   ├── knowledge_consistency.py
│   │   ├── long_term_drift.py
│   │   ├── baseline_comparison.py
│   │   └── resource_usage.py
│   ├── data/
│   │   ├── models/                # GGUF 模型文件
│   │   └── test_corpus/           # 测试语料
│   └── results/                   # 实验结果输出
├── .venv/                         # Python 虚拟环境（F 盘）
├── TOPOMEM_PLAN.md                # 项目计划
└── TOPOMEM_STATUS.md              # 进度跟踪
```

---

## 8. 实现顺序与阶段对应

| 顺序 | 文件 | 对应 Phase | 依赖 |
|------|------|-----------|------|
| 1 | `config.py` | Phase 0 ✅ | 无 |
| 2 | `embedding.py` | Phase 1 | 无 |
| 3 | `topology.py` | Phase 1 | 无 |
| 4 | `memory.py` | Phase 2 | embedding, topology |
| 5 | `engine.py` | Phase 3 | embedding |
| 6 | `self_awareness.py` | Phase 5 | topology, memory |
| 7 | `guard.py` | Phase 5 | memory, self_awareness, embedding |
| 8 | `adapters.py` | Phase 4 | memory, embedding |
| 9 | `system.py` | Phase 6 | 全部 |
