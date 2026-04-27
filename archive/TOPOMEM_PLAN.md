# TopoMem Reasoner v0.1 — 完整实施计划

**文档角色**：项目蓝图与执行合同  
**作者**：首席架构师  
**日期**：2026-04-04  
**状态**：待确认

---

## 0. 硬件现实与架构约束

| 资源 | 实际值 | 约束影响 |
|------|--------|----------|
| GPU | **无独立 GPU**（Intel UHD 630 集显，1GB） | 不可能运行 3B+ 模型 on-device |
| RAM | 16 GB（可用 ~1.6 GB，系统占用重） | CPU 推理上限 ~1B 模型 Q4 量化 |
| 磁盘 | F: 盘 146 GB 可用 | 模型文件 + 数据充足 |
| Python | 3.14.2 | 前沿版本，部分库兼容性需验证 |
| 已有库 | torch, transformers, chromadb, networkx, scipy, sentence-transformers, onnxruntime(CPU) | 基础设施充足 |
| 缺失 | 无 Ollama/llama.cpp, 无 giotto-tda/GUDHI, 无 PEFT | 需逐步安装 |

### 关键决策：模型选型

**无 GPU 意味着原计划中的 "Qwen3-8B / Phi-4 量化小模型" 不可行。** 在 16GB RAM CPU-only 环境下：

| 方案 | 模型大小 | 推理速度 | 可行性 |
|------|---------|---------|--------|
| ~~Qwen3-8B Q4~~ | ~4.5 GB | 极慢（~1 tok/s CPU） | ❌ 内存不足 |
| ~~Phi-4 Q4~~ | ~7 GB | 不可用 | ❌ |
| Qwen2.5-1.5B Q4 | ~1 GB | ~5-8 tok/s CPU | ✅ 勉强可用 |
| Qwen2.5-0.5B Q4 | ~400 MB | ~15-20 tok/s CPU | ✅ 推荐 MVP |
| SmolLM2-360M | ~360 MB | ~20+ tok/s CPU | ✅ 最快选项 |
| **ONNX 量化 embedding 模型** | ~90 MB | 快 | ✅ 特征提取首选 |

**MVP 选型决策**：
- **特征提取 / embedding**：用 ONNX 量化的小 embedding 模型（all-MiniLM-L6-v2，90MB），速度快，占用小
- **推理引擎**：用 Qwen2.5-0.5B Q4（via llama-cpp-python CPU），仅在需要文本生成/推理时调用
- **验证后升级**：一旦机制验证通过，可以在有 GPU 的机器上换 3B-8B 模型

> **核心原则**：MVP 的目标是验证架构，不是验证模型能力。用小模型证明"拓扑记忆 + 动态塑造"的增益是真实的，然后 scale up。

---

## 1. 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    TopoMem Reasoner v0.1                    │
│                                                             │
│  ┌──────────────┐  ┌───────────────────┐  ┌──────────────┐ │
│  │   Layer 1    │  │     Layer 2       │  │   Layer 3    │ │
│  │  Attention   │  │  Topological      │  │  Dynamic     │ │
│  │  & Reasoning │  │  Memory System    │  │  Shaping &   │ │
│  │  Engine      │  │                   │  │  Self-Aware  │ │
│  │              │  │  ┌─────────────┐  │  │              │ │
│  │  Qwen2.5    │  │  │ Persistent  │  │  │  LoRA Pool   │ │
│  │  -0.5B Q4   │  │  │ Homology    │  │  │  Manager     │ │
│  │  (CPU)      │  │  │ Engine      │  │  │              │ │
│  │              │  │  └─────────────┘  │  │  Topo        │ │
│  │  Embedding: │  │  ┌─────────────┐  │  │  Fingerprint │ │
│  │  MiniLM-L6  │  │  │ Graph       │  │  │  & Drift     │ │
│  │  (ONNX)     │  │  │ Memory      │  │  │  Guard       │ │
│  │              │  │  │ (NetworkX)  │  │  │              │ │
│  │              │  │  └─────────────┘  │  │              │ │
│  │              │  │  ┌─────────────┐  │  │              │ │
│  │              │  │  │ Vector      │  │  │              │ │
│  │              │  │  │ Store       │  │  │              │ │
│  │              │  │  │ (ChromaDB)  │  │  │              │ │
│  │              │  │  └─────────────┘  │  │              │ │
│  └──────────────┘  └───────────────────┘  └──────────────┘ │
│                                                             │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              Orchestration Layer                        ││
│  │              (Pipeline Controller)                      ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 阶段划分

### Phase 0: 基础设施搭建（Day 1-2）

**目标**：建立项目骨架、安装依赖、验证所有底层工具可用。

| 任务 | 具体内容 | 验证标准 |
|------|---------|---------|
| 0.1 项目结构 | 创建 `topomem/` 包结构，与现有 `core/` 分离 | `import topomem` 成功 |
| 0.2 依赖安装 | giotto-tda, llama-cpp-python(CPU), GUDHI | `pip install` 成功 + import 测试 |
| 0.3 小模型下载 | Qwen2.5-0.5B-Instruct GGUF Q4 | 文件存在 + 能加载 |
| 0.4 embedding 验证 | all-MiniLM-L6-v2 via sentence-transformers | 编码一段文本返回 384 维向量 |
| 0.5 TDA 冒烟测试 | giotto-tda 计算一组随机点的 PH | 返回持久图 |
| 0.6 ChromaDB 验证 | 创建 collection, 插入/查询 | CRUD 操作通过 |

**产出**：`topomem/__init__.py`, `topomem/tests/test_infra.py` 全部通过

**风险**：
- giotto-tda 可能不支持 Python 3.14（较新），备选方案：用 `ripser` 或 `persim` 或直接用 GUDHI
- llama-cpp-python CPU 编译在 Windows 上可能有问题，备选：用 ctransformers 或直接用 transformers + torch CPU

**预计时间**：1-2 天

---

### Phase 1: Embedding 特征层 + 拓扑分析引擎（Day 3-6）

**目标**：实现"输入 → embedding → 拓扑特征提取"的完整管线。

#### 1.1 Embedding Manager (`topomem/embedding.py`)

```python
class EmbeddingManager:
    """管理文本/多模态输入到特征向量的转换"""
    def encode(text: str) -> np.ndarray          # → 384-dim
    def encode_batch(texts: list) -> np.ndarray   # 批量编码
    def similarity(a, b) -> float                 # 余弦相似度
```

- 使用 sentence-transformers (all-MiniLM-L6-v2)
- 后续可替换为更好的 embedding 模型

#### 1.2 拓扑特征引擎 (`topomem/topology.py`)

```python
class TopologyEngine:
    """从特征向量集合中提取持久拓扑特征"""
    
    def compute_persistence(
        points: np.ndarray,          # (N, dim) 点云
        max_dim: int = 1             # 计算 H0, H1
    ) -> PersistenceDiagram:
        """计算 Persistent Homology"""
    
    def extract_persistent_features(
        diagram: PersistenceDiagram,
        persistence_threshold: float = None  # 默认用中位数
    ) -> List[TopologicalFeature]:
        """过滤噪声，只保留持久特征"""
    
    def wasserstein_distance(
        diag_a: PersistenceDiagram,
        diag_b: PersistenceDiagram
    ) -> float:
        """两个持久图之间的距离 = 拓扑变化度"""
    
    def topological_summary(
        diagram: PersistenceDiagram
    ) -> np.ndarray:
        """将持久图转为固定长度向量（拓扑指纹）"""
```

**关键设计决策**：
- filtration: Vietoris-Rips 复形，metric = L2 距离
- 持久性阈值: `median(persistence)` 以上视为信号
- 拓扑指纹: 用 Persistence Landscape 或 Betti Curve 转为向量

#### 1.3 验证实验

| 实验 | 方法 | 期望结果 |
|------|------|---------|
| 同主题文本聚类 | 编码 50 段文本 → 计算 PH → 检查 H0 | 同主题文本形成持久连通分支 |
| 主题切换检测 | 两组不同主题文本 → 拓扑指纹 → Wasserstein 距离 | 切换点的距离明显更大 |
| 噪声鲁棒性 | 在 embedding 上加高斯噪声 → 检查持久特征是否稳定 | 持久特征（高 persistence）不受影响 |

**产出**：`topomem/embedding.py`, `topomem/topology.py`, `topomem/tests/test_topo.py`

**预计时间**：3-4 天

---

### Phase 2: 图结构记忆系统（Day 7-11）

**目标**：实现带拓扑关系的结构化记忆存储。

#### 2.1 记忆节点设计 (`topomem/memory.py`)

```python
@dataclass
class MemoryNode:
    id: str
    content: str                          # 原始文本
    embedding: np.ndarray                 # 特征向量
    created_at: float                     # 创建时间
    last_accessed: float                  # 最近访问
    access_count: int                     # 访问次数
    persistence_score: float              # 拓扑持久性分数
    topological_features: dict            # H0/H1 特征标签
    metadata: dict                        # 任意元数据

class MemoryGraph:
    """基于 NetworkX 的图结构记忆"""
    
    def add_memory(content, embedding, metadata) -> MemoryNode
    def query_similar(embedding, k=5) -> List[MemoryNode]       # 向量相似度查询
    def query_topological_neighbors(node_id) -> List[MemoryNode] # 拓扑邻居
    def get_cluster(node_id) -> List[MemoryNode]                 # 所属拓扑簇
    
    def update_topology(self):
        """重新计算所有节点的拓扑关系"""
        # 1. 收集所有 embedding
        # 2. 调用 TopologyEngine.compute_persistence()
        # 3. 根据持久 H0 分支建立/更新边
        # 4. 标记每个节点的 persistence_score
    
    def prune(self, max_nodes: int):
        """移除低持久性 + 低访问量的节点"""
    
    def get_topological_summary(self) -> np.ndarray:
        """全局拓扑指纹"""
```

#### 2.2 向量索引层

- **ChromaDB** 作为向量检索后端（已安装）
- **NetworkX** 存储拓扑关系图
- 两层配合：ChromaDB 做快速 ANN 检索，NetworkX 存储拓扑结构

#### 2.3 验证实验

| 实验 | 方法 | 期望结果 |
|------|------|---------|
| 渐进记忆构建 | 逐条插入 100 条不同主题的文本 | 自动形成主题簇 |
| 拓扑引导检索 vs 纯向量检索 | 同一查询，比较两种检索的结果 | 拓扑检索返回更连贯的上下文 |
| 容量管理 | 持续插入直到需要 prune | 低持久性节点被正确移除，高持久性节点保留 |

**产出**：`topomem/memory.py`, `topomem/tests/test_memory.py`

**预计时间**：4-5 天

---

### Phase 3: 推理引擎集成（Day 12-17）

**目标**：接入小型 LLM，实现"记忆增强推理"。

#### 3.1 LLM 推理接口 (`topomem/engine.py`)

```python
class ReasoningEngine:
    """轻量推理引擎，封装小模型调用"""
    
    def __init__(self, model_path: str):
        # 加载 Qwen2.5-0.5B GGUF via llama-cpp-python
        # 或 fallback: transformers + torch CPU
    
    def generate(
        prompt: str,
        context: List[MemoryNode] = None,  # 注入的记忆上下文
        max_tokens: int = 256,
        temperature: float = 0.7
    ) -> str:
        """带记忆上下文的文本生成"""
    
    def reason(
        query: str,
        memory_graph: MemoryGraph,
        retrieval_strategy: str = "topological"  # or "vector" or "hybrid"
    ) -> ReasoningResult:
        """完整推理流程：
        1. 编码 query
        2. 从 memory_graph 检索相关记忆
        3. 构造带上下文的 prompt
        4. 生成回答
        5. 将新知识写回 memory_graph
        """
```

#### 3.2 检索策略对比

实现三种检索策略，用于后续对比实验：

| 策略 | 方法 | 预期优势 |
|------|------|---------|
| `vector` | 纯 ChromaDB 余弦相似度 top-k | 基线，速度快 |
| `topological` | 先找拓扑簇，再在簇内检索 | 上下文更连贯 |
| `hybrid` | vector top-k + 拓扑邻居扩展 | 兼顾相关性和连贯性 |

#### 3.3 验证实验

| 实验 | 方法 | 期望结果 |
|------|------|---------|
| 基础问答 | 先存入一批知识，再提问 | 能基于检索到的记忆回答 |
| 三策略对比 | 同一组问题，三种策略 | topological/hybrid 在需要多条连贯知识的问题上更优 |
| 延迟容忍测试 | 完整 query → answer 的端到端延迟 | < 30 秒 / query (CPU 限制下) |

**产出**：`topomem/engine.py`, `topomem/tests/test_engine.py`

**预计时间**：5-6 天

---

### Phase 4: 动态塑造机制（Day 18-23）

**目标**：实现推理时动态调整模型行为的机制。

> **重要调整**：在无 GPU + 0.5B 模型的约束下，传统 LoRA adapter 的动态插入不现实（模型太小，adapter 的相对开销太大）。替代方案：**Prompt Adapter Pool** — 用不同的系统提示/指令模板作为"软 adapter"，根据拓扑分析动态选择。

#### 4.1 Prompt Adapter Pool (`topomem/adapters.py`)

```python
@dataclass
class PromptAdapter:
    id: str
    name: str
    system_prompt: str          # 定制化系统提示
    domain_keywords: List[str]  # 关联的领域关键词
    topological_cluster: int    # 对应的拓扑簇 ID
    usage_count: int
    effectiveness_score: float  # 用户反馈或自评分数

class AdapterPool:
    """动态管理 prompt adapter 的创建、选择和进化"""
    
    def select_adapter(
        query_embedding: np.ndarray,
        memory_graph: MemoryGraph
    ) -> PromptAdapter:
        """根据 query 的拓扑位置选择最合适的 adapter"""
    
    def create_adapter(
        cluster_id: int,
        representative_memories: List[MemoryNode]
    ) -> PromptAdapter:
        """从拓扑簇中自动生成新 adapter"""
    
    def evolve_adapter(
        adapter: PromptAdapter,
        feedback: float
    ):
        """根据反馈调整 adapter 的 effectiveness_score"""
    
    def prune_adapters(self, min_usage: int = 3):
        """移除低效 adapter"""
```

**设计理由**：
- 0.5B 模型对 system prompt 的敏感度比大模型更高，精心设计的 prompt adapter 可以显著改变行为
- Prompt adapter 零参数开销，不增加内存
- 与拓扑簇绑定，实现"不同知识域用不同的推理策略"
- 这种机制在升级到更大模型后可以平滑过渡为真正的 LoRA adapter

#### 4.2 动态适应逻辑

从 unified-sel 的 surprise/tension 机制迁移：
- **surprise 信号**：query embedding 与最近拓扑簇中心的距离 → 高 surprise = 新领域
- **tension 信号**：最近 N 次推理的拓扑指纹变化率 → 高 tension = 知识快速变化
- 决策逻辑：
  - 低 surprise + 低 tension → 使用现有 adapter
  - 高 surprise + 低 tension → 创建新 adapter（新领域出现）
  - 任意 surprise + 高 tension → 触发记忆整理（拓扑重计算 + adapter 重评估）

#### 4.3 验证实验

| 实验 | 方法 | 期望结果 |
|------|------|---------|
| adapter 自动创建 | 交替输入两个截然不同的领域问题 | 系统自动创建 2 个 adapter |
| adapter 选择准确性 | 混合领域问题流 | 正确率 > 80% 选对 adapter |
| 与无 adapter 对比 | 同一任务，有/无 adapter pool | 多领域混合时有 adapter 更连贯 |

**产出**：`topomem/adapters.py`, `topomem/tests/test_adapters.py`

**预计时间**：5-6 天

---

### Phase 5: 自我认知与一致性守护（Day 24-28）

**目标**：实现系统的"拓扑自我意识"，防止记忆漂移。

#### 5.1 自我认知层 (`topomem/self_awareness.py`)

```python
class SelfAwareness:
    """维护系统的拓扑自我模型"""
    
    def __init__(self):
        self.fingerprint_history: List[np.ndarray] = []  # 拓扑指纹历史
        self.drift_threshold: float = 0.1                 # Wasserstein 距离阈值
        self.calibration_interval: int = 50               # 每 N 步校准一次
    
    def update_fingerprint(self, memory_graph: MemoryGraph):
        """计算当前全局拓扑指纹并追加到历史"""
        current = memory_graph.get_topological_summary()
        self.fingerprint_history.append(current)
    
    def detect_drift(self) -> DriftReport:
        """检测认知漂移
        
        方法：
        1. 计算最近 fingerprint 与历史基线的 Wasserstein 距离
        2. 计算滑动窗口内的距离变化趋势
        3. 分类：stable / drifting / restructuring
        """
    
    def calibrate(self, memory_graph: MemoryGraph, engine: ReasoningEngine):
        """一致性校准
        
        1. 从记忆中采样代表性节点
        2. 让模型基于这些记忆生成"自我描述"
        3. 比较当前自我描述与历史自我描述的一致性
        4. 如果不一致 → 标记漂移源 → 建议修复
        """
    
    def get_identity_vector(self) -> np.ndarray:
        """返回当前的"自我指纹"向量
        = 持久图中 top-K 持久特征的 (birth, death) 拼接"""
```

#### 5.2 一致性守护机制

```python
class ConsistencyGuard:
    """拦截可能破坏一致性的操作"""
    
    def should_accept_memory(
        new_memory: MemoryNode,
        memory_graph: MemoryGraph,
        self_awareness: SelfAwareness
    ) -> Tuple[bool, str]:
        """判断新记忆是否应该被接受
        
        拒绝条件：
        - 与现有高持久性记忆直接矛盾（embedding 接近但内容语义相反）
        - 会导致拓扑指纹剧烈变化（预估 Wasserstein 距离 > 阈值）
        
        返回 (accept, reason)
        """
    
    def recommend_consolidation(
        memory_graph: MemoryGraph
    ) -> List[ConsolidationAction]:
        """建议的记忆整理操作
        - 合并重复记忆
        - 强化高持久性特征
        - 移除低持久性 + 低访问量节点
        """
```

#### 5.3 验证实验

| 实验 | 方法 | 期望结果 |
|------|------|---------|
| 漂移检测 | 分三阶段输入不同领域知识，观察 drift 报告 | 阶段切换时检测到 drift |
| 一致性保护 | 输入与现有知识矛盾的内容 | ConsistencyGuard 正确标记 |
| 长期稳定性 | 300+ 轮次交互，跟踪拓扑指纹 | 指纹变化收敛，不无限漂移 |

**产出**：`topomem/self_awareness.py`, `topomem/tests/test_self.py`

**预计时间**：4-5 天

---

### Phase 6: 端到端集成 + 基准测试（Day 29-35）

**目标**：将所有组件集成为完整系统，运行对比实验。

#### 6.1 系统集成 (`topomem/system.py`)

```python
class TopoMemSystem:
    """TopoMem Reasoner v0.1 完整系统"""
    
    def __init__(self, config: TopoMemConfig):
        self.embedding = EmbeddingManager(config.embedding_model)
        self.topology = TopologyEngine(config.tda_params)
        self.memory = MemoryGraph(config.memory_params)
        self.engine = ReasoningEngine(config.model_path)
        self.adapters = AdapterPool(config.adapter_params)
        self.self_aware = SelfAwareness(config.awareness_params)
        self.guard = ConsistencyGuard(config.guard_params)
    
    def process(self, input_text: str) -> ProcessResult:
        """完整的输入处理流程"""
        # 1. 编码输入
        # 2. 选择 adapter
        # 3. 检索记忆（拓扑引导）
        # 4. 推理生成
        # 5. 新知识写入记忆（经 ConsistencyGuard 审查）
        # 6. 定期更新拓扑指纹 + 漂移检测
    
    def get_status(self) -> SystemStatus:
        """系统状态报告"""
        # 记忆节点数、拓扑簇数、adapter 数
        # 当前拓扑指纹、漂移状态
        # 资源占用
```

#### 6.2 基准测试

**Benchmark 1：知识一致性测试**
- 输入 100 条领域知识（分 5 个领域，每领域 20 条）
- 随后提出 50 个问题（混合领域）
- 测量：回答准确率、跨领域混淆率

**Benchmark 2：长期漂移测试**
- 500 轮交互，前 250 轮主要是领域 A，后 250 轮切换到领域 B
- 测量：领域 A 知识的遗忘率、拓扑指纹的变化曲线

**Benchmark 3：与基线对比**
| 系统 | 配置 |
|------|------|
| TopoMem (topological) | 完整系统，拓扑检索 |
| TopoMem (vector-only) | 同系统，但只用向量检索 |
| Naive RAG | 同模型 + ChromaDB，无拓扑层 |
| Pure LLM | 同模型，无记忆系统 |

对比指标：
- 回答准确率
- 长期一致性分数（300 轮后的自评一致性）
- 资源占用（RAM、延迟）
- 遗忘率（旧知识保持率）

**Benchmark 4：资源占用**
- 目标：总 RAM < 4 GB，单次推理延迟 < 30s

#### 6.3 验证标准（Phase 6 通过标准 = MVP 成功）

| 指标 | 目标 | 必须/期望 |
|------|------|----------|
| 系统可启动并完整运行 | 无崩溃完成 500 轮 | 必须 |
| TopoMem(topo) > TopoMem(vector) 准确率 | 差值 > 0 | 必须 |
| TopoMem(topo) > Naive RAG 一致性 | 差值 > 0 | 必须 |
| 遗忘率 < Naive RAG | TopoMem 更低 | 期望 |
| RAM 占用 | < 4 GB 总计 | 必须 |
| 单次延迟 | < 60s (CPU) | 期望 <30s |
| 漂移检测有效 | 在领域切换时触发 | 必须 |

**产出**：`topomem/system.py`, `topomem/benchmarks/`, 实验报告

**预计时间**：6-7 天

---

## 3. 项目结构

```
F:\unified-sel\
├── core/                          # 现有 unified-sel 代码（保留不动）
├── topomem/                       # 新系统
│   ├── __init__.py
│   ├── config.py                  # 全局配置
│   ├── embedding.py               # Phase 1: embedding 管理
│   ├── topology.py                # Phase 1: TDA 引擎
│   ├── memory.py                  # Phase 2: 图结构记忆
│   ├── engine.py                  # Phase 3: LLM 推理引擎
│   ├── adapters.py                # Phase 4: Prompt Adapter Pool
│   ├── self_awareness.py          # Phase 5: 自我认知
│   ├── guard.py                   # Phase 5: 一致性守护
│   ├── system.py                  # Phase 6: 集成系统
│   ├── tests/
│   │   ├── test_infra.py          # Phase 0
│   │   ├── test_topo.py           # Phase 1
│   │   ├── test_memory.py         # Phase 2
│   │   ├── test_engine.py         # Phase 3
│   │   ├── test_adapters.py       # Phase 4
│   │   ├── test_self.py           # Phase 5
│   │   └── test_integration.py    # Phase 6
│   ├── benchmarks/
│   │   ├── knowledge_consistency.py
│   │   ├── long_term_drift.py
│   │   ├── baseline_comparison.py
│   │   └── resource_usage.py
│   └── data/
│       ├── models/                # GGUF 模型文件
│       └── test_corpus/           # 测试语料
├── TOPOMEM_PLAN.md                # 本文件
└── TOPOMEM_STATUS.md              # 实施进度跟踪
```

---

## 4. 技术选型汇总

| 组件 | 选型 | 原因 | 备选 |
|------|------|------|------|
| 推理模型 | Qwen2.5-0.5B-Instruct Q4 GGUF | 0.5B 是 CPU 能跑的最大合理选择 | SmolLM2-360M, TinyLlama-1.1B |
| 模型运行时 | llama-cpp-python (CPU) | 高效 CPU 推理 | transformers + torch CPU |
| Embedding | all-MiniLM-L6-v2 (sentence-transformers) | 已安装，384 维，速度快 | bge-small-en-v1.5 |
| TDA | giotto-tda (主) + GUDHI (备) | Python 接口好，社区活跃 | ripser + persim |
| 向量存储 | ChromaDB | 已安装，轻量 | FAISS |
| 图存储 | NetworkX | 已安装，纯 Python | igraph |
| 序列化 | JSON + pickle | 简单，MVP 够用 | SQLite |

---

## 5. 风险登记表

| # | 风险 | 概率 | 影响 | 缓解策略 |
|---|------|------|------|---------|
| R1 | giotto-tda 不兼容 Python 3.14 | 高 | 中 | 备选 GUDHI 或 ripser；最坏情况手写简单 VR 复形 |
| R2 | 0.5B 模型推理能力太弱 | 高 | 高 | MVP 目标是验证架构增益，不是绝对能力；实验中与"同模型无记忆"对比即可 |
| R3 | CPU 推理延迟过长 | 中 | 中 | 用 GGUF Q4 压缩 + 限制 max_tokens；embedding 用 ONNX 加速 |
| R4 | 16GB RAM 不足 | 中 | 高 | 监控内存使用；ChromaDB 用磁盘模式；限制记忆容量上限 |
| R5 | llama-cpp-python Windows 编译失败 | 中 | 中 | 备选 ctransformers 或用 transformers 直接加载 |
| R6 | 拓扑计算开销在大记忆池中过大 | 低 | 中 | 增量计算 + 定期重计算策略；限制 MVP 记忆上限 500 节点 |
| R7 | 实验结果拓扑检索不优于向量检索 | 中 | 高 | 这是负面结果但仍有研究价值；分析原因可能指向更好的融合策略 |

---

## 6. 时间总览（已确认修订版）

```
        Day 1-2     Day 3-5     Day 6-8     Day 9-12    Day 13-15   Day 16-18   Day 19-21
        ┌───────┐   ┌───────┐   ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐
        │Phase 0│   │Phase 1│   │Phase 2 │  │Phase 3 │  │Phase 5 │  │Phase 4 │  │Phase 6 │
        │基础设施│   │Embed+ │   │图结构  │  │推理引擎│  │自我认知│  │动态塑造│  │集成+   │
        │搭建   │   │TDA引擎│   │记忆系统│  │集成    │  │守护层  │  │机制    │  │基准测试│
        └───────┘   └───────┘   └────────┘  └────────┘  └────────┘  └────────┘  └────────┘
        
        ▲ 调整：Phase 5(自我认知) 提前到 Phase 4(动态塑造) 之前
        ▲ 目标：先做扎实记忆系统 + 自我认知，再做动态塑造
        ▲ 所有接口保留未来 LoRA 适配能力
```

**总预计工期**：21 天（约 3 周），优先保证端到端可运行

---

## 7. 与最终愿景的关系

MVP v0.1 验证的核心命题：

> 在固定的小模型上，拓扑结构化记忆是否比纯向量检索提供更好的长期一致性和知识组织？

如果 MVP 证明**是**：
- Phase 7+ → 换更大模型（3B-8B，需要 GPU）
- Phase 8+ → 真正的 LoRA adapter 动态插入替代 Prompt Adapter
- Phase 9+ → 多模态特征记忆（图像/音频 embedding 也进入拓扑分析）
- Phase 10+ → AlphaEvolve 驱动的规则自动进化

如果 MVP 证明**否**：
- 分析失败原因（模型太小？拓扑粒度不对？Persistent Homology 不适合 embedding 空间？）
- 调整方向，仍然有"在 embedding 空间上做 TDA 的系统性实验数据"作为研究产出

---

## 8. 用户确认记录（2026-04-04）

1. ✅ 模型选型：Qwen2.5-0.5B Q4 GGUF 确认
2. ✅ Prompt Adapter Pool 替代 LoRA：确认，但要求保留未来 LoRA 切换接口
3. ✅ 时间线：压缩至 18-21 天
4. ✅ 阶段顺序调整：Phase 0→1→2→3→5→4→6（先做扎实记忆系统）
5. ✅ 架构确认：三层架构认可，强调"全局拓扑摘要"作为自我认知核心指纹

**状态：已确认，开始执行。**
