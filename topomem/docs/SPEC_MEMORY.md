# SPEC_MEMORY.md — 图结构记忆系统技术规格

> **对应模块**：`topomem/memory.py`  
> **对应阶段**：Phase 2  
> **前置依赖**：embedding.py, topology.py, networkx, chromadb  
> **被依赖方**：engine.py, self_awareness.py, guard.py, adapters.py, system.py

---

## 1. 模块职责

MemoryGraph 是整个系统的**知识仓库**。它做三件事：

1. **存储**：将知识（文本 + embedding + 元数据）持久化为图节点
2. **组织**：利用 TopologyEngine 的 PH 结果建立节点间的拓扑关系（簇、持久性分数）
3. **检索**：支持三种策略（vector / topological / hybrid），供推理引擎调用

**核心设计决策**：双层索引
- **ChromaDB**：负责向量 ANN 检索（速度快，O(log N)）
- **NetworkX**：负责拓扑关系图（簇归属、邻居查询、全局结构）
- 两层通过共享的 `node_id` 保持同步

---

## 2. 数据结构

### 2.1 MemoryNode

```python
@dataclass
class MemoryNode:
    """一个记忆节点 = 一条知识 + 其结构化标注"""
    
    # === 核心字段（创建时确定）===
    id: str                    # UUID，全局唯一
    content: str               # 原始文本内容
    embedding: np.ndarray      # 384 维特征向量（由 EmbeddingManager 生成）
    created_at: float          # 创建时间戳（time.time()）
    metadata: dict             # 任意用户元数据（来源、类型、标签等）
    
    # === 访问统计（动态更新）===
    last_accessed: float       # 最近一次被检索到的时间戳
    access_count: int          # 被检索到的总次数
    
    # === 拓扑标注（由 update_topology 更新）===
    cluster_id: int            # 所属拓扑簇 ID（-1 = 未分配）
    persistence_score: float   # 该节点在其所属簇中的拓扑持久性
                               # 越高 = 越是该簇的"核心成员"
    
    # === 衍生 ===
    importance_score: float    # 综合重要性 = f(persistence, access_count, recency)
                               # 用于 prune 决策
```

### 2.2 重要性分数计算

```python
def compute_importance(node: MemoryNode, current_time: float) -> float:
    """综合重要性 = 拓扑持久性 × 访问频率 × 时间衰减
    
    importance = (
        0.5 * persistence_score_normalized +     # 拓扑维度：越持久越重要
        0.3 * log(1 + access_count) / log(1 + max_access) +  # 访问维度
        0.2 * exp(-decay * (current_time - last_accessed))    # 时间衰减维度
    )
    
    权重 (0.5, 0.3, 0.2) 的设计理由：
    - 拓扑持久性是本系统的核心差异，赋予最高权重
    - 访问频率是实际使用信号
    - 时间衰减防止古老且不再有用的知识霸占位置
    
    decay 参数：建议 0.001 (每 1000 秒 / ~17 分钟衰减到 e^-1)
    """
```

---

## 3. MemoryGraph 完整 API

```python
class MemoryGraph:
    """带拓扑关系的图结构记忆系统。"""
    
    def __init__(self, config: MemoryConfig, embedding_mgr: EmbeddingManager):
        """
        初始化：
        1. 创建 NetworkX 图 (self._graph = nx.Graph())
        2. 连接 ChromaDB (persistent client, persist_dir=config.chroma_persist_dir)
        3. 创建或获取 ChromaDB collection ("topomem_memory")
        4. 保存 config 和 embedding_mgr 引用
        5. 初始化计数器 self._inserts_since_topo_update = 0
        """
    
    # =========== 写入 ===========
    
    def add_memory(
        self, 
        content: str, 
        embedding: np.ndarray, 
        metadata: Optional[dict] = None
    ) -> MemoryNode:
        """添加一条记忆。
        
        步骤：
        1. 生成 UUID
        2. 创建 MemoryNode（cluster_id=-1, persistence_score=0）
        3. 添加到 NetworkX 图（节点属性 = MemoryNode 全部字段）
        4. 添加到 ChromaDB（id, embedding, content 作为 document）
        5. self._inserts_since_topo_update += 1
        6. 如果 _inserts_since_topo_update >= config.topo_recompute_interval:
              self.update_topology(topo_engine)  ← 需要传入或在 init 时绑定
              self._inserts_since_topo_update = 0
        7. 返回 MemoryNode
        
        注意：不需要调用方提供 embedding，但 ARCHITECTURE.md 中定义了
        调用方（system.py）负责先编码再传入。这样 memory.py 不依赖 embedding.py。
        但考虑到便利性，同时提供一个 add_memory_from_text() 的封装。
        """
    
    def add_memory_from_text(
        self, 
        content: str, 
        metadata: Optional[dict] = None
    ) -> MemoryNode:
        """便利方法：自动编码文本后添加。
        依赖 self._embedding_mgr。
        """
    
    # =========== 检索 ===========
    
    def retrieve(
        self,
        query_embedding: np.ndarray,
        strategy: str = "hybrid",    # "vector" | "topological" | "hybrid"
        k: int = 5
    ) -> List[MemoryNode]:
        """根据策略检索相关记忆。
        
        三种策略详细实现见下方 §4。
        
        所有策略最终都返回 List[MemoryNode]，按相关性降序排列。
        每个被返回的节点的 access_count += 1, last_accessed = now。
        """
    
    def retrieve_by_cluster(self, cluster_id: int) -> List[MemoryNode]:
        """返回指定拓扑簇中的所有记忆。"""
    
    def get_cluster_centers(self) -> Dict[int, np.ndarray]:
        """返回每个簇的中心 embedding（= 簇内所有节点 embedding 的均值）。"""
    
    # =========== 拓扑管理 ===========
    
    def update_topology(self, topo_engine: TopologyEngine) -> TopologyResult:
        """重新计算所有节点的拓扑关系。
        
        步骤：
        1. 收集所有节点的 embedding → (N, 384) 矩阵
        2. 调用 topo_engine.compute_persistence(points)
        3. 调用 topo_engine.extract_persistent_features(diagram)
        4. 调用 topo_engine.cluster_labels_from_h0(diagram, points)
        5. 更新每个节点的 cluster_id 和 persistence_score:
           - cluster_id = labels[i]
           - persistence_score = 该节点所属 H0 分支的 persistence 值
             （同一簇内的所有节点共享相同的 persistence_score）
        6. 更新 NetworkX 图中的边：
           - 清除所有旧边
           - 同一 cluster_id 内的节点两两建立边
             （如果 N 太大，改为每个节点只连接簇内 k-nearest 邻居）
        7. 计算并返回 TopologyResult
        
        性能：
        - 这是最昂贵的操作（O(N^2) 距离矩阵 + O(N^3) PH）
        - 只在每 topo_recompute_interval 次插入后触发
        - 或被 SelfAwareness 手动触发
        """
    
    def get_topological_summary(
        self, 
        topo_engine: TopologyEngine
    ) -> TopologicalFingerprint:
        """返回当前记忆图的全局拓扑指纹。
        
        如果最近已计算过（缓存有效），直接返回缓存。
        否则调用 update_topology 后返回。
        """
    
    # =========== 容量管理 ===========
    
    def prune(self, max_nodes: Optional[int] = None) -> List[str]:
        """移除低重要性节点，释放容量。
        
        策略：
        1. 如果 node_count <= max_nodes → 不做任何操作
        2. 计算所有节点的 importance_score
        3. 按 importance 升序排序
        4. 移除最低的 (node_count - max_nodes) 个节点
        5. 从 NetworkX 和 ChromaDB 中同时删除
        6. 返回被删除的 node_id 列表
        
        保护规则：
        - 每个 cluster 至少保留 1 个节点（防止整个知识簇消失）
        - 最近 10 步内创建的节点不删除（新知识保护期）
        """
    
    def node_count(self) -> int:
        """当前节点总数。"""
    
    # =========== 序列化 ===========
    
    def save(self, path: str) -> None:
        """持久化到磁盘。
        
        保存内容：
        - NetworkX 图 → JSON (node_link_data)
        - ChromaDB 已由 persistent client 自动持久化
        - 元数据（inserts_since_update, topology 缓存）→ JSON
        
        embeddings 存储策略：
        - embedding 同时存在于 NetworkX 节点属性和 ChromaDB 中
        - 保存时 NetworkX 的 embedding 转为 list（JSON 序列化）
        - 加载时转回 np.ndarray
        """
    
    def load(self, path: str) -> None:
        """从磁盘加载。"""
```

---

## 4. 三种检索策略详细设计

### 4.1 Vector 检索（基线）

```
输入：query_embedding, k=5

1. 调用 ChromaDB collection.query(
       query_embeddings=[query_embedding.tolist()],
       n_results=k
   )
2. 通过返回的 id 从 NetworkX 图中获取完整 MemoryNode
3. 按 ChromaDB 返回的距离排序（已默认排序）

优点：快速，O(log N)
缺点：只看局部相似度，不考虑知识簇的全局结构
```

### 4.2 Topological 检索

```
输入：query_embedding, k=5

1. 获取所有 cluster 的中心 embedding（get_cluster_centers）
2. 计算 query 与每个簇中心的余弦距离
3. 选择最近的 top-1 或 top-2 个簇
4. 在选中的簇内，按余弦相似度排序，取 top-k 个节点
5. 返回这些节点

优点：返回的上下文来自同一知识簇，更连贯
缺点：如果 query 跨越多个簇，可能错过某些相关信息
使用场景：需要深度理解某一领域的问题
```

### 4.3 Hybrid 检索（推荐默认）

```
输入：query_embedding, k=5

1. Vector 检索得到 vector_results (top-k)
2. Topological 检索得到 topo_results (top-k)
3. 合并去重（用 node_id）
4. 重新排序：
   score = 0.6 * vector_similarity + 0.4 * topological_coherence
   
   其中 topological_coherence = 
     如果节点与 vector_results[0] 在同一簇 → 1.0
     如果节点与 vector_results[0] 的簇 persistence 差异小 → 0.5
     否则 → 0.0
   
5. 取合并后的 top-k 返回

优点：兼顾相关性和连贯性
缺点：计算略贵（两次检索）
```

---

## 5. ChromaDB 与 NetworkX 同步协议

```
不变量（invariant）：
  ChromaDB collection 中的 id 集合 == NetworkX 图的节点 id 集合

同步操作：
  add_memory:     同时写入两者
  prune/delete:   同时从两者删除
  update_topology: 只修改 NetworkX（ChromaDB 中不存储拓扑信息）

冲突处理：
  如果发现两者不一致（启动时检查）：
  - 以 ChromaDB 为准（因为它有持久化）
  - 重建 NetworkX 图
```

---

## 6. 性能约束

| 操作 | 目标延迟 | 节点数条件 |
|------|---------|-----------|
| add_memory | < 10ms | 任意 |
| retrieve (vector) | < 50ms | N ≤ 500 |
| retrieve (topological) | < 100ms | N ≤ 500 |
| retrieve (hybrid) | < 150ms | N ≤ 500 |
| update_topology | < 5s | N ≤ 500 |
| prune | < 100ms | 任意 |
| save/load | < 2s | N ≤ 500 |

---

## 7. 边界情况与错误处理

| 场景 | 处理方式 |
|------|---------|
| 空图（0 节点）时检索 | 返回空列表 |
| 只有 1-2 个节点时 update_topology | 跳过 PH 计算，所有节点 cluster_id=0 |
| retrieve 的 k > 节点总数 | 返回全部节点 |
| content 为空字符串 | 允许（但记录 warning）|
| embedding 维度不是 384 | 抛出 ValueError |
| ChromaDB 持久化目录不存在 | 自动创建 |
| 重复 content | 允许（不同的 node_id，embedding 相同）|

---

## 8. 单元测试清单

```python
# test_memory.py

def test_add_and_count():
    """添加 5 条记忆，node_count() 应为 5"""

def test_add_from_text():
    """用 add_memory_from_text 添加文本，验证 embedding 自动生成"""

def test_retrieve_vector():
    """插入 10 条，vector 检索应返回最相关的"""

def test_retrieve_topological():
    """插入 3 组不同主题各 5 条，拓扑检索应返回同组的"""

def test_retrieve_hybrid():
    """hybrid 应兼具 vector 和 topo 的优点"""

def test_update_topology_clusters():
    """3 组明显分离的数据，update_topology 应产生 3 个簇"""

def test_prune_removes_low_importance():
    """prune 应移除 importance 最低的节点"""

def test_prune_protects_clusters():
    """prune 后每个 cluster 至少保留 1 个节点"""

def test_access_count_update():
    """每次被 retrieve 返回后 access_count 应 +1"""

def test_save_and_load():
    """save 后 load，所有数据应完整恢复"""

def test_chromadb_networkx_sync():
    """删除节点后两个存储应保持一致"""

def test_topological_summary():
    """get_topological_summary 应返回 (100,) 归一化向量"""

def test_empty_graph_retrieve():
    """空图检索应返回空列表而非报错"""
```
