# SPEC_SELF_AWARENESS.md — 自我认知与一致性守护技术规格

> **对应模块**：`topomem/self_awareness.py` + `topomem/guard.py`  
> **对应阶段**：Phase 5（在本项目中提前到 Phase 3 之后执行）  
> **前置依赖**：topology.py, memory.py, embedding.py  
> **被依赖方**：system.py

---

## 1. 核心理念

自我认知层回答一个根本问题：

> **"我现在是谁？我和之前的我一样吗？"**

人类的自我认知建立在对自身认知结构的稳定感知之上。当你的世界观、知识体系、思维模式发生渐进变化时，你通常感觉不到；但如果突然、剧烈地变化，你会感到"不对劲"。

本层用**全局拓扑摘要**作为系统的自我指纹：

- 记忆图的拓扑结构（几个簇？簇之间什么关系？有环吗？）= 系统的"认知地图"
- 拓扑指纹的变化速率 = 认知漂移的度量
- 指纹的稳定核心 = 系统的身份基础

---

## 2. 全局拓扑摘要 — 核心指纹

### 2.1 指纹是什么

```
全局拓扑摘要 = TopologicalFingerprint = (100,) 维归一化向量

它编码了：
- 前 50 维：H0 的 Betti Curve（连通分支随尺度的演变）
  → "我的知识分成几个域？域之间有多远？"
- 后 50 维：H1 的 Betti Curve（环结构随尺度的演变）
  → "我的知识中有循环引用或冗余路径吗？"

指纹的含义举例：
- 指纹前半部分高且平坦 → 有多个稳定分离的知识簇
- 指纹前半部分快速下降 → 知识簇之间紧密连接
- 指纹后半部分有尖峰 → 存在环路结构（某些知识形成循环依赖）
- 指纹后半部分为零 → 知识结构是树形的，没有冗余
```

### 2.2 指纹的计算时机

```
触发条件（满足任一）：
1. MemoryGraph.update_topology() 被调用后（每 topo_recompute_interval 次插入）
2. SelfAwareness.calibrate() 被手动触发
3. ConsistencyGuard 在 should_accept_memory 中需要预估影响时

不在每次 process() 中计算（太昂贵）。
```

---

## 3. SelfAwareness 完整 API

```python
class SelfAwareness:
    """系统的自我认知模块。
    
    维护拓扑指纹的历史序列，通过 Wasserstein 距离
    检测认知漂移，通过校准流程维护一致性。
    """
    
    def __init__(self, config: SelfAwarenessConfig):
        """
        初始化：
        - self._fingerprint_history: List[TimestampedFingerprint] = []
        - self._diagram_history: List[TimestampedDiagram] = []
        - self._drift_reports: List[DriftReport] = []
        - self._step_count: int = 0
        - self._baseline_fingerprint: Optional[np.ndarray] = None
        
        config 字段：
        - fingerprint_history_size: 最多保留多少条历史指纹（默认 100）
        - drift_threshold: Wasserstein 距离超过此值视为漂移（默认 0.1）
        - calibration_interval: 每 N 步做一次完整校准（默认 50）
        - top_k_features: 身份向量取 top-K 特征（默认 10）
        """
    
    def update_fingerprint(
        self,
        memory_graph: MemoryGraph,
        topo_engine: TopologyEngine
    ) -> None:
        """记录新的拓扑指纹到历史。
        
        步骤：
        1. fingerprint = memory_graph.get_topological_summary(topo_engine)
        2. diagram = 最近一次 PH 计算的结果（从 memory_graph 缓存获取）
        3. 追加到 self._fingerprint_history（带时间戳）
        4. 追加到 self._diagram_history
        5. 如果历史超过 fingerprint_history_size，移除最旧的
        6. self._step_count += 1
        7. 如果 _baseline_fingerprint 为 None（首次）：
              self._baseline_fingerprint = fingerprint.copy()
        """
    
    def detect_drift(self) -> DriftReport:
        """检测认知漂移。
        
        漂移检测算法：
        
        1. 短期漂移（最近 vs 上一次）：
           short_drift = wasserstein_distance(
               diagram_history[-1], diagram_history[-2]
           )
        
        2. 长期漂移（最近 vs 基线）：
           long_drift = wasserstein_distance(
               diagram_history[-1], baseline_diagram
           )
        
        3. 趋势检测（最近 5 次的短期漂移序列）：
           recent_drifts = [short_drift for last 5 updates]
           trend = "accelerating" if 单调递增
                   "decelerating" if 单调递减
                   "stable" if 所有 < threshold
                   "oscillating" otherwise
        
        4. 状态判定：
           if short_drift < threshold * 0.5 and long_drift < threshold:
               status = "stable"       # 系统稳定
           elif short_drift > threshold:
               status = "drifting"      # 正在漂移（需要警惕）
           elif long_drift > threshold * 2:
               status = "restructured"  # 已经发生结构性变化（可能是正常的领域切换）
           else:
               status = "evolving"      # 缓慢演变（正常）
        
        返回 DriftReport 对象。
        """
    
    def get_identity_vector(self) -> np.ndarray:
        """返回当前的身份向量。
        
        身份向量 = 最近拓扑指纹中 top-K 最持久特征的 (birth, death) 拼接。
        
        与 topological_summary 的区别：
        - topological_summary (Betti Curve) 编码全局统计信息
        - identity_vector 编码最显著的个体特征
        - 前者用于漂移检测（连续变化度量）
        - 后者用于身份识别（"我有哪些核心知识簇？"）
        
        实现：
        1. 从最近的 diagram 中取 H0 + H1 的所有特征
        2. 按 persistence 降序排序
        3. 取 top-K 个
        4. 拼接 (birth_1, death_1, birth_2, death_2, ...) → (2*K,) 向量
        5. 不足 K 个时补 0
        
        输出形状：(2 * config.top_k_features,) = (20,)
        """
    
    def calibrate(
        self,
        memory_graph: MemoryGraph,
        topo_engine: TopologyEngine,
        engine: Optional[ReasoningEngine] = None
    ) -> CalibrationReport:
        """完整的自我校准流程。
        
        步骤：
        
        1. 强制重计算拓扑：
           memory_graph.update_topology(topo_engine)
           self.update_fingerprint(memory_graph, topo_engine)
        
        2. 漂移检测：
           drift = self.detect_drift()
        
        3. 结构性分析：
           n_clusters = len(set(node.cluster_id for node in graph))
           cluster_sizes = Counter(node.cluster_id for node in graph)
           cluster_balance = std(cluster_sizes.values()) / mean(cluster_sizes.values())
        
        4. 记忆健康度：
           avg_persistence = mean(node.persistence_score for node in graph)
           orphan_ratio = count(node.cluster_id == -1) / total_nodes
        
        5.（可选，如果有 engine）自我描述一致性：
           a. 从每个簇中采样 1 条代表性记忆
           b. 让 engine 生成"系统能力自述"
           c. 与上次校准的自述做 embedding 相似度
           d. 相似度 < 0.7 → 标记自述不一致
        
        返回 CalibrationReport 对象。
        """
    
    def should_calibrate(self) -> bool:
        """判断是否需要校准。
        
        触发条件（满足任一）：
        1. step_count % calibration_interval == 0
        2. 最近一次 detect_drift() 返回 "drifting"
        3. 从未校准过
        """
    
    # =========== 序列化 ===========
    
    def save(self, path: str) -> None:
        """保存指纹历史和配置。"""
    
    def load(self, path: str) -> None:
        """加载指纹历史。"""
```

---

## 4. ConsistencyGuard 完整 API

```python
class ConsistencyGuard:
    """一致性守护。在记忆写入前做预检，防止有害变更。
    
    设计哲学：
    - 宁可误拒（false reject）也不误入（false accept）
    - 但不能太严格导致系统无法学习新知识
    - 所有拒绝都返回原因，方便调试和人工覆盖
    """
    
    def __init__(self, config: SelfAwarenessConfig):
        """
        配置：复用 SelfAwarenessConfig 的 drift_threshold。
        """
    
    def should_accept_memory(
        self,
        new_content: str,
        new_embedding: np.ndarray,
        memory_graph: MemoryGraph,
        self_awareness: SelfAwareness,
        topo_engine: TopologyEngine
    ) -> Tuple[bool, str]:
        """判断是否接受新记忆。
        
        检查清单：
        
        检查 1: 重复检测
        - 在 memory_graph 中检索与 new_embedding 最相似的 1 条
        - 如果 similarity > 0.95 → 拒绝（"duplicate: similarity {sim} with node {id}"）
        
        检查 2: 矛盾检测（简化版）
        - 检索 top-3 最相似的记忆
        - 用 embedding 距离 + 简单规则判断是否矛盾：
          - 如果 similarity > 0.8 但 new_content 包含否定词（not, never, false, 不, 非, 没有）
            而最相似记忆不包含 → 标记为"potential contradiction"
          - 不阻塞（只是 warning），accept=True 但 reason 中说明
        
        检查 3: 拓扑稳定性预估（可选，计算较贵）
        - 如果 self_awareness 最近的 drift_status == "drifting"：
          → 提高警惕，要求 similarity > 0.3 才接受（必须与某些已有知识相关）
          → 避免在漂移期间引入完全无关的噪声知识
        
        检查 4: 容量检查
        - 如果 memory_graph.node_count() >= config.max_nodes:
          → accept=True 但建议 prune
          → reason = "at capacity, prune recommended"
        
        返回 (accept, reason)
        """
    
    def recommend_consolidation(
        self,
        memory_graph: MemoryGraph,
        topo_engine: TopologyEngine
    ) -> List[ConsolidationAction]:
        """建议的记忆整理操作。
        
        扫描策略：
        
        1. 合并候选：
           - 找到 similarity > 0.9 的节点对
           - 建议合并（保留 access_count 更高的，合并 content）
        
        2. 强化候选：
           - 找到 persistence_score > 90th percentile 的节点
           - 建议标记为"核心知识"（不可被 prune）
        
        3. 清理候选：
           - 找到 importance_score < 10th percentile 的节点
           - 建议删除
        
        4. 孤儿处理：
           - 找到 cluster_id == -1 的节点
           - 建议重新分配或删除
        
        返回 List[ConsolidationAction]，每个包含 action_type, node_ids, reason
        """
```

---

## 5. 数据类型定义

```python
@dataclass
class TimestampedFingerprint:
    timestamp: float
    step: int
    fingerprint: np.ndarray  # (100,)

@dataclass
class TimestampedDiagram:
    timestamp: float
    step: int
    diagram: PersistenceDiagram

@dataclass
class DriftReport:
    timestamp: float
    step: int
    short_drift: float        # 与上次的 Wasserstein 距离
    long_drift: float         # 与基线的 Wasserstein 距离
    trend: str                # "accelerating" | "decelerating" | "stable" | "oscillating"
    status: str               # "stable" | "evolving" | "drifting" | "restructured"
    n_clusters: int           # 当前簇数
    message: str              # 人可读的描述

@dataclass
class CalibrationReport:
    timestamp: float
    drift: DriftReport
    n_clusters: int
    cluster_sizes: Dict[int, int]
    cluster_balance: float     # 0=完全平衡, 越大越不平衡
    avg_persistence: float
    orphan_ratio: float        # 未分配簇的节点比例
    self_description_consistency: Optional[float]  # 与上次自述的相似度
    recommendations: List[str]  # 建议操作列表

@dataclass
class ConsolidationAction:
    action_type: str           # "merge" | "strengthen" | "remove" | "reassign"
    node_ids: List[str]
    reason: str
```

---

## 6. 漂移检测算法可视化

```
时间轴：
  t0     t1     t2     t3     t4     t5     t6     t7
  │      │      │      │      │      │      │      │
  ▼      ▼      ▼      ▼      ▼      ▼      ▼      ▼
  FP₀    FP₁    FP₂    FP₃    FP₄    FP₅    FP₆    FP₇
  
  short_drift(t7) = Wasserstein(Diag₇, Diag₆)
  long_drift(t7)  = Wasserstein(Diag₇, Diag₀)  ← Diag₀ 是基线
  
  recent_drifts = [sd(t3), sd(t4), sd(t5), sd(t6), sd(t7)]
  
  状态判定矩阵：
  ┌────────────────┬──────────────────┬──────────────────┐
  │                │ long_drift < T   │ long_drift > 2T  │
  ├────────────────┼──────────────────┼──────────────────┤
  │ short < T/2    │ STABLE           │ RESTRUCTURED     │
  │ short > T      │ DRIFTING         │ DRIFTING         │
  │ T/2<short<T    │ EVOLVING         │ EVOLVING         │
  └────────────────┴──────────────────┴──────────────────┘
  
  T = config.drift_threshold (默认 0.1)
```

---

## 7. 校准流程时序图

```
system.py                   self_awareness.py             memory_graph         topo_engine
    │                              │                          │                    │
    │  should_calibrate()?         │                          │                    │
    │─────────────────────────────>│                          │                    │
    │  True                        │                          │                    │
    │<─────────────────────────────│                          │                    │
    │                              │                          │                    │
    │  calibrate(graph, topo, eng) │                          │                    │
    │─────────────────────────────>│                          │                    │
    │                              │  update_topology(topo)   │                    │
    │                              │─────────────────────────>│                    │
    │                              │                          │ compute_persistence│
    │                              │                          │───────────────────>│
    │                              │                          │  diagram           │
    │                              │                          │<───────────────────│
    │                              │  TopologyResult          │                    │
    │                              │<─────────────────────────│                    │
    │                              │                          │                    │
    │                              │  update_fingerprint()    │                    │
    │                              │─── (内部) ──────────────>│                    │
    │                              │                          │                    │
    │                              │  detect_drift()          │                    │
    │                              │─── (内部) ───>           │                    │
    │                              │  DriftReport             │                    │
    │                              │                          │                    │
    │  CalibrationReport           │                          │                    │
    │<─────────────────────────────│                          │                    │
```

---

## 8. 单元测试清单

```python
# test_self.py

# === SelfAwareness 测试 ===

def test_initial_fingerprint():
    """首次 update 后应设置 baseline_fingerprint"""

def test_fingerprint_history_size():
    """历史超过 max 时应丢弃最旧的"""

def test_detect_drift_stable():
    """相同数据多次更新 → status=stable"""

def test_detect_drift_drifting():
    """突然插入大量新领域数据 → status=drifting"""

def test_detect_drift_restructured():
    """完全替换数据后 → status=restructured"""

def test_identity_vector_shape():
    """应返回 (20,) 向量"""

def test_identity_vector_stability():
    """小变化不应大幅改变身份向量"""

def test_calibrate_returns_report():
    """calibrate 应返回完整的 CalibrationReport"""

def test_should_calibrate_interval():
    """达到 calibration_interval 时应返回 True"""

def test_save_load_roundtrip():
    """save 后 load，指纹历史应完整恢复"""

# === ConsistencyGuard 测试 ===

def test_reject_duplicate():
    """similarity > 0.95 的重复内容应被拒绝"""

def test_accept_new_knowledge():
    """全新领域的知识应被接受"""

def test_warn_contradiction():
    """潜在矛盾应 accept=True 但 reason 包含 warning"""

def test_drifting_state_raises_threshold():
    """漂移状态下应拒绝完全无关的知识"""

def test_capacity_warning():
    """满容量时应建议 prune"""

def test_consolidation_merge():
    """高相似度节点对应被建议合并"""

def test_consolidation_strengthen():
    """高持久性节点应被建议强化"""

def test_consolidation_remove():
    """低重要性节点应被建议删除"""
```
