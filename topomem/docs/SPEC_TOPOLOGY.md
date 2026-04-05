# SPEC_TOPOLOGY.md — 拓扑分析引擎技术规格

> **对应模块**：`topomem/topology.py`  
> **对应阶段**：Phase 1  
> **前置依赖**：numpy, scipy, ripser, persim, gudhi（备选）  
> **被依赖方**：memory.py, self_awareness.py, guard.py

---

## 1. 模块职责

TopologyEngine 是整个系统的**数学核心**。它的唯一职责是：

> 给定高维空间中的一组点（embedding 向量），提取其**拓扑不变量**——即在噪声和连续变形下仍然稳定的几何结构信息。

为什么需要它（而非只用向量相似度）：
- 余弦相似度只能判断"两点是否接近"
- PH 能回答"这组点形成了几个天然的簇？簇有多稳定？是否存在环路（冗余关系）？"
- 这种**全局结构信息**是记忆组织和自我认知的基础

---

## 2. 核心概念速查

| 概念 | 含义 | 在本系统中的用途 |
|------|------|----------------|
| Persistent Homology (PH) | 跟踪拓扑特征随尺度变化的出现和消失 | 识别记忆中的稳定结构 |
| H0 | 连通分支数 | 记忆中有多少个知识簇 |
| H1 | 1 维环 | 记忆中是否有循环/冗余依赖 |
| Persistence | death - birth | 特征的稳定程度（越大越真实） |
| Persistence Diagram | (birth, death) 散点图 | PH 的标准输出格式 |
| Wasserstein Distance | 两个持久图之间的最优传输距离 | 度量拓扑变化程度（漂移检测） |
| Topological Fingerprint | 持久图的固定长度向量化 | 自我认知的核心指纹 |

---

## 3. 完整 API 规格

```python
class TopologyEngine:
    """拓扑数据分析引擎。
    
    基于 Persistent Homology，从高维点云中提取
    稳定的拓扑特征。
    
    主要库选择：
    - ripser（主选）：速度快，API 简洁，专注 VR 复形
    - gudhi（备选）：功能更全，支持更多复形类型
    """
    
    def __init__(self, config: TopologyConfig):
        """
        参数说明：
        - config.max_homology_dim: 计算到 H 几（默认 1，即 H0+H1）
        - config.persistence_threshold: 持久性过滤阈值
          - None → 自动使用 median(persistence) 作为阈值
          - float → 固定阈值
        - config.metric: 距离度量（默认 "euclidean"）
        """
    
    def compute_persistence(
        self,
        points: np.ndarray,      # shape: (N, D), N 个 D 维点
    ) -> PersistenceDiagram:
        """计算 Persistent Homology。
        
        实现步骤：
        1. 调用 ripser(points, maxdim=self.max_homology_dim)
        2. 返回 result['dgms']，即 List[np.ndarray]
           - dgms[0]: H0 diagram, shape (n, 2)
           - dgms[1]: H1 diagram, shape (m, 2)
        
        注意事项：
        - H0 中会有一个 (birth=0, death=inf) 的点，表示全局连通分支
          → 在后续处理中需要过滤掉 death=inf 的点
        - 当 N < 3 时，跳过计算，返回空 diagram
        - 当 N > 500 时，记录警告（计算复杂度 O(N^3)）
        
        性能约束：
        - N=100: < 0.1s
        - N=500: < 5s
        - N=1000: 不建议（可能 > 30s）
        """
    
    def extract_persistent_features(
        self,
        diagram: PersistenceDiagram,
        threshold: Optional[float] = None
    ) -> List[TopologicalFeature]:
        """从持久图中提取有意义的拓扑特征。
        
        过滤策略：
        1. 移除 death=inf 的点（H0 中的全局连通分支）
        2. 计算所有点的 persistence = death - birth
        3. 如果 threshold 为 None：
           - 计算 median_persistence = median(所有 persistence)
           - 阈值 = max(median_persistence, 0.01)  # 防止全 0
        4. 保留 persistence > 阈值 的特征
        5. 按 persistence 降序排序
        
        返回 List[TopologicalFeature]，每个包含：
        - dimension: 0 或 1
        - birth, death, persistence
        - representative: 构成该特征的点索引（如果 ripser 支持）
        
        边界情况：
        - 如果过滤后为空 → 返回空列表（说明数据中没有显著拓扑结构）
        - 如果所有 persistence 相同 → 全部保留或全部丢弃取决于阈值
        """
    
    def wasserstein_distance(
        self,
        diag_a: PersistenceDiagram,
        diag_b: PersistenceDiagram,
        dim: int = 0
    ) -> float:
        """计算两个持久图在指定维度上的 Wasserstein 距离。
        
        实现：
        1. 使用 persim.wasserstein(diag_a[dim], diag_b[dim])
        2. 过滤掉 death=inf 的点后再计算
        
        用途：
        - 自我认知层用来度量"拓扑结构变化了多少"
        - 距离越大 → 记忆结构变化越剧烈 → 可能发生漂移
        
        返回：float >= 0
        - 0 = 完全相同的拓扑结构
        - 越大 = 差异越大
        
        注意：
        - 如果某个 diagram 为空，返回对方所有点到对角线距离之和
        - 两个都为空 → 返回 0
        """
    
    def topological_summary(
        self,
        diagram: PersistenceDiagram,
        method: str = "betti_curve"
    ) -> TopologicalFingerprint:
        """将持久图转为固定长度向量（拓扑指纹）。
        
        这是自我认知的核心：一个固定长度的向量，编码了整个记忆图的拓扑结构。
        
        方法选择：
        
        方法 1: "betti_curve"（推荐，MVP 使用）
        - 原理：在 filtration 轴上均匀采样 K 个点，统计每个点处存活的特征数
        - 步骤：
          1. 确定 filtration 范围 [0, max_death]（排除 inf）
          2. 均匀采样 K=50 个 filtration 值 t_1, ..., t_K
          3. 对每个 t_i，统计 birth <= t_i < death 的特征数 → betti_number(t_i)
          4. 对 H0 和 H1 分别计算 → 拼接为 (2*K,) = (100,) 维向量
        - 优点：直观、稳定、快速
        - 输出形状：(100,) 
        
        方法 2: "persistence_landscape"（未来可选）
        - 更精细但计算更重
        
        方法 3: "top_k_features"（简单备选）
        - 取 top-K 个最持久特征的 (birth, death) → (2*K,) 向量
        - 不足 K 个时补 0
        - 输出形状：(2 * top_k_features,)
        
        归一化：
        - 对输出向量做 L2 归一化，使得不同规模的记忆图的指纹可以直接用余弦相似度比较
        """
    
    def cluster_labels_from_h0(
        self,
        diagram: PersistenceDiagram,
        points: np.ndarray,
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """从 H0 持久分支推导聚类标签。
        
        原理：
        - H0 的每个 (birth, death) 对应一个连通分支的合并事件
        - persistence > threshold 的分支 = "真实的"独立簇
        - 其余 = 被合并到更大簇中
        
        实现策略（基于 single-linkage 等价性）：
        1. 计算点之间的 pairwise 距离矩阵
        2. 做 single-linkage 层次聚类
        3. 在 threshold 对应的距离处切割树 → 得到簇标签
        
        注意：Vietoris-Rips H0 等价于 single-linkage 聚类，
        所以可以直接用 scipy.cluster.hierarchy 实现。
        
        输入：
        - diagram: compute_persistence 的输出
        - points: 原始点云 (N, D)
        - threshold: 切割阈值（None 则自动选择）
        
        输出：np.ndarray, shape (N,)，每个点的簇标签（从 0 开始）
        
        自动阈值选择：
        - 取 H0 中 persistence 的 gap（排序后相邻 persistence 差值最大处）
        - 该 gap 对应的 death 值就是最佳切割距离
        """
```

---

## 4. 数据类型定义

```python
from dataclasses import dataclass
from typing import List, Optional

# 持久图类型
PersistenceDiagram = List[np.ndarray]
# dgms[0]: H0, shape (n, 2)
# dgms[1]: H1, shape (m, 2)
# 每行 = (birth, death)

# 拓扑指纹类型
TopologicalFingerprint = np.ndarray  # shape: (100,) for betti_curve, L2 归一化

@dataclass
class TopologicalFeature:
    """一个拓扑特征（一个持久的 hole）"""
    dimension: int          # 0=连通分支, 1=环
    birth: float            # 出现的 filtration 值
    death: float            # 消失的 filtration 值
    persistence: float      # = death - birth，稳定程度
    representative: Optional[List[int]] = None  # 构成该特征的点索引

@dataclass
class TopologyResult:
    """compute_persistence 的完整返回（可选的富结果）"""
    diagram: PersistenceDiagram
    features: List[TopologicalFeature]
    fingerprint: TopologicalFingerprint
    n_clusters: int                    # H0 持久分支数（= 有意义的簇数）
    cluster_labels: Optional[np.ndarray] = None  # (N,) 聚类标签
```

---

## 5. 算法细节

### 5.1 Vietoris-Rips 复形构建

```
输入：N 个点，每个 D 维（在本系统中 D=384）

1. 计算 N×N 的 pairwise 距离矩阵（euclidean）
2. ripser 内部使用 clearing optimization 避免显式构建完整复形
3. 时间复杂度：O(N^2) 距离矩阵 + O(N^3) 最坏情况 PH 计算
4. 实际性能：ripser 有大量优化，500 点通常 < 5s
```

### 5.2 持久性阈值自动选择

```
给定所有特征的 persistence 值 P = [p_1, p_2, ..., p_k]

方法 1: Median（默认）
  threshold = median(P)
  简单鲁棒，但对极端分布可能不好

方法 2: Gap（推荐用于 cluster_labels_from_h0）
  1. 将 P 排序
  2. 计算相邻差值 gaps = [P[i+1] - P[i]]
  3. threshold = P[argmax(gaps)]
  意义：persistence 值的最大跳跃点分隔"噪声"和"信号"

方法 3: Percentile
  threshold = percentile(P, 75)
  只保留最持久的 25% 特征
```

### 5.3 Betti Curve 计算

```
输入：PersistenceDiagram, K=50 采样点

对 H0:
  1. max_death_h0 = max(death for (b,d) in dgm[0] if d != inf)
  2. t_values = linspace(0, max_death_h0, K)
  3. for each t in t_values:
       betti_0(t) = count of features where birth <= t < death
  4. betti_curve_h0 = [betti_0(t_1), ..., betti_0(t_K)]

对 H1:
  同理计算 betti_curve_h1

拼接：
  fingerprint = concat(betti_curve_h0, betti_curve_h1)  # (2K,) = (100,)

归一化：
  fingerprint = fingerprint / (||fingerprint|| + 1e-10)
```

---

## 6. 错误处理

| 场景 | 处理方式 |
|------|---------|
| N < 3 个点 | 返回空 PersistenceDiagram（两个空 ndarray）|
| N > 500 | 记录 warning，正常计算（不阻塞）|
| 所有点相同 | ripser 会返回一个 H0 点 (0, inf)，其余为空 |
| NaN/Inf 在输入中 | 抛出 ValueError("points contains NaN or Inf") |
| ripser 崩溃 | 捕获异常，fallback 到 gudhi 实现 |
| persim wasserstein 崩溃 | 捕获异常，返回 float('inf') |

---

## 7. 单元测试清单

```python
# test_topology.py 应包含以下测试用例：

def test_empty_points():
    """0 个或 1 个点 → 返回空 diagram"""

def test_two_points():
    """2 个点 → H0 有一个 merge 事件"""

def test_three_clusters():
    """3 组明显分开的点 → H0 应有 3 个高 persistence 特征"""

def test_circle():
    """圆上的点 → H1 应有一个高 persistence 环"""

def test_noise_robustness():
    """同一数据加不同程度噪声 → 高 persistence 特征应稳定"""

def test_wasserstein_self_zero():
    """同一 diagram 的 Wasserstein 距离应为 0"""

def test_wasserstein_different():
    """不同 diagram 的距离应 > 0"""

def test_fingerprint_deterministic():
    """同一数据两次计算的指纹应完全相同"""

def test_fingerprint_shape():
    """指纹维度应为 (100,) 且 L2 归一化"""

def test_cluster_labels_count():
    """3 组分开的点 → 应返回 3 种标签"""

def test_persistence_threshold_auto():
    """自动阈值应在合理范围内"""

def test_gudhi_fallback():
    """模拟 ripser 失败 → 应 fallback 到 gudhi"""
```

---

## 8. 与其他模块的交互

### 被 memory.py 调用

```python
# 在 MemoryGraph.update_topology() 中：
embeddings = self._collect_all_embeddings()       # (N, 384)
diagram = topo_engine.compute_persistence(embeddings)
features = topo_engine.extract_persistent_features(diagram)
labels = topo_engine.cluster_labels_from_h0(diagram, embeddings)
fingerprint = topo_engine.topological_summary(diagram)

# 用 labels 更新每个节点的 cluster_id
# 用 features 更新每个节点的 persistence_score
# 用 fingerprint 传递给 SelfAwareness
```

### 被 self_awareness.py 调用

```python
# 在 SelfAwareness.update_fingerprint() 中：
current_fp = topo_engine.topological_summary(diagram)
prev_fp = self.fingerprint_history[-1]
drift = topo_engine.wasserstein_distance(current_diagram, prev_diagram)
```
