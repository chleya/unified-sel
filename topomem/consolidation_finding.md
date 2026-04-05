# TopoMem Consolidation 发现总结

## 实验时间
2026-04-05

## 核心发现

### 1. ChromaDB Metadata 同步 Bug ✅ 已修复

**问题**：`update_topology()` 更新了 NetworkX node 属性（`cluster_id`, `persistence_score`）但没有同步到 ChromaDB metadata。

**后果**：consolidation_pass 从 ChromaDB 读取时，`cluster_id` 默认 -1，导致所有节点被错误标记为孤立节点。

**修复**：在 `memory.py update_topology()` 里，每次计算完 cluster_labels 后调用 `self._collection.update()` 同步到 ChromaDB。

### 2. TDA H0 Clustering 问题 ⚠️ 根本性限制

**问题**：TDA H0 barcode 把每个节点都当作独立的 connected component。13 个语义相近的 Python 知识 → 13 个独立的簇。

**根因**：
- H0 只检测连通分量，在高维 embedding 空间（384维）中，每个点都是孤立点
- TDA 的强大之处是检测"环"（H1），不是检测"连接"
- 当前 topology engine 没有用到 H1 的信息来做聚类

**影响**：
- `consolidation_pass` 的 merge_candidates 逻辑基于簇 centroid similarity，在当前 TDA 输出下永远找不到多节点簇 → merge_candidates 永远为空
- `orphan_detection` 正常工作（`cluster_id=-1` 的节点会被正确识别）

**需要探索的方向**：
1. 用 DBSCAN 或 KNN 在 embedding 空间做 pre-clustering，再用 TDA 分析拓扑
2. 用 H1 barcode 的 birth-death pairs 来度量簇间相似性
3. 直接用 embedding cosine similarity 做层次聚类

## 代码改动

### memory.py
- 添加了 ChromaDB metadata 同步（`update_topology` 中）

### system.py  
- 修复了 `consolidation_pass` 的 logging（`cluster_count` → `n_clusters`）

## 关键配置参数
- `orphan_threshold`: 0.05（默认）
- `merge_centroid_threshold`: 0.92（默认）— 在单节点簇情况下无意义

## 待验证
- `persistence_score` 的实际含义和分布
- H1 barcode 在 embedding 空间的实际效果
